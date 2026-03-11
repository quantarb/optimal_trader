from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from django.utils.text import slugify

from ml.multitask import derive_oracle_cluster_labels

from .experiments import expand_model_cohort_configs
from .models import Artifact, PipelineRun, StrategyDefinition
from .services import _stable_payload_hash, execute_pipeline_run
from .strategy_definitions import upsert_strategy_definition


FIT_TO_SCORE_JOB = {
    "fit_classifier": "score_classifier",
    "fit_regressor": "score_regressor",
    "fit_autoencoder": "score_autoencoder",
    "fit_mtl": "score_mtl",
}

FIT_DEFAULTS = {
    "fit_classifier": {"target_col": "label"},
    "fit_regressor": {"target_col": "trade_return"},
    "fit_autoencoder": {"target_col": "trade_return"},
    "fit_mtl": {"target_col": "label"},
}

DEFAULT_VALIDATION_CONFIG = {
    "min_trained_rows": 50,
    "min_rows_scored": 25,
    "min_selected_rows": 10,
    "min_trades": 10,
    "min_benchmark_days": 20,
    "max_drawdown_abs": 0.75,
    "min_valid_fold_rate": 0.6,
    "max_fold_excess_std": None,
}

COHORT_SUMMARY_SCHEMA_VERSION = 3


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_cached_payload(path: Path, required_keys: Sequence[str], *, schema_version: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if any(key not in payload for key in required_keys):
        return None
    if int(payload.get("schema_version") or 0) != int(schema_version):
        return None
    return payload


def _artifact_uri_exists(artifact: Artifact | None) -> bool:
    if artifact is None:
        return False
    uri = str(artifact.uri or "").strip()
    return bool(uri) and Path(uri).exists()


def _latest_cached_artifact(
    *,
    artifact_type: str,
    metadata_key: str,
    cache_key: str,
    extra_metadata_filters: dict[str, Any] | None = None,
) -> Artifact | None:
    filters = {
        "artifact_type": artifact_type,
        "pipeline_run__status": PipelineRun.Status.SUCCEEDED,
        f"metadata__{metadata_key}": str(cache_key),
    }
    for key, value in dict(extra_metadata_filters or {}).items():
        filters[f"metadata__{key}"] = value
    for artifact in Artifact.objects.filter(**filters).select_related("pipeline_run").order_by("-created_at", "-id"):
        if _artifact_uri_exists(artifact):
            return artifact
    return None


def _resolve_or_build_universe_artifact(*, symbols: Sequence[str], output_basename: str) -> Artifact:
    normalized_symbols = [str(symbol).strip().upper() for symbol in list(symbols or []) if str(symbol).strip()]
    cache_key = _stable_payload_hash({"symbols": normalized_symbols, "filters": {}})
    cached = _latest_cached_artifact(
        artifact_type="UNIVERSE",
        metadata_key="universe_cache_key",
        cache_key=cache_key,
    )
    if cached is not None:
        return cached
    return _run_pipeline_job(
        name=f"{output_basename}-universe",
        requested_job="universe",
        config={"symbols": normalized_symbols},
    )


def _resolve_or_build_label_artifact(
    *,
    universe_artifact: Artifact,
    symbols: Sequence[str],
    base_model_config: dict[str, Any],
    output_basename: str,
) -> Artifact:
    normalized_symbols = [str(symbol).strip().upper() for symbol in list(symbols or []) if str(symbol).strip()]
    label_k_values = _collect_all_k_values(base_model_config)
    min_profit_pct = float(base_model_config.get("min_profit_pct") or 0.0)
    min_profit_decimal = max(0.0, min_profit_pct) / 100.0
    cache_key = _stable_payload_hash(
        {
            "source_universe_artifact_id": int(universe_artifact.id),
            "symbols": normalized_symbols,
            "k_params": {"YE": label_k_values or [1]},
            "min_profit_decimal": min_profit_decimal,
            "buy_col": "adj_high",
            "sell_col": "adj_low",
            "short_col": "adj_low",
            "cover_col": "adj_high",
            "dedup_mode": "exact",
        }
    )
    cached = _latest_cached_artifact(
        artifact_type="LABELS",
        metadata_key="labels_cache_key",
        cache_key=cache_key,
        extra_metadata_filters={"source_universe_artifact_id": int(universe_artifact.id)},
    )
    if cached is not None:
        return cached
    return _run_pipeline_job(
        name=f"{output_basename}-labels",
        requested_job="labels",
        config={
            "k_params": {"YE": label_k_values or [1]},
            "min_profit_pct": min_profit_pct,
        },
        input_ids=[int(universe_artifact.id)],
    )


def _resolve_or_build_feature_artifact(
    *,
    universe_artifact: Artifact,
    symbols: Sequence[str],
    feature_config: dict[str, Any] | None,
    output_basename: str,
) -> Artifact:
    normalized_symbols = [str(symbol).strip().upper() for symbol in list(symbols or []) if str(symbol).strip()]
    resolved_feature_config = dict(feature_config or {})
    cache_key = _stable_payload_hash(
        {
            "source_universe_artifact_id": int(universe_artifact.id),
            "symbols": normalized_symbols,
            "feature_config": resolved_feature_config,
        }
    )
    cached = _latest_cached_artifact(
        artifact_type="FEATURES",
        metadata_key="features_cache_key",
        cache_key=cache_key,
        extra_metadata_filters={"source_universe_artifact_id": int(universe_artifact.id)},
    )
    if cached is not None:
        return cached
    return _run_pipeline_job(
        name=f"{output_basename}-features",
        requested_job="features",
        config=resolved_feature_config,
        input_ids=[int(universe_artifact.id)],
    )


def _load_strategy_rows(strategy_artifact: Artifact) -> list[dict[str, Any]]:
    path = Path(str(strategy_artifact.uri or ""))
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _build_equal_weight_benchmark(strategy_artifact: Artifact) -> dict[str, Any]:
    rows = _load_strategy_rows(strategy_artifact)
    if not rows:
        return {
            "benchmark_days": 0,
            "benchmark_final_equity": 1.0,
            "benchmark_cumulative_return": 0.0,
            "benchmark_max_drawdown": 0.0,
        }

    df = pd.DataFrame(rows)
    if df.empty or "date" not in df.columns or "ret_1" not in df.columns:
        return {
            "benchmark_days": 0,
            "benchmark_final_equity": 1.0,
            "benchmark_cumulative_return": 0.0,
            "benchmark_max_drawdown": 0.0,
        }
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ret_1"] = pd.to_numeric(df["ret_1"], errors="coerce")
    df = df.dropna(subset=["date", "ret_1"]).copy()
    if df.empty:
        return {
            "benchmark_days": 0,
            "benchmark_final_equity": 1.0,
            "benchmark_cumulative_return": 0.0,
            "benchmark_max_drawdown": 0.0,
        }

    daily = (
        df.groupby(df["date"].dt.strftime("%Y-%m-%d"))["ret_1"]
        .mean()
        .sort_index()
    )
    equity = 1.0
    max_equity = 1.0
    max_drawdown = 0.0
    for daily_ret in daily.tolist():
        equity *= 1.0 + float(daily_ret)
        max_equity = max(max_equity, equity)
        if max_equity > 0:
            max_drawdown = min(max_drawdown, (equity / max_equity) - 1.0)
    return {
        "benchmark_days": int(len(daily)),
        "benchmark_final_equity": round(float(equity), 8),
        "benchmark_cumulative_return": round(float(equity - 1.0), 8),
        "benchmark_max_drawdown": round(float(max_drawdown), 8),
    }


def _aggregate_walk_forward_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault(str(row.get("variant_name") or ""), []).append(dict(row))

    aggregate_rows: list[dict[str, Any]] = []
    for variant_name, rows in grouped.items():
        rows = sorted(rows, key=lambda item: str(item.get("fold_name") or ""))
        strategy_equity = 1.0
        benchmark_equity = 1.0
        worst_drawdown = 0.0
        benchmark_worst_drawdown = 0.0
        valid_fold_count = 0
        invalid_fold_reasons: list[str] = []
        fold_excess_returns: list[float] = []
        fold_cumulative_returns: list[float] = []
        fold_drawdowns_abs: list[float] = []
        for row in rows:
            cumulative_return = float(row.get("cumulative_return") or 0.0)
            benchmark_cumulative_return = float(row.get("benchmark_cumulative_return") or 0.0)
            max_drawdown = float(row.get("max_drawdown") or 0.0)
            strategy_equity *= 1.0 + cumulative_return
            benchmark_equity *= 1.0 + benchmark_cumulative_return
            worst_drawdown = min(worst_drawdown, max_drawdown)
            benchmark_worst_drawdown = min(benchmark_worst_drawdown, float(row.get("benchmark_max_drawdown") or 0.0))
            fold_cumulative_returns.append(cumulative_return)
            fold_excess_returns.append(cumulative_return - benchmark_cumulative_return)
            fold_drawdowns_abs.append(abs(max_drawdown))
            if bool(row.get("passed_validity_gates")):
                valid_fold_count += 1
            else:
                for reason in list(row.get("gate_reasons") or []):
                    if str(reason) not in invalid_fold_reasons:
                        invalid_fold_reasons.append(str(reason))

        avg_dataset_build_seconds = sum(float(row.get("dataset_build_seconds") or 0.0) for row in rows) / float(len(rows))
        avg_fit_seconds = sum(float(row.get("fit_seconds") or 0.0) for row in rows) / float(len(rows))
        avg_score_seconds = sum(float(row.get("score_seconds") or 0.0) for row in rows) / float(len(rows))
        avg_strategy_build_seconds = sum(float(row.get("strategy_build_seconds") or 0.0) for row in rows) / float(len(rows))
        avg_backtest_seconds = sum(float(row.get("backtest_seconds") or 0.0) for row in rows) / float(len(rows))
        avg_total_runtime_seconds = sum(float(row.get("total_runtime_seconds") or 0.0) for row in rows) / float(len(rows))
        valid_fold_rate = (float(valid_fold_count) / float(len(rows))) if rows else 0.0
        latest_row = rows[-1] if rows else {}
        mean_fold_excess = sum(fold_excess_returns) / float(len(fold_excess_returns)) if fold_excess_returns else 0.0
        mean_fold_return = sum(fold_cumulative_returns) / float(len(fold_cumulative_returns)) if fold_cumulative_returns else 0.0
        fold_excess_std = float(pd.Series(fold_excess_returns).std(ddof=0)) if fold_excess_returns else 0.0
        fold_return_std = float(pd.Series(fold_cumulative_returns).std(ddof=0)) if fold_cumulative_returns else 0.0
        fold_drawdown_std = float(pd.Series(fold_drawdowns_abs).std(ddof=0)) if fold_drawdowns_abs else 0.0
        aggregate_rows.append(
            {
                "variant_name": variant_name,
                "fold_count": int(len(rows)),
                "valid_fold_count": int(valid_fold_count),
                "valid_fold_rate": round(float(valid_fold_rate), 8),
                "fold_names": [str(row.get("fold_name") or "") for row in rows],
                "feature_families": rows[0].get("feature_families") or [],
                "label_ks": rows[0].get("label_ks") or [],
                "oracle_cluster_scope": str(rows[0].get("oracle_cluster_scope") or "generalist"),
                "oracle_cluster_keys": list(rows[0].get("oracle_cluster_keys") or []),
                "oracle_cluster_rows": int(rows[0].get("oracle_cluster_rows") or 0),
                "walk_forward_final_equity": round(float(strategy_equity), 8),
                "walk_forward_cumulative_return": round(float(strategy_equity - 1.0), 8),
                "walk_forward_max_drawdown": round(float(worst_drawdown), 8),
                "benchmark_walk_forward_final_equity": round(float(benchmark_equity), 8),
                "benchmark_walk_forward_cumulative_return": round(float(benchmark_equity - 1.0), 8),
                "benchmark_walk_forward_max_drawdown": round(float(benchmark_worst_drawdown), 8),
                "walk_forward_excess_cumulative_return": round(float(strategy_equity - benchmark_equity), 8),
                "mean_fold_cumulative_return": round(float(mean_fold_return), 8),
                "mean_fold_excess_cumulative_return": round(float(mean_fold_excess), 8),
                "fold_excess_cumulative_return_std": round(float(fold_excess_std), 8),
                "fold_cumulative_return_std": round(float(fold_return_std), 8),
                "fold_drawdown_abs_std": round(float(fold_drawdown_std), 8),
                "min_fold_excess_cumulative_return": round(float(min(fold_excess_returns) if fold_excess_returns else 0.0), 8),
                "max_fold_excess_cumulative_return": round(float(max(fold_excess_returns) if fold_excess_returns else 0.0), 8),
                "avg_dataset_build_seconds": round(float(avg_dataset_build_seconds), 6),
                "avg_fit_seconds": round(float(avg_fit_seconds), 6),
                "avg_score_seconds": round(float(avg_score_seconds), 6),
                "avg_strategy_build_seconds": round(float(avg_strategy_build_seconds), 6),
                "avg_backtest_seconds": round(float(avg_backtest_seconds), 6),
                "avg_total_runtime_seconds": round(float(avg_total_runtime_seconds), 6),
                "invalid_fold_reasons": invalid_fold_reasons,
                "model_artifact_id": int(latest_row.get("model_artifact_id") or 0),
                "prediction_artifact_id": int(latest_row.get("prediction_artifact_id") or 0),
                "strategy_artifact_id": int(latest_row.get("strategy_artifact_id") or 0),
                "backtest_artifact_id": int(latest_row.get("backtest_artifact_id") or 0),
            }
        )
    aggregate_rows.sort(
        key=lambda row: (
            float(row.get("mean_fold_excess_cumulative_return") or 0.0),
            -float(row.get("fold_excess_cumulative_return_std") or 0.0),
            float(row.get("valid_fold_rate") or 0.0),
            float(row.get("walk_forward_final_equity") or 0.0),
        ),
        reverse=True,
    )
    return aggregate_rows


def _evaluate_variant_gates(row: dict[str, Any], validation_config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULT_VALIDATION_CONFIG)
    cfg.update(dict(validation_config or {}))
    reasons: list[str] = []
    if int(row.get("trained_rows") or 0) < int(cfg["min_trained_rows"]):
        reasons.append("trained_rows_below_min")
    if int(row.get("rows_scored") or 0) < int(cfg["min_rows_scored"]):
        reasons.append("rows_scored_below_min")
    if int(row.get("selected_rows") or 0) < int(cfg["min_selected_rows"]):
        reasons.append("selected_rows_below_min")
    if int(row.get("trades") or 0) < int(cfg["min_trades"]):
        reasons.append("trades_below_min")
    if int(row.get("benchmark_days") or 0) < int(cfg["min_benchmark_days"]):
        reasons.append("benchmark_days_below_min")
    if abs(float(row.get("max_drawdown") or 0.0)) > float(cfg["max_drawdown_abs"]):
        reasons.append("drawdown_above_max")
    return {
        "passed_validity_gates": len(reasons) == 0,
        "gate_reasons": reasons,
    }


def _apply_walk_forward_gates(aggregate_rows: list[dict[str, Any]], validation_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = dict(DEFAULT_VALIDATION_CONFIG)
    cfg.update(dict(validation_config or {}))
    out: list[dict[str, Any]] = []
    for row in aggregate_rows:
        reasons = list(row.get("invalid_fold_reasons") or [])
        if float(row.get("valid_fold_rate") or 0.0) < float(cfg["min_valid_fold_rate"]):
            reasons.append("valid_fold_rate_below_min")
        max_fold_excess_std = cfg.get("max_fold_excess_std")
        if max_fold_excess_std not in (None, ""):
            if float(row.get("fold_excess_cumulative_return_std") or 0.0) > float(max_fold_excess_std):
                reasons.append("fold_excess_std_above_max")
        item = dict(row)
        item["passed_stability_gates"] = len(reasons) == 0
        item["stability_gate_reasons"] = list(dict.fromkeys(reasons))
        out.append(item)
    return out


def _validate_walk_forward_fold(fold: dict[str, Any]) -> None:
    fold_name = str(fold.get("name") or fold.get("fold_name") or "fold").strip()
    train_end = str(fold.get("train_end_date") or "").strip()
    test_start = str(fold.get("backtest_start_date") or "").strip()
    test_end = str(fold.get("backtest_end_date") or "").strip()
    if not train_end or not test_start or not test_end:
        raise ValueError(f"Fold {fold_name!r} is missing train_end_date/backtest_start_date/backtest_end_date.")
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)
    if train_end_ts >= test_start_ts:
        raise ValueError(f"Fold {fold_name!r} violates chronology: train_end_date must be before backtest_start_date.")
    if test_start_ts > test_end_ts:
        raise ValueError(f"Fold {fold_name!r} violates chronology: backtest_start_date must be <= backtest_end_date.")


def _run_pipeline_job(
    *,
    name: str,
    requested_job: str,
    config: dict[str, Any] | None = None,
    input_ids: Sequence[int] | None = None,
) -> Artifact:
    run = PipelineRun.objects.create(
        name=name,
        requested_job=requested_job,
        mode=PipelineRun.Mode.STRICT,
        status=PipelineRun.Status.PENDING,
        config=dict(config or {}),
    )
    return execute_pipeline_run(
        pipeline_run=run,
        target_job=requested_job,
        mode="strict",
        config=dict(config or {}),
        input_artifact_ids=list(input_ids or []),
    )


def _collect_all_k_values(base_config: dict[str, Any]) -> list[int]:
    out: list[int] = []
    for value in list(base_config.get("label_ks") or []):
        try:
            parsed = int(value)
        except Exception:
            continue
        if parsed > 0 and parsed not in out:
            out.append(parsed)
    for group in list(base_config.get("label_k_groups") or []):
        for value in list(group or []):
            try:
                parsed = int(value)
            except Exception:
                continue
            if parsed > 0 and parsed not in out:
                out.append(parsed)
    raw_k = base_config.get("label_k")
    try:
        parsed = int(raw_k)
        if parsed > 0 and parsed not in out:
            out.append(parsed)
    except Exception:
        pass
    return sorted(out)


def _resolve_top_oracle_cluster_groups(
    *,
    label_artifact: Artifact,
    train_end_date: str,
    top_n: int,
    min_rows: int,
    label_ks: Sequence[int] = (),
    min_abs_trade_return_pct: float | None = None,
    max_hold_days: int | None = None,
) -> list[list[str]]:
    uri = str(label_artifact.uri or "").strip()
    path = Path(uri)
    if not uri or not path.exists():
        return []
    try:
        label_df = pd.read_csv(path)
    except Exception:
        return []
    if label_df.empty or not {"trade_return", "hold_days"}.issubset(set(label_df.columns)):
        return []
    if "date" in label_df.columns:
        label_df["date"] = pd.to_datetime(label_df["date"], errors="coerce")
        label_df = label_df.dropna(subset=["date"])
        if train_end_date:
            label_df = label_df[label_df["date"] <= pd.Timestamp(str(train_end_date))].copy()
    selected_ks = {int(value) for value in list(label_ks or []) if int(value) > 0}
    if selected_ks and "k" in label_df.columns:
        label_df = label_df[pd.to_numeric(label_df["k"], errors="coerce").isin(selected_ks)].copy()
    if min_abs_trade_return_pct not in (None, "") and "trade_return" in label_df.columns:
        min_abs_value = max(0.0, float(min_abs_trade_return_pct) / 100.0)
        label_df["trade_return"] = pd.to_numeric(label_df["trade_return"], errors="coerce")
        label_df = label_df[label_df["trade_return"].abs() >= min_abs_value].copy()
    if max_hold_days not in (None, "") and "hold_days" in label_df.columns:
        max_hold_value = max(1, int(max_hold_days))
        label_df["hold_days"] = pd.to_numeric(label_df["hold_days"], errors="coerce")
        label_df = label_df[label_df["hold_days"].fillna(max_hold_value + 1) <= max_hold_value].copy()
    if label_df.empty:
        return []
    label_df["oracle_cluster_key"] = derive_oracle_cluster_labels(label_df)
    grouped = (
        label_df.groupby("oracle_cluster_key", observed=True)
        .agg(
            rows=("oracle_cluster_key", "size"),
            avg_trade_return=("trade_return", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["rows"] >= max(int(min_rows), 1)].copy()
    if grouped.empty:
        return []
    grouped = grouped.sort_values(["rows", "avg_trade_return"], ascending=[False, False]).head(max(int(top_n), 0))
    return [[str(value)] for value in grouped["oracle_cluster_key"].astype(str).tolist()]


def _expand_cluster_specialist_variants(
    *,
    variant_configs: list[dict[str, Any]],
    base_model_config: dict[str, Any],
    label_artifact: Artifact,
    train_end_date: str,
    fit_job: str,
) -> list[dict[str, Any]]:
    mode = str(base_model_config.get("oracle_cluster_mode") or "").strip().lower()
    if mode not in {"top_clusters", "specialist_top_clusters"} or str(fit_job).strip() not in {"fit_mtl", "fit_regressor"}:
        return [dict(variant) for variant in variant_configs]
    include_generalist = bool(base_model_config.get("include_cluster_generalist", True))
    out: list[dict[str, Any]] = []
    for variant in variant_configs:
        generalist = dict(variant)
        generalist["oracle_cluster_scope"] = "generalist"
        generalist["oracle_cluster_keys"] = []
        if include_generalist:
            out.append(generalist)
        cluster_groups = _resolve_top_oracle_cluster_groups(
            label_artifact=label_artifact,
            train_end_date=str(train_end_date or ""),
            top_n=int(base_model_config.get("oracle_cluster_top_n") or 0),
            min_rows=int(base_model_config.get("oracle_cluster_min_rows") or 1),
            label_ks=list(variant.get("label_ks") or []),
            min_abs_trade_return_pct=variant.get("min_abs_trade_return_pct", base_model_config.get("min_abs_trade_return_pct")),
            max_hold_days=variant.get("max_hold_days", base_model_config.get("max_hold_days")),
        )
        for cluster_group in cluster_groups:
            specialist = dict(variant)
            specialist["oracle_cluster_scope"] = "specialist"
            specialist["oracle_cluster_keys"] = list(cluster_group)
            cluster_slug = slugify(cluster_group[0])[:48] or f"cluster-{len(out) + 1}"
            model_name = str(specialist.get("model_name") or "model").strip() or "model"
            specialist["model_name"] = f"{model_name}__cluster__{cluster_slug}"
            out.append(specialist)
    return out or [dict(variant) for variant in variant_configs]


def run_model_cohort_backtests(
    *,
    symbols: Sequence[str],
    fit_job: str,
    base_model_config: dict[str, Any],
    train_end_date: str,
    backtest_start_date: str,
    backtest_end_date: str,
    universe_artifact: Artifact | None = None,
    label_artifact: Artifact | None = None,
    feature_artifact: Artifact | None = None,
    feature_config: dict[str, Any] | None = None,
    strategy_definition: StrategyDefinition | None = None,
    strategy_definition_slug: str = "mag7-cohort-backtest",
    strategy_definition_name: str = "MAG7 Cohort Backtest Strategy",
    strategy_config: dict[str, Any] | None = None,
    validation_config: dict[str, Any] | None = None,
    transaction_cost_bps: float = 10.0,
    backtest_config: dict[str, Any] | None = None,
    output_basename: str = "mag7_cohort_backtest_summary",
    resume_existing: bool = False,
) -> dict[str, Any]:
    output_dir = Path("data") / "pipeline_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{output_basename}.json"
    csv_path = output_dir / f"{output_basename}.csv"
    if resume_existing:
        cached_payload = _load_cached_payload(
            json_path,
            required_keys=("summary_rows", "base_artifacts"),
            schema_version=COHORT_SUMMARY_SCHEMA_VERSION,
        )
        if cached_payload is not None:
            cached_payload["summary_json_path"] = str(json_path)
            cached_payload["summary_csv_path"] = str(csv_path)
            return cached_payload

    fit_job_value = str(fit_job).strip()
    if fit_job_value not in FIT_TO_SCORE_JOB:
        raise ValueError(f"Unsupported fit job for cohort backtest: {fit_job!r}")
    score_job = FIT_TO_SCORE_JOB[fit_job_value]

    if strategy_definition is None:
        strategy_definition = upsert_strategy_definition(
            slug=strategy_definition_slug,
            name=strategy_definition_name,
            strategy_type="notebook_topk_v1",
            description="Strategy used by the cohort backtest runner.",
            config=dict(strategy_config or {}),
        )

    if universe_artifact is None and (label_artifact is None or feature_artifact is None):
        universe_artifact = _resolve_or_build_universe_artifact(symbols=symbols, output_basename=output_basename)
    if label_artifact is None:
        if universe_artifact is None:
            raise ValueError("A universe artifact is required when label_artifact is not provided.")
        label_artifact = _resolve_or_build_label_artifact(
            universe_artifact=universe_artifact,
            symbols=symbols,
            base_model_config=base_model_config,
            output_basename=output_basename,
        )
    if feature_artifact is None:
        if universe_artifact is None:
            raise ValueError("A universe artifact is required when feature_artifact is not provided.")
        feature_artifact = _resolve_or_build_feature_artifact(
            universe_artifact=universe_artifact,
            symbols=symbols,
            feature_config=feature_config,
            output_basename=output_basename,
        )

    variant_configs = expand_model_cohort_configs(
        base_config=base_model_config,
        feature_artifact=feature_artifact,
        label_artifact=label_artifact,
    )
    variant_configs = _expand_cluster_specialist_variants(
        variant_configs=variant_configs,
        base_model_config=base_model_config,
        label_artifact=label_artifact,
        train_end_date=str(train_end_date or ""),
        fit_job=fit_job_value,
    )

    summary_rows: list[dict[str, Any]] = []
    variant_outputs: list[dict[str, Any]] = []
    failed_variants: list[dict[str, Any]] = []
    resolved_backtest_config = dict(backtest_config or {})
    resolved_backtest_config.setdefault("transaction_cost_bps", float(transaction_cost_bps))
    for index, variant in enumerate(variant_configs, start=1):
        variant_name = str(variant.get("model_name") or f"variant_{index}").strip()
        variant_slug = slugify(variant_name) or f"variant-{index}"
        fit_config = dict(FIT_DEFAULTS.get(fit_job_value, {}))
        fit_config.update(dict(base_model_config))
        fit_config.update(dict(variant))
        fit_config["train_end_date"] = str(train_end_date)

        try:
            model_artifact = _run_pipeline_job(
                name=f"{output_basename}-{variant_slug}-fit",
                requested_job=fit_job_value,
                config=fit_config,
                input_ids=[int(feature_artifact.id), int(label_artifact.id)],
            )
            score_artifact = _run_pipeline_job(
                name=f"{output_basename}-{variant_slug}-score",
                requested_job=score_job,
                config={
                    "score_start_date": str(backtest_start_date),
                    "score_end_date": str(backtest_end_date),
                    "label_artifact_id": int(label_artifact.id),
                },
                input_ids=[int(model_artifact.id), int(feature_artifact.id)],
            )
            strategy_artifact = _run_pipeline_job(
                name=f"{output_basename}-{variant_slug}-strategy",
                requested_job="build_strategy_dataset",
                config={
                    "strategy_definition_id": int(strategy_definition.id),
                    "label_artifact_id": int(label_artifact.id),
                    "prediction_artifact_ids": [int(score_artifact.id)],
                    "strategy_start_date": str(backtest_start_date),
                    "strategy_end_date": str(backtest_end_date),
                },
                input_ids=[int(feature_artifact.id)],
            )
            backtest_artifact = _run_pipeline_job(
                name=f"{output_basename}-{variant_slug}-backtest",
                requested_job="backtest_strategy",
                config={
                    "backtest_start_date": str(backtest_start_date),
                    "backtest_end_date": str(backtest_end_date),
                    **resolved_backtest_config,
                },
                input_ids=[int(strategy_artifact.id)],
            )
        except Exception as exc:
            failed_variants.append(
                {
                    "variant_name": variant_name,
                    "fit_job": fit_job_value,
                    "feature_families": list(variant.get("feature_families") or []),
                    "label_ks": list(variant.get("label_ks") or []),
                    "oracle_cluster_scope": str(variant.get("oracle_cluster_scope") or "generalist"),
                    "oracle_cluster_keys": list(variant.get("oracle_cluster_keys") or []),
                    "error": str(exc),
                }
            )
            continue

        model_meta = dict(model_artifact.metadata or {})
        score_meta = dict(score_artifact.metadata or {})
        strategy_meta = dict(strategy_artifact.metadata or {})
        backtest_meta = dict(backtest_artifact.metadata or {})
        backtest_content = dict(backtest_artifact.content or {})
        benchmark = _build_equal_weight_benchmark(strategy_artifact)
        row = {
            "variant_name": variant_name,
            "fit_job": fit_job_value,
            "score_job": score_job,
            "feature_families": list(model_meta.get("feature_families") or []),
            "label_ks": list(model_meta.get("label_ks") or []),
            "dataset_build_seconds": float(model_meta.get("dataset_build_seconds") or 0.0),
            "fit_seconds": float(model_meta.get("fit_seconds") or 0.0),
            "score_seconds": float(score_meta.get("score_seconds") or 0.0),
            "strategy_build_seconds": float(strategy_meta.get("strategy_build_seconds") or 0.0),
            "backtest_seconds": float(backtest_meta.get("backtest_seconds") or 0.0),
            "coverage_start_date": str(model_meta.get("coverage_start_date") or ""),
            "coverage_end_date": str(model_meta.get("coverage_end_date") or ""),
            "coverage_rows": int(model_meta.get("coverage_rows") or 0),
            "label_rows_after_filters": int(model_meta.get("label_rows_after_filters") or 0),
            "oracle_cluster_scope": str(model_meta.get("oracle_cluster_scope") or fit_config.get("oracle_cluster_scope") or "generalist"),
            "oracle_cluster_keys": list(model_meta.get("oracle_cluster_keys") or fit_config.get("oracle_cluster_keys") or []),
            "oracle_cluster_rows": int(model_meta.get("cluster_rows_after_filter") or 0),
            "trained_rows": int(model_artifact.content.get("trained_rows") or 0),
            "rows_scored": int(score_meta.get("rows_scored") or 0),
            "selected_rows": int(strategy_artifact.content.get("selected_rows") or 0),
            "final_equity": float(backtest_content.get("final_equity") or 0.0),
            "cumulative_return": float(backtest_content.get("cumulative_return") or 0.0),
            "max_drawdown": float(backtest_content.get("max_drawdown") or 0.0),
            "trades": int(backtest_content.get("trades") or 0),
            "benchmark_days": int(benchmark.get("benchmark_days") or 0),
            "benchmark_final_equity": float(benchmark.get("benchmark_final_equity") or 0.0),
            "benchmark_cumulative_return": float(benchmark.get("benchmark_cumulative_return") or 0.0),
            "benchmark_max_drawdown": float(benchmark.get("benchmark_max_drawdown") or 0.0),
            "backtest_fee_bps": float((backtest_meta.get("backtest_config") or {}).get("fee_bps") or resolved_backtest_config.get("fee_bps") or 0.0),
            "backtest_slippage_bps": float((backtest_meta.get("backtest_config") or {}).get("slippage_bps") or resolved_backtest_config.get("slippage_bps") or 0.0),
            "excess_cumulative_return": round(
                float(backtest_content.get("cumulative_return") or 0.0) - float(benchmark.get("benchmark_cumulative_return") or 0.0),
                8,
            ),
            "relative_final_equity": round(
                float(backtest_content.get("final_equity") or 0.0) - float(benchmark.get("benchmark_final_equity") or 0.0),
                8,
            ),
            "model_artifact_id": int(model_artifact.id),
            "prediction_artifact_id": int(score_artifact.id),
            "strategy_artifact_id": int(strategy_artifact.id),
            "backtest_artifact_id": int(backtest_artifact.id),
        }
        row["total_runtime_seconds"] = round(
            float(row["dataset_build_seconds"])
            + float(row["fit_seconds"])
            + float(row["score_seconds"])
            + float(row["strategy_build_seconds"])
            + float(row["backtest_seconds"]),
            6,
        )
        row.update(_evaluate_variant_gates(row, validation_config=validation_config))
        summary_rows.append(row)
        variant_outputs.append(
            {
                "variant": row,
                "artifacts": {
                    "model": int(model_artifact.id),
                    "prediction": int(score_artifact.id),
                    "strategy": int(strategy_artifact.id),
                    "backtest": int(backtest_artifact.id),
                },
            }
        )

    payload = {
        "schema_version": COHORT_SUMMARY_SCHEMA_VERSION,
        "base_artifacts": {
            "universe": int(universe_artifact.id) if universe_artifact is not None else 0,
            "labels": int(label_artifact.id),
            "features": int(feature_artifact.id),
        },
        "strategy_definition": {
            "id": int(strategy_definition.id),
            "name": str(strategy_definition.name),
            "slug": str(strategy_definition.slug),
            "strategy_type": str(strategy_definition.strategy_type),
        },
        "fit_job": fit_job_value,
        "score_job": score_job,
        "validation_config": dict(DEFAULT_VALIDATION_CONFIG | dict(validation_config or {})),
        "backtest_config": resolved_backtest_config,
        "variants": variant_outputs,
        "failed_variants": failed_variants,
        "summary_rows": summary_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_rows_csv(csv_path, summary_rows)
    payload["summary_json_path"] = str(json_path)
    payload["summary_csv_path"] = str(csv_path)
    return payload


def run_walk_forward_model_cohort_backtests(
    *,
    symbols: Sequence[str],
    fit_job: str,
    base_model_config: dict[str, Any],
    folds: Sequence[dict[str, Any]],
    universe_artifact: Artifact | None = None,
    label_artifact: Artifact | None = None,
    feature_artifact: Artifact | None = None,
    feature_config: dict[str, Any] | None = None,
    strategy_definition: StrategyDefinition | None = None,
    strategy_definition_slug: str = "walk-forward-cohort-backtest",
    strategy_definition_name: str = "Walk Forward Cohort Backtest Strategy",
    strategy_config: dict[str, Any] | None = None,
    validation_config: dict[str, Any] | None = None,
    transaction_cost_bps: float = 10.0,
    backtest_config: dict[str, Any] | None = None,
    output_basename: str = "walk_forward_cohort_summary",
    resume_existing: bool = False,
) -> dict[str, Any]:
    if not folds:
        raise ValueError("run_walk_forward_model_cohort_backtests requires at least one fold.")
    for fold in folds:
        _validate_walk_forward_fold(dict(fold))

    output_dir = Path("data") / "pipeline_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{output_basename}.json"
    csv_path = output_dir / f"{output_basename}.csv"
    if resume_existing:
        cached_payload = _load_cached_payload(
            json_path,
            required_keys=("folds", "aggregate_rows", "summary_rows"),
            schema_version=COHORT_SUMMARY_SCHEMA_VERSION,
        )
        if cached_payload is not None:
            cached_payload["summary_json_path"] = str(json_path)
            cached_payload["summary_csv_path"] = str(csv_path)
            return cached_payload

    all_summary_rows: list[dict[str, Any]] = []
    fold_outputs: list[dict[str, Any]] = []
    shared_kwargs = {
        "symbols": symbols,
        "fit_job": fit_job,
        "base_model_config": dict(base_model_config or {}),
        "universe_artifact": universe_artifact,
        "label_artifact": label_artifact,
        "feature_artifact": feature_artifact,
        "feature_config": dict(feature_config or {}),
        "strategy_definition": strategy_definition,
        "strategy_definition_slug": strategy_definition_slug,
        "strategy_definition_name": strategy_definition_name,
        "strategy_config": dict(strategy_config or {}),
        "validation_config": dict(validation_config or {}),
        "transaction_cost_bps": float(transaction_cost_bps),
        "backtest_config": dict(backtest_config or {}),
    }

    for index, fold in enumerate(folds, start=1):
        fold_name = str(fold.get("name") or fold.get("fold_name") or f"fold_{index}").strip()
        fold_summary = run_model_cohort_backtests(
            **shared_kwargs,
            train_end_date=str(fold.get("train_end_date") or ""),
            backtest_start_date=str(fold.get("backtest_start_date") or ""),
            backtest_end_date=str(fold.get("backtest_end_date") or ""),
            output_basename=f"{output_basename}__{slugify(fold_name) or index}",
            resume_existing=resume_existing,
        )
        for row in list(fold_summary.get("summary_rows") or []):
            item = dict(row)
            item["fold_name"] = fold_name
            item["train_end_date"] = str(fold.get("train_end_date") or "")
            item["backtest_start_date"] = str(fold.get("backtest_start_date") or "")
            item["backtest_end_date"] = str(fold.get("backtest_end_date") or "")
            all_summary_rows.append(item)
        fold_outputs.append(
            {
                "fold_name": fold_name,
                "train_end_date": str(fold.get("train_end_date") or ""),
                "backtest_start_date": str(fold.get("backtest_start_date") or ""),
                "backtest_end_date": str(fold.get("backtest_end_date") or ""),
                "summary_json_path": str(fold_summary.get("summary_json_path") or ""),
                "summary_csv_path": str(fold_summary.get("summary_csv_path") or ""),
            }
        )
        if universe_artifact is None:
            universe_id = int((fold_summary.get("base_artifacts") or {}).get("universe") or 0)
            universe_artifact = Artifact.objects.filter(pk=universe_id).first()
        if label_artifact is None:
            label_id = int((fold_summary.get("base_artifacts") or {}).get("labels") or 0)
            label_artifact = Artifact.objects.filter(pk=label_id).first()
        if feature_artifact is None:
            feature_id = int((fold_summary.get("base_artifacts") or {}).get("features") or 0)
            feature_artifact = Artifact.objects.filter(pk=feature_id).first()
        if strategy_definition is None:
            strategy_id = int((fold_summary.get("strategy_definition") or {}).get("id") or 0)
            strategy_definition = StrategyDefinition.objects.filter(pk=strategy_id).first()

    aggregate_rows = _apply_walk_forward_gates(
        _aggregate_walk_forward_rows(all_summary_rows),
        validation_config=validation_config,
    )
    payload = {
        "schema_version": COHORT_SUMMARY_SCHEMA_VERSION,
        "fit_job": str(fit_job),
        "base_artifacts": {
            "universe": int(universe_artifact.id) if universe_artifact is not None else 0,
            "labels": int(label_artifact.id) if label_artifact is not None else 0,
            "features": int(feature_artifact.id) if feature_artifact is not None else 0,
        },
        "strategy_definition": {
            "id": int(strategy_definition.id) if strategy_definition is not None else 0,
            "name": str(strategy_definition.name) if strategy_definition is not None else "",
            "slug": str(strategy_definition.slug) if strategy_definition is not None else "",
            "strategy_type": str(strategy_definition.strategy_type) if strategy_definition is not None else "",
        },
        "folds": fold_outputs,
        "summary_rows": all_summary_rows,
        "aggregate_rows": aggregate_rows,
        "validation_config": dict(DEFAULT_VALIDATION_CONFIG | dict(validation_config or {})),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_rows_csv(csv_path, aggregate_rows)
    payload["summary_json_path"] = str(json_path)
    payload["summary_csv_path"] = str(csv_path)
    return payload
