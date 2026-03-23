from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .cohort_runner import (
    _build_equal_weight_benchmark,
    _load_cached_payload,
    _resolve_or_build_feature_artifact,
    _resolve_or_build_label_artifact,
    _resolve_or_build_universe_artifact,
    _run_pipeline_job,
)
from .direct_strategy_runner import _resolved_backtest_cost, _summarize_backtest_artifact
from .models import Artifact, StrategyDefinition
from .service_runtime import read_frame_artifact
from .strategy_definitions import upsert_strategy_definition
from .symbol_diagnostics import (
    aggregate_symbol_diagnostic_rows,
    compute_symbol_strategy_diagnostics,
)
from .symbol_filters import build_symbol_metadata_filter_summary, select_symbols_with_metadata_filter
from .universe_selection import DEFAULT_US_EXCHANGES, MARKET_CAP_TIERS, resolve_market_cap_tier_symbols


TIME_SERIES_MOMENTUM_ORACLE_COMPARISON_SCHEMA_VERSION = 1
DEFAULT_LABEL_KS: tuple[int, ...] = (1, 2, 4, 8)
STABLE_METADATA_CATEGORICAL_COLS: tuple[str, ...] = ("sector", "industry", "country", "exchange")

TIER_LABELS: dict[str, str] = {
    "1t": "1T+ market cap",
    "100b": "100B+ market cap",
    "10b": "10B+ market cap",
}

STRATEGY_DISPLAY_NAMES: dict[str, str] = {
    "baseline": "Baseline TSMOM",
    "ml_all_data": "ML Oracle RF (All Data)",
    "ml_pre_backtest": "ML Oracle RF (Pre-2020 Train)",
}

FILTER_DISPLAY_NAMES: dict[str, str] = {
    "no_filter": "No Filter",
    "metadata_filter": "Decision-Tree Symbol Filter",
}


@dataclass(frozen=True)
class ModelVariantSpec:
    key: str
    display_name: str
    train_end_date: str


@dataclass(frozen=True)
class StrategyArtifacts:
    strategy_artifact: Artifact
    pre_backtest_artifact: Artifact
    evaluation_backtest_artifact: Artifact
    model_artifact: Artifact | None = None
    prediction_artifact: Artifact | None = None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _pct(value: Any) -> str:
    return f"{_safe_float(value) * 100.0:.2f}%"


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_rows = [dict(row) for row in list(rows or [])]
    if not serializable_rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in serializable_rows:
        for fieldname in row.keys():
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(serializable_rows)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    rule_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header_line, rule_line, *body_lines])


def _pre_backtest_end_date(backtest_start_date: str) -> str:
    start_ts = pd.Timestamp(str(backtest_start_date))
    return (start_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _variant_name(strategy_name: str, filter_name: str) -> str:
    return f"{str(strategy_name).strip()}__{str(filter_name).strip()}"


def _baseline_strategy_config() -> dict[str, Any]:
    return {
        "rebalance_freq": "M",
        "gross_exposure": 1.0,
        "selection_side": "long_short",
        "signal_combination": "direct",
        "combined_score_expr": "(1.0 + px__ret_252_d) / (1.0 + px__ret_21_d) - 1.0",
        "action_transform": "sign",
        "action_threshold": 0.0,
    }


def _model_strategy_config() -> dict[str, Any]:
    return {
        "rebalance_freq": "M",
        "gross_exposure": 1.0,
        "selection_side": "long_short",
        "signal_combination": "direct",
        "combined_score_expr": "direction",
        "action_transform": "sign",
        "action_threshold": 0.0,
    }


def _default_backtest_config(
    *,
    fee_bps: float,
    slippage_bps: float,
    short_borrow_bps_annual: float,
    execution_delay_days: int,
) -> dict[str, Any]:
    return {
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "short_borrow_bps_annual": float(short_borrow_bps_annual),
        "execution_delay_days": int(execution_delay_days),
        "turnover_half_l1": True,
        "use_lagged_weights": True,
        "min_price": 5.0,
        "min_dollar_volume": 5_000_000.0,
    }


def _default_model_fit_config(
    *,
    variant: ModelVariantSpec,
    label_ks: Sequence[int],
) -> dict[str, Any]:
    return {
        "model_name": f"tsmom_oracle_direction_rf__{variant.key}",
        "algorithm": "random_forest_classifier",
        "task_type": "classification",
        "target_col": "market_position",
        "split_ratio": 1.0,
        "train_end_date": str(variant.train_end_date or ""),
        "label_ks": [int(value) for value in list(label_ks or []) if int(value) > 0],
        "sample_weight_mode": "uniform",
        "missing_feature_policy": "complete_case",
        "params": {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 10,
            "n_jobs": -1,
        },
    }


def _strategy_definition(
    *,
    slug: str,
    name: str,
    description: str,
    config: Mapping[str, Any],
) -> StrategyDefinition:
    return upsert_strategy_definition(
        slug=slug,
        name=name,
        strategy_type="notebook_topk_v1",
        description=description,
        config=dict(config or {}),
    )


def _resolve_strategy_frame(strategy_artifact: Artifact) -> pd.DataFrame:
    return read_frame_artifact(
        strategy_artifact,
        parse_dates=False,
        normalize_symbols=True,
    )


def _strategy_row_counts(strategy_artifact: Artifact, *, allowed_symbols: Sequence[str] | None = None) -> tuple[int, int]:
    strategy_df = _resolve_strategy_frame(strategy_artifact)
    if strategy_df.empty:
        return 0, 0
    if allowed_symbols:
        allowed = {str(symbol).strip().upper() for symbol in list(allowed_symbols or []) if str(symbol).strip()}
        strategy_df = strategy_df[strategy_df["symbol"].astype(str).isin(allowed)].copy()
    rows_scored = int(len(strategy_df))
    selected_rows = int((pd.to_numeric(strategy_df.get("strategy_signal"), errors="coerce").fillna(0.0) != 0.0).sum())
    return rows_scored, selected_rows


def _run_backtest(
    *,
    strategy_artifact: Artifact,
    backtest_config: Mapping[str, Any],
    output_name: str,
    start_date: str,
    end_date: str,
    allowed_symbols: Sequence[str] | None = None,
) -> Artifact:
    config = {
        "backtest_start_date": str(start_date or ""),
        "backtest_end_date": str(end_date or ""),
        **dict(backtest_config or {}),
    }
    if allowed_symbols is not None:
        config["allowed_symbols"] = [str(symbol) for symbol in list(allowed_symbols or [])]
    return _run_pipeline_job(
        name=output_name,
        requested_job="backtest_strategy",
        config=config,
        input_ids=[int(strategy_artifact.id)],
    )


def _build_strategy_artifact(
    *,
    feature_artifact: Artifact,
    strategy_definition: StrategyDefinition,
    output_name: str,
    label_artifact: Artifact | None = None,
    prediction_artifact: Artifact | None = None,
) -> Artifact:
    config: dict[str, Any] = {
        "strategy_definition_id": int(strategy_definition.id),
        "strategy_start_date": "",
        "strategy_end_date": "",
    }
    if label_artifact is not None:
        config["label_artifact_id"] = int(label_artifact.id)
    if prediction_artifact is not None:
        config["prediction_artifact_ids"] = [int(prediction_artifact.id)]
    return _run_pipeline_job(
        name=output_name,
        requested_job="build_strategy_dataset",
        config=config,
        input_ids=[int(feature_artifact.id)],
    )


def build_profitable_symbol_targets(strategy_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for row in list(strategy_rows or []):
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        cumulative_return = _safe_float(row.get("cumulative_return"))
        targets.append(
            {
                "symbol": symbol,
                "strategy_total_return": round(cumulative_return, 8),
                "symbol_profitable": int(cumulative_return > 0.0),
            }
        )
    targets.sort(key=lambda row: str(row["symbol"]))
    return targets


def select_symbols_with_stable_metadata_filter(
    *,
    metadata_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    minimum_selected_symbols: int,
    max_depth: int,
    min_samples_leaf: int,
) -> dict[str, Any]:
    return select_symbols_with_metadata_filter(
        metadata_rows=metadata_rows,
        target_rows=target_rows,
        target_col="symbol_profitable",
        minimum_selected_symbols=int(minimum_selected_symbols),
        categorical_cols=STABLE_METADATA_CATEGORICAL_COLS,
        numeric_cols=(),
        max_depth=int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
    )


def _count_values(rows: Sequence[Mapping[str, Any]], field: str, selected_symbols: set[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol not in selected_symbols:
            continue
        value = str(row.get(field) or "Unknown").strip() or "Unknown"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _build_filter_diagnostic_row(
    *,
    universe_key: str,
    universe_label: str,
    strategy_name: str,
    metadata_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    filter_result: Mapping[str, Any],
    filter_training_time_sec: float,
) -> dict[str, Any]:
    selected_symbols = {
        str(symbol).strip().upper()
        for symbol in list(filter_result.get("selected_symbols") or [])
        if str(symbol).strip()
    }
    target_lookup = {
        str(row.get("symbol") or "").strip().upper(): dict(row)
        for row in list(target_rows or [])
        if str(row.get("symbol") or "").strip()
    }
    selected_target_rows = [target_lookup[symbol] for symbol in selected_symbols if symbol in target_lookup]
    profitable_rate = (
        sum(int(_safe_float(row.get("symbol_profitable"))) for row in selected_target_rows) / float(len(selected_target_rows))
        if selected_target_rows
        else 0.0
    )
    return {
        "universe_key": str(universe_key),
        "universe_label": str(universe_label),
        "strategy_name": str(strategy_name),
        "filter_name": "metadata_filter",
        "target_col": str(filter_result.get("target_col") or "symbol_profitable"),
        "universe_symbol_count": int(len(list(metadata_rows or []))),
        "selection_count": int(filter_result.get("selection_count") or 0),
        "selected_symbols_preview": sorted(selected_symbols)[:15],
        "trained_symbols": int(filter_result.get("trained_symbols") or 0),
        "positive_target_count": int(filter_result.get("positive_target_count") or 0),
        "positive_target_rate": round(_safe_float(filter_result.get("positive_target_rate")), 6),
        "selected_profitable_rate": round(float(profitable_rate), 6),
        "used_fallback": bool(filter_result.get("used_fallback", False)),
        "fallback_reason": str(filter_result.get("fallback_reason") or ""),
        "tree_depth": int(filter_result.get("tree_depth") or 0),
        "leaf_count": int(filter_result.get("leaf_count") or 0),
        "feature_count": int(filter_result.get("feature_count") or 0),
        "feature_columns": list(filter_result.get("feature_columns") or []),
        "top_features": list(filter_result.get("top_features") or []),
        "tree_rules": str(filter_result.get("tree_rules") or ""),
        "filter_training_time_sec": round(float(filter_training_time_sec), 6),
        "selected_sector_counts": dict(filter_result.get("selected_sector_counts") or _count_values(metadata_rows, "sector", selected_symbols)),
        "selected_industry_counts": dict(filter_result.get("selected_industry_counts") or _count_values(metadata_rows, "industry", selected_symbols)),
        "selected_country_counts": dict(filter_result.get("selected_country_counts") or _count_values(metadata_rows, "country", selected_symbols)),
        "selected_exchange_counts": dict(filter_result.get("selected_exchange_counts") or _count_values(metadata_rows, "exchange", selected_symbols)),
    }


def _build_variant_row(
    *,
    universe_key: str,
    universe_label: str,
    strategy_name: str,
    filter_name: str,
    display_name: str,
    train_end_date: str,
    backtest_start_date: str,
    backtest_end_date: str,
    strategy_artifact: Artifact,
    backtest_artifact: Artifact,
    backtest_config: Mapping[str, Any],
    selected_symbols: Sequence[str],
    selection_metadata: Mapping[str, Any] | None = None,
    model_artifact: Artifact | None = None,
    prediction_artifact: Artifact | None = None,
    filter_training_time_sec: float = 0.0,
) -> dict[str, Any]:
    runtime_summary = _summarize_backtest_artifact(backtest_artifact)
    backtest_content = dict(backtest_artifact.content or {})
    backtest_meta = dict(backtest_artifact.metadata or {})
    benchmark = _build_equal_weight_benchmark(strategy_artifact, allowed_symbols=selected_symbols)
    rows_scored, selected_rows = _strategy_row_counts(strategy_artifact, allowed_symbols=selected_symbols)
    dataset_build_seconds = _safe_float((model_artifact.metadata or {}).get("dataset_build_seconds") if model_artifact is not None else 0.0)
    fit_seconds = _safe_float((model_artifact.metadata or {}).get("fit_seconds") if model_artifact is not None else 0.0)
    score_seconds = _safe_float((prediction_artifact.metadata or {}).get("score_seconds") if prediction_artifact is not None else 0.0)
    strategy_build_seconds = _safe_float((strategy_artifact.metadata or {}).get("strategy_build_seconds"))
    backtest_seconds = _safe_float(backtest_meta.get("backtest_seconds"))
    row = {
        "universe_key": str(universe_key),
        "universe_label": str(universe_label),
        "strategy_name": str(strategy_name),
        "strategy_display_name": str(display_name),
        "filter_name": str(filter_name),
        "filter_display_name": FILTER_DISPLAY_NAMES.get(str(filter_name), str(filter_name)),
        "variant_name": _variant_name(strategy_name, filter_name),
        "train_end_date": str(train_end_date or ""),
        "backtest_start_date": str(backtest_start_date or ""),
        "backtest_end_date": str(backtest_end_date or ""),
        "trained_rows": int(_safe_float((model_artifact.content or {}).get("trained_rows") if model_artifact is not None else rows_scored)),
        "rows_scored": int(rows_scored),
        "selected_rows": int(selected_rows),
        "selected_symbol_count": int(len(list(selected_symbols or []))),
        "selected_symbols_preview": [str(symbol) for symbol in list(selected_symbols or [])[:10]],
        "final_equity": float(backtest_content.get("final_equity") or 1.0),
        "cumulative_return": float(backtest_content.get("cumulative_return") or 0.0),
        "max_drawdown": float(backtest_content.get("max_drawdown") or 0.0),
        "trades": int(backtest_content.get("trades") or 0),
        "sharpe": float(runtime_summary.get("sharpe") or 0.0),
        "avg_turnover": float(runtime_summary.get("avg_turnover") or 0.0),
        "total_turnover": float(runtime_summary.get("total_turnover") or 0.0),
        "positive_days": int(runtime_summary.get("positive_days") or 0),
        "negative_days": int(runtime_summary.get("negative_days") or 0),
        "benchmark_days": int(benchmark.get("benchmark_days") or 0),
        "benchmark_final_equity": float(benchmark.get("benchmark_final_equity") or 0.0),
        "benchmark_cumulative_return": float(benchmark.get("benchmark_cumulative_return") or 0.0),
        "benchmark_max_drawdown": float(benchmark.get("benchmark_max_drawdown") or 0.0),
        "excess_cumulative_return": round(
            float(backtest_content.get("cumulative_return") or 0.0) - float(benchmark.get("benchmark_cumulative_return") or 0.0),
            8,
        ),
        "relative_final_equity": round(
            float(backtest_content.get("final_equity") or 0.0) - float(benchmark.get("benchmark_final_equity") or 0.0),
            8,
        ),
        "dataset_build_seconds": round(float(dataset_build_seconds), 6),
        "fit_seconds": round(float(fit_seconds), 6),
        "score_seconds": round(float(score_seconds), 6),
        "strategy_build_seconds": round(float(strategy_build_seconds), 6),
        "backtest_seconds": round(float(backtest_seconds), 6),
        "filter_training_time_sec": round(float(filter_training_time_sec), 6),
        "total_runtime_seconds": round(
            float(dataset_build_seconds)
            + float(fit_seconds)
            + float(score_seconds)
            + float(strategy_build_seconds)
            + float(backtest_seconds)
            + float(filter_training_time_sec),
            6,
        ),
        "backtest_fee_bps": _resolved_backtest_cost(backtest_meta, dict(backtest_config), "fee_bps"),
        "backtest_slippage_bps": _resolved_backtest_cost(backtest_meta, dict(backtest_config), "slippage_bps"),
        "backtest_artifact_id": int(backtest_artifact.id),
        "strategy_artifact_id": int(strategy_artifact.id),
        "model_artifact_id": int(model_artifact.id) if model_artifact is not None else 0,
        "prediction_artifact_id": int(prediction_artifact.id) if prediction_artifact is not None else 0,
    }
    if selection_metadata:
        row["filter_target_col"] = str(selection_metadata.get("target_col") or "")
        row["filter_model_kind"] = str(selection_metadata.get("model_kind") or "")
        row["filter_used_fallback"] = bool(selection_metadata.get("used_fallback", False))
    return row


def _build_empty_variant_row(
    *,
    universe_key: str,
    universe_label: str,
    strategy_name: str,
    filter_name: str,
    display_name: str,
    train_end_date: str,
    backtest_start_date: str,
    backtest_end_date: str,
    strategy_artifact: Artifact,
    backtest_config: Mapping[str, Any],
    selected_symbols: Sequence[str],
    selection_metadata: Mapping[str, Any] | None = None,
    model_artifact: Artifact | None = None,
    prediction_artifact: Artifact | None = None,
    filter_training_time_sec: float = 0.0,
) -> dict[str, Any]:
    rows_scored, selected_rows = _strategy_row_counts(strategy_artifact, allowed_symbols=selected_symbols)
    dataset_build_seconds = _safe_float((model_artifact.metadata or {}).get("dataset_build_seconds") if model_artifact is not None else 0.0)
    fit_seconds = _safe_float((model_artifact.metadata or {}).get("fit_seconds") if model_artifact is not None else 0.0)
    score_seconds = _safe_float((prediction_artifact.metadata or {}).get("score_seconds") if prediction_artifact is not None else 0.0)
    strategy_build_seconds = _safe_float((strategy_artifact.metadata or {}).get("strategy_build_seconds"))
    row = {
        "universe_key": str(universe_key),
        "universe_label": str(universe_label),
        "strategy_name": str(strategy_name),
        "strategy_display_name": str(display_name),
        "filter_name": str(filter_name),
        "filter_display_name": FILTER_DISPLAY_NAMES.get(str(filter_name), str(filter_name)),
        "variant_name": _variant_name(strategy_name, filter_name),
        "train_end_date": str(train_end_date or ""),
        "backtest_start_date": str(backtest_start_date or ""),
        "backtest_end_date": str(backtest_end_date or ""),
        "trained_rows": int(_safe_float((model_artifact.content or {}).get("trained_rows") if model_artifact is not None else rows_scored)),
        "rows_scored": int(rows_scored),
        "selected_rows": int(selected_rows),
        "selected_symbol_count": int(len(list(selected_symbols or []))),
        "selected_symbols_preview": [str(symbol) for symbol in list(selected_symbols or [])[:10]],
        "final_equity": 1.0,
        "cumulative_return": 0.0,
        "max_drawdown": 0.0,
        "trades": 0,
        "sharpe": 0.0,
        "avg_turnover": 0.0,
        "total_turnover": 0.0,
        "positive_days": 0,
        "negative_days": 0,
        "benchmark_days": 0,
        "benchmark_final_equity": 0.0,
        "benchmark_cumulative_return": 0.0,
        "benchmark_max_drawdown": 0.0,
        "excess_cumulative_return": 0.0,
        "relative_final_equity": 0.0,
        "dataset_build_seconds": round(float(dataset_build_seconds), 6),
        "fit_seconds": round(float(fit_seconds), 6),
        "score_seconds": round(float(score_seconds), 6),
        "strategy_build_seconds": round(float(strategy_build_seconds), 6),
        "backtest_seconds": 0.0,
        "filter_training_time_sec": round(float(filter_training_time_sec), 6),
        "total_runtime_seconds": round(
            float(dataset_build_seconds)
            + float(fit_seconds)
            + float(score_seconds)
            + float(strategy_build_seconds)
            + float(filter_training_time_sec),
            6,
        ),
        "backtest_fee_bps": _safe_float(dict(backtest_config).get("fee_bps")),
        "backtest_slippage_bps": _safe_float(dict(backtest_config).get("slippage_bps")),
        "backtest_artifact_id": 0,
        "strategy_artifact_id": int(strategy_artifact.id),
        "model_artifact_id": int(model_artifact.id) if model_artifact is not None else 0,
        "prediction_artifact_id": int(prediction_artifact.id) if prediction_artifact is not None else 0,
        "error": "no_active_portfolio_rows",
    }
    if selection_metadata:
        row["filter_target_col"] = str(selection_metadata.get("target_col") or "")
        row["filter_model_kind"] = str(selection_metadata.get("model_kind") or "")
        row["filter_used_fallback"] = bool(selection_metadata.get("used_fallback", False))
    return row


def _build_test_symbol_rows(
    *,
    backtest_artifact: Artifact,
    universe_key: str,
    universe_label: str,
    strategy_name: str,
    filter_name: str,
    backtest_start_date: str,
    backtest_end_date: str,
) -> list[dict[str, Any]]:
    return [
        {
            **dict(row),
            "universe_key": str(universe_key),
            "universe_label": str(universe_label),
        }
        for row in compute_symbol_strategy_diagnostics(
            backtest_artifact,
            strategy_name=strategy_name,
            filter_name=filter_name,
            evaluation_scope="test",
            fold_name="single_split",
            backtest_start_date=backtest_start_date,
            backtest_end_date=backtest_end_date,
        )
    ]


def _top_count_rows(rows: Sequence[Mapping[str, Any]], *, field: str, limit: int = 8) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for row in rows:
        for name, value in dict(row.get(field) or {}).items():
            counts[str(name)] = counts.get(str(name), 0) + int(value)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: max(int(limit), 0)]


def _top_symbol_rows(rows: Sequence[Mapping[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    ordered = sorted(
        [dict(row) for row in list(rows or [])],
        key=lambda row: (
            _safe_float(row.get("sharpe")),
            _safe_float(row.get("avg_trade_return")),
            _safe_float(row.get("trade_count")),
        ),
        reverse=True,
    )
    return ordered[: max(int(limit), 0)]


def _resolve_baseline_strategy_artifacts(
    *,
    universe_key: str,
    feature_artifact: Artifact,
    backtest_start_date: str,
    backtest_end_date: str,
    pre_backtest_end_date: str,
    backtest_config: Mapping[str, Any],
    output_basename: str,
) -> StrategyArtifacts:
    strategy_definition = _strategy_definition(
        slug="tsmom-paper-style-12-1-single-split",
        name="TSMOM Paper Style 12-1",
        description="Single-split paper-style time-series momentum strategy.",
        config=_baseline_strategy_config(),
    )
    strategy_artifact = _build_strategy_artifact(
        feature_artifact=feature_artifact,
        strategy_definition=strategy_definition,
        output_name=f"{output_basename}__{universe_key}__baseline__strategy",
    )
    pre_backtest_artifact = _run_backtest(
        strategy_artifact=strategy_artifact,
        backtest_config=backtest_config,
        output_name=f"{output_basename}__{universe_key}__baseline__pre_backtest",
        start_date="",
        end_date=pre_backtest_end_date,
    )
    evaluation_backtest_artifact = _run_backtest(
        strategy_artifact=strategy_artifact,
        backtest_config=backtest_config,
        output_name=f"{output_basename}__{universe_key}__baseline__evaluation",
        start_date=backtest_start_date,
        end_date=backtest_end_date,
    )
    return StrategyArtifacts(
        strategy_artifact=strategy_artifact,
        pre_backtest_artifact=pre_backtest_artifact,
        evaluation_backtest_artifact=evaluation_backtest_artifact,
    )


def _resolve_model_strategy_artifacts(
    *,
    universe_key: str,
    variant: ModelVariantSpec,
    feature_artifact: Artifact,
    label_artifact: Artifact,
    label_ks: Sequence[int],
    backtest_start_date: str,
    backtest_end_date: str,
    pre_backtest_end_date: str,
    backtest_config: Mapping[str, Any],
    output_basename: str,
) -> StrategyArtifacts:
    model_artifact = _run_pipeline_job(
        name=f"{output_basename}__{universe_key}__{variant.key}__fit",
        requested_job="fit_classifier",
        config=_default_model_fit_config(variant=variant, label_ks=label_ks),
        input_ids=[int(label_artifact.id), int(feature_artifact.id)],
    )
    prediction_artifact = _run_pipeline_job(
        name=f"{output_basename}__{universe_key}__{variant.key}__score",
        requested_job="score_classifier",
        config={
            "score_start_date": "",
            "score_end_date": "",
            "label_artifact_id": int(label_artifact.id),
        },
        input_ids=[int(model_artifact.id), int(feature_artifact.id)],
    )
    strategy_definition = _strategy_definition(
        slug=f"tsmom-oracle-direction-rf-single-split-{variant.key}",
        name=f"TSMOM Oracle Direction RF {variant.display_name}",
        description="Single-split oracle-direction classifier policy for time-series momentum comparison.",
        config=_model_strategy_config(),
    )
    strategy_artifact = _build_strategy_artifact(
        feature_artifact=feature_artifact,
        strategy_definition=strategy_definition,
        label_artifact=label_artifact,
        prediction_artifact=prediction_artifact,
        output_name=f"{output_basename}__{universe_key}__{variant.key}__strategy",
    )
    pre_backtest_artifact = _run_backtest(
        strategy_artifact=strategy_artifact,
        backtest_config=backtest_config,
        output_name=f"{output_basename}__{universe_key}__{variant.key}__pre_backtest",
        start_date="",
        end_date=pre_backtest_end_date,
    )
    evaluation_backtest_artifact = _run_backtest(
        strategy_artifact=strategy_artifact,
        backtest_config=backtest_config,
        output_name=f"{output_basename}__{universe_key}__{variant.key}__evaluation",
        start_date=backtest_start_date,
        end_date=backtest_end_date,
    )
    return StrategyArtifacts(
        strategy_artifact=strategy_artifact,
        pre_backtest_artifact=pre_backtest_artifact,
        evaluation_backtest_artifact=evaluation_backtest_artifact,
        model_artifact=model_artifact,
        prediction_artifact=prediction_artifact,
    )


def _sort_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tier_order = {"1t": 0, "100b": 1, "10b": 2}
    variant_order = {
        "baseline__no_filter": 0,
        "baseline__metadata_filter": 1,
        "ml_all_data__no_filter": 2,
        "ml_all_data__metadata_filter": 3,
        "ml_pre_backtest__no_filter": 4,
        "ml_pre_backtest__metadata_filter": 5,
    }
    return sorted(
        [dict(row) for row in list(rows or [])],
        key=lambda row: (
            tier_order.get(str(row.get("universe_key") or ""), 999),
            variant_order.get(str(row.get("variant_name") or ""), 999),
        ),
    )


def _runtime_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        rows.append(
            {
                "universe_key": row.get("universe_key", ""),
                "universe_label": row.get("universe_label", ""),
                "strategy_name": row.get("strategy_name", ""),
                "filter_name": row.get("filter_name", ""),
                "variant_name": row.get("variant_name", ""),
                "total_runtime_seconds": row.get("total_runtime_seconds", 0.0),
                "dataset_build_seconds": row.get("dataset_build_seconds", 0.0),
                "fit_seconds": row.get("fit_seconds", 0.0),
                "score_seconds": row.get("score_seconds", 0.0),
                "strategy_build_seconds": row.get("strategy_build_seconds", 0.0),
                "backtest_seconds": row.get("backtest_seconds", 0.0),
                "filter_training_time_sec": row.get("filter_training_time_sec", 0.0),
            }
        )
    return rows


def run_time_series_momentum_oracle_comparison_experiment(
    *,
    tiers: Sequence[str],
    backtest_start_date: str = "2020-01-01",
    backtest_end_date: str = "",
    fee_bps: float = 2.0,
    slippage_bps: float = 8.0,
    short_borrow_bps_annual: float = 25.0,
    execution_delay_days: int = 1,
    country: str = "US",
    exchanges: Sequence[str] = DEFAULT_US_EXCHANGES,
    max_symbols_per_tier: int | None = None,
    minimum_filter_symbols: int = 5,
    filter_max_depth: int = 3,
    filter_min_samples_leaf: int = 3,
    label_ks: Sequence[int] = DEFAULT_LABEL_KS,
    min_profit_pct: float = 10.0,
    output_basename: str = "time_series_momentum_oracle_comparison",
    resume_existing: bool = False,
) -> dict[str, Any]:
    output_dir = Path("data") / "pipeline_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = output_dir / f"{output_basename}.json"
    summary_csv_path = output_dir / f"{output_basename}.csv"
    filter_diag_csv_path = output_dir / f"{output_basename}__filter_diagnostics.csv"
    symbol_diag_csv_path = output_dir / f"{output_basename}__symbol_diagnostics.csv"
    symbol_diag_agg_csv_path = output_dir / f"{output_basename}__symbol_diagnostics_aggregate.csv"
    runtime_csv_path = output_dir / f"{output_basename}__runtime.csv"

    if resume_existing:
        cached_payload = _load_cached_payload(
            summary_json_path,
            required_keys=("aggregate_rows", "universe_rows", "filter_diagnostic_rows"),
            schema_version=TIME_SERIES_MOMENTUM_ORACLE_COMPARISON_SCHEMA_VERSION,
        )
        if cached_payload is not None:
            cached_payload["summary_json_path"] = str(summary_json_path)
            cached_payload["summary_csv_path"] = str(summary_csv_path)
            cached_payload["filter_diagnostics_csv_path"] = str(filter_diag_csv_path)
            cached_payload["symbol_diagnostics_csv_path"] = str(symbol_diag_csv_path)
            cached_payload["symbol_diagnostics_aggregate_csv_path"] = str(symbol_diag_agg_csv_path)
            cached_payload["runtime_analysis_csv_path"] = str(runtime_csv_path)
            return cached_payload

    pre_cutoff = _pre_backtest_end_date(backtest_start_date)
    backtest_config = _default_backtest_config(
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        short_borrow_bps_annual=short_borrow_bps_annual,
        execution_delay_days=execution_delay_days,
    )
    label_config = {
        "label_ks": [int(value) for value in list(label_ks or []) if int(value) > 0],
        "min_profit_pct": float(min_profit_pct),
    }
    model_variants = (
        ModelVariantSpec(key="ml_all_data", display_name=STRATEGY_DISPLAY_NAMES["ml_all_data"], train_end_date=""),
        ModelVariantSpec(key="ml_pre_backtest", display_name=STRATEGY_DISPLAY_NAMES["ml_pre_backtest"], train_end_date=pre_cutoff),
    )

    aggregate_rows: list[dict[str, Any]] = []
    symbol_diagnostic_rows: list[dict[str, Any]] = []
    filter_diagnostic_rows: list[dict[str, Any]] = []
    universe_rows: list[dict[str, Any]] = []

    for tier_key in [str(value).strip().lower() for value in list(tiers or []) if str(value).strip()]:
        universe_label = TIER_LABELS.get(tier_key, tier_key)
        symbols = resolve_market_cap_tier_symbols(
            tier_key=tier_key,
            country=country,
            exchanges=list(exchanges),
            limit=max_symbols_per_tier,
            exclude_pooled_vehicles=True,
        )
        if not symbols:
            universe_rows.append(
                {
                    "universe_key": tier_key,
                    "universe_label": universe_label,
                    "symbol_count": 0,
                    "status": "no_symbols",
                }
            )
            continue

        universe_artifact = _resolve_or_build_universe_artifact(
            symbols=symbols,
            output_basename=f"{output_basename}__{tier_key}",
        )
        feature_artifact = _resolve_or_build_feature_artifact(
            universe_artifact=universe_artifact,
            symbols=symbols,
            feature_config={},
            output_basename=f"{output_basename}__{tier_key}",
        )
        label_artifact = _resolve_or_build_label_artifact(
            universe_artifact=universe_artifact,
            symbols=symbols,
            base_model_config=label_config,
            output_basename=f"{output_basename}__{tier_key}",
        )
        metadata_rows = build_symbol_metadata_filter_summary(
            feature_artifact,
            end_date=pre_cutoff,
            symbols=symbols,
        )

        baseline_artifacts = _resolve_baseline_strategy_artifacts(
            universe_key=tier_key,
            feature_artifact=feature_artifact,
            backtest_start_date=backtest_start_date,
            backtest_end_date=backtest_end_date,
            pre_backtest_end_date=pre_cutoff,
            backtest_config=backtest_config,
            output_basename=output_basename,
        )
        aggregate_rows.append(
            _build_variant_row(
                universe_key=tier_key,
                universe_label=universe_label,
                strategy_name="baseline",
                filter_name="no_filter",
                display_name=STRATEGY_DISPLAY_NAMES["baseline"],
                train_end_date=pre_cutoff,
                backtest_start_date=backtest_start_date,
                backtest_end_date=backtest_end_date,
                strategy_artifact=baseline_artifacts.strategy_artifact,
                backtest_artifact=baseline_artifacts.evaluation_backtest_artifact,
                backtest_config=backtest_config,
                selected_symbols=symbols,
            )
        )
        symbol_diagnostic_rows.extend(
            _build_test_symbol_rows(
                backtest_artifact=baseline_artifacts.evaluation_backtest_artifact,
                universe_key=tier_key,
                universe_label=universe_label,
                strategy_name="baseline",
                filter_name="no_filter",
                backtest_start_date=backtest_start_date,
                backtest_end_date=backtest_end_date,
            )
        )
        baseline_targets = build_profitable_symbol_targets(
            compute_symbol_strategy_diagnostics(
                baseline_artifacts.pre_backtest_artifact,
                strategy_name="baseline",
                filter_name="training_unfiltered",
                evaluation_scope="train",
                fold_name="single_split",
                backtest_end_date=pre_cutoff,
            )
        )
        started = time.perf_counter()
        baseline_filter = select_symbols_with_stable_metadata_filter(
            metadata_rows=metadata_rows,
            target_rows=baseline_targets,
            minimum_selected_symbols=minimum_filter_symbols,
            max_depth=filter_max_depth,
            min_samples_leaf=filter_min_samples_leaf,
        )
        baseline_filter_seconds = time.perf_counter() - started
        filter_diagnostic_rows.append(
            _build_filter_diagnostic_row(
                universe_key=tier_key,
                universe_label=universe_label,
                strategy_name="baseline",
                metadata_rows=metadata_rows,
                target_rows=baseline_targets,
                filter_result=baseline_filter,
                filter_training_time_sec=baseline_filter_seconds,
            )
        )
        baseline_selected_symbols = [str(symbol) for symbol in list(baseline_filter.get("selected_symbols") or []) if str(symbol).strip()]
        filtered_rows_scored, filtered_selected_rows = _strategy_row_counts(
            baseline_artifacts.strategy_artifact,
            allowed_symbols=baseline_selected_symbols,
        )
        if baseline_selected_symbols and filtered_rows_scored > 0 and filtered_selected_rows > 0:
            filtered_backtest = _run_backtest(
                strategy_artifact=baseline_artifacts.strategy_artifact,
                backtest_config=backtest_config,
                output_name=f"{output_basename}__{tier_key}__baseline__metadata_filter__evaluation",
                start_date=backtest_start_date,
                end_date=backtest_end_date,
                allowed_symbols=baseline_selected_symbols,
            )
            aggregate_rows.append(
                _build_variant_row(
                    universe_key=tier_key,
                    universe_label=universe_label,
                    strategy_name="baseline",
                    filter_name="metadata_filter",
                    display_name=STRATEGY_DISPLAY_NAMES["baseline"],
                    train_end_date=pre_cutoff,
                    backtest_start_date=backtest_start_date,
                    backtest_end_date=backtest_end_date,
                    strategy_artifact=baseline_artifacts.strategy_artifact,
                    backtest_artifact=filtered_backtest,
                    backtest_config=backtest_config,
                    selected_symbols=baseline_selected_symbols,
                    selection_metadata=baseline_filter,
                    filter_training_time_sec=baseline_filter_seconds,
                )
            )
            symbol_diagnostic_rows.extend(
                _build_test_symbol_rows(
                    backtest_artifact=filtered_backtest,
                    universe_key=tier_key,
                    universe_label=universe_label,
                    strategy_name="baseline",
                    filter_name="metadata_filter",
                    backtest_start_date=backtest_start_date,
                    backtest_end_date=backtest_end_date,
                )
            )
        else:
            aggregate_rows.append(
                _build_empty_variant_row(
                    universe_key=tier_key,
                    universe_label=universe_label,
                    strategy_name="baseline",
                    filter_name="metadata_filter",
                    display_name=STRATEGY_DISPLAY_NAMES["baseline"],
                    train_end_date=pre_cutoff,
                    backtest_start_date=backtest_start_date,
                    backtest_end_date=backtest_end_date,
                    strategy_artifact=baseline_artifacts.strategy_artifact,
                    backtest_config=backtest_config,
                    selected_symbols=baseline_selected_symbols,
                    selection_metadata=baseline_filter,
                    filter_training_time_sec=baseline_filter_seconds,
                )
            )

        for variant in model_variants:
            model_artifacts = _resolve_model_strategy_artifacts(
                universe_key=tier_key,
                variant=variant,
                feature_artifact=feature_artifact,
                label_artifact=label_artifact,
                label_ks=label_config["label_ks"],
                backtest_start_date=backtest_start_date,
                backtest_end_date=backtest_end_date,
                pre_backtest_end_date=pre_cutoff,
                backtest_config=backtest_config,
                output_basename=output_basename,
            )
            aggregate_rows.append(
                _build_variant_row(
                    universe_key=tier_key,
                    universe_label=universe_label,
                    strategy_name=variant.key,
                    filter_name="no_filter",
                    display_name=variant.display_name,
                    train_end_date=variant.train_end_date,
                    backtest_start_date=backtest_start_date,
                    backtest_end_date=backtest_end_date,
                    strategy_artifact=model_artifacts.strategy_artifact,
                    backtest_artifact=model_artifacts.evaluation_backtest_artifact,
                    backtest_config=backtest_config,
                    selected_symbols=symbols,
                    model_artifact=model_artifacts.model_artifact,
                    prediction_artifact=model_artifacts.prediction_artifact,
                )
            )
            symbol_diagnostic_rows.extend(
                _build_test_symbol_rows(
                    backtest_artifact=model_artifacts.evaluation_backtest_artifact,
                    universe_key=tier_key,
                    universe_label=universe_label,
                    strategy_name=variant.key,
                    filter_name="no_filter",
                    backtest_start_date=backtest_start_date,
                    backtest_end_date=backtest_end_date,
                )
            )
            model_targets = build_profitable_symbol_targets(
                compute_symbol_strategy_diagnostics(
                    model_artifacts.pre_backtest_artifact,
                    strategy_name=variant.key,
                    filter_name="training_unfiltered",
                    evaluation_scope="train",
                    fold_name="single_split",
                    backtest_end_date=pre_cutoff,
                )
            )
            started = time.perf_counter()
            model_filter = select_symbols_with_stable_metadata_filter(
                metadata_rows=metadata_rows,
                target_rows=model_targets,
                minimum_selected_symbols=minimum_filter_symbols,
                max_depth=filter_max_depth,
                min_samples_leaf=filter_min_samples_leaf,
            )
            model_filter_seconds = time.perf_counter() - started
            filter_diagnostic_rows.append(
                _build_filter_diagnostic_row(
                    universe_key=tier_key,
                    universe_label=universe_label,
                    strategy_name=variant.key,
                    metadata_rows=metadata_rows,
                    target_rows=model_targets,
                    filter_result=model_filter,
                    filter_training_time_sec=model_filter_seconds,
                )
            )
            selected_symbols = [str(symbol) for symbol in list(model_filter.get("selected_symbols") or []) if str(symbol).strip()]
            filtered_rows_scored, filtered_selected_rows = _strategy_row_counts(
                model_artifacts.strategy_artifact,
                allowed_symbols=selected_symbols,
            )
            if selected_symbols and filtered_rows_scored > 0 and filtered_selected_rows > 0:
                filtered_backtest = _run_backtest(
                    strategy_artifact=model_artifacts.strategy_artifact,
                    backtest_config=backtest_config,
                    output_name=f"{output_basename}__{tier_key}__{variant.key}__metadata_filter__evaluation",
                    start_date=backtest_start_date,
                    end_date=backtest_end_date,
                    allowed_symbols=selected_symbols,
                )
                aggregate_rows.append(
                    _build_variant_row(
                        universe_key=tier_key,
                        universe_label=universe_label,
                        strategy_name=variant.key,
                        filter_name="metadata_filter",
                        display_name=variant.display_name,
                        train_end_date=variant.train_end_date,
                        backtest_start_date=backtest_start_date,
                        backtest_end_date=backtest_end_date,
                        strategy_artifact=model_artifacts.strategy_artifact,
                        backtest_artifact=filtered_backtest,
                        backtest_config=backtest_config,
                        selected_symbols=selected_symbols,
                        selection_metadata=model_filter,
                        model_artifact=model_artifacts.model_artifact,
                        prediction_artifact=model_artifacts.prediction_artifact,
                        filter_training_time_sec=model_filter_seconds,
                    )
                )
                symbol_diagnostic_rows.extend(
                    _build_test_symbol_rows(
                        backtest_artifact=filtered_backtest,
                        universe_key=tier_key,
                        universe_label=universe_label,
                        strategy_name=variant.key,
                        filter_name="metadata_filter",
                        backtest_start_date=backtest_start_date,
                        backtest_end_date=backtest_end_date,
                    )
                )
            else:
                aggregate_rows.append(
                    _build_empty_variant_row(
                        universe_key=tier_key,
                        universe_label=universe_label,
                        strategy_name=variant.key,
                        filter_name="metadata_filter",
                        display_name=variant.display_name,
                        train_end_date=variant.train_end_date,
                        backtest_start_date=backtest_start_date,
                        backtest_end_date=backtest_end_date,
                        strategy_artifact=model_artifacts.strategy_artifact,
                        backtest_config=backtest_config,
                        selected_symbols=selected_symbols,
                        selection_metadata=model_filter,
                        model_artifact=model_artifacts.model_artifact,
                        prediction_artifact=model_artifacts.prediction_artifact,
                        filter_training_time_sec=model_filter_seconds,
                    )
                )

        universe_rows.append(
            {
                "universe_key": tier_key,
                "universe_label": universe_label,
                "symbol_count": int(len(symbols)),
                "min_market_cap": float(MARKET_CAP_TIERS.get(tier_key) or 0.0),
                "source_universe_artifact_id": int(universe_artifact.id),
                "source_feature_artifact_id": int(feature_artifact.id),
                "source_label_artifact_id": int(label_artifact.id),
                "status": "succeeded",
                "symbols_preview": list(symbols[:20]),
            }
        )

    aggregate_rows = _sort_summary_rows(aggregate_rows)
    symbol_diagnostics_aggregate_rows = aggregate_symbol_diagnostic_rows(
        symbol_diagnostic_rows,
        group_keys=("universe_key", "universe_label", "strategy_name", "filter_name", "symbol"),
    )
    runtime_rows = _runtime_rows(aggregate_rows)

    payload = {
        "schema_version": TIME_SERIES_MOMENTUM_ORACLE_COMPARISON_SCHEMA_VERSION,
        "mode": "time_series_momentum_oracle_comparison",
        "tiers": [str(value).strip().lower() for value in list(tiers or []) if str(value).strip()],
        "backtest_start_date": str(backtest_start_date or ""),
        "backtest_end_date": str(backtest_end_date or ""),
        "pre_backtest_end_date": str(pre_cutoff),
        "label_config": {
            "k_params": {"YE": list(label_config["label_ks"])},
            "min_profit_pct": float(min_profit_pct),
        },
        "feature_usage": {
            "artifact_scope": "full_feature_artifact",
            "missing_feature_policy": "complete_case",
            "imputation": "none",
        },
        "backtest_config": dict(backtest_config),
        "universe_rows": universe_rows,
        "aggregate_rows": aggregate_rows,
        "summary_rows": aggregate_rows,
        "symbol_diagnostic_rows": symbol_diagnostic_rows,
        "symbol_diagnostics_aggregate_rows": symbol_diagnostics_aggregate_rows,
        "filter_diagnostic_rows": filter_diagnostic_rows,
        "runtime_rows": runtime_rows,
        "summary_json_path": str(summary_json_path),
        "summary_csv_path": str(summary_csv_path),
        "filter_diagnostics_csv_path": str(filter_diag_csv_path),
        "symbol_diagnostics_csv_path": str(symbol_diag_csv_path),
        "symbol_diagnostics_aggregate_csv_path": str(symbol_diag_agg_csv_path),
        "runtime_analysis_csv_path": str(runtime_csv_path),
    }

    summary_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_rows_csv(summary_csv_path, aggregate_rows)
    _write_rows_csv(filter_diag_csv_path, filter_diagnostic_rows)
    _write_rows_csv(symbol_diag_csv_path, symbol_diagnostic_rows)
    _write_rows_csv(symbol_diag_agg_csv_path, symbol_diagnostics_aggregate_rows)
    _write_rows_csv(runtime_csv_path, runtime_rows)
    return payload


def write_time_series_momentum_oracle_comparison_report(
    *,
    report_path: Path,
    payload: Mapping[str, Any],
) -> None:
    aggregate_rows = _sort_summary_rows(list(payload.get("aggregate_rows") or payload.get("summary_rows") or []))
    filter_rows = [dict(row) for row in list(payload.get("filter_diagnostic_rows") or [])]
    symbol_rows = [dict(row) for row in list(payload.get("symbol_diagnostics_aggregate_rows") or [])]
    universe_rows = [dict(row) for row in list(payload.get("universe_rows") or [])]
    runtime_rows = [dict(row) for row in list(payload.get("runtime_rows") or [])]
    feature_usage = dict(payload.get("feature_usage") or {})

    performance_table = _markdown_table(
        ["Universe", "Strategy", "Filter", "Sharpe", "Return", "Max DD", "Trades", "Selected Symbols", "Runtime"],
        [
            [
                str(row.get("universe_label") or ""),
                STRATEGY_DISPLAY_NAMES.get(str(row.get("strategy_name") or ""), str(row.get("strategy_name") or "")),
                FILTER_DISPLAY_NAMES.get(str(row.get("filter_name") or ""), str(row.get("filter_name") or "")),
                f"{_safe_float(row.get('sharpe')):.3f}",
                _pct(row.get("cumulative_return")),
                _pct(row.get("max_drawdown")),
                str(int(_safe_float(row.get("trades")))),
                str(int(_safe_float(row.get("selected_symbol_count")))),
                f"{_safe_float(row.get('total_runtime_seconds')):.2f}s",
            ]
            for row in aggregate_rows
        ],
    ) if aggregate_rows else "_No completed variants._"

    filter_table = _markdown_table(
        ["Universe", "Strategy", "Symbols Before", "Symbols After", "Positive Target Rate", "Tree Depth", "Top Features"],
        [
            [
                str(row.get("universe_label") or ""),
                STRATEGY_DISPLAY_NAMES.get(str(row.get("strategy_name") or ""), str(row.get("strategy_name") or "")),
                str(int(_safe_float(row.get("universe_symbol_count")))),
                str(int(_safe_float(row.get("selection_count")))),
                f"{_safe_float(row.get('positive_target_rate')):.2%}",
                str(int(_safe_float(row.get("tree_depth")))),
                ", ".join(str(name) for name, _value in list(row.get("top_features") or [])[:4]) or "n/a",
            ]
            for row in filter_rows
        ],
    ) if filter_rows else "_No filter diagnostics available._"

    runtime_table = _markdown_table(
        ["Universe", "Variant", "Dataset", "Fit", "Score", "Backtest", "Filter", "Total"],
        [
            [
                str(row.get("universe_label") or ""),
                (
                    f"{STRATEGY_DISPLAY_NAMES.get(str(row.get('strategy_name') or ''), str(row.get('strategy_name') or ''))} / "
                    f"{FILTER_DISPLAY_NAMES.get(str(row.get('filter_name') or ''), str(row.get('filter_name') or ''))}"
                ),
                f"{_safe_float(row.get('dataset_build_seconds')):.2f}s",
                f"{_safe_float(row.get('fit_seconds')):.2f}s",
                f"{_safe_float(row.get('score_seconds')):.2f}s",
                f"{_safe_float(row.get('backtest_seconds')):.2f}s",
                f"{_safe_float(row.get('filter_training_time_sec')):.2f}s",
                f"{_safe_float(row.get('total_runtime_seconds')):.2f}s",
            ]
            for row in runtime_rows
        ],
    ) if runtime_rows else "_No runtime diagnostics available._"

    top_symbols_by_variant: list[str] = []
    grouped_symbols: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in symbol_rows:
        grouped_symbols.setdefault(
            (str(row.get("universe_label") or ""), str(row.get("strategy_name") or "")),
            [],
        ).append(dict(row))
    for (universe_label, strategy_name), rows in sorted(grouped_symbols.items()):
        top_rows = _top_symbol_rows(rows, limit=5)
        preview = ", ".join(
            f"{row.get('symbol')} (Sharpe {_safe_float(row.get('sharpe')):.2f}, avg trade {_pct(row.get('avg_trade_return'))})"
            for row in top_rows
        ) or "n/a"
        top_symbols_by_variant.append(
            f"- {universe_label} / {STRATEGY_DISPLAY_NAMES.get(strategy_name, strategy_name)}: {preview}"
        )
    if not top_symbols_by_variant:
        top_symbols_by_variant = ["_No symbol-level diagnostics available._"]

    top_sectors = ", ".join(f"{name} ({count})" for name, count in _top_count_rows(filter_rows, field="selected_sector_counts")) or "n/a"
    top_industries = ", ".join(f"{name} ({count})" for name, count in _top_count_rows(filter_rows, field="selected_industry_counts")) or "n/a"
    top_countries = ", ".join(f"{name} ({count})" for name, count in _top_count_rows(filter_rows, field="selected_country_counts")) or "n/a"
    top_exchanges = ", ".join(f"{name} ({count})" for name, count in _top_count_rows(filter_rows, field="selected_exchange_counts")) or "n/a"

    tree_sections: list[str] = []
    for row in filter_rows:
        rules = str(row.get("tree_rules") or "").strip()
        if not rules:
            continue
        tree_sections.extend(
            [
                f"### {row.get('universe_label', '')} / {STRATEGY_DISPLAY_NAMES.get(str(row.get('strategy_name') or ''), str(row.get('strategy_name') or ''))}",
                "",
                f"- Symbols selected: {int(_safe_float(row.get('selection_count')))}",
                f"- Positive target rate in training symbols: {_safe_float(row.get('positive_target_rate')):.2%}",
                f"- Top features: {', '.join(str(name) for name, _value in list(row.get('top_features') or [])[:6]) or 'n/a'}",
                "",
                "```text",
                rules,
                "```",
                "",
            ]
        )
    if not tree_sections:
        tree_sections = ["_No decision-tree rules were captured._", ""]

    answers: list[str] = []
    for universe in universe_rows:
        universe_label = str(universe.get("universe_label") or "")
        baseline = next(
            (row for row in aggregate_rows if str(row.get("universe_label") or "") == universe_label and str(row.get("variant_name") or "") == "baseline__no_filter"),
            {},
        )
        optimistic = next(
            (row for row in aggregate_rows if str(row.get("universe_label") or "") == universe_label and str(row.get("variant_name") or "") == "ml_all_data__no_filter"),
            {},
        )
        realistic = next(
            (row for row in aggregate_rows if str(row.get("universe_label") or "") == universe_label and str(row.get("variant_name") or "") == "ml_pre_backtest__no_filter"),
            {},
        )
        answers.append(
            f"- {universe_label}: baseline {_pct(baseline.get('cumulative_return'))} / Sharpe {_safe_float(baseline.get('sharpe')):.3f}; "
            f"ML all-data {_pct(optimistic.get('cumulative_return'))} / Sharpe {_safe_float(optimistic.get('sharpe')):.3f}; "
            f"ML pre-2020 {_pct(realistic.get('cumulative_return'))} / Sharpe {_safe_float(realistic.get('sharpe')):.3f}."
        )
    if not answers:
        answers = ["- No completed universes were available for comparison."]

    universe_summary_line = (
        "- Universes: "
        + ", ".join(
            f"{row.get('universe_label', '')} ({int(_safe_float(row.get('symbol_count')))} symbols)"
            for row in universe_rows
        )
        if universe_rows
        else "- Universes: n/a"
    )

    lines = [
        "# Time Series Momentum Oracle Comparison Report",
        "",
        "## Experiment Goal",
        "",
        "- Compare a paper-style Time Series Momentum strategy against oracle-label machine-learning variants inspired by Moskowitz, Ooi, and Pedersen (2012), \"Time Series Momentum.\"",
        "- Measure results across the platform's 1T+, 100B+, and 10B+ market-cap universes.",
        "- Learn where the strategies work by training a symbol filter from stable metadata only.",
        "",
        "## Experiment Setup",
        "",
        f"- Backtest window: `{payload.get('backtest_start_date') or ''}` through `{payload.get('backtest_end_date') or 'latest available date'}`.",
        f"- Pre-backtest history window ends: `{payload.get('pre_backtest_end_date') or ''}`.",
        universe_summary_line,
        f"- Oracle labeling: `YE={list(((payload.get('label_config') or {}).get('k_params') or {}).get('YE') or [])}`, `min_profit_pct={_safe_float(((payload.get('label_config') or {}).get('min_profit_pct'))):.1f}`.",
        "- Baseline policy: monthly paper-style 12-1 time-series momentum signal, long when the 12-1 return is positive and short when it is negative.",
        "- ML target: `market_position` from the oracle labels, so the classifier predicts true long/short directions.",
        "- ML model: `RandomForestClassifier` on the full feature artifact.",
        "- Model training variants: all-data training (optimistic, includes overlap with evaluation) and pre-backtest training only (`train_data < 2020-01-01`).",
        "- Walk-forward optimization: not used; this experiment runs a single split to avoid the runtime cost of WFO.",
        f"- Feature artifact usage: `{feature_usage.get('artifact_scope') or 'full_feature_artifact'}`.",
        f"- Missing data policy: `{feature_usage.get('missing_feature_policy') or 'complete_case'}` only for model features; rows with missing model inputs are dropped and not imputed.",
        "- Symbol filter: `DecisionTreeClassifier` on `sector`, `industry`, `country`, and `exchange` only; no numeric metadata such as market cap is used in the filter.",
        "",
        "## Performance Comparison",
        "",
        performance_table,
        "",
        "## Filter Diagnostics",
        "",
        filter_table,
        "",
        f"- Sectors most frequently selected by the decision-tree filters: {top_sectors}.",
        f"- Industries most frequently selected by the decision-tree filters: {top_industries}.",
        f"- Countries most frequently selected by the decision-tree filters: {top_countries}.",
        f"- Exchanges most frequently selected by the decision-tree filters: {top_exchanges}.",
        "",
        "## Decision Tree Summaries",
        "",
        *tree_sections,
        "## Runtime Comparison",
        "",
        runtime_table,
        "",
        "## Symbol-Level Performance",
        "",
        *top_symbols_by_variant,
        "",
        "## Answers",
        "",
        "- Does oracle-label ML outperform TSMOM? Compare the no-filter rows below by universe:",
        *answers,
        "- Which sectors, industries, countries, or exchanges look strongest? Use the filter diagnostics and decision-tree rules above; the selected cohort counts summarize the most frequent stable categories.",
        "- Can simple symbol-level filters help? Compare each strategy's `Decision-Tree Symbol Filter` row against its own `No Filter` row in the performance table.",
        "- How different are optimistic vs realistic ML training? Compare `ML Oracle RF (All Data)` against `ML Oracle RF (Pre-2020 Train)` in the same universe.",
        "",
        "## Output Artifacts",
        "",
        f"- Summary JSON: `{payload.get('summary_json_path') or ''}`",
        f"- Summary CSV: `{payload.get('summary_csv_path') or ''}`",
        f"- Filter diagnostics CSV: `{payload.get('filter_diagnostics_csv_path') or ''}`",
        f"- Symbol diagnostics CSV: `{payload.get('symbol_diagnostics_csv_path') or ''}`",
        f"- Symbol diagnostics aggregate CSV: `{payload.get('symbol_diagnostics_aggregate_csv_path') or ''}`",
        f"- Runtime CSV: `{payload.get('runtime_analysis_csv_path') or ''}`",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_LABEL_KS",
    "DEFAULT_US_EXCHANGES",
    "STABLE_METADATA_CATEGORICAL_COLS",
    "run_time_series_momentum_oracle_comparison_experiment",
    "select_symbols_with_stable_metadata_filter",
    "write_time_series_momentum_oracle_comparison_report",
]
