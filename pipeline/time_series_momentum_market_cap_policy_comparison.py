from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from pipeline.performance import PerformanceTracer
from tools.performance_analysis.models import RuntimeProfileReport
from tools.performance_analysis.reports.hotspot_report import runtime_hotspots_markdown
from tools.performance_analysis.utils.report_utils import utc_timestamp, write_json, write_markdown

from .cohort_runner import (
    _aggregate_walk_forward_rows,
    _apply_walk_forward_gates,
    _build_equal_weight_benchmark,
    _evaluate_variant_gates,
    _load_cached_payload,
    _resolve_or_build_feature_artifact,
    _resolve_or_build_label_artifact,
    _resolve_or_build_universe_artifact,
    _run_pipeline_job,
    run_model_cohort_backtests,
)
from .direct_strategy_runner import (
    _resolved_backtest_cost,
    _summarize_backtest_artifact,
    _summarize_walk_forward_metrics,
    run_direct_feature_strategy_backtests,
)
from .models import Artifact
from .service_runtime import read_frame_artifact
from .symbol_diagnostics import (
    aggregate_symbol_diagnostic_rows,
    compute_symbol_buy_hold_diagnostics,
    compute_symbol_strategy_diagnostics,
)
from .symbol_filters import build_symbol_metadata_filter_summary, select_symbols_with_metadata_filter
from .time_series_momentum_policy_comparison import build_yearly_folds
from .universe_selection import DEFAULT_US_EXCHANGES, MARKET_CAP_TIERS, resolve_market_cap_tier_symbols


MARKET_CAP_POLICY_COMPARISON_SCHEMA_VERSION = 1

TIER_LABELS: dict[str, str] = {
    "1t": "1T+ market cap",
    "100b": "100B+ market cap",
    "10b": "10B+ market cap",
}

FILTER_VARIANTS: tuple[tuple[str, str], ...] = (
    ("profitable_filter", "symbol_profitable"),
    ("beats_buy_hold_filter", "beats_buy_hold"),
)

FILTER_DISPLAY_NAMES: dict[str, str] = {
    "no_filter": "No Filter",
    "profitable_filter": "Profitable Filter",
    "beats_buy_hold_filter": "Beats Buy-and-Hold Filter",
}


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
    rows = [dict(row) for row in rows]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for fieldname in row.keys():
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        "action_transform": "sign",
        "action_threshold": 0.0,
    }


def _default_model_config() -> dict[str, Any]:
    return {
        "model_name": "tsmom_oracle_trade_rf",
        "algorithm": "random_forest_regressor",
        "task_type": "regression",
        "target_col": "trade_return",
        "split_ratio": 1.0,
        "label_ks": [1],
        "min_profit_pct": 0.0,
        "sample_weight_mode": "trade_return_abs",
        "params": {
            "n_estimators": 40,
            "max_depth": 4,
            "min_samples_leaf": 10,
            "n_jobs": -1,
        },
    }


def _default_validation_config() -> dict[str, Any]:
    return {
        "min_trained_rows": 200,
        "min_rows_scored": 100,
        "min_selected_rows": 20,
        "min_trades": 20,
        "min_benchmark_days": 50,
        "min_valid_fold_rate": 0.6,
        "max_fold_excess_std": 0.75,
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


def _single_summary_row(summary: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    rows = [dict(row) for row in list(summary.get("summary_rows") or [])]
    if not rows:
        failed = list(summary.get("failed_variants") or [])
        detail = failed[0].get("error") if failed else "no summary rows produced"
        raise ValueError(f"{label} did not produce a usable summary row: {detail}")
    return dict(rows[0])


def _resolve_artifact(artifact_id: Any, *, label: str) -> Artifact:
    artifact = Artifact.objects.filter(pk=int(artifact_id or 0)).first()
    if artifact is None:
        raise ValueError(f"{label} artifact #{artifact_id} was not found.")
    return artifact


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
    selected_rows = int((strategy_df["strategy_signal"].fillna(0).astype(float) != 0.0).sum()) if "strategy_signal" in strategy_df.columns else 0
    return rows_scored, selected_rows


def _variant_name(strategy_name: str, filter_name: str) -> str:
    return f"{str(strategy_name).strip()}__{str(filter_name).strip()}"


def _annotate_variant_row(
    row: Mapping[str, Any],
    *,
    universe_key: str,
    universe_label: str,
    strategy_name: str,
    filter_name: str,
    selected_symbols: Sequence[str],
    selection_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(row)
    out["universe_key"] = str(universe_key)
    out["universe_label"] = str(universe_label)
    out["variant_name"] = _variant_name(strategy_name, filter_name)
    out["strategy_name"] = str(strategy_name)
    out["policy_name"] = str(strategy_name)
    out["filter_name"] = str(filter_name)
    out["selected_symbol_count"] = int(len(selected_symbols))
    out["selected_symbols_preview"] = [str(symbol) for symbol in list(selected_symbols or [])[:10]]
    if selection_metadata:
        out["filter_target_col"] = str(selection_metadata.get("target_col") or "")
        out["filter_model_kind"] = str(selection_metadata.get("model_kind") or "")
        out["filter_used_fallback"] = bool(selection_metadata.get("used_fallback", False))
    return out


def _run_filtered_backtest(
    *,
    strategy_artifact: Artifact,
    backtest_config: Mapping[str, Any],
    output_name: str,
) -> Artifact:
    return _run_pipeline_job(
        name=output_name,
        requested_job="backtest_strategy",
        config=dict(backtest_config),
        input_ids=[int(strategy_artifact.id)],
    )


def _build_strategy_target_rows(
    strategy_rows: Sequence[Mapping[str, Any]],
    buy_hold_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    strategy_map = {
        str(row.get("symbol") or "").strip().upper(): dict(row)
        for row in strategy_rows
        if str(row.get("symbol") or "").strip()
    }
    buy_hold_map = {
        str(row.get("symbol") or "").strip().upper(): dict(row)
        for row in buy_hold_rows
        if str(row.get("symbol") or "").strip()
    }
    out: list[dict[str, Any]] = []
    for symbol, strategy_row in strategy_map.items():
        buy_hold_row = dict(buy_hold_map.get(symbol) or {})
        strategy_total_return = _safe_float(strategy_row.get("cumulative_return"))
        buy_hold_return = _safe_float(buy_hold_row.get("buy_and_hold_return"))
        out.append(
            {
                "symbol": symbol,
                "strategy_total_return": round(strategy_total_return, 8),
                "buy_and_hold_return": round(buy_hold_return, 8),
                "symbol_profitable": int(strategy_total_return > 0.0),
                "beats_buy_hold": int(strategy_total_return > buy_hold_return),
            }
        )
    out.sort(key=lambda row: str(row["symbol"]))
    return out


def _build_variant_row(
    *,
    base_row: Mapping[str, Any],
    strategy_artifact: Artifact,
    backtest_artifact: Artifact,
    backtest_config: Mapping[str, Any],
    validation_config: Mapping[str, Any],
    universe_key: str,
    universe_label: str,
    strategy_name: str,
    filter_name: str,
    selected_symbols: Sequence[str],
    selection_metadata: Mapping[str, Any] | None = None,
    training_prep_row: Mapping[str, Any] | None = None,
    filter_training_time_sec: float = 0.0,
) -> dict[str, Any]:
    row = dict(base_row)
    backtest_content = dict(backtest_artifact.content or {})
    backtest_meta = dict(backtest_artifact.metadata or {})
    runtime_summary = _summarize_backtest_artifact(backtest_artifact)
    benchmark = _build_equal_weight_benchmark(
        strategy_artifact,
        allowed_symbols=selected_symbols,
    )
    rows_scored, selected_rows = _strategy_row_counts(strategy_artifact, allowed_symbols=selected_symbols)
    row.update(
        {
            "rows_scored": int(rows_scored),
            "selected_rows": int(selected_rows),
            "final_equity": float(backtest_content.get("final_equity") or 0.0),
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
            "backtest_seconds": float(backtest_meta.get("backtest_seconds") or 0.0),
            "backtest_fee_bps": _resolved_backtest_cost(backtest_meta, dict(backtest_config), "fee_bps"),
            "backtest_slippage_bps": _resolved_backtest_cost(backtest_meta, dict(backtest_config), "slippage_bps"),
            "excess_cumulative_return": round(
                float(backtest_content.get("cumulative_return") or 0.0) - float(benchmark.get("benchmark_cumulative_return") or 0.0),
                8,
            ),
            "relative_final_equity": round(
                float(backtest_content.get("final_equity") or 0.0) - float(benchmark.get("benchmark_final_equity") or 0.0),
                8,
            ),
            "backtest_artifact_id": int(backtest_artifact.id),
        }
    )
    pipeline_runtime_seconds = round(
        _safe_float(row.get("dataset_build_seconds"))
        + _safe_float(row.get("fit_seconds"))
        + _safe_float(row.get("score_seconds"))
        + _safe_float(row.get("strategy_build_seconds"))
        + _safe_float(row.get("backtest_seconds")),
        6,
    )
    prep_total_runtime = _safe_float((training_prep_row or {}).get("total_runtime_seconds"))
    prep_fit_seconds = _safe_float((training_prep_row or {}).get("fit_seconds"))
    prep_backtest_seconds = _safe_float((training_prep_row or {}).get("backtest_seconds"))
    row["pipeline_runtime_seconds"] = pipeline_runtime_seconds
    row["filter_preparation_time_sec"] = round(prep_total_runtime, 6)
    row["filter_training_time_sec"] = round(_safe_float(filter_training_time_sec), 6)
    row["model_training_time_sec"] = round(_safe_float(row.get("fit_seconds")) + prep_fit_seconds, 6)
    row["backtest_time_sec"] = round(_safe_float(row.get("backtest_seconds")) + prep_backtest_seconds, 6)
    row["total_runtime_seconds"] = round(pipeline_runtime_seconds + prep_total_runtime + _safe_float(filter_training_time_sec), 6)
    row.update(_evaluate_variant_gates(row, validation_config=dict(validation_config)))
    return _annotate_variant_row(
        row,
        universe_key=universe_key,
        universe_label=universe_label,
        strategy_name=strategy_name,
        filter_name=filter_name,
        selected_symbols=selected_symbols,
        selection_metadata=selection_metadata,
    )


def _stage_seconds(tracer: PerformanceTracer, previous_count: int) -> float:
    stages = tracer.stages
    if len(stages) <= previous_count:
        return 0.0
    return round(float(stages[-1].wall_seconds), 6)


def _build_filter_diagnostic_row(
    *,
    universe_key: str,
    universe_label: str,
    fold_name: str,
    strategy_name: str,
    filter_name: str,
    metadata_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    filter_result: Mapping[str, Any],
    filter_training_time_sec: float,
) -> dict[str, Any]:
    metadata_lookup = {
        str(row.get("symbol") or "").strip().upper(): dict(row)
        for row in metadata_rows
        if str(row.get("symbol") or "").strip()
    }
    target_lookup = {
        str(row.get("symbol") or "").strip().upper(): dict(row)
        for row in target_rows
        if str(row.get("symbol") or "").strip()
    }
    selected_symbols = [str(symbol).strip().upper() for symbol in list(filter_result.get("selected_symbols") or []) if str(symbol).strip()]
    selected_metadata = [metadata_lookup[symbol] for symbol in selected_symbols if symbol in metadata_lookup]
    selected_target_rows = [target_lookup[symbol] for symbol in selected_symbols if symbol in target_lookup]
    beats_buy_hold_rate = (
        sum(int(_safe_float(row.get("beats_buy_hold"))) for row in selected_target_rows) / float(len(selected_target_rows))
        if selected_target_rows
        else 0.0
    )
    profitable_rate = (
        sum(int(_safe_float(row.get("symbol_profitable"))) for row in selected_target_rows) / float(len(selected_target_rows))
        if selected_target_rows
        else 0.0
    )
    return {
        "universe_key": str(universe_key),
        "universe_label": str(universe_label),
        "fold_name": str(fold_name),
        "strategy_name": str(strategy_name),
        "filter_name": str(filter_name),
        "target_col": str(filter_result.get("target_col") or ""),
        "selection_count": int(filter_result.get("selection_count") or 0),
        "selected_symbols": selected_symbols,
        "selected_symbols_preview": selected_symbols[:15],
        "trained_symbols": int(filter_result.get("trained_symbols") or 0),
        "positive_target_count": int(filter_result.get("positive_target_count") or 0),
        "positive_target_rate": round(_safe_float(filter_result.get("positive_target_rate")), 6),
        "used_fallback": bool(filter_result.get("used_fallback", False)),
        "fallback_reason": str(filter_result.get("fallback_reason") or ""),
        "tree_depth": int(filter_result.get("tree_depth") or 0),
        "leaf_count": int(filter_result.get("leaf_count") or 0),
        "feature_count": int(filter_result.get("feature_count") or 0),
        "feature_columns": list(filter_result.get("feature_columns") or []),
        "top_features": list(filter_result.get("top_features") or []),
        "tree_rules": str(filter_result.get("tree_rules") or ""),
        "score_rows": list(filter_result.get("score_rows") or []),
        "filter_training_time_sec": round(_safe_float(filter_training_time_sec), 6),
        "selected_profitable_rate": round(float(profitable_rate), 6),
        "selected_beats_buy_hold_rate": round(float(beats_buy_hold_rate), 6),
        "selected_sector_counts": dict(filter_result.get("selected_sector_counts") or {}),
        "selected_industry_counts": dict(filter_result.get("selected_industry_counts") or {}),
        "selected_exchange_counts": dict(filter_result.get("selected_exchange_counts") or {}),
    }


def _top_count_rows(rows: Sequence[Mapping[str, Any]], *, field: str, limit: int = 8) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for row in rows:
        for name, value in dict(row.get(field) or {}).items():
            counts[str(name)] = counts.get(str(name), 0) + int(value)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: max(int(limit), 0)]


def _top_symbol_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    ordered = sorted(
        [dict(row) for row in rows],
        key=lambda item: (
            _safe_float(item.get("sharpe")),
            _safe_float(item.get("avg_trade_return")),
            _safe_float(item.get("trade_count")),
        ),
        reverse=True,
    )
    return ordered[: max(int(limit), 0)]


def _comparison_signal(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> str:
    candidate_sharpe = _safe_float(candidate.get("sharpe"))
    baseline_sharpe = _safe_float(baseline.get("sharpe"))
    candidate_return = _safe_float(candidate.get("total_return"))
    baseline_return = _safe_float(baseline.get("total_return"))
    candidate_stability = _safe_float(candidate.get("positive_fold_rate"))
    baseline_stability = _safe_float(baseline.get("positive_fold_rate"))
    better_sharpe = candidate_sharpe > baseline_sharpe
    better_return = candidate_return > baseline_return
    better_stability = candidate_stability >= baseline_stability
    if better_sharpe and better_return and better_stability:
        return "yes"
    if better_sharpe or better_return:
        return "mixed"
    return "no"


def _aggregate_variant_rows(
    *,
    summary_rows: Sequence[Mapping[str, Any]],
    validation_config: Mapping[str, Any],
    universe_runtime_rows: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        key = (str(row.get("universe_key") or ""), str(row.get("variant_name") or ""))
        grouped.setdefault(key, []).append(dict(row))

    out: list[dict[str, Any]] = []
    for (universe_key, variant_name), rows in grouped.items():
        rows = sorted(rows, key=lambda item: str(item.get("fold_name") or ""))
        base_rows = _apply_walk_forward_gates(
            _aggregate_walk_forward_rows([dict(row) for row in rows]),
            validation_config=dict(validation_config),
        )
        base_row = dict(base_rows[0]) if base_rows else {}
        walk_forward_metrics = _summarize_walk_forward_metrics(rows)
        positive_folds = sum(1 for item in rows if _safe_float(item.get("cumulative_return")) > 0.0)
        sharpe_values = [_safe_float(item.get("sharpe")) for item in rows]
        selected_counts = [int(_safe_float(item.get("selected_symbol_count"))) for item in rows]
        universe_runtime = dict(universe_runtime_rows.get(universe_key) or {})
        item = dict(base_row)
        item.update(
            {
                "universe_key": universe_key,
                "universe_label": str(rows[0].get("universe_label") or "") if rows else "",
                "min_market_cap": float(universe_runtime.get("min_market_cap") or 0.0),
                "universe_symbol_count": int(universe_runtime.get("symbol_count") or 0),
                "variant_name": variant_name,
                "strategy_name": str(rows[0].get("strategy_name") or "") if rows else "",
                "policy_name": str(rows[0].get("policy_name") or "") if rows else "",
                "filter_name": str(rows[0].get("filter_name") or "") if rows else "",
                "sharpe": float(walk_forward_metrics.get("sharpe") or 0.0),
                "total_return": float(walk_forward_metrics.get("total_return") or 0.0),
                "final_equity": float(walk_forward_metrics.get("final_equity") or 1.0),
                "max_drawdown": float(walk_forward_metrics.get("max_drawdown") or 0.0),
                "avg_turnover": float(walk_forward_metrics.get("avg_turnover") or 0.0),
                "total_turnover": float(walk_forward_metrics.get("total_turnover") or 0.0),
                "trade_count": int(walk_forward_metrics.get("trade_count") or 0),
                "walk_forward_start_date": str(walk_forward_metrics.get("start_date") or ""),
                "walk_forward_end_date": str(walk_forward_metrics.get("end_date") or ""),
                "positive_fold_count": int(positive_folds),
                "positive_fold_rate": round(float(positive_folds / float(len(rows))) if rows else 0.0, 8),
                "mean_fold_sharpe": round(float(sum(sharpe_values) / len(sharpe_values)) if sharpe_values else 0.0, 8),
                "fold_sharpe_std": round(float(pd.Series(sharpe_values).std(ddof=0)) if sharpe_values else 0.0, 8),
                "mean_selected_symbol_count": round(float(sum(selected_counts) / len(selected_counts)) if selected_counts else 0.0, 4),
                "min_selected_symbol_count": min(selected_counts) if selected_counts else 0,
                "max_selected_symbol_count": max(selected_counts) if selected_counts else 0,
                "total_runtime_sec": round(sum(_safe_float(row.get("total_runtime_seconds")) for row in rows), 6),
                "pipeline_runtime_sec": round(sum(_safe_float(row.get("pipeline_runtime_seconds")) for row in rows), 6),
                "model_training_time_sec": round(sum(_safe_float(row.get("model_training_time_sec")) for row in rows), 6),
                "filter_training_time_sec": round(sum(_safe_float(row.get("filter_training_time_sec")) for row in rows), 6),
                "filter_preparation_time_sec": round(sum(_safe_float(row.get("filter_preparation_time_sec")) for row in rows), 6),
                "backtest_time_sec": round(sum(_safe_float(row.get("backtest_time_sec")) for row in rows), 6),
                "feature_artifact_loading_time_sec": round(_safe_float(universe_runtime.get("feature_artifact_loading_time_sec")), 6),
            }
        )
        out.append(item)

    tier_order = {"1t": 0, "100b": 1, "10b": 2}
    variant_order = {
        "baseline__no_filter": 0,
        "baseline__profitable_filter": 1,
        "baseline__beats_buy_hold_filter": 2,
        "model__no_filter": 3,
        "model__profitable_filter": 4,
        "model__beats_buy_hold_filter": 5,
    }
    out.sort(
        key=lambda item: (
            tier_order.get(str(item.get("universe_key") or ""), 999),
            variant_order.get(str(item.get("variant_name") or ""), 999),
        )
    )
    return out


def _runtime_stage_rows(
    *,
    universe_key: str,
    universe_label: str,
    tracer_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in list(tracer_summary.get("stages") or []):
        row = dict(stage)
        row["universe_key"] = str(universe_key)
        row["universe_label"] = str(universe_label)
        rows.append(row)
    return rows


def _category_wall_seconds(tracer_summary: Mapping[str, Any], category: str) -> float:
    return round(_safe_float(((tracer_summary.get("by_category") or {}).get(category) or {}).get("wall_seconds")), 6)


def _write_runtime_profile(
    *,
    output_basename: str,
    universe_key: str,
    tracer_summary: Mapping[str, Any],
) -> dict[str, str]:
    output_dir = Path("docs") / "performance"
    report = RuntimeProfileReport(
        generated_at=utc_timestamp(),
        engine="performance_tracer",
        target=f"tsmom_market_cap_policy_comparison__{universe_key}",
        total_seconds=round(_safe_float(tracer_summary.get("total_runtime_seconds")), 6),
        raw_output_path="",
        hotspots=[],
        stage_hotspots=sorted(
            [dict(row) for row in list(tracer_summary.get("stages") or [])],
            key=lambda row: _safe_float(row.get("wall_seconds")),
            reverse=True,
        )[:12],
        notes=["Generated from the experiment's existing PerformanceTracer stages and rendered with tools/performance_analysis."],
    )
    json_path = output_dir / f"{output_basename}__{universe_key}__runtime_hotspots.json"
    md_path = output_dir / f"{output_basename}__{universe_key}__runtime_hotspots.md"
    write_json(json_path, report)
    write_markdown(md_path, runtime_hotspots_markdown(report))
    return {
        "runtime_report_json_path": str(json_path),
        "runtime_report_markdown_path": str(md_path),
    }


def _run_single_universe_experiment(
    *,
    universe_key: str,
    universe_label: str,
    symbols: Sequence[str],
    folds: Sequence[Mapping[str, Any]],
    feature_config: Mapping[str, Any],
    baseline_strategy_config: Mapping[str, Any],
    model_strategy_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    validation_config: Mapping[str, Any],
    backtest_config: Mapping[str, Any],
    minimum_filter_symbols: int,
    filter_max_depth: int,
    filter_min_samples_leaf: int,
    output_basename: str,
    resume_existing: bool,
) -> dict[str, Any]:
    resolved_symbols = [str(symbol).strip().upper() for symbol in list(symbols or []) if str(symbol).strip()]
    tracer = PerformanceTracer(enabled=True)

    stage_count = len(tracer.stages)
    with tracer.stage(f"universe.artifact.{universe_key}", category="universe", workload_type=universe_key):
        universe_artifact = _resolve_or_build_universe_artifact(symbols=resolved_symbols, output_basename=output_basename)
    stage_count = len(tracer.stages)
    with tracer.stage(f"feature.artifact.{universe_key}", category="feature_loading", workload_type=universe_key):
        feature_artifact = _resolve_or_build_feature_artifact(
            universe_artifact=universe_artifact,
            symbols=resolved_symbols,
            feature_config=dict(feature_config),
            output_basename=output_basename,
        )
    feature_artifact_loading_time_sec = _stage_seconds(tracer, stage_count)
    stage_count = len(tracer.stages)
    with tracer.stage(f"feature.frame_load.{universe_key}", category="feature_loading", workload_type=universe_key):
        feature_frame = read_frame_artifact(
            feature_artifact,
            parse_dates=False,
            normalize_symbols=True,
        )
    feature_frame_loading_time_sec = _stage_seconds(tracer, stage_count)
    stage_count = len(tracer.stages)
    with tracer.stage(f"label.artifact.{universe_key}", category="label_loading", workload_type=universe_key):
        label_artifact = _resolve_or_build_label_artifact(
            universe_artifact=universe_artifact,
            symbols=resolved_symbols,
            base_model_config=dict(model_config),
            output_basename=output_basename,
        )
    label_artifact_loading_time_sec = _stage_seconds(tracer, stage_count)

    summary_rows: list[dict[str, Any]] = []
    test_symbol_rows: list[dict[str, Any]] = []
    filter_diagnostic_rows: list[dict[str, Any]] = []

    for fold in folds:
        fold_name = str(fold.get("name") or fold.get("fold_name") or "").strip()
        train_end_date = str(fold.get("train_end_date") or "")
        backtest_start_date = str(fold.get("backtest_start_date") or "")
        backtest_end_date = str(fold.get("backtest_end_date") or "")

        stage_count = len(tracer.stages)
        with tracer.stage(
            f"filter.metadata.{universe_key}.{fold_name}",
            category="filter_training",
            workload_type=universe_key,
            metadata={"fold_name": fold_name},
        ):
            metadata_rows = build_symbol_metadata_filter_summary(
                feature_frame,
                end_date=train_end_date,
                symbols=resolved_symbols,
            )
        filter_metadata_time_sec = _stage_seconds(tracer, stage_count)

        stage_count = len(tracer.stages)
        with tracer.stage(
            f"baseline.train.{universe_key}.{fold_name}",
            category="backtest",
            workload_type=universe_key,
            metadata={"policy": "baseline", "scope": "train", "fold_name": fold_name},
        ):
            baseline_train_summary = run_direct_feature_strategy_backtests(
                symbols=resolved_symbols,
                train_end_date=train_end_date,
                backtest_start_date="",
                backtest_end_date=train_end_date,
                universe_artifact=universe_artifact,
                feature_artifact=feature_artifact,
                feature_config=dict(feature_config),
                strategy_definition_slug="tsmom-baseline-market-cap-train",
                strategy_definition_name="TSMOM Baseline Market Cap Train",
                strategy_config=dict(baseline_strategy_config),
                validation_config=dict(validation_config),
                backtest_config=dict(backtest_config),
                output_basename=f"{output_basename}__baseline_train__{fold_name}",
                resume_existing=resume_existing,
            )
        baseline_train_row = _single_summary_row(baseline_train_summary, label=f"{universe_key} {fold_name} baseline train")
        baseline_train_backtest = _resolve_artifact(baseline_train_row.get("backtest_artifact_id"), label=f"{universe_key} {fold_name} baseline train backtest")
        baseline_train_diags = compute_symbol_strategy_diagnostics(
            baseline_train_backtest,
            strategy_name="baseline",
            filter_name="training_unfiltered",
            evaluation_scope="train",
            fold_name=fold_name,
            backtest_end_date=train_end_date,
            backtest_config=backtest_config,
        )
        buy_hold_train_rows = compute_symbol_buy_hold_diagnostics(
            baseline_train_backtest,
            evaluation_scope="train",
            fold_name=fold_name,
            backtest_end_date=train_end_date,
        )
        baseline_target_rows = _build_strategy_target_rows(baseline_train_diags, buy_hold_train_rows)
        baseline_filters: dict[str, dict[str, Any]] = {}
        for filter_name, target_col in FILTER_VARIANTS:
            stage_count = len(tracer.stages)
            with tracer.stage(
                f"filter.train.{universe_key}.baseline.{filter_name}.{fold_name}",
                category="filter_training",
                workload_type=universe_key,
                metadata={"policy": "baseline", "filter_name": filter_name, "fold_name": fold_name},
            ):
                filter_result = select_symbols_with_metadata_filter(
                    metadata_rows=metadata_rows,
                    target_rows=baseline_target_rows,
                    target_col=target_col,
                    minimum_selected_symbols=minimum_filter_symbols,
                    max_depth=filter_max_depth,
                    min_samples_leaf=filter_min_samples_leaf,
                )
            filter_training_time_sec = _stage_seconds(tracer, stage_count)
            baseline_filters[filter_name] = {
                **dict(filter_result),
                "filter_training_time_sec": round(filter_training_time_sec + filter_metadata_time_sec, 6),
            }
            filter_diagnostic_rows.append(
                _build_filter_diagnostic_row(
                    universe_key=universe_key,
                    universe_label=universe_label,
                    fold_name=fold_name,
                    strategy_name="baseline",
                    filter_name=filter_name,
                    metadata_rows=metadata_rows,
                    target_rows=baseline_target_rows,
                    filter_result=filter_result,
                    filter_training_time_sec=round(filter_training_time_sec + filter_metadata_time_sec, 6),
                )
            )

        stage_count = len(tracer.stages)
        with tracer.stage(
            f"baseline.test.{universe_key}.{fold_name}",
            category="backtest",
            workload_type=universe_key,
            metadata={"policy": "baseline", "scope": "test", "fold_name": fold_name},
        ):
            baseline_test_summary = run_direct_feature_strategy_backtests(
                symbols=resolved_symbols,
                train_end_date=train_end_date,
                backtest_start_date=backtest_start_date,
                backtest_end_date=backtest_end_date,
                universe_artifact=universe_artifact,
                feature_artifact=feature_artifact,
                feature_config=dict(feature_config),
                strategy_definition_slug="tsmom-baseline-market-cap-test",
                strategy_definition_name="TSMOM Baseline Market Cap Test",
                strategy_config=dict(baseline_strategy_config),
                validation_config=dict(validation_config),
                backtest_config=dict(backtest_config),
                output_basename=f"{output_basename}__baseline_test__{fold_name}",
                resume_existing=resume_existing,
            )
        baseline_test_row = _single_summary_row(baseline_test_summary, label=f"{universe_key} {fold_name} baseline test")
        baseline_strategy_artifact = _resolve_artifact(baseline_test_row.get("strategy_artifact_id"), label=f"{universe_key} {fold_name} baseline strategy")
        baseline_test_backtest = _resolve_artifact(baseline_test_row.get("backtest_artifact_id"), label=f"{universe_key} {fold_name} baseline test backtest")
        baseline_base_row = {
            **baseline_test_row,
            "fold_name": fold_name,
            "train_end_date": train_end_date,
            "backtest_start_date": backtest_start_date,
            "backtest_end_date": backtest_end_date,
        }
        summary_rows.append(
            _build_variant_row(
                base_row=baseline_base_row,
                strategy_artifact=baseline_strategy_artifact,
                backtest_artifact=baseline_test_backtest,
                backtest_config=dict(backtest_config),
                validation_config=dict(validation_config),
                universe_key=universe_key,
                universe_label=universe_label,
                strategy_name="baseline",
                filter_name="no_filter",
                selected_symbols=resolved_symbols,
            )
        )
        test_symbol_rows.extend(
            [
                {
                    **dict(row),
                    "universe_key": universe_key,
                    "universe_label": universe_label,
                }
                for row in compute_symbol_strategy_diagnostics(
                    baseline_test_backtest,
                    strategy_name="baseline",
                    filter_name="no_filter",
                    evaluation_scope="test",
                    fold_name=fold_name,
                    backtest_start_date=backtest_start_date,
                    backtest_end_date=backtest_end_date,
                    backtest_config=backtest_config,
                )
            ]
        )
        for filter_name, _target_col in FILTER_VARIANTS:
            filter_result = dict(baseline_filters[filter_name])
            selected_symbols = [str(symbol) for symbol in list(filter_result.get("selected_symbols") or [])]
            stage_count = len(tracer.stages)
            with tracer.stage(
                f"backtest.{universe_key}.baseline.{filter_name}.{fold_name}",
                category="backtest",
                workload_type=universe_key,
                metadata={"policy": "baseline", "filter_name": filter_name, "fold_name": fold_name},
            ):
                filtered_backtest = _run_filtered_backtest(
                    strategy_artifact=baseline_strategy_artifact,
                    backtest_config={**dict(backtest_config), "allowed_symbols": selected_symbols},
                    output_name=f"{output_basename}__baseline_{filter_name}__{fold_name}",
                )
            summary_rows.append(
                _build_variant_row(
                    base_row=baseline_base_row,
                    strategy_artifact=baseline_strategy_artifact,
                    backtest_artifact=filtered_backtest,
                    backtest_config={**dict(backtest_config), "allowed_symbols": selected_symbols},
                    validation_config=dict(validation_config),
                    universe_key=universe_key,
                    universe_label=universe_label,
                    strategy_name="baseline",
                    filter_name=filter_name,
                    selected_symbols=selected_symbols,
                    selection_metadata=filter_result,
                    training_prep_row=baseline_train_row,
                    filter_training_time_sec=_safe_float(filter_result.get("filter_training_time_sec")),
                )
            )
            test_symbol_rows.extend(
                [
                    {
                        **dict(row),
                        "universe_key": universe_key,
                        "universe_label": universe_label,
                    }
                    for row in compute_symbol_strategy_diagnostics(
                        filtered_backtest,
                        strategy_name="baseline",
                        filter_name=filter_name,
                        evaluation_scope="test",
                        fold_name=fold_name,
                        backtest_start_date=backtest_start_date,
                        backtest_end_date=backtest_end_date,
                        backtest_config={**dict(backtest_config), "allowed_symbols": selected_symbols},
                    )
                ]
            )

        stage_count = len(tracer.stages)
        with tracer.stage(
            f"model.train.{universe_key}.{fold_name}",
            category="model_training",
            workload_type=universe_key,
            metadata={"policy": "model", "scope": "train", "fold_name": fold_name},
        ):
            model_train_summary = run_model_cohort_backtests(
                symbols=resolved_symbols,
                fit_job="fit_regressor",
                base_model_config=dict(model_config),
                train_end_date=train_end_date,
                backtest_start_date="",
                backtest_end_date=train_end_date,
                universe_artifact=universe_artifact,
                label_artifact=label_artifact,
                feature_artifact=feature_artifact,
                feature_config=dict(feature_config),
                strategy_definition_slug="tsmom-model-market-cap-train",
                strategy_definition_name="TSMOM Model Market Cap Train",
                strategy_config=dict(model_strategy_config),
                validation_config=dict(validation_config),
                backtest_config=dict(backtest_config),
                output_basename=f"{output_basename}__model_train__{fold_name}",
                resume_existing=resume_existing,
            )
        model_train_row = _single_summary_row(model_train_summary, label=f"{universe_key} {fold_name} model train")
        model_train_backtest = _resolve_artifact(model_train_row.get("backtest_artifact_id"), label=f"{universe_key} {fold_name} model train backtest")
        model_train_diags = compute_symbol_strategy_diagnostics(
            model_train_backtest,
            strategy_name="model",
            filter_name="training_unfiltered",
            evaluation_scope="train",
            fold_name=fold_name,
            backtest_end_date=train_end_date,
            backtest_config=backtest_config,
        )
        model_target_rows = _build_strategy_target_rows(model_train_diags, buy_hold_train_rows)
        model_filters: dict[str, dict[str, Any]] = {}
        for filter_name, target_col in FILTER_VARIANTS:
            stage_count = len(tracer.stages)
            with tracer.stage(
                f"filter.train.{universe_key}.model.{filter_name}.{fold_name}",
                category="filter_training",
                workload_type=universe_key,
                metadata={"policy": "model", "filter_name": filter_name, "fold_name": fold_name},
            ):
                filter_result = select_symbols_with_metadata_filter(
                    metadata_rows=metadata_rows,
                    target_rows=model_target_rows,
                    target_col=target_col,
                    minimum_selected_symbols=minimum_filter_symbols,
                    max_depth=filter_max_depth,
                    min_samples_leaf=filter_min_samples_leaf,
                )
            filter_training_time_sec = _stage_seconds(tracer, stage_count)
            model_filters[filter_name] = {
                **dict(filter_result),
                "filter_training_time_sec": round(filter_training_time_sec + filter_metadata_time_sec, 6),
            }
            filter_diagnostic_rows.append(
                _build_filter_diagnostic_row(
                    universe_key=universe_key,
                    universe_label=universe_label,
                    fold_name=fold_name,
                    strategy_name="model",
                    filter_name=filter_name,
                    metadata_rows=metadata_rows,
                    target_rows=model_target_rows,
                    filter_result=filter_result,
                    filter_training_time_sec=round(filter_training_time_sec + filter_metadata_time_sec, 6),
                )
            )

        stage_count = len(tracer.stages)
        with tracer.stage(
            f"model.test.{universe_key}.{fold_name}",
            category="model_training",
            workload_type=universe_key,
            metadata={"policy": "model", "scope": "test", "fold_name": fold_name},
        ):
            model_test_summary = run_model_cohort_backtests(
                symbols=resolved_symbols,
                fit_job="fit_regressor",
                base_model_config=dict(model_config),
                train_end_date=train_end_date,
                backtest_start_date=backtest_start_date,
                backtest_end_date=backtest_end_date,
                universe_artifact=universe_artifact,
                label_artifact=label_artifact,
                feature_artifact=feature_artifact,
                feature_config=dict(feature_config),
                strategy_definition_slug="tsmom-model-market-cap-test",
                strategy_definition_name="TSMOM Model Market Cap Test",
                strategy_config=dict(model_strategy_config),
                validation_config=dict(validation_config),
                backtest_config=dict(backtest_config),
                output_basename=f"{output_basename}__model_test__{fold_name}",
                resume_existing=resume_existing,
            )
        model_test_row = _single_summary_row(model_test_summary, label=f"{universe_key} {fold_name} model test")
        model_strategy_artifact = _resolve_artifact(model_test_row.get("strategy_artifact_id"), label=f"{universe_key} {fold_name} model strategy")
        model_test_backtest = _resolve_artifact(model_test_row.get("backtest_artifact_id"), label=f"{universe_key} {fold_name} model test backtest")
        model_base_row = {
            **model_test_row,
            "fold_name": fold_name,
            "train_end_date": train_end_date,
            "backtest_start_date": backtest_start_date,
            "backtest_end_date": backtest_end_date,
        }
        summary_rows.append(
            _build_variant_row(
                base_row=model_base_row,
                strategy_artifact=model_strategy_artifact,
                backtest_artifact=model_test_backtest,
                backtest_config=dict(backtest_config),
                validation_config=dict(validation_config),
                universe_key=universe_key,
                universe_label=universe_label,
                strategy_name="model",
                filter_name="no_filter",
                selected_symbols=resolved_symbols,
            )
        )
        test_symbol_rows.extend(
            [
                {
                    **dict(row),
                    "universe_key": universe_key,
                    "universe_label": universe_label,
                }
                for row in compute_symbol_strategy_diagnostics(
                    model_test_backtest,
                    strategy_name="model",
                    filter_name="no_filter",
                    evaluation_scope="test",
                    fold_name=fold_name,
                    backtest_start_date=backtest_start_date,
                    backtest_end_date=backtest_end_date,
                    backtest_config=backtest_config,
                )
            ]
        )
        for filter_name, _target_col in FILTER_VARIANTS:
            filter_result = dict(model_filters[filter_name])
            selected_symbols = [str(symbol) for symbol in list(filter_result.get("selected_symbols") or [])]
            stage_count = len(tracer.stages)
            with tracer.stage(
                f"backtest.{universe_key}.model.{filter_name}.{fold_name}",
                category="backtest",
                workload_type=universe_key,
                metadata={"policy": "model", "filter_name": filter_name, "fold_name": fold_name},
            ):
                filtered_backtest = _run_filtered_backtest(
                    strategy_artifact=model_strategy_artifact,
                    backtest_config={**dict(backtest_config), "allowed_symbols": selected_symbols},
                    output_name=f"{output_basename}__model_{filter_name}__{fold_name}",
                )
            summary_rows.append(
                _build_variant_row(
                    base_row=model_base_row,
                    strategy_artifact=model_strategy_artifact,
                    backtest_artifact=filtered_backtest,
                    backtest_config={**dict(backtest_config), "allowed_symbols": selected_symbols},
                    validation_config=dict(validation_config),
                    universe_key=universe_key,
                    universe_label=universe_label,
                    strategy_name="model",
                    filter_name=filter_name,
                    selected_symbols=selected_symbols,
                    selection_metadata=filter_result,
                    training_prep_row=model_train_row,
                    filter_training_time_sec=_safe_float(filter_result.get("filter_training_time_sec")),
                )
            )
            test_symbol_rows.extend(
                [
                    {
                        **dict(row),
                        "universe_key": universe_key,
                        "universe_label": universe_label,
                    }
                    for row in compute_symbol_strategy_diagnostics(
                        filtered_backtest,
                        strategy_name="model",
                        filter_name=filter_name,
                        evaluation_scope="test",
                        fold_name=fold_name,
                        backtest_start_date=backtest_start_date,
                        backtest_end_date=backtest_end_date,
                        backtest_config={**dict(backtest_config), "allowed_symbols": selected_symbols},
                    )
                ]
            )

    tracer_summary = tracer.summary()
    runtime_paths = _write_runtime_profile(
        output_basename=output_basename,
        universe_key=universe_key,
        tracer_summary=tracer_summary,
    )
    return {
        "universe_key": universe_key,
        "universe_label": universe_label,
        "min_market_cap": float(MARKET_CAP_TIERS[universe_key]),
        "symbols": resolved_symbols,
        "symbol_count": int(len(resolved_symbols)),
        "summary_rows": summary_rows,
        "symbol_diagnostics_test_rows": test_symbol_rows,
        "filter_diagnostic_rows": filter_diagnostic_rows,
        "performance": tracer_summary,
        "feature_artifact_loading_time_sec": round(feature_artifact_loading_time_sec + feature_frame_loading_time_sec, 6),
        "label_artifact_loading_time_sec": round(label_artifact_loading_time_sec, 6),
        **runtime_paths,
    }


def run_market_cap_policy_comparison_experiment(
    *,
    tiers: Sequence[str] = ("1t", "100b", "10b"),
    folds: Sequence[Mapping[str, Any]],
    feature_config: Mapping[str, Any] | None = None,
    baseline_strategy_config: Mapping[str, Any] | None = None,
    model_strategy_config: Mapping[str, Any] | None = None,
    model_config: Mapping[str, Any] | None = None,
    validation_config: Mapping[str, Any] | None = None,
    backtest_config: Mapping[str, Any] | None = None,
    country: str = "US",
    exchanges: Sequence[str] = DEFAULT_US_EXCHANGES,
    max_symbols_per_tier: int | None = None,
    minimum_filter_symbols: int = 5,
    filter_max_depth: int = 3,
    filter_min_samples_leaf: int = 3,
    output_basename: str = "time_series_momentum_market_cap_policy_comparison",
    resume_existing: bool = False,
) -> dict[str, Any]:
    output_dir = Path("data") / "pipeline_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{output_basename}.json"
    summary_csv_path = output_dir / f"{output_basename}.csv"
    fold_csv_path = output_dir / f"{output_basename}__fold_rows.csv"
    symbol_test_csv_path = output_dir / f"{output_basename}__symbol_diagnostics_test.csv"
    symbol_agg_csv_path = output_dir / f"{output_basename}__symbol_diagnostics_aggregate.csv"
    filter_diag_csv_path = output_dir / f"{output_basename}__filter_diagnostics.csv"
    runtime_csv_path = output_dir / f"{output_basename}__runtime_analysis.csv"
    runtime_stage_csv_path = output_dir / f"{output_basename}__runtime_stages.csv"

    if resume_existing:
        cached_payload = _load_cached_payload(
            json_path,
            required_keys=("aggregate_rows", "summary_rows", "universe_rows"),
            schema_version=MARKET_CAP_POLICY_COMPARISON_SCHEMA_VERSION,
        )
        if cached_payload is not None:
            cached_payload["summary_json_path"] = str(json_path)
            cached_payload["summary_csv_path"] = str(summary_csv_path)
            cached_payload["fold_results_csv_path"] = str(fold_csv_path)
            cached_payload["symbol_diagnostics_test_csv_path"] = str(symbol_test_csv_path)
            cached_payload["symbol_diagnostics_aggregate_csv_path"] = str(symbol_agg_csv_path)
            cached_payload["filter_diagnostics_csv_path"] = str(filter_diag_csv_path)
            cached_payload["runtime_analysis_csv_path"] = str(runtime_csv_path)
            cached_payload["runtime_stage_csv_path"] = str(runtime_stage_csv_path)
            return cached_payload

    resolved_feature_config = dict(feature_config or {})
    resolved_baseline_strategy_config = dict(_baseline_strategy_config())
    resolved_baseline_strategy_config.update(dict(baseline_strategy_config or {}))
    resolved_model_strategy_config = dict(_model_strategy_config())
    resolved_model_strategy_config.update(dict(model_strategy_config or {}))
    resolved_model_config = dict(_default_model_config())
    resolved_model_config.update(dict(model_config or {}))
    resolved_validation_config = dict(_default_validation_config())
    resolved_validation_config.update(dict(validation_config or {}))
    resolved_backtest_config = dict(backtest_config or {})

    universe_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    symbol_test_rows: list[dict[str, Any]] = []
    filter_diagnostic_rows: list[dict[str, Any]] = []
    runtime_stage_rows: list[dict[str, Any]] = []

    for raw_tier in tiers:
        tier_key = str(raw_tier or "").strip().lower()
        if tier_key not in MARKET_CAP_TIERS:
            raise ValueError(f"Unknown market cap tier: {raw_tier!r}")
        symbols = resolve_market_cap_tier_symbols(
            tier_key=tier_key,
            country=str(country or "US").strip(),
            exchanges=list(exchanges or DEFAULT_US_EXCHANGES),
            limit=int(max_symbols_per_tier) if max_symbols_per_tier not in (None, 0) else None,
            exclude_pooled_vehicles=True,
        )
        if not symbols:
            universe_rows.append(
                {
                    "universe_key": tier_key,
                    "universe_label": TIER_LABELS.get(tier_key, tier_key),
                    "min_market_cap": float(MARKET_CAP_TIERS[tier_key]),
                    "symbol_count": 0,
                    "symbols_preview": [],
                    "status": "skipped",
                    "error": "no_symbols",
                }
            )
            continue
        result = _run_single_universe_experiment(
            universe_key=tier_key,
            universe_label=TIER_LABELS.get(tier_key, tier_key),
            symbols=symbols,
            folds=folds,
            feature_config=resolved_feature_config,
            baseline_strategy_config=resolved_baseline_strategy_config,
            model_strategy_config=resolved_model_strategy_config,
            model_config=resolved_model_config,
            validation_config=resolved_validation_config,
            backtest_config=resolved_backtest_config,
            minimum_filter_symbols=minimum_filter_symbols,
            filter_max_depth=filter_max_depth,
            filter_min_samples_leaf=filter_min_samples_leaf,
            output_basename=f"{output_basename}__{tier_key}",
            resume_existing=resume_existing,
        )
        universe_rows.append(
            {
                "universe_key": tier_key,
                "universe_label": TIER_LABELS.get(tier_key, tier_key),
                "min_market_cap": float(MARKET_CAP_TIERS[tier_key]),
                "symbol_count": int(result.get("symbol_count") or 0),
                "symbols_preview": list(result.get("symbols") or [])[:20],
                "status": "succeeded",
                "feature_artifact_loading_time_sec": round(_safe_float(result.get("feature_artifact_loading_time_sec")), 6),
                "label_artifact_loading_time_sec": round(_safe_float(result.get("label_artifact_loading_time_sec")), 6),
                "total_runtime_sec": round(_safe_float((result.get("performance") or {}).get("total_runtime_seconds")), 6),
                "runtime_report_json_path": str(result.get("runtime_report_json_path") or ""),
                "runtime_report_markdown_path": str(result.get("runtime_report_markdown_path") or ""),
            }
        )
        summary_rows.extend(list(result.get("summary_rows") or []))
        symbol_test_rows.extend(list(result.get("symbol_diagnostics_test_rows") or []))
        filter_diagnostic_rows.extend(list(result.get("filter_diagnostic_rows") or []))
        runtime_stage_rows.extend(
            _runtime_stage_rows(
                universe_key=tier_key,
                universe_label=TIER_LABELS.get(tier_key, tier_key),
                tracer_summary=dict(result.get("performance") or {}),
            )
        )

    universe_runtime_lookup = {
        str(row.get("universe_key") or ""): dict(row)
        for row in universe_rows
        if str(row.get("status") or "") == "succeeded"
    }
    aggregate_rows = _aggregate_variant_rows(
        summary_rows=summary_rows,
        validation_config=resolved_validation_config,
        universe_runtime_rows=universe_runtime_lookup,
    )
    aggregate_symbol_rows = aggregate_symbol_diagnostic_rows(
        symbol_test_rows,
        group_keys=("universe_key", "universe_label", "strategy_name", "filter_name", "symbol"),
    )
    runtime_rows = [
        {
            "universe_key": row.get("universe_key", ""),
            "universe_label": row.get("universe_label", ""),
            "policy_name": row.get("policy_name", ""),
            "variant_name": row.get("variant_name", ""),
            "filter_name": row.get("filter_name", ""),
            "feature_artifact_loading_time_sec": row.get("feature_artifact_loading_time_sec", 0.0),
            "total_runtime_sec": row.get("total_runtime_sec", 0.0),
            "model_training_time_sec": row.get("model_training_time_sec", 0.0),
            "filter_training_time_sec": row.get("filter_training_time_sec", 0.0),
            "backtest_time_sec": row.get("backtest_time_sec", 0.0),
        }
        for row in aggregate_rows
    ]

    payload = {
        "schema_version": MARKET_CAP_POLICY_COMPARISON_SCHEMA_VERSION,
        "mode": "time_series_momentum_market_cap_policy_comparison",
        "folds": [dict(fold) for fold in folds],
        "country": str(country or "US").strip(),
        "exchanges": [str(exchange).strip().upper() for exchange in list(exchanges or DEFAULT_US_EXCHANGES)],
        "tiers": [str(token).strip().lower() for token in list(tiers or [])],
        "feature_config": resolved_feature_config,
        "baseline_strategy_config": resolved_baseline_strategy_config,
        "model_strategy_config": resolved_model_strategy_config,
        "model_config": resolved_model_config,
        "validation_config": resolved_validation_config,
        "backtest_config": resolved_backtest_config,
        "minimum_filter_symbols": int(minimum_filter_symbols),
        "filter_max_depth": int(filter_max_depth),
        "filter_min_samples_leaf": int(filter_min_samples_leaf),
        "universe_rows": universe_rows,
        "summary_rows": summary_rows,
        "aggregate_rows": aggregate_rows,
        "symbol_diagnostics_test_rows": symbol_test_rows,
        "symbol_diagnostics_aggregate_rows": aggregate_symbol_rows,
        "filter_diagnostic_rows": filter_diagnostic_rows,
        "runtime_rows": runtime_rows,
        "runtime_stage_rows": runtime_stage_rows,
        "summary_json_path": str(json_path),
        "summary_csv_path": str(summary_csv_path),
        "fold_results_csv_path": str(fold_csv_path),
        "symbol_diagnostics_test_csv_path": str(symbol_test_csv_path),
        "symbol_diagnostics_aggregate_csv_path": str(symbol_agg_csv_path),
        "filter_diagnostics_csv_path": str(filter_diag_csv_path),
        "runtime_analysis_csv_path": str(runtime_csv_path),
        "runtime_stage_csv_path": str(runtime_stage_csv_path),
    }
    write_json(json_path, payload)
    _write_rows_csv(summary_csv_path, aggregate_rows)
    _write_rows_csv(fold_csv_path, summary_rows)
    _write_rows_csv(symbol_test_csv_path, symbol_test_rows)
    _write_rows_csv(symbol_agg_csv_path, aggregate_symbol_rows)
    _write_rows_csv(filter_diag_csv_path, filter_diagnostic_rows)
    _write_rows_csv(runtime_csv_path, runtime_rows)
    _write_rows_csv(runtime_stage_csv_path, runtime_stage_rows)
    return payload


def write_market_cap_policy_comparison_report(
    *,
    report_path: Path,
    payload: Mapping[str, Any],
) -> None:
    universe_rows = [dict(row) for row in list(payload.get("universe_rows") or []) if str(row.get("status") or "") == "succeeded"]
    aggregate_rows = [dict(row) for row in list(payload.get("aggregate_rows") or [])]
    symbol_rows = [dict(row) for row in list(payload.get("symbol_diagnostics_aggregate_rows") or [])]
    filter_rows = [dict(row) for row in list(payload.get("filter_diagnostic_rows") or [])]
    runtime_rows = [dict(row) for row in list(payload.get("runtime_rows") or [])]

    best_ml_by_universe: dict[str, dict[str, Any]] = {}
    baseline_by_universe: dict[str, dict[str, Any]] = {}
    best_filter_by_policy_universe: dict[tuple[str, str], dict[str, Any]] = {}
    for row in aggregate_rows:
        universe_key = str(row.get("universe_key") or "")
        strategy_name = str(row.get("strategy_name") or "")
        filter_name = str(row.get("filter_name") or "")
        if strategy_name == "baseline" and filter_name == "no_filter":
            baseline_by_universe[universe_key] = dict(row)
        if strategy_name == "model" and filter_name == "no_filter":
            best_ml_by_universe[universe_key] = dict(row)
        if filter_name != "no_filter":
            key = (universe_key, strategy_name)
            best = best_filter_by_policy_universe.get(key)
            if best is None or (
                _safe_float(row.get("sharpe")) > _safe_float(best.get("sharpe"))
                and _safe_float(row.get("total_return")) >= _safe_float(best.get("total_return"))
            ):
                best_filter_by_policy_universe[key] = dict(row)

    ml_vs_baseline_signals = [
        _comparison_signal(best_ml_by_universe.get(universe_key, {}), baseline_by_universe.get(universe_key, {}))
        for universe_key in sorted(best_ml_by_universe.keys())
        if universe_key in baseline_by_universe
    ]
    filter_help_signals = [
        _comparison_signal(best_filter_by_policy_universe.get((universe_key, policy_name), {}), aggregate_row)
        for aggregate_row in aggregate_rows
        if str(aggregate_row.get("filter_name") or "") == "no_filter"
        for universe_key, policy_name in [(str(aggregate_row.get("universe_key") or ""), str(aggregate_row.get("strategy_name") or ""))]
        if (universe_key, policy_name) in best_filter_by_policy_universe
    ]
    best_universe_row = max(
        aggregate_rows,
        key=lambda row: (_safe_float(row.get("sharpe")), _safe_float(row.get("total_return"))),
        default={},
    )
    model_runtime_ratios = []
    for universe_key, model_row in best_ml_by_universe.items():
        baseline_row = baseline_by_universe.get(universe_key)
        if baseline_row is None:
            continue
        baseline_runtime = max(_safe_float(baseline_row.get("total_runtime_sec")), 1e-9)
        model_runtime_ratios.append(_safe_float(model_row.get("total_runtime_sec")) / baseline_runtime)
    avg_runtime_ratio = sum(model_runtime_ratios) / float(len(model_runtime_ratios)) if model_runtime_ratios else 0.0
    if ml_vs_baseline_signals and all(signal == "yes" for signal in ml_vs_baseline_signals) and avg_runtime_ratio <= 3.0:
        complexity_signal = "yes"
    elif any(signal in {"yes", "mixed"} for signal in ml_vs_baseline_signals):
        complexity_signal = "mixed"
    else:
        complexity_signal = "no"

    strongest_symbols = _top_symbol_rows(symbol_rows, limit=10)
    top_selected_sectors = _top_count_rows(filter_rows, field="selected_sector_counts", limit=8)
    top_selected_industries = _top_count_rows(filter_rows, field="selected_industry_counts", limit=8)

    filter_summary_rows: list[str] = []
    filter_table_rows = sorted(
        filter_rows,
        key=lambda row: (
            str(row.get("universe_key") or ""),
            str(row.get("strategy_name") or ""),
            str(row.get("filter_name") or ""),
            str(row.get("fold_name") or ""),
        ),
    )
    latest_filter_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in filter_table_rows:
        key = (
            str(row.get("universe_key") or ""),
            str(row.get("strategy_name") or ""),
            str(row.get("filter_name") or ""),
        )
        latest_filter_rows[key] = dict(row)
    for (_universe_key, _strategy_name, _filter_name), row in sorted(latest_filter_rows.items()):
        top_features = ", ".join(
            f"{name} ({_safe_float(value):.2f})"
            for name, value in list(row.get("top_features") or [])[:3]
        ) or "n/a"
        filter_summary_rows.append(
            "| "
            + f"{row.get('universe_label', '')} | "
            + f"{row.get('strategy_name', '')} | "
            + f"{FILTER_DISPLAY_NAMES.get(str(row.get('filter_name') or ''), str(row.get('filter_name') or ''))} | "
            + f"{int(_safe_float(row.get('tree_depth')))} | "
            + f"{int(_safe_float(row.get('selection_count')))} | "
            + f"{top_features} |"
        )

    lines = [
        "# Time Series Momentum Market-Cap Policy Comparison",
        "",
        "## Experiment Setup",
        "",
        "- Universes: " + ", ".join(
            f"{row.get('universe_label', '')} ({int(row.get('symbol_count') or 0)} symbols)"
            for row in universe_rows
        ),
        "- Policies: `baseline` uses the platform's TSMOM 12-1 signal; `model` uses a RandomForestRegressor trained on oracle `trade_return` labels with the full feature artifact.",
        "- Filters: `no_filter`, `profitable_filter`, and `beats_buy_hold_filter`.",
        "- Filter features: sector, industry, exchange, and average market cap through the training window.",
        "- Walk-forward methodology: yearly folds with training through December 31 of year N, then frozen out-of-sample evaluation during year N+1.",
        "",
        "## Performance Comparison",
        "",
        "| Universe | Policy | Variant | Sharpe | Return | Max DD | Turnover | Trades | Positive Fold Rate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        lines.append(
            "| "
            + f"{row.get('universe_label', '')} | "
            + f"{row.get('policy_name', '')} | "
            + f"{FILTER_DISPLAY_NAMES.get(str(row.get('filter_name') or ''), str(row.get('filter_name') or ''))} | "
            + f"{_safe_float(row.get('sharpe')):.3f} | "
            + f"{_pct(row.get('total_return'))} | "
            + f"{_pct(row.get('max_drawdown'))} | "
            + f"{_safe_float(row.get('total_turnover')):.2f} | "
            + f"{int(_safe_float(row.get('trade_count')))} | "
            + f"{_safe_float(row.get('positive_fold_rate')):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Runtime Comparison",
            "",
            "| Universe | Policy | Variant | Runtime | Model Train | Filter Train | Backtest |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in runtime_rows:
        lines.append(
            "| "
            + f"{row.get('universe_label', '')} | "
            + f"{row.get('policy_name', '')} | "
            + f"{FILTER_DISPLAY_NAMES.get(str(row.get('filter_name') or ''), str(row.get('filter_name') or ''))} | "
            + f"{_safe_float(row.get('total_runtime_sec')):.3f}s | "
            + f"{_safe_float(row.get('model_training_time_sec')):.3f}s | "
            + f"{_safe_float(row.get('filter_training_time_sec')):.3f}s | "
            + f"{_safe_float(row.get('backtest_time_sec')):.3f}s |"
        )

    lines.extend(
        [
            "",
            "## Symbol Insights",
            "",
            "- Strongest aggregate test-fold symbols: "
            + ", ".join(
                f"{row.get('symbol', '')} ({row.get('universe_label', '')}, {row.get('strategy_name', '')}, Sharpe {_safe_float(row.get('sharpe')):.2f})"
                for row in strongest_symbols
            ),
            "- Sectors most frequently selected by the metadata filters: "
            + ", ".join(f"{name} ({count})" for name, count in top_selected_sectors),
            "- Industries most frequently selected by the metadata filters: "
            + ", ".join(f"{name} ({count})" for name, count in top_selected_industries),
            "",
            "## Filter Interpretation",
            "",
            "| Universe | Policy | Variant | Depth | Symbols Selected | Top Features |",
            "| --- | --- | --- | ---: | ---: | --- |",
            *filter_summary_rows,
            "",
            "Rule previews are in the filter diagnostics artifact; each row stores the exact shallow-tree rules used for that fold.",
            "",
            "## Conclusions",
            "",
            f"- Does the ML policy beat the baseline? {', '.join(ml_vs_baseline_signals) if ml_vs_baseline_signals else 'n/a'} across the resolved universes.",
            f"- Does filtering help? {', '.join(filter_help_signals) if filter_help_signals else 'n/a'} when each policy is compared against its own no-filter baseline.",
            f"- Which universe works best? `{best_universe_row.get('universe_label', 'n/a')}` with `{best_universe_row.get('variant_name', 'n/a')}` delivered the strongest combined Sharpe/return result in this run.",
            f"- Does added complexity justify runtime cost? {complexity_signal}. Average model/no-filter runtime multiple vs baseline/no-filter was {avg_runtime_ratio:.2f}x.",
            "",
            "## Output Artifacts",
            "",
            f"- Summary JSON: `{payload.get('summary_json_path') or ''}`",
            f"- Summary CSV: `{payload.get('summary_csv_path') or ''}`",
            f"- Fold results CSV: `{payload.get('fold_results_csv_path') or ''}`",
            f"- Symbol diagnostics (test folds): `{payload.get('symbol_diagnostics_test_csv_path') or ''}`",
            f"- Aggregate symbol diagnostics: `{payload.get('symbol_diagnostics_aggregate_csv_path') or ''}`",
            f"- Filter diagnostics: `{payload.get('filter_diagnostics_csv_path') or ''}`",
            f"- Runtime analysis CSV: `{payload.get('runtime_analysis_csv_path') or ''}`",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_US_EXCHANGES",
    "FILTER_DISPLAY_NAMES",
    "MARKET_CAP_POLICY_COMPARISON_SCHEMA_VERSION",
    "TIER_LABELS",
    "build_yearly_folds",
    "run_market_cap_policy_comparison_experiment",
    "write_market_cap_policy_comparison_report",
]
