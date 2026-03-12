from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ml.execution import load_artifact_csv_frame

from .cohort_runner import (
    _aggregate_walk_forward_rows,
    _apply_walk_forward_gates,
    _load_cached_payload,
    _resolve_or_build_feature_artifact,
    _resolve_or_build_universe_artifact,
)
from .cross_sectional_rank_labels import (
    CrossSectionalRankLabelSpec,
    first_available_column,
    resolve_or_build_cross_sectional_rank_label_artifact,
)
from .direct_strategy_runner import _summarize_walk_forward_metrics, run_direct_feature_strategy_backtests
from .experiments import available_feature_families
from .models import Artifact
from .ranking_diagnostics import (
    aggregate_bucket_overlap_rows,
    aggregate_bucket_return_rows,
    aggregate_ranking_summary_rows,
    aggregate_top_bucket_cohort_rows,
    assign_cross_sectional_buckets,
    build_expression_score_frame,
    build_signal_ranking_panel,
    build_symbol_metadata_lookup,
    compute_bucket_overlap_rows,
    compute_bucket_return_rows,
    compute_ranking_summary_rows,
    compute_top_bucket_cohort_rows,
    compute_top_bucket_stability_rows,
    top_bucket_rows,
)
from .cohort_runner import run_model_cohort_backtests
from .universe_selection import filter_symbols_by_price_history, resolve_symbol_universe, summarize_symbol_price_history


ORACLE_RANKING_SIGNAL_SCHEMA_VERSION = 1
DEFAULT_EXCLUDED_SYMBOL_PREFIXES = ("TIER",)
DEFAULT_PRICE_ONLY_FAMILIES = ("prices_div_adj",)


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
    items = [dict(row) for row in rows]
    if not items:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in items:
        for field in row.keys():
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(items)


def build_yearly_folds(start_year: int, end_year: int) -> list[dict[str, str]]:
    return [
        {
            "name": f"wf_{year}",
            "train_end_date": f"{year - 1}-12-31",
            "backtest_start_date": f"{year}-01-01",
            "backtest_end_date": f"{year}-12-31",
        }
        for year in range(int(start_year), int(end_year) + 1)
    ]


def _required_history_window(
    *,
    test_start_year: int,
    test_end_year: int,
    lookback_days: int,
    forward_horizon_days: int,
    start_offset_days: int,
) -> tuple[str, str, int]:
    first_test_date = pd.Timestamp(f"{int(test_start_year)}-01-01")
    buffer_days = int(lookback_days) + int(forward_horizon_days) + int(start_offset_days) + 10
    required_start = first_test_date - pd.offsets.BDay(buffer_days)
    required_end = pd.Timestamp(f"{int(test_end_year)}-12-31")
    expected_days = len(pd.bdate_range(start=required_start, end=required_end))
    min_history_days = max(int(expected_days * 0.75), buffer_days)
    return required_start.strftime("%Y-%m-%d"), required_end.strftime("%Y-%m-%d"), int(min_history_days)


def resolve_research_symbols(
    *,
    requested_symbols: Sequence[str] | None = None,
    symbol_limit: int,
    candidate_limit: int,
    min_market_cap: float,
    test_start_year: int,
    test_end_year: int,
    lookback_days: int,
    forward_horizon_days: int,
    start_offset_days: int,
    exclude_symbol_prefixes: Sequence[str] = DEFAULT_EXCLUDED_SYMBOL_PREFIXES,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    if requested_symbols:
        candidate_symbols = [
            str(symbol).strip().upper()
            for symbol in list(requested_symbols or [])
            if str(symbol).strip()
        ]
    else:
        candidate_symbols = resolve_symbol_universe(
            min_market_cap=float(min_market_cap),
            country="US",
            exchanges=["NASDAQ", "NYSE", "AMEX"],
            limit=max(int(candidate_limit), int(symbol_limit)),
            exclude_pooled_vehicles=True,
            exclude_symbol_prefixes=list(exclude_symbol_prefixes),
        )
    required_start, required_end, min_history_days = _required_history_window(
        test_start_year=int(test_start_year),
        test_end_year=int(test_end_year),
        lookback_days=int(lookback_days),
        forward_horizon_days=int(forward_horizon_days),
        start_offset_days=int(start_offset_days),
    )
    filtered_symbols = filter_symbols_by_price_history(
        candidate_symbols,
        start_date=required_start,
        end_date=required_end,
        required_start_date=required_start,
        required_end_date=required_end,
        min_history_days=min_history_days,
    )
    selected_symbols = filtered_symbols[: max(int(symbol_limit), 1)]
    coverage_rows = summarize_symbol_price_history(
        selected_symbols,
        start_date=required_start,
        end_date=required_end,
        required_start_date=required_start,
        required_end_date=required_end,
        min_history_days=min_history_days,
    )
    missing_requested = [symbol for symbol in candidate_symbols if symbol not in filtered_symbols]
    return selected_symbols, coverage_rows, missing_requested


def _resolve_momentum_signal_spec(feature_artifact: Artifact) -> dict[str, Any]:
    feature_df = load_feature_columns_frame(feature_artifact)
    columns = list(feature_df.columns)
    long_col = first_available_column(columns, ("px__ret_252d", "px__ret_252_d", "ret_252d", "ret_252_d"))
    short_col = first_available_column(columns, ("px__ret_21d", "px__ret_21_d", "ret_21d", "ret_21_d"))
    if long_col and short_col:
        return {
            "signal_name": "twelve_minus_one_momentum",
            "expression": f"(1.0 + {long_col}) / (1.0 + {short_col}) - 1.0",
            "used_columns": [long_col, short_col],
        }
    fallback = first_available_column(
        columns,
        ("px__ret_252d", "px__ret_252_d", "px__ret_189d", "px__ret_189_d", "px__ret_126d", "px__ret_126_d", "ret_1"),
    )
    if not fallback:
        raise ValueError("Could not resolve a baseline momentum feature from the current feature artifact.")
    return {
        "signal_name": "trailing_return_momentum",
        "expression": fallback,
        "used_columns": [fallback],
    }


def load_feature_columns_frame(feature_artifact: Artifact) -> pd.DataFrame:
    if feature_artifact is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(str(feature_artifact.uri or ""), nrows=1)
    except Exception:
        try:
            return load_artifact_csv_frame(feature_artifact).head(1)
        except Exception:
            return pd.DataFrame()


def _baseline_strategy_config(*, bucket_count: int, score_expression: str) -> dict[str, Any]:
    return {
        "rebalance_freq": "M",
        "gross_exposure": 1.0,
        "selection_side": "long_short",
        "signal_combination": "direct",
        "combined_score_expr": str(score_expression),
        "portfolio_construction": "cross_sectional_quantiles",
        "cross_sectional_score_field": "strategy_score",
        "cross_sectional_bucket_count": int(bucket_count),
        "long_bucket": "top",
        "short_bucket": "bottom",
        "holding_period_rebalances": 1,
        "ranking_lag_days": 0,
        "higher_score_is_better": True,
    }


def _model_strategy_config(*, bucket_count: int) -> dict[str, Any]:
    return {
        "rebalance_freq": "M",
        "gross_exposure": 1.0,
        "selection_side": "long_short",
        "signal_combination": "direct",
        "action_source_field": "ranking",
        "portfolio_construction": "cross_sectional_quantiles",
        "cross_sectional_score_field": "strategy_score",
        "cross_sectional_bucket_count": int(bucket_count),
        "long_bucket": "top",
        "short_bucket": "bottom",
        "holding_period_rebalances": 1,
        "ranking_lag_days": 0,
        "higher_score_is_better": True,
    }


def _default_model_config(*, horizon_days: int) -> dict[str, Any]:
    return {
        "algorithm": "random_forest_regressor",
        "task_type": "regression",
        "target_col": "future_rank_pct",
        "label_k": int(horizon_days),
        "split_ratio": 1.0,
        "sample_weight_mode": "uniform",
        "params": {
            "n_estimators": 80,
            "max_depth": 5,
            "min_samples_leaf": 8,
            "n_jobs": -1,
        },
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
        "min_dollar_volume": 10_000_000.0,
    }


def _default_validation_config() -> dict[str, Any]:
    return {
        "min_trained_rows": 100,
        "min_rows_scored": 50,
        "min_selected_rows": 10,
        "min_trades": 10,
        "min_benchmark_days": 30,
        "min_valid_fold_rate": 0.6,
        "max_fold_excess_std": 0.75,
    }


def _single_summary_row(summary: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    rows = [dict(row) for row in list(summary.get("summary_rows") or [])]
    if rows:
        return rows[0]
    failed = list(summary.get("failed_variants") or [])
    detail = failed[0].get("error") if failed else "no summary rows produced"
    raise ValueError(f"{label} did not produce a usable summary row: {detail}")


def _resolve_artifact(artifact_id: Any, *, label: str) -> Artifact:
    artifact = Artifact.objects.filter(pk=int(artifact_id or 0)).first()
    if artifact is None:
        raise ValueError(f"{label} artifact #{artifact_id} was not found.")
    return artifact


def _build_model_variants(feature_artifact: Artifact, *, base_model_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    family_names = available_feature_families(feature_artifact)
    variants: list[dict[str, Any]] = [
        {
            "variant_name": "oracle_rank_rf_all_features",
            "variant_label": "ML All Features",
            "feature_scope": "all_features",
            "model_config": {
                **dict(base_model_config),
                "model_name": "oracle_rank_rf_all_features",
            },
        }
    ]
    if "prices_div_adj" in family_names:
        variants.append(
            {
                "variant_name": "oracle_rank_rf_prices_only",
                "variant_label": "ML Price Features",
                "feature_scope": "prices_only",
                "model_config": {
                    **dict(base_model_config),
                    "model_name": "oracle_rank_rf_prices_only",
                    "feature_families": list(DEFAULT_PRICE_ONLY_FAMILIES),
                },
            }
        )
    context_families = [family for family in family_names if family not in DEFAULT_PRICE_ONLY_FAMILIES]
    if context_families:
        variants.append(
            {
                "variant_name": "oracle_rank_rf_context_only",
                "variant_label": "ML Context Features",
                "feature_scope": "context_only",
                "model_config": {
                    **dict(base_model_config),
                    "model_name": "oracle_rank_rf_context_only",
                    "feature_families": list(context_families),
                },
            }
        )
    return variants


def _annotate_fold_row(
    row: Mapping[str, Any],
    *,
    variant_name: str,
    variant_kind: str,
    variant_label: str,
    feature_scope: str,
    fold_name: str,
    train_end_date: str,
    backtest_start_date: str,
    backtest_end_date: str,
) -> dict[str, Any]:
    out = dict(row)
    out["variant_name"] = str(variant_name)
    out["variant_kind"] = str(variant_kind)
    out["variant_label"] = str(variant_label)
    out["feature_scope"] = str(feature_scope)
    out["fold_name"] = str(fold_name)
    out["train_end_date"] = str(train_end_date)
    out["backtest_start_date"] = str(backtest_start_date)
    out["backtest_end_date"] = str(backtest_end_date)
    return out


def _aggregate_performance_rows(summary_rows: list[dict[str, Any]], validation_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    aggregate_rows = _apply_walk_forward_gates(
        _aggregate_walk_forward_rows([dict(row) for row in summary_rows]),
        validation_config=dict(validation_config),
    )
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        by_variant.setdefault(str(row.get("variant_name") or ""), []).append(dict(row))
    enriched: list[dict[str, Any]] = []
    for row in aggregate_rows:
        variant_name = str(row.get("variant_name") or "")
        variant_rows = by_variant.get(variant_name, [])
        walk_forward = _summarize_walk_forward_metrics(variant_rows)
        positive_folds = sum(1 for item in variant_rows if _safe_float(item.get("cumulative_return")) > 0.0)
        sharpe_values = [_safe_float(item.get("sharpe")) for item in variant_rows]
        item = dict(row)
        item.update(
            {
                "variant_kind": str(variant_rows[0].get("variant_kind") or "") if variant_rows else "",
                "variant_label": str(variant_rows[0].get("variant_label") or "") if variant_rows else "",
                "feature_scope": str(variant_rows[0].get("feature_scope") or "") if variant_rows else "",
                "sharpe": _safe_float(walk_forward.get("sharpe")),
                "total_return": _safe_float(walk_forward.get("total_return")),
                "final_equity": _safe_float(walk_forward.get("final_equity"), 1.0),
                "max_drawdown": _safe_float(walk_forward.get("max_drawdown")),
                "avg_turnover": _safe_float(walk_forward.get("avg_turnover")),
                "total_turnover": _safe_float(walk_forward.get("total_turnover")),
                "trade_count": int(_safe_float(walk_forward.get("trade_count"))),
                "walk_forward_start_date": str(walk_forward.get("start_date") or ""),
                "walk_forward_end_date": str(walk_forward.get("end_date") or ""),
                "positive_fold_count": int(positive_folds),
                "positive_fold_rate": round(float(positive_folds) / float(len(variant_rows)) if variant_rows else 0.0, 8),
                "mean_fold_sharpe": round(float(sum(sharpe_values) / len(sharpe_values)) if sharpe_values else 0.0, 8),
                "fold_sharpe_std": round(float(pd.Series(sharpe_values).std(ddof=0)) if sharpe_values else 0.0, 8),
            }
        )
        enriched.append(item)
    enriched.sort(
        key=lambda item: (
            _safe_float(item.get("sharpe")),
            _safe_float(item.get("total_return")),
            -abs(_safe_float(item.get("max_drawdown"))),
        ),
        reverse=True,
    )
    return enriched


def _top_bucket_return(aggregate_bucket_rows: Sequence[Mapping[str, Any]], variant_name: str, bucket: int) -> float:
    for row in aggregate_bucket_rows:
        if str(row.get("variant_name") or "") == str(variant_name) and int(row.get("bucket") or 0) == int(bucket):
            return _safe_float(row.get("avg_forward_return"))
    return 0.0


def _cohort_preview(rows: Sequence[Mapping[str, Any]], *, variant_name: str, cohort_kind: str, limit: int = 3) -> str:
    matched = [
        dict(row)
        for row in rows
        if str(row.get("variant_name") or "") == str(variant_name)
        and str(row.get("cohort_kind") or "") == str(cohort_kind)
    ]
    preview = matched[: max(int(limit), 0)]
    return ", ".join(
        f"{row['cohort_value']} ({_safe_float(row.get('selection_share')):.2f})"
        for row in preview
    )


def write_oracle_ranking_signal_report(
    *,
    report_path: Path,
    payload: Mapping[str, Any],
) -> None:
    aggregate_rows = [dict(row) for row in list(payload.get("aggregate_rows") or [])]
    ranking_rows = [dict(row) for row in list(payload.get("ranking_summary_aggregate_rows") or [])]
    bucket_rows = [dict(row) for row in list(payload.get("bucket_aggregate_rows") or [])]
    cohort_rows = [dict(row) for row in list(payload.get("cohort_aggregate_rows") or [])]
    overlap_rows = [dict(row) for row in list(payload.get("overlap_aggregate_rows") or [])]
    stability_rows = [dict(row) for row in list(payload.get("stability_summary_rows") or [])]
    symbols = [str(symbol) for symbol in list(payload.get("symbols") or [])]
    model_variants = [dict(row) for row in list(payload.get("model_variants") or [])]
    momentum_signal = dict(payload.get("momentum_signal") or {})
    label_meta = dict(payload.get("rank_label_metadata") or {})
    bucket_count = int(payload.get("bucket_count") or 10)
    if not aggregate_rows:
        raise ValueError("Expected aggregate rows to write the oracle ranking report.")

    baseline_row = next((row for row in aggregate_rows if str(row.get("variant_name")) == "baseline_momentum"), {})
    variant_lookup = {str(row.get("variant_name") or ""): row for row in aggregate_rows}
    ranking_lookup = {str(row.get("variant_name") or ""): row for row in ranking_rows}
    stability_lookup = {str(row.get("variant_name") or ""): row for row in stability_rows}
    best_model = max(
        [row for row in aggregate_rows if str(row.get("variant_kind") or "") == "model"],
        key=lambda row: (
            _safe_float(row.get("sharpe")),
            _safe_float(row.get("total_return")),
            _safe_float(ranking_lookup.get(str(row.get("variant_name") or ""), {}).get("mean_spearman_ic")),
        ),
        default={},
    )
    lines = [
        "# Oracle Ranking Signal Research Report",
        "",
        "## 1. Strategy implementation",
        "",
        "- Objective: test whether an oracle-trained regression model that predicts future cross-sectional rank percentiles can produce better cross-sectional portfolios than a simple momentum ranking baseline.",
        f"- Oracle target: `future_rank_pct`, computed on monthly rebalance dates from {int(label_meta.get('rebalance_dates') or 0)} labeled cross-sections using a {int(label_meta.get('horizon_days') or 0)}-day forward return horizon and a {int(label_meta.get('start_offset_days') or 0)}-day execution offset.",
        f"- Baseline ranking signal: `{momentum_signal.get('expression') or ''}`.",
        "- Portfolio construction: existing `portfolio_construction=\"cross_sectional_quantiles\"` with equal-weight long top bucket / short bottom bucket and monthly rebalancing.",
        f"- Universe: {len(symbols)} US large-cap symbols with sufficient price history.",
        "- Symbols: " + ", ".join(symbols),
        "",
        "## 2. Experiment results",
        "",
        f"- Baseline Sharpe {(_safe_float(baseline_row.get('sharpe'))):.3f}, total return {_pct(baseline_row.get('total_return'))}, max drawdown {_pct(baseline_row.get('max_drawdown'))}, positive fold rate {_safe_float(baseline_row.get('positive_fold_rate')):.2f}.",
        f"- Best model variant: {best_model.get('variant_name', '')} | Sharpe {(_safe_float(best_model.get('sharpe'))):.3f} | total return {_pct(best_model.get('total_return'))} | max drawdown {_pct(best_model.get('max_drawdown'))}.",
        "",
        "| Variant | Sharpe | Total Return | Max DD | Turnover | Trades | Mean IC | Mean Long-Short Spread | Positive Fold Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        ranking_row = ranking_lookup.get(str(row.get("variant_name") or ""), {})
        lines.append(
            "| "
            + f"{row.get('variant_name', '')} | "
            + f"{_safe_float(row.get('sharpe')):.3f} | "
            + f"{_pct(row.get('total_return'))} | "
            + f"{_pct(row.get('max_drawdown'))} | "
            + f"{_safe_float(row.get('total_turnover')):.2f} | "
            + f"{int(_safe_float(row.get('trade_count')))} | "
            + f"{_safe_float(ranking_row.get('mean_spearman_ic')):.3f} | "
            + f"{_pct(ranking_row.get('mean_long_short_spread'))} | "
            + f"{_safe_float(row.get('positive_fold_rate')):.2f} |"
        )
    lines.extend(
        [
            "",
            "## 3. Ranking diagnostics",
            "",
            f"- Baseline mean Spearman IC: {_safe_float(ranking_lookup.get('baseline_momentum', {}).get('mean_spearman_ic')):.3f}.",
            f"- Best model mean Spearman IC: {_safe_float(ranking_lookup.get(str(best_model.get('variant_name') or ''), {}).get('mean_spearman_ic')):.3f}.",
            f"- Baseline top-bottom bucket spread: {_pct(ranking_lookup.get('baseline_momentum', {}).get('mean_long_short_spread'))}.",
            f"- Best model top-bottom bucket spread: {_pct(ranking_lookup.get(str(best_model.get('variant_name') or ''), {}).get('mean_long_short_spread'))}.",
            f"- Baseline top bucket avg forward return: {_pct(_top_bucket_return(bucket_rows, 'baseline_momentum', bucket_count))}.",
            f"- Best model top bucket avg forward return: {_pct(_top_bucket_return(bucket_rows, str(best_model.get('variant_name') or ''), bucket_count))}.",
            "",
            "## 4. Cohort diagnostics",
            "",
            "- Baseline top-bucket sector mix: " + (_cohort_preview(cohort_rows, variant_name="baseline_momentum", cohort_kind="sector") or "n/a"),
            "- Best-model top-bucket sector mix: " + (_cohort_preview(cohort_rows, variant_name=str(best_model.get("variant_name") or ""), cohort_kind="sector") or "n/a"),
            "- Baseline stock/ETF mix: " + (_cohort_preview(cohort_rows, variant_name="baseline_momentum", cohort_kind="instrument_type", limit=2) or "n/a"),
            "- Best-model stock/ETF mix: " + (_cohort_preview(cohort_rows, variant_name=str(best_model.get("variant_name") or ""), cohort_kind="instrument_type", limit=2) or "n/a"),
        ]
    )
    for variant in model_variants:
        variant_name = str(variant.get("variant_name") or "")
        overlap = next((row for row in overlap_rows if str(row.get("right_variant_name") or "") == variant_name), {})
        stability = stability_lookup.get(variant_name, {})
        lines.append(
            f"- {variant_name}: overlap with baseline winners {_safe_float(overlap.get('jaccard')):.3f} average Jaccard, fold stability {_safe_float(stability.get('mean_pairwise_jaccard')):.3f}."
        )
    lines.extend(
        [
            "",
            "## 5. Interpretation",
            "",
            f"- Does ML beat simple momentum? {'yes' if _safe_float(best_model.get('sharpe')) > _safe_float(baseline_row.get('sharpe')) and _safe_float(best_model.get('total_return')) > _safe_float(baseline_row.get('total_return')) else 'mixed/no'}.",
            "- Does broader context add value beyond price-only features? Compare `oracle_rank_rf_all_features` against `oracle_rank_rf_prices_only` in the table above; the gap is the cleanest test because both use the same target, model family, and portfolio construction.",
            "- Are improvements consistent across folds? Use positive fold rate plus the fold-stability rows; higher Sharpe with weak IC or low fold consistency should be treated cautiously.",
            "- Is extra complexity justified? The answer depends on whether the best ML variant improves both portfolio outcomes and ranking IC over the baseline, not just one of them.",
            "",
            "## 6. Platform capabilities added",
            "",
            "- Reusable cross-sectional rank-percentile label artifacts backed by the standard `LABELS` artifact type.",
            "- Reusable ranking diagnostics for rank IC, bucket spreads, cohort composition, winner overlap, and fold stability.",
            "- A reusable oracle-ranking research runner that composes the existing feature, model, strategy, and backtest infrastructure instead of building a parallel notebook path.",
            "",
            "## 7. Output artifacts",
            "",
            f"- Summary JSON: `{payload.get('summary_json_path') or ''}`",
            f"- Aggregate results CSV: `{payload.get('summary_csv_path') or ''}`",
            f"- Fold results CSV: `{payload.get('fold_summary_csv_path') or ''}`",
            f"- Ranking summary CSV: `{payload.get('ranking_summary_aggregate_csv_path') or ''}`",
            f"- Bucket returns CSV: `{payload.get('bucket_aggregate_csv_path') or ''}`",
            f"- Cohort diagnostics CSV: `{payload.get('cohort_aggregate_csv_path') or ''}`",
            f"- Winner overlap CSV: `{payload.get('overlap_aggregate_csv_path') or ''}`",
            f"- Fold stability CSV: `{payload.get('stability_summary_csv_path') or ''}`",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_oracle_ranking_signal_research(
    *,
    requested_symbols: Sequence[str] | None = None,
    symbol_limit: int = 30,
    candidate_limit: int = 100,
    min_market_cap: float = 25_000_000_000.0,
    test_start_year: int = 2021,
    test_end_year: int = 2025,
    lookback_days: int = 252,
    forward_horizon_days: int = 21,
    start_offset_days: int = 1,
    bucket_count: int = 10,
    fee_bps: float = 2.0,
    slippage_bps: float = 8.0,
    short_borrow_bps_annual: float = 25.0,
    execution_delay_days: int = 1,
    output_basename: str = "oracle_ranking_signal_research",
    resume_existing: bool = False,
) -> dict[str, Any]:
    output_dir = Path("data") / "pipeline_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{output_basename}.json"
    csv_path = output_dir / f"{output_basename}.csv"
    fold_csv_path = output_dir / f"{output_basename}__fold_rows.csv"
    ranking_summary_csv_path = output_dir / f"{output_basename}__ranking_summary_rows.csv"
    ranking_summary_aggregate_csv_path = output_dir / f"{output_basename}__ranking_summary_aggregate.csv"
    bucket_rows_csv_path = output_dir / f"{output_basename}__bucket_rows.csv"
    bucket_aggregate_csv_path = output_dir / f"{output_basename}__bucket_aggregate.csv"
    cohort_rows_csv_path = output_dir / f"{output_basename}__cohort_rows.csv"
    cohort_aggregate_csv_path = output_dir / f"{output_basename}__cohort_aggregate.csv"
    overlap_rows_csv_path = output_dir / f"{output_basename}__overlap_rows.csv"
    overlap_aggregate_csv_path = output_dir / f"{output_basename}__overlap_aggregate.csv"
    top_bucket_rows_csv_path = output_dir / f"{output_basename}__top_bucket_rows.csv"
    stability_summary_csv_path = output_dir / f"{output_basename}__stability_summary.csv"
    stability_symbol_csv_path = output_dir / f"{output_basename}__stability_symbols.csv"
    coverage_csv_path = output_dir / f"{output_basename}__coverage.csv"
    if resume_existing:
        cached_payload = _load_cached_payload(
            json_path,
            required_keys=("summary_rows", "aggregate_rows", "symbols"),
            schema_version=ORACLE_RANKING_SIGNAL_SCHEMA_VERSION,
        )
        if cached_payload is not None:
            cached_payload["summary_json_path"] = str(json_path)
            cached_payload["summary_csv_path"] = str(csv_path)
            return cached_payload

    folds = build_yearly_folds(int(test_start_year), int(test_end_year))
    symbols, coverage_rows, missing_symbols = resolve_research_symbols(
        requested_symbols=requested_symbols,
        symbol_limit=int(symbol_limit),
        candidate_limit=int(candidate_limit),
        min_market_cap=float(min_market_cap),
        test_start_year=int(test_start_year),
        test_end_year=int(test_end_year),
        lookback_days=int(lookback_days),
        forward_horizon_days=int(forward_horizon_days),
        start_offset_days=int(start_offset_days),
    )
    if not symbols:
        raise ValueError("No symbols were available after applying the history screen.")

    universe_artifact = _resolve_or_build_universe_artifact(symbols=symbols, output_basename=output_basename)
    feature_artifact = _resolve_or_build_feature_artifact(
        universe_artifact=universe_artifact,
        symbols=symbols,
        feature_config={},
        output_basename=output_basename,
    )
    momentum_signal = _resolve_momentum_signal_spec(feature_artifact)
    rank_label_spec = CrossSectionalRankLabelSpec(
        horizon_days=int(forward_horizon_days),
        rebalance_freq="M",
        start_offset_days=int(start_offset_days),
        minimum_cross_section=max(2, int(min(len(symbols), max(10, bucket_count * 2)))),
        target_col="future_rank_pct",
        forward_return_col="trade_return",
    )
    label_artifact = resolve_or_build_cross_sectional_rank_label_artifact(
        feature_artifact=feature_artifact,
        spec=rank_label_spec,
        output_basename=f"{output_basename}__rank_labels",
    )
    symbol_metadata_lookup = build_symbol_metadata_lookup(symbols)
    baseline_strategy_config = _baseline_strategy_config(
        bucket_count=int(bucket_count),
        score_expression=str(momentum_signal.get("expression") or ""),
    )
    model_strategy_config = _model_strategy_config(bucket_count=int(bucket_count))
    backtest_config = _default_backtest_config(
        fee_bps=float(fee_bps),
        slippage_bps=float(slippage_bps),
        short_borrow_bps_annual=float(short_borrow_bps_annual),
        execution_delay_days=int(execution_delay_days),
    )
    validation_config = _default_validation_config()
    model_variants = _build_model_variants(
        feature_artifact,
        base_model_config=_default_model_config(horizon_days=int(forward_horizon_days)),
    )

    summary_rows: list[dict[str, Any]] = []
    ranking_summary_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    accumulated_top_bucket_rows: list[dict[str, Any]] = []

    for fold in folds:
        fold_name = str(fold.get("name") or "").strip()
        train_end_date = str(fold.get("train_end_date") or "")
        backtest_start_date = str(fold.get("backtest_start_date") or "")
        backtest_end_date = str(fold.get("backtest_end_date") or "")

        baseline_summary = run_direct_feature_strategy_backtests(
            symbols=symbols,
            train_end_date=train_end_date,
            backtest_start_date=backtest_start_date,
            backtest_end_date=backtest_end_date,
            universe_artifact=universe_artifact,
            feature_artifact=feature_artifact,
            feature_config={},
            strategy_definition_slug="oracle-ranking-baseline",
            strategy_definition_name="Oracle Ranking Baseline Momentum",
            strategy_config=baseline_strategy_config,
            validation_config=validation_config,
            backtest_config=backtest_config,
            output_basename=f"{output_basename}__baseline__{fold_name}",
            resume_existing=resume_existing,
        )
        baseline_row = _annotate_fold_row(
            _single_summary_row(baseline_summary, label=f"{fold_name} baseline"),
            variant_name="baseline_momentum",
            variant_kind="baseline",
            variant_label="Baseline Momentum",
            feature_scope="prices_only",
            fold_name=fold_name,
            train_end_date=train_end_date,
            backtest_start_date=backtest_start_date,
            backtest_end_date=backtest_end_date,
        )
        summary_rows.append(baseline_row)
        baseline_score_frame = build_expression_score_frame(
            feature_artifact,
            score_expression=str(momentum_signal.get("expression") or ""),
            start_date=backtest_start_date,
            end_date=backtest_end_date,
        )
        baseline_panel = build_signal_ranking_panel(
            baseline_score_frame,
            label_artifact,
            target_col="future_rank_pct",
            forward_return_col="trade_return",
            start_date=backtest_start_date,
            end_date=backtest_end_date,
            variant_name="baseline_momentum",
            fold_name=fold_name,
            feature_scope="prices_only",
            variant_kind="baseline",
            variant_label="Baseline Momentum",
            symbol_metadata_lookup=symbol_metadata_lookup,
        )
        baseline_bucketed = assign_cross_sectional_buckets(
            baseline_panel,
            bucket_count=int(bucket_count),
            higher_score_is_better=True,
        )
        ranking_summary_rows.extend(compute_ranking_summary_rows(baseline_bucketed, bucket_count=int(bucket_count)))
        bucket_rows.extend(compute_bucket_return_rows(baseline_bucketed))
        cohort_rows.extend(compute_top_bucket_cohort_rows(baseline_bucketed, bucket_count=int(bucket_count)))
        accumulated_top_bucket_rows.extend(top_bucket_rows(baseline_bucketed, bucket_count=int(bucket_count)))

        for variant in model_variants:
            variant_name = str(variant.get("variant_name") or "")
            variant_label = str(variant.get("variant_label") or variant_name)
            feature_scope = str(variant.get("feature_scope") or "")
            model_summary = run_model_cohort_backtests(
                symbols=symbols,
                fit_job="fit_regressor",
                base_model_config=dict(variant.get("model_config") or {}),
                train_end_date=train_end_date,
                backtest_start_date=backtest_start_date,
                backtest_end_date=backtest_end_date,
                universe_artifact=universe_artifact,
                label_artifact=label_artifact,
                feature_artifact=feature_artifact,
                feature_config={},
                strategy_definition_slug=f"{variant_name}-strategy",
                strategy_definition_name=f"{variant_label} Strategy",
                strategy_config=model_strategy_config,
                validation_config=validation_config,
                backtest_config=backtest_config,
                output_basename=f"{output_basename}__{variant_name}__{fold_name}",
                resume_existing=resume_existing,
            )
            model_row = _annotate_fold_row(
                _single_summary_row(model_summary, label=f"{fold_name} {variant_name}"),
                variant_name=variant_name,
                variant_kind="model",
                variant_label=variant_label,
                feature_scope=feature_scope,
                fold_name=fold_name,
                train_end_date=train_end_date,
                backtest_start_date=backtest_start_date,
                backtest_end_date=backtest_end_date,
            )
            summary_rows.append(model_row)
            prediction_artifact = _resolve_artifact(
                model_row.get("prediction_artifact_id"),
                label=f"{fold_name} {variant_name} prediction",
            )
            model_panel = build_signal_ranking_panel(
                prediction_artifact,
                label_artifact,
                target_col="future_rank_pct",
                forward_return_col="trade_return",
                start_date=backtest_start_date,
                end_date=backtest_end_date,
                variant_name=variant_name,
                fold_name=fold_name,
                feature_scope=feature_scope,
                variant_kind="model",
                variant_label=variant_label,
                symbol_metadata_lookup=symbol_metadata_lookup,
            )
            model_bucketed = assign_cross_sectional_buckets(
                model_panel,
                bucket_count=int(bucket_count),
                higher_score_is_better=True,
            )
            ranking_summary_rows.extend(compute_ranking_summary_rows(model_bucketed, bucket_count=int(bucket_count)))
            bucket_rows.extend(compute_bucket_return_rows(model_bucketed))
            cohort_rows.extend(compute_top_bucket_cohort_rows(model_bucketed, bucket_count=int(bucket_count)))
            overlap_rows.extend(
                compute_bucket_overlap_rows(
                    baseline_bucketed,
                    model_bucketed,
                    bucket_count=int(bucket_count),
                    left_variant_name="baseline_momentum",
                    right_variant_name=variant_name,
                )
            )
            accumulated_top_bucket_rows.extend(top_bucket_rows(model_bucketed, bucket_count=int(bucket_count)))

    aggregate_rows = _aggregate_performance_rows(summary_rows, validation_config=validation_config)
    ranking_summary_aggregate_rows = aggregate_ranking_summary_rows(ranking_summary_rows)
    bucket_aggregate_rows = aggregate_bucket_return_rows(bucket_rows)
    cohort_aggregate_rows = aggregate_top_bucket_cohort_rows(cohort_rows)
    overlap_aggregate_rows = aggregate_bucket_overlap_rows(overlap_rows)
    top_bucket_df = pd.DataFrame(accumulated_top_bucket_rows)
    stability_summary_rows, stability_symbol_rows = compute_top_bucket_stability_rows(
        assign_cross_sectional_buckets(top_bucket_df, bucket_count=int(bucket_count)) if "bucket" not in top_bucket_df.columns else top_bucket_df,
        bucket_count=int(bucket_count),
    )

    payload = {
        "schema_version": ORACLE_RANKING_SIGNAL_SCHEMA_VERSION,
        "mode": "oracle_ranking_signal_research",
        "symbols": symbols,
        "missing_requested_symbols": missing_symbols,
        "coverage_rows": coverage_rows,
        "folds": [dict(fold) for fold in folds],
        "base_artifacts": {
            "universe": int(universe_artifact.id),
            "features": int(feature_artifact.id),
            "labels": int(label_artifact.id),
        },
        "momentum_signal": momentum_signal,
        "rank_label_metadata": dict(label_artifact.metadata or {}),
        "model_variants": model_variants,
        "bucket_count": int(bucket_count),
        "backtest_config": backtest_config,
        "validation_config": validation_config,
        "summary_rows": summary_rows,
        "aggregate_rows": aggregate_rows,
        "ranking_summary_rows": ranking_summary_rows,
        "ranking_summary_aggregate_rows": ranking_summary_aggregate_rows,
        "bucket_rows": bucket_rows,
        "bucket_aggregate_rows": bucket_aggregate_rows,
        "cohort_rows": cohort_rows,
        "cohort_aggregate_rows": cohort_aggregate_rows,
        "overlap_rows": overlap_rows,
        "overlap_aggregate_rows": overlap_aggregate_rows,
        "top_bucket_rows": accumulated_top_bucket_rows,
        "stability_summary_rows": stability_summary_rows,
        "stability_symbol_rows": stability_symbol_rows,
        "summary_json_path": str(json_path),
        "summary_csv_path": str(csv_path),
        "fold_summary_csv_path": str(fold_csv_path),
        "ranking_summary_csv_path": str(ranking_summary_csv_path),
        "ranking_summary_aggregate_csv_path": str(ranking_summary_aggregate_csv_path),
        "bucket_rows_csv_path": str(bucket_rows_csv_path),
        "bucket_aggregate_csv_path": str(bucket_aggregate_csv_path),
        "cohort_rows_csv_path": str(cohort_rows_csv_path),
        "cohort_aggregate_csv_path": str(cohort_aggregate_csv_path),
        "overlap_rows_csv_path": str(overlap_rows_csv_path),
        "overlap_aggregate_csv_path": str(overlap_aggregate_csv_path),
        "top_bucket_rows_csv_path": str(top_bucket_rows_csv_path),
        "stability_summary_csv_path": str(stability_summary_csv_path),
        "stability_symbol_csv_path": str(stability_symbol_csv_path),
        "coverage_csv_path": str(coverage_csv_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_rows_csv(csv_path, aggregate_rows)
    _write_rows_csv(fold_csv_path, summary_rows)
    _write_rows_csv(ranking_summary_csv_path, ranking_summary_rows)
    _write_rows_csv(ranking_summary_aggregate_csv_path, ranking_summary_aggregate_rows)
    _write_rows_csv(bucket_rows_csv_path, bucket_rows)
    _write_rows_csv(bucket_aggregate_csv_path, bucket_aggregate_rows)
    _write_rows_csv(cohort_rows_csv_path, cohort_rows)
    _write_rows_csv(cohort_aggregate_csv_path, cohort_aggregate_rows)
    _write_rows_csv(overlap_rows_csv_path, overlap_rows)
    _write_rows_csv(overlap_aggregate_csv_path, overlap_aggregate_rows)
    _write_rows_csv(top_bucket_rows_csv_path, accumulated_top_bucket_rows)
    _write_rows_csv(stability_summary_csv_path, stability_summary_rows)
    _write_rows_csv(stability_symbol_csv_path, stability_symbol_rows)
    _write_rows_csv(coverage_csv_path, coverage_rows)
    return payload


__all__ = [
    "DEFAULT_EXCLUDED_SYMBOL_PREFIXES",
    "ORACLE_RANKING_SIGNAL_SCHEMA_VERSION",
    "build_yearly_folds",
    "run_oracle_ranking_signal_research",
    "write_oracle_ranking_signal_report",
]
