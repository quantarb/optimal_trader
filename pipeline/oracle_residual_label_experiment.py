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
    run_model_cohort_backtests,
)
from .cross_sectional_rank_labels import (
    CrossSectionalRankLabelSpec,
    first_available_column,
    resolve_or_build_cross_sectional_rank_label_artifact,
)
from .direct_strategy_runner import _summarize_walk_forward_metrics, run_direct_feature_strategy_backtests
from .models import Artifact
from .oracle_ranking_signal_research import build_yearly_folds, resolve_research_symbols
from .ranking_diagnostics import build_expression_score_frame


ORACLE_RESIDUAL_EXPERIMENT_SCHEMA_VERSION = 1
DEFAULT_SCORE_COL_CANDIDATES: tuple[str, ...] = (
    "score",
    "signal_score",
    "prediction_score",
    "prediction",
    "strategy_score",
    "ranking",
)


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
    items = [dict(row) for row in list(rows or [])]
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


def _resolve_momentum_signal_spec(feature_artifact: Artifact) -> dict[str, Any]:
    feature_df = load_artifact_csv_frame(feature_artifact).head(1)
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


def _default_model_config(*, target_col: str, horizon_days: int, model_name: str) -> dict[str, Any]:
    return {
        "algorithm": "random_forest_regressor",
        "task_type": "regression",
        "target_col": str(target_col),
        "label_k": int(horizon_days),
        "split_ratio": 1.0,
        "sample_weight_mode": "uniform",
        "model_name": str(model_name),
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


def _annotate_fold_row(
    row: Mapping[str, Any],
    *,
    variant_name: str,
    variant_kind: str,
    variant_label: str,
    label_type: str,
    fold_name: str,
    train_end_date: str,
    backtest_start_date: str,
    backtest_end_date: str,
) -> dict[str, Any]:
    out = dict(row)
    out["variant_name"] = str(variant_name)
    out["variant_kind"] = str(variant_kind)
    out["variant_label"] = str(variant_label)
    out["label_type"] = str(label_type)
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
                "label_type": str(variant_rows[0].get("label_type") or "") if variant_rows else "",
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


def _spearman(series_a: pd.Series, series_b: pd.Series) -> float | None:
    valid = pd.concat([pd.to_numeric(series_a, errors="coerce"), pd.to_numeric(series_b, errors="coerce")], axis=1).dropna()
    if len(valid) < 2:
        return None
    if valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return None
    value = valid.iloc[:, 0].rank().corr(valid.iloc[:, 1].rank())
    return None if pd.isna(value) else float(value)


def _load_score_frame(score_frame_or_artifact) -> pd.DataFrame:
    score_df = (
        load_artifact_csv_frame(score_frame_or_artifact)
        if hasattr(score_frame_or_artifact, "uri")
        else pd.DataFrame(score_frame_or_artifact).copy()
    )
    if score_df.empty:
        return pd.DataFrame(columns=["date", "symbol", "score"])
    score_col = "score" if "score" in score_df.columns else first_available_column(score_df.columns, DEFAULT_SCORE_COL_CANDIDATES)
    if not score_col:
        raise ValueError("Could not resolve a usable score column from the scoring output.")
    out = score_df[["date", "symbol", score_col]].copy().rename(columns={score_col: "score"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    return out.dropna(subset=["date", "symbol", "score"]).sort_values(["date", "symbol"]).reset_index(drop=True)


def compute_prediction_diagnostic_rows(
    score_frame_or_artifact,
    label_artifact: Artifact,
    *,
    variant_name: str,
    variant_kind: str,
    variant_label: str,
    label_type: str,
    fold_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
    raw_return_col: str = "trade_return",
    residual_return_col: str = "residual_return",
    target_col: str = "future_rank_pct",
) -> dict[str, Any]:
    score_df = _load_score_frame(score_frame_or_artifact)
    label_df = load_artifact_csv_frame(label_artifact)
    keep_cols = [
        column
        for column in ["date", "symbol", raw_return_col, residual_return_col, target_col]
        if column in label_df.columns
    ]
    if len(keep_cols) < 4:
        raise ValueError("Label artifact must contain raw return, residual return, and target columns for diagnostics.")
    label_panel = label_df[keep_cols].copy()
    label_panel["date"] = pd.to_datetime(label_panel["date"], errors="coerce")
    label_panel["symbol"] = label_panel["symbol"].astype(str).str.strip().str.upper()
    if start_date:
        start_ts = pd.Timestamp(str(start_date))
        score_df = score_df[score_df["date"] >= start_ts].copy()
        label_panel = label_panel[label_panel["date"] >= start_ts].copy()
    if end_date:
        end_ts = pd.Timestamp(str(end_date))
        score_df = score_df[score_df["date"] <= end_ts].copy()
        label_panel = label_panel[label_panel["date"] <= end_ts].copy()
    merged = score_df.merge(label_panel, on=["date", "symbol"], how="inner")
    if merged.empty:
        return {
            "variant_name": str(variant_name),
            "variant_kind": str(variant_kind),
            "variant_label": str(variant_label),
            "label_type": str(label_type),
            "fold_name": str(fold_name),
            "scored_rows": 0,
            "rebalance_dates": 0,
            "mean_forward_return_ic": 0.0,
            "mean_residual_return_ic": 0.0,
            "mean_target_rank_ic": 0.0,
        }
    daily_forward_ic: list[float] = []
    daily_residual_ic: list[float] = []
    daily_target_ic: list[float] = []
    for _date_value, group in merged.groupby("date", sort=True):
        forward_ic = _spearman(group["score"], group[raw_return_col])
        residual_ic = _spearman(group["score"], group[residual_return_col])
        target_ic = _spearman(group["score"], group[target_col])
        if forward_ic is not None:
            daily_forward_ic.append(forward_ic)
        if residual_ic is not None:
            daily_residual_ic.append(residual_ic)
        if target_ic is not None:
            daily_target_ic.append(target_ic)
    return {
        "variant_name": str(variant_name),
        "variant_kind": str(variant_kind),
        "variant_label": str(variant_label),
        "label_type": str(label_type),
        "fold_name": str(fold_name),
        "scored_rows": int(len(merged)),
        "rebalance_dates": int(merged["date"].nunique()),
        "mean_forward_return_ic": round(float(sum(daily_forward_ic) / len(daily_forward_ic)) if daily_forward_ic else 0.0, 8),
        "mean_residual_return_ic": round(float(sum(daily_residual_ic) / len(daily_residual_ic)) if daily_residual_ic else 0.0, 8),
        "mean_target_rank_ic": round(float(sum(daily_target_ic) / len(daily_target_ic)) if daily_target_ic else 0.0, 8),
    }


def aggregate_prediction_diagnostic_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame([dict(row) for row in list(rows or [])])
    if df.empty:
        return []
    grouped = (
        df.groupby(["variant_name", "variant_kind", "variant_label", "label_type"], dropna=False, sort=True)
        .agg(
            fold_count=("fold_name", "nunique"),
            scored_rows=("scored_rows", "sum"),
            rebalance_dates=("rebalance_dates", "sum"),
            mean_forward_return_ic=("mean_forward_return_ic", "mean"),
            mean_residual_return_ic=("mean_residual_return_ic", "mean"),
            mean_target_rank_ic=("mean_target_rank_ic", "mean"),
        )
        .reset_index()
    )
    numeric_cols = ["mean_forward_return_ic", "mean_residual_return_ic", "mean_target_rank_ic"]
    grouped[numeric_cols] = grouped[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return grouped.to_dict(orient="records")


def build_comparison_summary_rows(
    performance_rows: Sequence[Mapping[str, Any]],
    diagnostic_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    perf_df = pd.DataFrame([dict(row) for row in list(performance_rows or [])])
    diag_df = pd.DataFrame([dict(row) for row in list(diagnostic_rows or [])])
    if perf_df.empty:
        return []
    if diag_df.empty:
        return perf_df.to_dict(orient="records")
    merged = perf_df.merge(
        diag_df[
            [
                "variant_name",
                "mean_forward_return_ic",
                "mean_residual_return_ic",
                "mean_target_rank_ic",
            ]
        ].copy(),
        on="variant_name",
        how="left",
    )
    for column in ["mean_forward_return_ic", "mean_residual_return_ic", "mean_target_rank_ic"]:
        merged[column] = pd.to_numeric(merged.get(column), errors="coerce").fillna(0.0)
    return merged.to_dict(orient="records")


def write_oracle_residual_label_report(
    *,
    report_path: Path,
    payload: Mapping[str, Any],
) -> None:
    comparison_rows = [dict(row) for row in list(payload.get("comparison_summary_rows") or [])]
    diagnostic_rows = [dict(row) for row in list(payload.get("prediction_diagnostics_aggregate_rows") or [])]
    label_meta = dict(payload.get("rank_label_metadata") or {})
    symbols = [str(symbol) for symbol in list(payload.get("symbols") or [])]
    if not comparison_rows:
        raise ValueError("Expected comparison rows to write the oracle residual label report.")
    row_lookup = {str(row.get("variant_name") or ""): row for row in comparison_rows}
    raw_row = row_lookup.get("raw_label_model", {})
    residual_row = row_lookup.get("residual_label_model", {})
    baseline_row = row_lookup.get("baseline_momentum", {})
    lines = [
        "# Oracle Residual Label Experiment",
        "",
        "## 1. Design",
        "",
        "- Objective: compare the same oracle ranking pipeline trained on raw forward-return ranks versus factor-residualized forward-return ranks.",
        f"- Label horizon: {int(label_meta.get('horizon_days') or 0)} business days with monthly rebalance dates.",
        "- Raw target: `future_rank_pct`, the cross-sectional percentile rank of `trade_return`.",
        "- Residual target: `residual_rank_pct`, where `residual_return = trade_return - factor_expected_return` from a per-date cross-sectional regression.",
        "- Residualization proxies: existing size, momentum, volatility features plus sector dummies and stock/ETF type from symbol metadata.",
        f"- Universe: {len(symbols)} symbols.",
        "- Symbols: " + ", ".join(symbols),
        "",
        "## 2. Results",
        "",
        "| Variant | Sharpe | Total Return | Max DD | Turnover | Trades | IC vs Raw Return | IC vs Residual Return | Positive Fold Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison_rows:
        lines.append(
            "| "
            + f"{row.get('variant_name', '')} | "
            + f"{_safe_float(row.get('sharpe')):.3f} | "
            + f"{_pct(row.get('total_return'))} | "
            + f"{_pct(row.get('max_drawdown'))} | "
            + f"{_safe_float(row.get('total_turnover')):.2f} | "
            + f"{int(_safe_float(row.get('trade_count')))} | "
            + f"{_safe_float(row.get('mean_forward_return_ic')):.3f} | "
            + f"{_safe_float(row.get('mean_residual_return_ic')):.3f} | "
            + f"{_safe_float(row.get('positive_fold_rate')):.2f} |"
        )
    interpretation = "mixed"
    if _safe_float(residual_row.get("sharpe")) > _safe_float(raw_row.get("sharpe")) and _safe_float(residual_row.get("total_return")) > _safe_float(raw_row.get("total_return")):
        interpretation = "Residual labels helped: factor noise was hurting the learning signal."
    elif _safe_float(raw_row.get("sharpe")) > _safe_float(residual_row.get("sharpe")) and _safe_float(raw_row.get("total_return")) > _safe_float(residual_row.get("total_return")):
        interpretation = "Raw labels helped: the profitable edge appears to rely on systematic factor exposures."
    baseline_dominance = ""
    if (
        _safe_float(baseline_row.get("sharpe")) > max(_safe_float(raw_row.get("sharpe")), _safe_float(residual_row.get("sharpe")))
        and _safe_float(baseline_row.get("total_return")) > max(_safe_float(raw_row.get("total_return")), _safe_float(residual_row.get("total_return")))
    ):
        baseline_dominance = " Neither ML variant beat the simple momentum baseline in this pilot."
    lines.extend(
        [
            "",
            "## 3. Interpretation",
            "",
            f"- Momentum baseline: Sharpe {_safe_float(baseline_row.get('sharpe')):.3f}, total return {_pct(baseline_row.get('total_return'))}.",
            f"- Raw-label model: Sharpe {_safe_float(raw_row.get('sharpe')):.3f}, total return {_pct(raw_row.get('total_return'))}, raw-return IC {_safe_float(raw_row.get('mean_forward_return_ic')):.3f}.",
            f"- Residual-label model: Sharpe {_safe_float(residual_row.get('sharpe')):.3f}, total return {_pct(residual_row.get('total_return'))}, residual-return IC {_safe_float(residual_row.get('mean_residual_return_ic')):.3f}.",
            f"- Conclusion: {interpretation}{baseline_dominance}",
            "",
            "## 4. Artifacts",
            "",
            f"- Summary JSON: `{payload.get('summary_json_path') or ''}`",
            f"- Comparison summary CSV: `{payload.get('comparison_summary_csv_path') or ''}`",
            f"- Fold results CSV: `{payload.get('fold_results_csv_path') or ''}`",
            f"- Prediction diagnostics CSV: `{payload.get('prediction_diagnostics_csv_path') or ''}`",
            f"- Aggregate prediction diagnostics CSV: `{payload.get('prediction_diagnostics_aggregate_csv_path') or ''}`",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_oracle_residual_label_experiment(
    *,
    requested_symbols: Sequence[str] | None = None,
    symbol_limit: int = 20,
    candidate_limit: int = 60,
    min_market_cap: float = 25_000_000_000.0,
    test_start_year: int = 2022,
    test_end_year: int = 2025,
    lookback_days: int = 252,
    forward_horizon_days: int = 21,
    start_offset_days: int = 1,
    bucket_count: int = 10,
    fee_bps: float = 2.0,
    slippage_bps: float = 8.0,
    short_borrow_bps_annual: float = 25.0,
    execution_delay_days: int = 1,
    output_dirname: str = "oracle_residual_experiment",
    resume_existing: bool = False,
) -> dict[str, Any]:
    output_dir = Path("data") / "pipeline_artifacts" / str(output_dirname)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    comparison_summary_csv_path = output_dir / "comparison_summary.csv"
    fold_results_csv_path = output_dir / "fold_results.csv"
    prediction_diagnostics_csv_path = output_dir / "prediction_diagnostics.csv"
    prediction_diagnostics_aggregate_csv_path = output_dir / "prediction_diagnostics_aggregate.csv"
    if resume_existing:
        cached_payload = _load_cached_payload(
            json_path,
            required_keys=("summary_rows", "comparison_summary_rows", "prediction_diagnostic_rows"),
            schema_version=ORACLE_RESIDUAL_EXPERIMENT_SCHEMA_VERSION,
        )
        if cached_payload is not None:
            cached_payload["summary_json_path"] = str(json_path)
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

    universe_artifact = _resolve_or_build_universe_artifact(symbols=symbols, output_basename=f"{output_dirname}__base")
    feature_artifact = _resolve_or_build_feature_artifact(
        universe_artifact=universe_artifact,
        symbols=symbols,
        feature_config={},
        output_basename=f"{output_dirname}__base",
    )
    momentum_signal = _resolve_momentum_signal_spec(feature_artifact)
    label_artifact = resolve_or_build_cross_sectional_rank_label_artifact(
        feature_artifact=feature_artifact,
        spec=CrossSectionalRankLabelSpec(
            horizon_days=int(forward_horizon_days),
            rebalance_freq="M",
            start_offset_days=int(start_offset_days),
            minimum_cross_section=max(2, int(min(len(symbols), max(10, bucket_count * 2)))),
            label_variant="raw",
            target_col="future_rank_pct",
            forward_return_col="trade_return",
            residualize_targets=True,
            residual_target_col="residual_rank_pct",
            residual_return_col="residual_return",
            fitted_return_col="factor_expected_return",
        ),
        output_basename=f"{output_dirname}__rank_labels",
    )
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
    model_variants = [
        {
            "variant_name": "raw_label_model",
            "variant_label": "Raw Label Model",
            "label_type": "raw",
            "target_col": "future_rank_pct",
        },
        {
            "variant_name": "residual_label_model",
            "variant_label": "Residual Label Model",
            "label_type": "residual",
            "target_col": "residual_rank_pct",
        },
    ]

    summary_rows: list[dict[str, Any]] = []
    prediction_diagnostic_rows: list[dict[str, Any]] = []
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
            strategy_definition_slug="oracle-residual-baseline",
            strategy_definition_name="Oracle Residual Baseline Momentum",
            strategy_config=baseline_strategy_config,
            validation_config=validation_config,
            backtest_config=backtest_config,
            output_basename=f"{output_dirname}__baseline__{fold_name}",
            resume_existing=resume_existing,
        )
        summary_rows.append(
            _annotate_fold_row(
                _single_summary_row(baseline_summary, label=f"{fold_name} baseline"),
                variant_name="baseline_momentum",
                variant_kind="baseline",
                variant_label="Baseline Momentum",
                label_type="baseline",
                fold_name=fold_name,
                train_end_date=train_end_date,
                backtest_start_date=backtest_start_date,
                backtest_end_date=backtest_end_date,
            )
        )
        baseline_score_frame = build_expression_score_frame(
            feature_artifact,
            score_expression=str(momentum_signal.get("expression") or ""),
            start_date=backtest_start_date,
            end_date=backtest_end_date,
        )
        prediction_diagnostic_rows.append(
            compute_prediction_diagnostic_rows(
                baseline_score_frame,
                label_artifact,
                variant_name="baseline_momentum",
                variant_kind="baseline",
                variant_label="Baseline Momentum",
                label_type="baseline",
                fold_name=fold_name,
                start_date=backtest_start_date,
                end_date=backtest_end_date,
                target_col="future_rank_pct",
            )
        )

        for variant in model_variants:
            variant_name = str(variant.get("variant_name") or "")
            variant_label = str(variant.get("variant_label") or variant_name)
            target_col = str(variant.get("target_col") or "future_rank_pct")
            label_type = str(variant.get("label_type") or "raw")
            model_summary = run_model_cohort_backtests(
                symbols=symbols,
                fit_job="fit_regressor",
                base_model_config=_default_model_config(
                    target_col=target_col,
                    horizon_days=int(forward_horizon_days),
                    model_name=variant_name,
                ),
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
                output_basename=f"{output_dirname}__{variant_name}__{fold_name}",
                resume_existing=resume_existing,
            )
            model_row = _annotate_fold_row(
                _single_summary_row(model_summary, label=f"{fold_name} {variant_name}"),
                variant_name=variant_name,
                variant_kind="model",
                variant_label=variant_label,
                label_type=label_type,
                fold_name=fold_name,
                train_end_date=train_end_date,
                backtest_start_date=backtest_start_date,
                backtest_end_date=backtest_end_date,
            )
            summary_rows.append(model_row)
            prediction_artifact = Artifact.objects.filter(pk=int(model_row.get("prediction_artifact_id") or 0)).first()
            if prediction_artifact is None:
                raise ValueError(f"{variant_name} prediction artifact was not found.")
            prediction_diagnostic_rows.append(
                compute_prediction_diagnostic_rows(
                    prediction_artifact,
                    label_artifact,
                    variant_name=variant_name,
                    variant_kind="model",
                    variant_label=variant_label,
                    label_type=label_type,
                    fold_name=fold_name,
                    start_date=backtest_start_date,
                    end_date=backtest_end_date,
                    target_col=target_col,
                )
            )

    aggregate_rows = _aggregate_performance_rows(summary_rows, validation_config=validation_config)
    prediction_diagnostics_aggregate_rows = aggregate_prediction_diagnostic_rows(prediction_diagnostic_rows)
    comparison_summary_rows = build_comparison_summary_rows(aggregate_rows, prediction_diagnostics_aggregate_rows)
    payload = {
        "schema_version": ORACLE_RESIDUAL_EXPERIMENT_SCHEMA_VERSION,
        "mode": "oracle_residual_label_experiment",
        "symbols": symbols,
        "missing_requested_symbols": missing_symbols,
        "coverage_rows": coverage_rows,
        "folds": [dict(fold) for fold in folds],
        "base_artifacts": {
            "universe": int(universe_artifact.id),
            "features": int(feature_artifact.id),
            "labels": int(label_artifact.id),
        },
        "rank_label_metadata": dict(label_artifact.metadata or {}),
        "momentum_signal": momentum_signal,
        "summary_rows": summary_rows,
        "aggregate_rows": aggregate_rows,
        "prediction_diagnostic_rows": prediction_diagnostic_rows,
        "prediction_diagnostics_aggregate_rows": prediction_diagnostics_aggregate_rows,
        "comparison_summary_rows": comparison_summary_rows,
        "summary_json_path": str(json_path),
        "comparison_summary_csv_path": str(comparison_summary_csv_path),
        "fold_results_csv_path": str(fold_results_csv_path),
        "prediction_diagnostics_csv_path": str(prediction_diagnostics_csv_path),
        "prediction_diagnostics_aggregate_csv_path": str(prediction_diagnostics_aggregate_csv_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_rows_csv(comparison_summary_csv_path, comparison_summary_rows)
    _write_rows_csv(fold_results_csv_path, summary_rows)
    _write_rows_csv(prediction_diagnostics_csv_path, prediction_diagnostic_rows)
    _write_rows_csv(prediction_diagnostics_aggregate_csv_path, prediction_diagnostics_aggregate_rows)
    return payload


__all__ = [
    "ORACLE_RESIDUAL_EXPERIMENT_SCHEMA_VERSION",
    "aggregate_prediction_diagnostic_rows",
    "build_comparison_summary_rows",
    "compute_prediction_diagnostic_rows",
    "run_oracle_residual_label_experiment",
    "write_oracle_residual_label_report",
]
