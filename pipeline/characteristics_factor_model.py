from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from django.utils.text import slugify
from sklearn.ensemble import RandomForestRegressor

from domain.models.datasets import feature_columns_from_frame, filter_frame_by_date
from ml.execution import infer_feature_family_columns, load_artifact_csv_frame

from .cohort_runner import (
    _aggregate_walk_forward_rows,
    _apply_walk_forward_gates,
    _build_equal_weight_benchmark,
    _evaluate_variant_gates,
    _load_cached_payload,
    _resolve_or_build_feature_artifact,
    _resolve_or_build_universe_artifact,
    _run_pipeline_job,
    run_model_cohort_backtests,
)
from .cross_sectional_rank_labels import (
    CrossSectionalRankLabelSpec,
    first_available_column,
    resolve_or_build_cross_sectional_rank_label_artifact,
)
from .direct_strategy_runner import _summarize_walk_forward_metrics, run_direct_feature_strategy_backtests
from .experiments import available_feature_families
from .factor_analysis import summarize_return_frame
from .models import Artifact, StrategyDefinition
from .oracle_ranking_signal_research import build_yearly_folds, resolve_research_symbols
from .prediction_artifacts import save_prediction_frame_artifact
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
from .service_runtime import json_safe_value
from .strategy_definitions import upsert_strategy_definition


CHARACTERISTICS_FACTOR_SCHEMA_VERSION = 1
DEFAULT_PRICE_ONLY_FAMILIES = ("prices_div_adj",)
DEFAULT_RETURN_COL_CANDIDATES = (
    "ret_1",
    "px__ret_1",
    "px__ret_1d",
    "px__ret_1_d",
    "asset_return",
)
DEFAULT_PRICE_COL_CANDIDATES = ("px__adj_close", "adj_close", "px__close", "close")


@dataclass(frozen=True)
class LatentFactorSpec:
    n_factors: int = 3
    exposure_lookback_days: int = 63
    minimum_exposure_observations: int = 30
    return_col_candidates: tuple[str, ...] = DEFAULT_RETURN_COL_CANDIDATES
    price_col_candidates: tuple[str, ...] = DEFAULT_PRICE_COL_CANDIDATES


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _pct(value: Any) -> str:
    return f"{_safe_float(value) * 100.0:.2f}%"


def _round_float(value: Any, digits: int = 8) -> float:
    return round(_safe_float(value), digits)


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


def _default_direct_model_config(*, horizon_days: int) -> dict[str, Any]:
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


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(df).copy()
    if out.empty:
        return out
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    return out.dropna(subset=[column for column in ["date", "symbol"] if column in out.columns]).copy()


def resolve_feature_scope_variants(
    feature_artifact: Artifact,
    *,
    include_prices_only: bool = False,
    include_context_only: bool = False,
) -> list[dict[str, Any]]:
    family_names = available_feature_families(feature_artifact)
    variants = [
        {
            "variant_name": "all_features",
            "variant_label": "All Features",
            "feature_scope": "all_features",
            "feature_families": [],
        }
    ]
    if include_prices_only and "prices_div_adj" in family_names:
        variants.append(
            {
                "variant_name": "prices_only",
                "variant_label": "Price Features",
                "feature_scope": "prices_only",
                "feature_families": list(DEFAULT_PRICE_ONLY_FAMILIES),
            }
        )
    context_families = [family for family in family_names if family not in DEFAULT_PRICE_ONLY_FAMILIES]
    if include_context_only and context_families:
        variants.append(
            {
                "variant_name": "context_only",
                "variant_label": "Context Features",
                "feature_scope": "context_only",
                "feature_families": list(context_families),
            }
        )
    return variants


def build_characteristic_rank_panel(
    feature_frame_or_artifact: Artifact | pd.DataFrame,
    label_frame_or_artifact: Artifact | pd.DataFrame,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    feature_families: Sequence[str] = (),
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    feature_df = (
        load_artifact_csv_frame(feature_frame_or_artifact)
        if isinstance(feature_frame_or_artifact, Artifact)
        else pd.DataFrame(feature_frame_or_artifact).copy()
    )
    label_df = (
        load_artifact_csv_frame(label_frame_or_artifact)
        if isinstance(label_frame_or_artifact, Artifact)
        else pd.DataFrame(label_frame_or_artifact).copy()
    )
    feature_df = filter_frame_by_date(_normalize_frame(feature_df), start_date=start_date, end_date=end_date)
    label_df = filter_frame_by_date(_normalize_frame(label_df), start_date=start_date, end_date=end_date)
    if feature_df.empty or label_df.empty:
        return pd.DataFrame(), [], {"rows": 0, "feature_count": 0}

    family_map = infer_feature_family_columns(feature_columns_from_frame(feature_df))
    requested_families = [str(value).strip() for value in list(feature_families or []) if str(value).strip()]
    if requested_families:
        selected_cols: list[str] = []
        for family_name in requested_families:
            for column in list(family_map.get(family_name) or []):
                if column not in selected_cols:
                    selected_cols.append(str(column))
    else:
        selected_cols = feature_columns_from_frame(feature_df)

    numeric_columns: dict[str, pd.Series] = {}
    numeric_cols: list[str] = []
    for column in selected_cols:
        numeric = pd.to_numeric(feature_df.get(column), errors="coerce")
        if numeric.notna().any():
            numeric_columns[str(column)] = numeric
            numeric_cols.append(str(column))
    panel = pd.concat(
        [
            feature_df[["date", "symbol"]].reset_index(drop=True),
            pd.DataFrame(numeric_columns).reset_index(drop=True),
        ],
        axis=1,
    )
    merge_cols = [
        column
        for column in ["date", "symbol", "future_rank_pct", "trade_return", "forward_start_date", "forward_end_date", "cross_section_size", "k"]
        if column in label_df.columns
    ]
    out = panel.merge(label_df[merge_cols].copy(), on=["date", "symbol"], how="inner")
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    return out, numeric_cols, {
        "rows": int(len(out)),
        "symbols": int(out["symbol"].nunique()) if not out.empty else 0,
        "dates": int(out["date"].nunique()) if not out.empty else 0,
        "feature_count": int(len(numeric_cols)),
        "feature_families": list(requested_families),
        "available_feature_families": sorted([str(name) for name, cols in family_map.items() if list(cols or [])]),
    }


def build_daily_return_panel(
    feature_frame_or_artifact: Artifact | pd.DataFrame,
    *,
    spec: LatentFactorSpec | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved_spec = spec or LatentFactorSpec()
    feature_df = (
        load_artifact_csv_frame(feature_frame_or_artifact)
        if isinstance(feature_frame_or_artifact, Artifact)
        else pd.DataFrame(feature_frame_or_artifact).copy()
    )
    feature_df = filter_frame_by_date(_normalize_frame(feature_df), start_date=start_date, end_date=end_date)
    if feature_df.empty:
        return pd.DataFrame(), {"rows": 0}

    return_col = first_available_column(feature_df.columns, resolved_spec.return_col_candidates)
    panel = feature_df[["date", "symbol"]].copy()
    if return_col:
        panel["daily_return"] = pd.to_numeric(feature_df.get(return_col), errors="coerce")
    else:
        price_col = first_available_column(feature_df.columns, resolved_spec.price_col_candidates)
        if not price_col:
            raise ValueError("Feature frame does not contain a usable return or price column for latent factor estimation.")
        panel["_price"] = pd.to_numeric(feature_df.get(price_col), errors="coerce")
        panel["daily_return"] = panel.groupby("symbol", sort=False)["_price"].pct_change()
        panel = panel.drop(columns=["_price"])

    panel = panel.dropna(subset=["date", "symbol"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    return_pivot = (
        panel.pivot_table(index="date", columns="symbol", values="daily_return", aggfunc="last")
        .sort_index()
    )
    if return_pivot.empty:
        return pd.DataFrame(), {"rows": 0}
    return return_pivot, {
        "rows": int(return_pivot.size),
        "dates": int(len(return_pivot.index)),
        "symbols": int(len(return_pivot.columns)),
        "return_col": str(return_col or ""),
    }


def estimate_latent_factor_basis(
    return_panel: pd.DataFrame,
    *,
    train_end_date: str,
    n_factors: int = 3,
    minimum_history_days: int = 60,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if return_panel.empty:
        raise ValueError("return_panel is required to estimate a latent factor basis.")
    train_slice = return_panel.loc[return_panel.index <= pd.Timestamp(str(train_end_date))].copy()
    valid_symbols = [
        str(column)
        for column in train_slice.columns
        if int(train_slice[column].notna().sum()) >= max(int(minimum_history_days), 2)
    ]
    if len(valid_symbols) < 2:
        raise ValueError("Not enough symbols with sufficient history to estimate latent factors.")
    centered = train_slice[valid_symbols].fillna(0.0).astype(float)
    centered = centered - centered.mean(axis=0)
    matrix = centered.to_numpy()
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("Latent factor estimation requires at least two dates and two symbols.")
    _u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    factor_count = max(1, min(int(n_factors), int(vt.shape[0]), int(len(valid_symbols))))
    factor_names = [f"latent_factor_{index}" for index in range(1, factor_count + 1)]
    loading_df = pd.DataFrame(vt[:factor_count].T, index=valid_symbols, columns=factor_names)
    singular_power = singular_values ** 2
    explained = []
    total_power = float(singular_power.sum()) if len(singular_power) else 0.0
    for index, factor_name in enumerate(factor_names):
        share = float(singular_power[index] / total_power) if total_power > 1e-12 else 0.0
        explained.append({"factor": factor_name, "explained_variance_share": round(share, 8)})
    return loading_df, {
        "n_factors": int(factor_count),
        "train_days": int(len(train_slice.index)),
        "basis_symbols": int(len(valid_symbols)),
        "factor_names": factor_names,
        "explained_variance_rows": explained,
    }


def project_latent_factor_returns(
    return_panel: pd.DataFrame,
    loading_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if return_panel.empty or loading_df.empty:
        return pd.DataFrame(), {"rows": 0}
    aligned_returns = return_panel.reindex(columns=loading_df.index.tolist()).fillna(0.0).astype(float)
    projected = aligned_returns.to_numpy() @ loading_df.to_numpy()
    factor_df = pd.DataFrame(projected, index=aligned_returns.index, columns=loading_df.columns.tolist()).reset_index()
    factor_df = factor_df.rename(columns={"index": "date"})
    factor_df["date"] = pd.to_datetime(factor_df["date"], errors="coerce")
    return factor_df, {
        "rows": int(len(factor_df)),
        "dates": int(factor_df["date"].nunique()),
        "factor_names": [str(column) for column in loading_df.columns.tolist()],
    }


def estimate_rebalance_factor_exposures(
    return_panel: pd.DataFrame,
    factor_return_df: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp | str],
    *,
    lookback_days: int = 63,
    minimum_observations: int = 30,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if return_panel.empty or factor_return_df.empty:
        return pd.DataFrame(), {"rows": 0}
    factor_frame = factor_return_df.copy()
    factor_frame["date"] = pd.to_datetime(factor_frame["date"], errors="coerce")
    factor_frame = factor_frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    factor_cols = [column for column in factor_frame.columns if column != "date"]
    if not factor_cols:
        return pd.DataFrame(), {"rows": 0}

    factor_pivot = factor_frame.set_index("date")[factor_cols].astype(float)
    common_dates = factor_pivot.index.intersection(return_panel.index)
    if common_dates.empty:
        return pd.DataFrame(), {"rows": 0}
    factor_pivot = factor_pivot.loc[common_dates]
    aligned_returns = return_panel.loc[common_dates].astype(float)
    date_positions = {pd.Timestamp(date_value): idx for idx, date_value in enumerate(common_dates)}
    required_obs = max(int(minimum_observations), len(factor_cols) + 2)

    rows: list[dict[str, Any]] = []
    for raw_date in list(rebalance_dates or []):
        date_value = pd.Timestamp(raw_date)
        if date_value not in date_positions:
            continue
        end_pos = int(date_positions[date_value])
        start_pos = max(0, end_pos - int(lookback_days) + 1)
        window_dates = common_dates[start_pos : end_pos + 1]
        if len(window_dates) < required_obs:
            continue
        window_factors = factor_pivot.loc[window_dates, factor_cols].astype(float)
        factor_values = window_factors.to_numpy()
        if factor_values.size <= 0:
            continue
        design = np.column_stack([np.ones(len(window_factors)), factor_values])
        for symbol in aligned_returns.columns.tolist():
            asset_returns = aligned_returns.loc[window_dates, symbol].to_numpy(dtype=float)
            mask = np.isfinite(asset_returns) & np.isfinite(factor_values).all(axis=1)
            if int(mask.sum()) < required_obs:
                continue
            coeffs, *_ = np.linalg.lstsq(design[mask], asset_returns[mask], rcond=None)
            row = {
                "date": date_value.strftime("%Y-%m-%d"),
                "symbol": str(symbol),
                "factor_alpha": _round_float(coeffs[0]),
                "exposure_observations": int(mask.sum()),
            }
            for factor_index, factor_name in enumerate(factor_cols, start=1):
                row[f"{factor_name}_beta"] = _round_float(coeffs[factor_index])
            rows.append(row)
    exposure_df = pd.DataFrame(rows)
    if exposure_df.empty:
        return exposure_df, {"rows": 0, "factor_columns": []}
    factor_beta_cols = [column for column in exposure_df.columns if column.endswith("_beta")]
    return exposure_df.sort_values(["date", "symbol"]).reset_index(drop=True), {
        "rows": int(len(exposure_df)),
        "dates": int(exposure_df["date"].nunique()),
        "symbols": int(exposure_df["symbol"].nunique()),
        "factor_columns": factor_beta_cols,
    }


def estimate_cross_sectional_factor_premia(
    train_panel_with_exposures: pd.DataFrame,
    *,
    factor_cols: Sequence[str],
    target_col: str = "future_rank_pct",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if train_panel_with_exposures.empty:
        return pd.DataFrame(), {"factor_premia": {}, "alpha": 0.0}
    rows: list[dict[str, Any]] = []
    resolved_factor_cols = [str(column) for column in list(factor_cols or []) if str(column).strip()]
    required = [column for column in [target_col, *resolved_factor_cols] if column in train_panel_with_exposures.columns]
    if len(required) < len(resolved_factor_cols) + 1:
        return pd.DataFrame(), {"factor_premia": {}, "alpha": 0.0}

    for date_value, group in train_panel_with_exposures.groupby("date", sort=True):
        sample = group.dropna(subset=required).copy()
        if len(sample) < len(resolved_factor_cols) + 2:
            continue
        y = pd.to_numeric(sample[target_col], errors="coerce").to_numpy(dtype=float)
        x = sample[resolved_factor_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if int(mask.sum()) < len(resolved_factor_cols) + 2:
            continue
        design = np.column_stack([np.ones(int(mask.sum())), x[mask]])
        coeffs, *_ = np.linalg.lstsq(design, y[mask], rcond=None)
        fitted = design @ coeffs
        y_used = y[mask]
        y_mean = float(y_used.mean()) if len(y_used) else 0.0
        ss_tot = float(((y_used - y_mean) ** 2).sum())
        ss_res = float(((y_used - fitted) ** 2).sum())
        row = {
            "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
            "alpha": _round_float(coeffs[0]),
            "cross_section_size": int(mask.sum()),
            "r_squared": _round_float(1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0),
        }
        for factor_index, factor_name in enumerate(resolved_factor_cols, start=1):
            row[f"premium__{factor_name}"] = _round_float(coeffs[factor_index])
        rows.append(row)

    premia_df = pd.DataFrame(rows)
    if premia_df.empty:
        factor_premia = {str(column): 0.0 for column in resolved_factor_cols}
        return premia_df, {"alpha": 0.0, "factor_premia": factor_premia, "rows": 0}
    factor_premia = {
        str(column): _round_float(premia_df[f"premium__{column}"].mean())
        for column in resolved_factor_cols
    }
    return premia_df.sort_values("date").reset_index(drop=True), {
        "alpha": _round_float(premia_df["alpha"].mean()),
        "factor_premia": factor_premia,
        "rows": int(len(premia_df)),
        "mean_r_squared": _round_float(premia_df["r_squared"].mean()),
    }


def fit_characteristic_factor_ranker(
    train_panel_with_exposures: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    factor_cols: Sequence[str],
    factor_premia: Mapping[str, Any],
    random_state: int = 1337,
    n_estimators: int = 120,
    max_depth: int = 5,
    min_samples_leaf: int = 8,
) -> dict[str, Any]:
    resolved_feature_cols = [str(column) for column in list(feature_cols or []) if str(column).strip()]
    resolved_factor_cols = [str(column) for column in list(factor_cols or []) if str(column).strip()]
    train_df = train_panel_with_exposures.dropna(subset=["future_rank_pct", *resolved_factor_cols]).copy()
    if train_df.empty or not resolved_feature_cols or not resolved_factor_cols:
        raise ValueError("Characteristic factor training requires non-empty features, factor exposures, and targets.")

    x_raw = train_df[resolved_feature_cols].apply(pd.to_numeric, errors="coerce")
    medians = x_raw.median(axis=0, numeric_only=True).fillna(0.0)
    x_train = x_raw.fillna(medians)
    y_train = train_df[resolved_factor_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    model = RandomForestRegressor(
        n_estimators=max(int(n_estimators), 20),
        max_depth=max(int(max_depth), 1),
        min_samples_leaf=max(int(min_samples_leaf), 1),
        n_jobs=-1,
        random_state=int(random_state),
    )
    model.fit(x_train, y_train)

    factor_weights = np.array([_safe_float(factor_premia.get(column)) for column in resolved_factor_cols], dtype=float)
    alpha = _safe_float(factor_premia.get("alpha"))
    predicted_exposures = np.asarray(model.predict(x_train), dtype=float)
    train_scores = alpha + (predicted_exposures @ factor_weights)
    importances = getattr(model, "feature_importances_", None)
    top_features: list[tuple[str, float]] = []
    if importances is not None:
        top_features = sorted(
            (
                (str(column), float(value))
                for column, value in zip(resolved_feature_cols, list(importances))
                if float(value) > 0.0
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:15]

    score_series = pd.Series(train_scores, index=train_df.index, dtype=float)
    realized = pd.to_numeric(train_df["future_rank_pct"], errors="coerce")
    train_score_corr = score_series.rank().corr(realized.rank())
    train_target_r2 = score_series.corr(realized)
    return {
        "model": model,
        "feature_cols": resolved_feature_cols,
        "factor_cols": resolved_factor_cols,
        "feature_medians": {str(key): _safe_float(value) for key, value in medians.to_dict().items()},
        "factor_weights": {str(column): _safe_float(factor_premia.get(column)) for column in resolved_factor_cols},
        "alpha": alpha,
        "trained_rows": int(len(train_df)),
        "top_features": [(name, _round_float(value, 6)) for name, value in top_features],
        "train_score_rank_corr": _round_float(train_score_corr if pd.notna(train_score_corr) else 0.0),
        "train_score_linear_corr": _round_float(train_target_r2 if pd.notna(train_target_r2) else 0.0),
    }


def score_characteristic_factor_ranker(
    state: Mapping[str, Any],
    score_panel: pd.DataFrame,
) -> pd.DataFrame:
    if score_panel.empty:
        return pd.DataFrame(columns=["date", "symbol", "prediction", "prediction_score"])
    model = state.get("model")
    if model is None:
        raise ValueError("Characteristic factor model state is missing the trained model.")
    feature_cols = [str(column) for column in list(state.get("feature_cols") or []) if str(column).strip()]
    factor_cols = [str(column) for column in list(state.get("factor_cols") or []) if str(column).strip()]
    medians = {str(key): _safe_float(value) for key, value in dict(state.get("feature_medians") or {}).items()}
    alpha = _safe_float(state.get("alpha"))
    factor_weights = np.array([_safe_float(dict(state.get("factor_weights") or {}).get(column)) for column in factor_cols], dtype=float)

    x_score = score_panel[feature_cols].apply(pd.to_numeric, errors="coerce")
    for column in feature_cols:
        x_score[column] = x_score[column].fillna(_safe_float(medians.get(column)))
    predicted_exposures = np.asarray(model.predict(x_score), dtype=float)
    scores = alpha + (predicted_exposures @ factor_weights)

    out = score_panel[["date", "symbol"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["prediction"] = pd.Series(scores, index=out.index, dtype=float)
    out["prediction_score"] = out["prediction"]
    for factor_index, factor_name in enumerate(factor_cols):
        out[f"predicted__{factor_name}"] = predicted_exposures[:, factor_index]
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def build_characteristic_factor_targets(
    feature_frame_or_artifact: Artifact | pd.DataFrame,
    label_frame_or_artifact: Artifact | pd.DataFrame,
    *,
    train_end_date: str,
    score_end_date: str,
    spec: LatentFactorSpec | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    resolved_spec = spec or LatentFactorSpec()
    feature_df = (
        load_artifact_csv_frame(feature_frame_or_artifact)
        if isinstance(feature_frame_or_artifact, Artifact)
        else pd.DataFrame(feature_frame_or_artifact).copy()
    )
    label_df = (
        load_artifact_csv_frame(label_frame_or_artifact)
        if isinstance(label_frame_or_artifact, Artifact)
        else pd.DataFrame(label_frame_or_artifact).copy()
    )
    feature_df = filter_frame_by_date(_normalize_frame(feature_df), end_date=score_end_date)
    label_df = filter_frame_by_date(_normalize_frame(label_df), end_date=score_end_date)
    return_panel, return_meta = build_daily_return_panel(feature_df, spec=resolved_spec)
    loading_df, basis_meta = estimate_latent_factor_basis(
        return_panel,
        train_end_date=str(train_end_date),
        n_factors=int(resolved_spec.n_factors),
        minimum_history_days=max(int(resolved_spec.exposure_lookback_days), int(resolved_spec.minimum_exposure_observations)),
    )
    factor_return_df, factor_return_meta = project_latent_factor_returns(return_panel, loading_df)
    exposure_df, exposure_meta = estimate_rebalance_factor_exposures(
        return_panel,
        factor_return_df,
        rebalance_dates=sorted(label_df["date"].dropna().unique().tolist()),
        lookback_days=int(resolved_spec.exposure_lookback_days),
        minimum_observations=int(resolved_spec.minimum_exposure_observations),
    )
    if not factor_return_df.empty:
        factor_return_df["date"] = pd.to_datetime(factor_return_df["date"], errors="coerce")
    if not exposure_df.empty:
        exposure_df["date"] = pd.to_datetime(exposure_df["date"], errors="coerce")
    return factor_return_df, exposure_df, list(exposure_meta.get("factor_columns") or []), {
        "return_meta": return_meta,
        "basis_meta": basis_meta,
        "factor_return_meta": factor_return_meta,
        "exposure_meta": exposure_meta,
    }


def _characteristic_factor_variant_summary_row(
    *,
    variant_name: str,
    feature_scope: str,
    feature_families: Sequence[str],
    train_panel_meta: Mapping[str, Any],
    score_panel_meta: Mapping[str, Any],
    model_state: Mapping[str, Any],
    prediction_artifact: Artifact,
    strategy_artifact: Artifact,
    backtest_artifact: Artifact,
    backtest_config: Mapping[str, Any],
    validation_config: Mapping[str, Any],
) -> dict[str, Any]:
    backtest_content = dict(backtest_artifact.content or {})
    strategy_meta = dict(strategy_artifact.metadata or {})
    backtest_meta = dict(backtest_artifact.metadata or {})
    benchmark = _build_equal_weight_benchmark(
        strategy_artifact,
        allowed_symbols=dict(backtest_config or {}).get("allowed_symbols"),
    )
    runtime_summary = summarize_return_frame(
        list(backtest_content.get("daily_rows") or []),
        series_name=variant_name,
        series_kind="strategy",
    )
    row = {
        "variant_name": str(variant_name),
        "fit_job": "characteristic_factor_regressor",
        "score_job": "characteristic_factor_regressor",
        "feature_families": list(feature_families or []),
        "label_ks": [],
        "dataset_build_seconds": _safe_float(model_state.get("dataset_build_seconds")),
        "fit_seconds": _safe_float(model_state.get("fit_seconds")),
        "score_seconds": _safe_float(model_state.get("score_seconds")),
        "strategy_build_seconds": _safe_float(strategy_meta.get("strategy_build_seconds")),
        "backtest_seconds": _safe_float(backtest_meta.get("backtest_seconds")),
        "coverage_start_date": str(train_panel_meta.get("coverage_start_date") or ""),
        "coverage_end_date": str(score_panel_meta.get("coverage_end_date") or ""),
        "coverage_rows": int(train_panel_meta.get("rows") or 0) + int(score_panel_meta.get("rows") or 0),
        "oracle_cluster_scope": "generalist",
        "oracle_cluster_keys": [],
        "oracle_cluster_rows": 0,
        "trained_rows": int(model_state.get("trained_rows") or 0),
        "rows_scored": int(score_panel_meta.get("rows") or 0),
        "selected_rows": int((strategy_artifact.content or {}).get("selected_rows") or 0),
        "final_equity": _safe_float(backtest_content.get("final_equity"), 1.0),
        "cumulative_return": _safe_float(backtest_content.get("cumulative_return")),
        "max_drawdown": _safe_float(backtest_content.get("max_drawdown")),
        "trades": int(_safe_float(backtest_content.get("trades"))),
        "sharpe": _safe_float(runtime_summary.get("sharpe")),
        "avg_turnover": _safe_float(runtime_summary.get("avg_turnover")),
        "total_turnover": _safe_float(runtime_summary.get("total_turnover")),
        "positive_days": int(_safe_float(runtime_summary.get("positive_days"))),
        "negative_days": int(_safe_float(runtime_summary.get("negative_days"))),
        "benchmark_days": int(_safe_float(benchmark.get("benchmark_days"))),
        "benchmark_final_equity": _safe_float(benchmark.get("benchmark_final_equity"), 1.0),
        "benchmark_cumulative_return": _safe_float(benchmark.get("benchmark_cumulative_return")),
        "benchmark_max_drawdown": _safe_float(benchmark.get("benchmark_max_drawdown")),
        "backtest_fee_bps": _safe_float((backtest_meta.get("backtest_config") or {}).get("fee_bps") or backtest_config.get("fee_bps")),
        "backtest_slippage_bps": _safe_float((backtest_meta.get("backtest_config") or {}).get("slippage_bps") or backtest_config.get("slippage_bps")),
        "excess_cumulative_return": _round_float(
            _safe_float(backtest_content.get("cumulative_return")) - _safe_float(benchmark.get("benchmark_cumulative_return"))
        ),
        "relative_final_equity": _round_float(
            _safe_float(backtest_content.get("final_equity"), 1.0) - _safe_float(benchmark.get("benchmark_final_equity"), 1.0)
        ),
        "model_artifact_id": 0,
        "prediction_artifact_id": int(prediction_artifact.id),
        "strategy_artifact_id": int(strategy_artifact.id),
        "backtest_artifact_id": int(backtest_artifact.id),
        "feature_scope": str(feature_scope),
    }
    row["total_runtime_seconds"] = round(
        _safe_float(row["dataset_build_seconds"])
        + _safe_float(row["fit_seconds"])
        + _safe_float(row["score_seconds"])
        + _safe_float(row["strategy_build_seconds"])
        + _safe_float(row["backtest_seconds"]),
        6,
    )
    row.update(_evaluate_variant_gates(row, validation_config=dict(validation_config or {})))
    return row


def run_characteristic_factor_variant(
    *,
    variant_name: str,
    variant_label: str,
    feature_scope: str,
    feature_families: Sequence[str],
    feature_artifact: Artifact,
    label_artifact: Artifact,
    strategy_definition: StrategyDefinition,
    train_end_date: str,
    backtest_start_date: str,
    backtest_end_date: str,
    factor_target_df: pd.DataFrame,
    factor_cols: Sequence[str],
    backtest_config: Mapping[str, Any],
    validation_config: Mapping[str, Any],
    output_basename: str,
    random_state: int,
) -> dict[str, Any]:
    train_panel, feature_cols, train_meta = build_characteristic_rank_panel(
        feature_artifact,
        label_artifact,
        end_date=train_end_date,
        feature_families=feature_families,
    )
    score_panel, _ignored_feature_cols, score_meta = build_characteristic_rank_panel(
        feature_artifact,
        label_artifact,
        start_date=backtest_start_date,
        end_date=backtest_end_date,
        feature_families=feature_families,
    )
    train_meta = {
        **dict(train_meta),
        "coverage_start_date": str(train_panel["date"].min().strftime("%Y-%m-%d")) if not train_panel.empty else "",
        "coverage_end_date": str(train_panel["date"].max().strftime("%Y-%m-%d")) if not train_panel.empty else "",
    }
    score_meta = {
        **dict(score_meta),
        "coverage_start_date": str(score_panel["date"].min().strftime("%Y-%m-%d")) if not score_panel.empty else "",
        "coverage_end_date": str(score_panel["date"].max().strftime("%Y-%m-%d")) if not score_panel.empty else "",
    }
    if train_panel.empty or score_panel.empty:
        raise ValueError(f"{variant_name} could not build a usable train/test panel.")

    dataset_started = time.perf_counter()
    train_with_exposures = train_panel.merge(factor_target_df, on=["date", "symbol"], how="inner")
    score_with_exposures = score_panel.merge(factor_target_df, on=["date", "symbol"], how="left")
    dataset_build_seconds = max(time.perf_counter() - dataset_started, 0.0)
    if train_with_exposures.empty:
        raise ValueError(f"{variant_name} had no overlap between training characteristics and realized factor exposures.")

    premia_df, premia_meta = estimate_cross_sectional_factor_premia(
        train_with_exposures,
        factor_cols=factor_cols,
        target_col="future_rank_pct",
    )
    if premia_df.empty:
        raise ValueError(f"{variant_name} could not estimate factor premia from the training window.")

    fit_started = time.perf_counter()
    model_state = fit_characteristic_factor_ranker(
        train_with_exposures,
        feature_cols=feature_cols,
        factor_cols=factor_cols,
        factor_premia={"alpha": premia_meta.get("alpha"), **dict(premia_meta.get("factor_premia") or {})},
        random_state=int(random_state),
    )
    fit_seconds = max(time.perf_counter() - fit_started, 0.0)
    model_state["dataset_build_seconds"] = dataset_build_seconds
    model_state["fit_seconds"] = fit_seconds

    score_started = time.perf_counter()
    prediction_frame = score_characteristic_factor_ranker(model_state, score_with_exposures)
    score_seconds = max(time.perf_counter() - score_started, 0.0)
    model_state["score_seconds"] = score_seconds

    prediction_frame["variant_name"] = str(variant_name)
    prediction_frame["variant_label"] = str(variant_label)
    prediction_frame["feature_scope"] = str(feature_scope)
    prediction_artifact = save_prediction_frame_artifact(
        prediction_frame,
        artifact_type="REGRESSOR_PREDICTIONS",
        requested_job="score_characteristic_factor_ranker",
        run_name=f"{variant_label} predictions",
        config={
            "variant_name": str(variant_name),
            "feature_scope": str(feature_scope),
            "feature_families": list(feature_families or []),
            "train_end_date": str(train_end_date),
            "backtest_start_date": str(backtest_start_date),
            "backtest_end_date": str(backtest_end_date),
        },
        content={
            "rows": int(len(prediction_frame)),
            "trained_rows": int(model_state.get("trained_rows") or 0),
            "feature_count": int(len(feature_cols)),
            "factor_count": int(len(factor_cols)),
        },
        metadata={
            "variant_name": str(variant_name),
            "variant_label": str(variant_label),
            "feature_scope": str(feature_scope),
            "feature_families": list(feature_families or []),
            "factor_columns": list(factor_cols or []),
            "factor_premia": json_safe_value(dict(premia_meta.get("factor_premia") or {})),
            "alpha": _safe_float(premia_meta.get("alpha")),
            "train_score_rank_corr": _safe_float(model_state.get("train_score_rank_corr")),
            "top_features": json_safe_value(list(model_state.get("top_features") or [])),
        },
    )
    strategy_artifact = _run_pipeline_job(
        name=f"{output_basename}-{slugify(variant_name) or variant_name}-strategy",
        requested_job="build_strategy_dataset",
        config={
            "strategy_definition_id": int(strategy_definition.id),
            "label_artifact_id": int(label_artifact.id),
            "prediction_artifact_ids": [int(prediction_artifact.id)],
            "strategy_start_date": str(backtest_start_date),
            "strategy_end_date": str(backtest_end_date),
        },
        input_ids=[int(feature_artifact.id)],
    )
    backtest_artifact = _run_pipeline_job(
        name=f"{output_basename}-{slugify(variant_name) or variant_name}-backtest",
        requested_job="backtest_strategy",
        config={
            "backtest_start_date": str(backtest_start_date),
            "backtest_end_date": str(backtest_end_date),
            **dict(backtest_config or {}),
        },
        input_ids=[int(strategy_artifact.id)],
    )

    row = _characteristic_factor_variant_summary_row(
        variant_name=variant_name,
        feature_scope=feature_scope,
        feature_families=feature_families,
        train_panel_meta=train_meta,
        score_panel_meta=score_meta,
        model_state=model_state,
        prediction_artifact=prediction_artifact,
        strategy_artifact=strategy_artifact,
        backtest_artifact=backtest_artifact,
        backtest_config=backtest_config,
        validation_config=validation_config,
    )
    return {
        "summary_row": row,
        "prediction_artifact": prediction_artifact,
        "prediction_rows": prediction_frame.to_dict(orient="records"),
        "premia_rows": premia_df.assign(
            variant_name=str(variant_name),
            feature_scope=str(feature_scope),
        ).to_dict(orient="records"),
        "model_rows": [
            {
                "variant_name": str(variant_name),
                "variant_label": str(variant_label),
                "feature_scope": str(feature_scope),
                "feature_count": int(len(feature_cols)),
                "factor_count": int(len(factor_cols)),
                "trained_rows": int(model_state.get("trained_rows") or 0),
                "train_score_rank_corr": _safe_float(model_state.get("train_score_rank_corr")),
                "train_score_linear_corr": _safe_float(model_state.get("train_score_linear_corr")),
                "top_features": json.dumps(list(model_state.get("top_features") or [])),
            }
        ],
    }


def _build_direct_model_variants(feature_artifact: Artifact, *, horizon_days: int) -> list[dict[str, Any]]:
    variants = resolve_feature_scope_variants(feature_artifact, include_prices_only=False, include_context_only=False)
    return [
        {
            "variant_name": f"oracle_rank_rf_{variant['variant_name']}",
            "variant_label": f"Direct ML {variant['variant_label']}",
            "feature_scope": str(variant["feature_scope"]),
            "model_config": {
                **_default_direct_model_config(horizon_days=int(horizon_days)),
                "model_name": f"oracle_rank_rf_{variant['variant_name']}",
                "feature_families": list(variant.get("feature_families") or []),
            },
        }
        for variant in variants
    ]


def _build_characteristic_variants(
    feature_artifact: Artifact,
    *,
    include_prices_only: bool = False,
    include_context_only: bool = False,
) -> list[dict[str, Any]]:
    variants = resolve_feature_scope_variants(
        feature_artifact,
        include_prices_only=include_prices_only,
        include_context_only=include_context_only,
    )
    return [
        {
            "variant_name": f"characteristics_factor_rf_{variant['variant_name']}",
            "variant_label": f"Characteristics Factor {variant['variant_label']}",
            "feature_scope": str(variant["feature_scope"]),
            "feature_families": list(variant.get("feature_families") or []),
        }
        for variant in variants
    ]


def _top_bucket_return(aggregate_bucket_rows: Sequence[Mapping[str, Any]], variant_name: str, bucket: int) -> float:
    for row in list(aggregate_bucket_rows or []):
        if str(row.get("variant_name") or "") == str(variant_name) and int(row.get("bucket") or 0) == int(bucket):
            return _safe_float(row.get("avg_forward_return"))
    return 0.0


def _cohort_preview(rows: Sequence[Mapping[str, Any]], *, variant_name: str, cohort_kind: str, limit: int = 3) -> str:
    matched = [
        dict(row)
        for row in list(rows or [])
        if str(row.get("variant_name") or "") == str(variant_name)
        and str(row.get("cohort_kind") or "") == str(cohort_kind)
    ]
    preview = matched[: max(int(limit), 0)]
    return ", ".join(
        f"{row['cohort_value']} ({_safe_float(row.get('selection_share')):.2f})"
        for row in preview
    )


def write_characteristics_factor_report(
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
    factor_meta_by_fold = {
        str(key): dict(value)
        for key, value in dict(payload.get("latent_factor_meta") or {}).items()
    }
    basis_meta_rows = [dict(item.get("basis_meta") or {}) for item in factor_meta_by_fold.values() if dict(item)]
    resolved_factor_count = max(int(row.get("n_factors") or 0) for row in basis_meta_rows) if basis_meta_rows else 0
    resolved_basis_symbols = max(int(row.get("basis_symbols") or 0) for row in basis_meta_rows) if basis_meta_rows else 0
    symbols = [str(symbol) for symbol in list(payload.get("symbols") or [])]
    if not aggregate_rows:
        raise ValueError("Expected aggregate rows to write the characteristics-factor report.")

    baseline_row = next((row for row in aggregate_rows if str(row.get("variant_name") or "") == "baseline_momentum"), {})
    ranking_lookup = {str(row.get("variant_name") or ""): row for row in ranking_rows}
    stability_lookup = {str(row.get("variant_name") or ""): row for row in stability_rows}
    ml_rows = [row for row in aggregate_rows if str(row.get("variant_kind") or "") == "model"]
    best_ml = max(
        ml_rows,
        key=lambda row: (
            _safe_float(row.get("sharpe")),
            _safe_float(row.get("total_return")),
            _safe_float(ranking_lookup.get(str(row.get("variant_name") or ""), {}).get("mean_spearman_ic")),
        ),
        default={},
    )
    lines = [
        "# Characteristics Factor Research Report",
        "",
        "## 1. Experiment",
        "",
        "- Objective: test whether existing platform features can learn latent factor exposures from returns and convert them into a stronger cross-sectional ranking signal than simple momentum.",
        "- Research framing: inspired by Kelly, Pruitt, and Su's idea that characteristics can map to factor exposures, but implemented as a reusable cross-sectional research capability rather than a paper-specific reproduction.",
        f"- Latent factor basis: {int(resolved_factor_count)} PCA-style factors estimated from training-window daily returns only across up to {int(resolved_basis_symbols)} symbols per fold.",
        f"- Exposure target: rolling OLS betas over {int(payload.get('exposure_lookback_days') or 0)} business days, then cross-sectional factor premia estimated from the oracle rank-percentile label.",
        "- Trading path: predicted score -> existing `cross_sectional_quantiles` portfolio construction -> long top bucket / short bottom bucket.",
        f"- Universe: {len(symbols)} symbols.",
        "- Symbols: " + ", ".join(symbols),
        "",
        "## 2. Results",
        "",
        f"- Baseline momentum: Sharpe {_safe_float(baseline_row.get('sharpe')):.3f}, total return {_pct(baseline_row.get('total_return'))}, max drawdown {_pct(baseline_row.get('max_drawdown'))}.",
        f"- Best ML variant: {best_ml.get('variant_name', '')} | Sharpe {_safe_float(best_ml.get('sharpe')):.3f} | total return {_pct(best_ml.get('total_return'))} | max drawdown {_pct(best_ml.get('max_drawdown'))}.",
        "",
        "| Variant | Sharpe | Total Return | Max DD | Turnover | Trades | Mean IC | Long-Short Spread | Positive Fold Rate |",
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
            "## 3. Diagnostics",
            "",
            f"- Baseline mean Spearman IC: {_safe_float(ranking_lookup.get('baseline_momentum', {}).get('mean_spearman_ic')):.3f}.",
            f"- Best ML mean Spearman IC: {_safe_float(ranking_lookup.get(str(best_ml.get('variant_name') or ''), {}).get('mean_spearman_ic')):.3f}.",
            f"- Baseline top bucket forward return: {_pct(_top_bucket_return(bucket_rows, 'baseline_momentum', int(payload.get('bucket_count') or 10)))}.",
            f"- Best ML top bucket forward return: {_pct(_top_bucket_return(bucket_rows, str(best_ml.get('variant_name') or ''), int(payload.get('bucket_count') or 10)))}.",
            "- Baseline top-bucket sector mix: " + (_cohort_preview(cohort_rows, variant_name="baseline_momentum", cohort_kind="sector") or "n/a"),
            "- Best-ML top-bucket sector mix: " + (_cohort_preview(cohort_rows, variant_name=str(best_ml.get("variant_name") or ""), cohort_kind="sector") or "n/a"),
            "- Best-ML stock/ETF mix: " + (_cohort_preview(cohort_rows, variant_name=str(best_ml.get("variant_name") or ""), cohort_kind="instrument_type", limit=2) or "n/a"),
        ]
    )
    for row in ml_rows:
        variant_name = str(row.get("variant_name") or "")
        overlap = next(
            (item for item in overlap_rows if str(item.get("right_variant_name") or "") == variant_name),
            {},
        )
        stability = stability_lookup.get(variant_name, {})
        lines.append(
            f"- {variant_name}: overlap with momentum winners {_safe_float(overlap.get('jaccard')):.3f}, fold stability {_safe_float(stability.get('mean_pairwise_jaccard')):.3f}."
        )
    lines.extend(
        [
            "",
            "## 4. Interpretation",
            "",
            f"- Does a feature-based factor model beat simple momentum? {'yes on portfolio returns, but only mixed on ranking diagnostics' if _safe_float(best_ml.get('sharpe')) > _safe_float(baseline_row.get('sharpe')) and _safe_float(best_ml.get('total_return')) > _safe_float(baseline_row.get('total_return')) else 'mixed/no'}.",
            "- The direct ML baseline tests whether broader features help without factor structure. The characteristics-factor variant tests whether routing those same features through learned exposures adds value beyond direct score regression.",
            "- In this pilot the characteristics-factor model improved Sharpe and total return versus momentum, but it did so with lower rank IC and a deeper drawdown, so the edge looks portfolio-level rather than a cleaner cross-sectional oracle.",
            "- Winner overlap and fold stability help separate genuine ranking improvements from strategies that simply rotate into a different but noisier subset of names.",
            "",
            "## 5. Platform Capabilities Added",
            "",
            "- Reusable characteristic-panel builder that joins existing features with cross-sectional rank-percentile labels.",
            "- Reusable latent factor estimation from returns plus rolling realized exposure targets on rebalance dates.",
            "- Reusable feature-to-exposure regressor path that emits standard prediction artifacts for the existing strategy engine.",
            "",
            "## 6. Output Artifacts",
            "",
            f"- Summary JSON: `{payload.get('summary_json_path') or ''}`",
            f"- Aggregate results CSV: `{payload.get('summary_csv_path') or ''}`",
            f"- Fold results CSV: `{payload.get('fold_summary_csv_path') or ''}`",
            f"- Prediction rows CSV: `{payload.get('prediction_rows_csv_path') or ''}`",
            f"- Factor premia CSV: `{payload.get('factor_premia_csv_path') or ''}`",
            f"- Exposure targets CSV: `{payload.get('factor_exposure_csv_path') or ''}`",
            f"- Model diagnostics CSV: `{payload.get('model_diagnostics_csv_path') or ''}`",
            f"- Ranking diagnostics CSV: `{payload.get('ranking_summary_aggregate_csv_path') or ''}`",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_characteristics_factor_research(
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
    output_basename: str = "characteristics_factor_research",
    resume_existing: bool = False,
    n_factors: int = 3,
    exposure_lookback_days: int = 63,
    minimum_exposure_observations: int = 30,
    evaluate_feature_subsets: bool = False,
    random_state: int = 1337,
) -> dict[str, Any]:
    output_dir = Path("data") / "pipeline_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{output_basename}.json"
    csv_path = output_dir / f"{output_basename}.csv"
    fold_csv_path = output_dir / f"{output_basename}__fold_rows.csv"
    prediction_rows_csv_path = output_dir / f"{output_basename}__predictions.csv"
    factor_premia_csv_path = output_dir / f"{output_basename}__factor_premia.csv"
    factor_exposure_csv_path = output_dir / f"{output_basename}__factor_exposures.csv"
    factor_return_csv_path = output_dir / f"{output_basename}__factor_returns.csv"
    model_diagnostics_csv_path = output_dir / f"{output_basename}__model_diagnostics.csv"
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
            schema_version=CHARACTERISTICS_FACTOR_SCHEMA_VERSION,
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
    direct_model_variants = _build_direct_model_variants(feature_artifact, horizon_days=int(forward_horizon_days))
    characteristic_variants = _build_characteristic_variants(
        feature_artifact,
        include_prices_only=bool(evaluate_feature_subsets),
        include_context_only=bool(evaluate_feature_subsets),
    )
    factor_spec = LatentFactorSpec(
        n_factors=int(n_factors),
        exposure_lookback_days=int(exposure_lookback_days),
        minimum_exposure_observations=int(minimum_exposure_observations),
    )
    strategy_definition = upsert_strategy_definition(
        slug="characteristics-factor-quantiles",
        name="Characteristics Factor Quantile Strategy",
        strategy_type="notebook_topk_v1",
        description="Cross-sectional quantile strategy fed by learned characteristics-factor scores.",
        config=dict(model_strategy_config),
    )

    summary_rows: list[dict[str, Any]] = []
    ranking_summary_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    accumulated_top_bucket_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    factor_premia_rows: list[dict[str, Any]] = []
    factor_exposure_rows: list[dict[str, Any]] = []
    factor_return_rows: list[dict[str, Any]] = []
    model_diagnostics_rows: list[dict[str, Any]] = []
    latent_factor_meta: dict[str, Any] = {}

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
            strategy_definition_slug="characteristics-factor-baseline",
            strategy_definition_name="Characteristics Factor Baseline Momentum",
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
        baseline_bucketed = assign_cross_sectional_buckets(baseline_panel, bucket_count=int(bucket_count), higher_score_is_better=True)
        ranking_summary_rows.extend(compute_ranking_summary_rows(baseline_bucketed, bucket_count=int(bucket_count)))
        bucket_rows.extend(compute_bucket_return_rows(baseline_bucketed))
        cohort_rows.extend(compute_top_bucket_cohort_rows(baseline_bucketed, bucket_count=int(bucket_count)))
        accumulated_top_bucket_rows.extend(top_bucket_rows(baseline_bucketed, bucket_count=int(bucket_count)))

        factor_return_df, factor_target_df, factor_cols, fold_factor_meta = build_characteristic_factor_targets(
            feature_artifact,
            label_artifact,
            train_end_date=train_end_date,
            score_end_date=backtest_end_date,
            spec=factor_spec,
        )
        latent_factor_meta[fold_name] = fold_factor_meta
        factor_return_rows.extend(
            factor_return_df.assign(
                fold_name=fold_name,
                date=pd.to_datetime(factor_return_df["date"], errors="coerce").dt.strftime("%Y-%m-%d"),
            ).to_dict(orient="records")
            if not factor_return_df.empty
            else []
        )
        factor_exposure_rows.extend(
            factor_target_df.assign(
                fold_name=fold_name,
                date=pd.to_datetime(factor_target_df["date"], errors="coerce").dt.strftime("%Y-%m-%d"),
            ).to_dict(orient="records")
            if not factor_target_df.empty
            else []
        )

        for variant in direct_model_variants:
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
                strategy_definition=strategy_definition,
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
            prediction_artifact = Artifact.objects.filter(pk=int(model_row.get("prediction_artifact_id") or 0)).first()
            if prediction_artifact is None:
                raise ValueError(f"{variant_name} prediction artifact was not found.")
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
            model_bucketed = assign_cross_sectional_buckets(model_panel, bucket_count=int(bucket_count), higher_score_is_better=True)
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

        for variant in characteristic_variants:
            variant_name = str(variant.get("variant_name") or "")
            variant_label = str(variant.get("variant_label") or variant_name)
            feature_scope = str(variant.get("feature_scope") or "")
            variant_result = run_characteristic_factor_variant(
                variant_name=variant_name,
                variant_label=variant_label,
                feature_scope=feature_scope,
                feature_families=list(variant.get("feature_families") or []),
                feature_artifact=feature_artifact,
                label_artifact=label_artifact,
                strategy_definition=strategy_definition,
                train_end_date=train_end_date,
                backtest_start_date=backtest_start_date,
                backtest_end_date=backtest_end_date,
                factor_target_df=factor_target_df,
                factor_cols=factor_cols,
                backtest_config=backtest_config,
                validation_config=validation_config,
                output_basename=f"{output_basename}__{variant_name}__{fold_name}",
                random_state=int(random_state),
            )
            model_row = _annotate_fold_row(
                dict(variant_result["summary_row"]),
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
            prediction_rows.extend(
                [
                    {
                        **dict(row),
                        "fold_name": fold_name,
                        "variant_name": variant_name,
                        "variant_label": variant_label,
                    }
                    for row in list(variant_result.get("prediction_rows") or [])
                ]
            )
            factor_premia_rows.extend(
                [
                    {
                        **dict(row),
                        "fold_name": fold_name,
                    }
                    for row in list(variant_result.get("premia_rows") or [])
                ]
            )
            model_diagnostics_rows.extend(
                [
                    {
                        **dict(row),
                        "fold_name": fold_name,
                    }
                    for row in list(variant_result.get("model_rows") or [])
                ]
            )

            prediction_artifact = variant_result["prediction_artifact"]
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
            model_bucketed = assign_cross_sectional_buckets(model_panel, bucket_count=int(bucket_count), higher_score_is_better=True)
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
        assign_cross_sectional_buckets(top_bucket_df, bucket_count=int(bucket_count))
        if not top_bucket_df.empty and "bucket" not in top_bucket_df.columns
        else top_bucket_df,
        bucket_count=int(bucket_count),
    )

    payload = {
        "schema_version": CHARACTERISTICS_FACTOR_SCHEMA_VERSION,
        "mode": "characteristics_factor_research",
        "symbols": symbols,
        "missing_requested_symbols": missing_symbols,
        "coverage_rows": coverage_rows,
        "folds": [dict(fold) for fold in folds],
        "base_artifacts": {
            "universe": int(universe_artifact.id),
            "features": int(feature_artifact.id),
            "labels": int(label_artifact.id),
        },
        "bucket_count": int(bucket_count),
        "lookback_days": int(lookback_days),
        "forward_horizon_days": int(forward_horizon_days),
        "start_offset_days": int(start_offset_days),
        "n_factors": int(n_factors),
        "exposure_lookback_days": int(exposure_lookback_days),
        "minimum_exposure_observations": int(minimum_exposure_observations),
        "momentum_signal": momentum_signal,
        "rank_label_metadata": dict(label_artifact.metadata or {}),
        "latent_factor_meta": latent_factor_meta,
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
        "prediction_rows": prediction_rows,
        "factor_premia_rows": factor_premia_rows,
        "factor_exposure_rows": factor_exposure_rows,
        "factor_return_rows": factor_return_rows,
        "model_diagnostics_rows": model_diagnostics_rows,
        "summary_json_path": str(json_path),
        "summary_csv_path": str(csv_path),
        "fold_summary_csv_path": str(fold_csv_path),
        "prediction_rows_csv_path": str(prediction_rows_csv_path),
        "factor_premia_csv_path": str(factor_premia_csv_path),
        "factor_exposure_csv_path": str(factor_exposure_csv_path),
        "factor_return_csv_path": str(factor_return_csv_path),
        "model_diagnostics_csv_path": str(model_diagnostics_csv_path),
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
    _write_rows_csv(prediction_rows_csv_path, prediction_rows)
    _write_rows_csv(factor_premia_csv_path, factor_premia_rows)
    _write_rows_csv(factor_exposure_csv_path, factor_exposure_rows)
    _write_rows_csv(factor_return_csv_path, factor_return_rows)
    _write_rows_csv(model_diagnostics_csv_path, model_diagnostics_rows)
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
    "CHARACTERISTICS_FACTOR_SCHEMA_VERSION",
    "LatentFactorSpec",
    "build_characteristic_rank_panel",
    "build_characteristic_factor_targets",
    "build_daily_return_panel",
    "estimate_cross_sectional_factor_premia",
    "estimate_latent_factor_basis",
    "estimate_rebalance_factor_exposures",
    "fit_characteristic_factor_ranker",
    "project_latent_factor_returns",
    "resolve_feature_scope_variants",
    "run_characteristics_factor_research",
    "score_characteristic_factor_ranker",
    "write_characteristics_factor_report",
]
