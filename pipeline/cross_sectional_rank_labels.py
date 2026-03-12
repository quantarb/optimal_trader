from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import uuid
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from django.utils import timezone

from ml.execution import load_artifact_csv_frame

from .models import Artifact, PipelineRun
from .service_runtime import artifact_payload_hash, stable_payload_hash, write_frame_artifact


RANK_LABELS_SCHEMA_VERSION = 1
DEFAULT_PRICE_COL_CANDIDATES: tuple[str, ...] = ("px__adj_close", "adj_close", "px__close", "close")
DEFAULT_SIZE_COL_CANDIDATES: tuple[str, ...] = ("km__marketcap", "marketcap", "market_cap")
DEFAULT_MOMENTUM_LONG_COL_CANDIDATES: tuple[str, ...] = ("px__ret_252d", "px__ret_252_d", "ret_252d", "ret_252_d")
DEFAULT_MOMENTUM_SHORT_COL_CANDIDATES: tuple[str, ...] = ("px__ret_21d", "px__ret_21_d", "ret_21d", "ret_21_d")
DEFAULT_MOMENTUM_FALLBACK_COL_CANDIDATES: tuple[str, ...] = (
    "px__ret_189d",
    "px__ret_189_d",
    "px__ret_126d",
    "px__ret_126_d",
    "px__ret_63d",
    "px__ret_63_d",
    "ret_1",
)
DEFAULT_VOLATILITY_COL_CANDIDATES: tuple[str, ...] = (
    "px__vol_20",
    "px__vol_21",
    "px__vol_63",
    "px__atr_pct_20",
    "px__atr_pct_14",
    "vol_20",
    "vol_21",
    "vol_63",
    "vol_10",
    "vol_5",
)


@dataclass(frozen=True)
class CrossSectionalRankLabelSpec:
    horizon_days: int = 21
    rebalance_freq: str = "M"
    start_offset_days: int = 1
    minimum_cross_section: int = 10
    label_variant: str = "raw"
    target_col: str = "future_rank_pct"
    forward_return_col: str = "trade_return"
    residualize_targets: bool = False
    residual_target_col: str = "residual_rank_pct"
    residual_return_col: str = "residual_return"
    fitted_return_col: str = "factor_expected_return"
    price_col_candidates: tuple[str, ...] = DEFAULT_PRICE_COL_CANDIDATES
    size_col_candidates: tuple[str, ...] = DEFAULT_SIZE_COL_CANDIDATES
    momentum_long_col_candidates: tuple[str, ...] = DEFAULT_MOMENTUM_LONG_COL_CANDIDATES
    momentum_short_col_candidates: tuple[str, ...] = DEFAULT_MOMENTUM_SHORT_COL_CANDIDATES
    momentum_fallback_col_candidates: tuple[str, ...] = DEFAULT_MOMENTUM_FALLBACK_COL_CANDIDATES
    volatility_col_candidates: tuple[str, ...] = DEFAULT_VOLATILITY_COL_CANDIDATES


def first_available_column(columns: Iterable[str], candidates: Sequence[str]) -> str:
    available = {str(column).strip() for column in list(columns) if str(column).strip()}
    for candidate in list(candidates or []):
        value = str(candidate or "").strip()
        if value and value in available:
            return value
    return ""


def rebalance_dates(unique_dates: pd.DatetimeIndex, rebalance_freq: str) -> set[pd.Timestamp]:
    freq_value = str(rebalance_freq or "M").strip().upper() or "M"
    if freq_value == "D":
        return set(pd.DatetimeIndex(unique_dates))
    if freq_value == "W":
        return set(
            pd.Series(unique_dates, index=unique_dates).groupby(unique_dates.to_period("W")).head(1).tolist()
        )
    return set(
        pd.Series(unique_dates, index=unique_dates).groupby(unique_dates.to_period("M")).head(1).tolist()
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _rank_percentile(series: pd.Series) -> pd.Series:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    out = pd.Series(0.5, index=series.index, dtype=float)
    if valid.empty:
        return out
    if len(valid) <= 1:
        out.loc[valid.index] = 0.5
        return out
    rank_positions = valid.rank(method="average", ascending=True)
    out.loc[rank_positions.index] = (rank_positions - 1.0) / float(len(rank_positions) - 1)
    return out.clip(lower=0.0, upper=1.0)


def _normalize_cross_sectional_signal(series: pd.Series) -> pd.Series:
    ranked = _rank_percentile(series)
    return (ranked - 0.5).astype(float)


def _resolve_label_target_column(spec: CrossSectionalRankLabelSpec) -> str:
    variant = str(spec.label_variant or "raw").strip().lower()
    if variant == "residual":
        return str(spec.residual_target_col or "residual_rank_pct")
    return str(spec.target_col or "future_rank_pct")


def _resolve_residual_factor_columns(
    columns: Iterable[str],
    spec: CrossSectionalRankLabelSpec,
) -> dict[str, Any]:
    column_names = list(columns)
    size_col = first_available_column(column_names, spec.size_col_candidates)
    momentum_long = first_available_column(column_names, spec.momentum_long_col_candidates)
    momentum_short = first_available_column(column_names, spec.momentum_short_col_candidates)
    momentum_fallback = first_available_column(column_names, spec.momentum_fallback_col_candidates)
    volatility_col = first_available_column(column_names, spec.volatility_col_candidates)
    required_columns = [column for column in [size_col, momentum_long, momentum_short, momentum_fallback, volatility_col] if column]
    return {
        "size_col": str(size_col or ""),
        "momentum_long_col": str(momentum_long or ""),
        "momentum_short_col": str(momentum_short or ""),
        "momentum_fallback_col": str(momentum_fallback or ""),
        "volatility_col": str(volatility_col or ""),
        "required_feature_columns": list(dict.fromkeys(required_columns)),
    }


def _symbol_metadata_lookup(symbols: Sequence[str]) -> dict[str, dict[str, str]]:
    from fmp.models import Symbol

    normalized = [str(symbol).strip().upper() for symbol in list(symbols or []) if str(symbol).strip()]
    if not normalized:
        return {}
    rows = Symbol.objects.filter(symbol__in=normalized).only("symbol", "sector", "company_name", "payload")
    lookup: dict[str, dict[str, str]] = {}
    for row in rows:
        payload = row.payload if isinstance(row.payload, dict) else {}
        name = str(row.company_name or "").strip().lower()
        is_etf = bool(payload.get("isEtf")) or bool(payload.get("isETF")) or bool(payload.get("isFund")) or (" etf" in f" {name}")
        symbol = str(row.symbol or "").strip().upper()
        if not symbol:
            continue
        lookup[symbol] = {
            "sector": str(row.sector or "").strip() or "Unknown",
            "instrument_type": "etf" if is_etf else "stock",
        }
    return lookup


def _compute_momentum_proxy(panel: pd.DataFrame, factor_cols: Mapping[str, Any]) -> pd.Series:
    long_col = str(factor_cols.get("momentum_long_col") or "")
    short_col = str(factor_cols.get("momentum_short_col") or "")
    fallback_col = str(factor_cols.get("momentum_fallback_col") or "")
    if long_col and short_col and long_col in panel.columns and short_col in panel.columns:
        long_values = pd.to_numeric(panel[long_col], errors="coerce")
        short_values = pd.to_numeric(panel[short_col], errors="coerce")
        return ((1.0 + long_values) / (1.0 + short_values)) - 1.0
    if fallback_col and fallback_col in panel.columns:
        return pd.to_numeric(panel[fallback_col], errors="coerce")
    return pd.Series(index=panel.index, dtype=float)


def _build_factor_residual_columns(
    label_df: pd.DataFrame,
    *,
    spec: CrossSectionalRankLabelSpec,
    factor_cols: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = label_df.copy()
    out[spec.fitted_return_col] = pd.Series(index=out.index, dtype=float)
    out[spec.residual_return_col] = pd.Series(index=out.index, dtype=float)
    out[spec.residual_target_col] = pd.Series(index=out.index, dtype=float)
    out["size_proxy"] = pd.to_numeric(out.get(str(factor_cols.get("size_col") or "")), errors="coerce")
    out["momentum_proxy"] = _compute_momentum_proxy(out, factor_cols)
    out["volatility_proxy"] = pd.to_numeric(out.get(str(factor_cols.get("volatility_col") or "")), errors="coerce")
    metadata_lookup = _symbol_metadata_lookup(out["symbol"].unique().tolist())
    out["sector"] = out["symbol"].map(lambda symbol: str(metadata_lookup.get(str(symbol), {}).get("sector") or "Unknown"))
    out["instrument_type"] = out["symbol"].map(
        lambda symbol: str(metadata_lookup.get(str(symbol), {}).get("instrument_type") or "stock")
    )

    group_rows: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for date_value, group in out.groupby("date", sort=True):
        work = group.copy()
        y = pd.to_numeric(work[spec.forward_return_col], errors="coerce")
        size_signal = _normalize_cross_sectional_signal(np.log1p(pd.to_numeric(work["size_proxy"], errors="coerce").clip(lower=0.0)))
        momentum_signal = _normalize_cross_sectional_signal(pd.to_numeric(work["momentum_proxy"], errors="coerce"))
        volatility_signal = _normalize_cross_sectional_signal(pd.to_numeric(work["volatility_proxy"], errors="coerce"))
        design_parts: list[pd.DataFrame] = []
        numeric_part = pd.DataFrame(
            {
                "size_proxy": size_signal,
                "momentum_proxy": momentum_signal,
                "volatility_proxy": volatility_signal,
            },
            index=work.index,
        )
        numeric_part = numeric_part.loc[:, numeric_part.notna().any(axis=0)]
        if not numeric_part.empty:
            design_parts.append(numeric_part.fillna(0.0))
        if work["sector"].astype(str).nunique() > 1:
            sector_dummies = pd.get_dummies(work["sector"].fillna("Unknown"), prefix="sector", dtype=float)
            if sector_dummies.shape[1] > 1:
                sector_dummies = sector_dummies.iloc[:, 1:]
                if not sector_dummies.empty:
                    design_parts.append(sector_dummies)
        if work["instrument_type"].astype(str).nunique() > 1:
            design_parts.append(
                pd.DataFrame(
                    {"is_etf": work["instrument_type"].eq("etf").astype(float)},
                    index=work.index,
                )
            )

        x_design = pd.concat(design_parts, axis=1) if design_parts else pd.DataFrame(index=work.index)
        x_values = x_design.to_numpy(dtype=float) if not x_design.empty else np.empty((len(work), 0), dtype=float)
        valid_mask = np.isfinite(y.to_numpy(dtype=float))
        if x_values.shape[1] > 0:
            valid_mask &= np.isfinite(x_values).all(axis=1)
        if int(valid_mask.sum()) <= 1:
            fitted_values = pd.Series(float(y.mean()) if y.notna().any() else 0.0, index=work.index, dtype=float)
        else:
            design_matrix = np.ones((int(valid_mask.sum()), 1), dtype=float)
            if x_values.shape[1] > 0:
                design_matrix = np.column_stack([design_matrix, x_values[valid_mask]])
            coeffs, *_ = np.linalg.lstsq(design_matrix, y.to_numpy(dtype=float)[valid_mask], rcond=None)
            full_design = np.ones((len(work), 1), dtype=float)
            if x_values.shape[1] > 0:
                full_design = np.column_stack([full_design, x_values])
            fitted_values = pd.Series(full_design @ coeffs, index=work.index, dtype=float)
        work[spec.fitted_return_col] = fitted_values.astype(float)
        work[spec.residual_return_col] = (pd.to_numeric(work[spec.forward_return_col], errors="coerce") - fitted_values).astype(float)
        work[spec.residual_target_col] = _rank_percentile(work[spec.residual_return_col])
        group_rows.append(work)
        diagnostics.append(
            {
                "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                "cross_section_size": int(len(work)),
                "design_columns": int(x_design.shape[1]),
                "sector_count": int(work["sector"].astype(str).nunique()),
                "instrument_type_count": int(work["instrument_type"].astype(str).nunique()),
                "residual_mean": round(float(pd.to_numeric(work[spec.residual_return_col], errors="coerce").mean()), 8),
                "residual_std": round(
                    float(pd.to_numeric(work[spec.residual_return_col], errors="coerce").std(ddof=0))
                    if len(work) > 1
                    else 0.0,
                    8,
                ),
            }
        )

    if not group_rows:
        return out, {
            "residualization_enabled": False,
            "residualization_dates": 0,
        }
    residual_df = pd.concat(group_rows, ignore_index=True, sort=False).sort_values(["date", "symbol"]).reset_index(drop=True)
    residual_values = pd.to_numeric(residual_df[spec.residual_return_col], errors="coerce")
    target_values = pd.to_numeric(residual_df[spec.residual_target_col], errors="coerce")
    return residual_df, {
        "residualization_enabled": True,
        "label_variant": str(spec.label_variant or "raw").strip().lower() or "raw",
        "residualization_dates": int(len(diagnostics)),
        "residual_return_col": str(spec.residual_return_col),
        "fitted_return_col": str(spec.fitted_return_col),
        "residual_target_col": str(spec.residual_target_col),
        "size_col": str(factor_cols.get("size_col") or ""),
        "momentum_long_col": str(factor_cols.get("momentum_long_col") or ""),
        "momentum_short_col": str(factor_cols.get("momentum_short_col") or ""),
        "momentum_fallback_col": str(factor_cols.get("momentum_fallback_col") or ""),
        "volatility_col": str(factor_cols.get("volatility_col") or ""),
        "residual_return_mean": round(float(residual_values.mean()) if residual_values.notna().any() else 0.0, 8),
        "residual_return_std": round(float(residual_values.std(ddof=0)) if len(residual_values) > 1 else 0.0, 8),
        "residual_target_mean": round(float(target_values.mean()) if target_values.notna().any() else 0.0, 8),
        "residual_target_std": round(float(target_values.std(ddof=0)) if len(target_values) > 1 else 0.0, 8),
        "residualization_diagnostics": diagnostics,
    }


def build_cross_sectional_rank_label_frame(
    feature_frame_or_artifact: Artifact | pd.DataFrame,
    *,
    spec: CrossSectionalRankLabelSpec | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved_spec = spec or CrossSectionalRankLabelSpec()
    feature_df = (
        load_artifact_csv_frame(feature_frame_or_artifact)
        if isinstance(feature_frame_or_artifact, Artifact)
        else pd.DataFrame(feature_frame_or_artifact).copy()
    )
    if feature_df.empty:
        return pd.DataFrame(), {"rows": 0}
    if "date" not in feature_df.columns or "symbol" not in feature_df.columns:
        raise ValueError("Feature frame must contain 'date' and 'symbol' columns.")

    price_col = first_available_column(feature_df.columns, resolved_spec.price_col_candidates)
    if not price_col:
        raise ValueError(
            "Feature frame does not contain a usable price column. "
            f"Tried: {', '.join(resolved_spec.price_col_candidates)}."
        )

    horizon_days = max(int(resolved_spec.horizon_days), 1)
    start_offset_days = max(int(resolved_spec.start_offset_days), 0)
    minimum_cross_section = max(int(resolved_spec.minimum_cross_section), 2)
    active_target_col = _resolve_label_target_column(resolved_spec)
    factor_cols = _resolve_residual_factor_columns(feature_df.columns, resolved_spec)

    panel_columns = ["date", "symbol", price_col] + list(factor_cols.get("required_feature_columns") or [])
    panel = feature_df.loc[:, [column for column in panel_columns if column in feature_df.columns]].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["symbol"] = panel["symbol"].astype(str).str.strip().str.upper()
    panel[price_col] = pd.to_numeric(panel[price_col], errors="coerce")
    panel = panel.dropna(subset=["date", "symbol", price_col]).sort_values(["symbol", "date"]).reset_index(drop=True)
    if start_date:
        panel = panel[panel["date"] >= pd.Timestamp(str(start_date))].copy()
    if end_date:
        panel = panel[panel["date"] <= pd.Timestamp(str(end_date))].copy()
    if panel.empty:
        return pd.DataFrame(), {"rows": 0, "price_col": price_col}

    grouped_price = panel.groupby("symbol", sort=False)[price_col]
    grouped_dates = panel.groupby("symbol", sort=False)["date"]
    panel[resolved_spec.forward_return_col] = (
        grouped_price.shift(-(start_offset_days + horizon_days)) / grouped_price.shift(-start_offset_days)
    ) - 1.0
    panel["forward_start_date"] = grouped_dates.shift(-start_offset_days)
    panel["forward_end_date"] = grouped_dates.shift(-(start_offset_days + horizon_days))

    unique_dates = pd.DatetimeIndex(sorted(panel["date"].dropna().unique()))
    rebalance_date_set = rebalance_dates(unique_dates, resolved_spec.rebalance_freq)
    label_df = panel[panel["date"].isin(rebalance_date_set)].copy()
    label_df = label_df.dropna(
        subset=[resolved_spec.forward_return_col, "forward_start_date", "forward_end_date"]
    ).reset_index(drop=True)
    if label_df.empty:
        return pd.DataFrame(), {"rows": 0, "price_col": price_col}

    kept_groups: list[pd.DataFrame] = []
    cross_section_sizes: list[int] = []
    for _date_value, group in label_df.groupby("date", sort=True):
        valid = group.dropna(subset=[resolved_spec.forward_return_col]).copy()
        if len(valid) < minimum_cross_section:
            continue
        valid[resolved_spec.target_col] = _rank_percentile(valid[resolved_spec.forward_return_col])
        valid["cross_section_size"] = int(len(valid))
        cross_section_sizes.append(int(len(valid)))
        kept_groups.append(valid)

    if not kept_groups:
        return pd.DataFrame(), {"rows": 0, "price_col": price_col}

    out = pd.concat(kept_groups, ignore_index=True, sort=False)
    residual_meta: dict[str, Any] = {}
    if bool(resolved_spec.residualize_targets):
        out, residual_meta = _build_factor_residual_columns(
            out,
            spec=resolved_spec,
            factor_cols=factor_cols,
        )
    elif active_target_col != str(resolved_spec.target_col):
        raise ValueError(
            "CrossSectionalRankLabelSpec.label_variant='residual' requires residualize_targets=True "
            "so the residual target can be computed."
        )
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["forward_start_date"] = pd.to_datetime(out["forward_start_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["forward_end_date"] = pd.to_datetime(out["forward_end_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["label"] = (pd.to_numeric(out[active_target_col], errors="coerce") >= 0.5).astype(int)
    out["market_position"] = out["label"].astype(int)
    out["hold_days"] = int(horizon_days)
    out["freq"] = str(resolved_spec.rebalance_freq).strip().upper() or "M"
    out["k"] = int(horizon_days)
    out["side"] = out["label"].map({1: "long", 0: "short"})
    output_columns = [
        "date",
        "symbol",
        resolved_spec.target_col,
        resolved_spec.forward_return_col,
        resolved_spec.fitted_return_col,
        resolved_spec.residual_return_col,
        resolved_spec.residual_target_col,
        "label",
        "market_position",
        "hold_days",
        "freq",
        "k",
        "forward_start_date",
        "forward_end_date",
        "cross_section_size",
    ]
    output_columns = [column for column in output_columns if column in out.columns]
    out = out[output_columns].sort_values(["date", "symbol"]).reset_index(drop=True)

    forward_returns = pd.to_numeric(out[resolved_spec.forward_return_col], errors="coerce")
    target_values = pd.to_numeric(out[resolved_spec.target_col], errors="coerce")
    residual_returns = (
        pd.to_numeric(out[resolved_spec.residual_return_col], errors="coerce")
        if resolved_spec.residual_return_col in out.columns
        else pd.Series(dtype=float)
    )
    residual_targets = (
        pd.to_numeric(out[resolved_spec.residual_target_col], errors="coerce")
        if resolved_spec.residual_target_col in out.columns
        else pd.Series(dtype=float)
    )
    return out, {
        "schema_version": RANK_LABELS_SCHEMA_VERSION,
        "rows": int(len(out)),
        "symbols": int(out["symbol"].nunique()),
        "rebalance_dates": int(out["date"].nunique()),
        "price_col": price_col,
        "horizon_days": int(horizon_days),
        "start_offset_days": int(start_offset_days),
        "minimum_cross_section": int(minimum_cross_section),
        "label_variant": str(resolved_spec.label_variant or "raw").strip().lower() or "raw",
        "active_target_col": str(active_target_col),
        "rebalance_freq": str(resolved_spec.rebalance_freq).strip().upper() or "M",
        "target_col": str(resolved_spec.target_col),
        "forward_return_col": str(resolved_spec.forward_return_col),
        "label_start_date": str(out["date"].min() or ""),
        "label_end_date": str(out["date"].max() or ""),
        "mean_cross_section_size": round(float(sum(cross_section_sizes) / len(cross_section_sizes)), 6),
        "min_cross_section_size": min(cross_section_sizes) if cross_section_sizes else 0,
        "max_cross_section_size": max(cross_section_sizes) if cross_section_sizes else 0,
        "forward_return_mean": round(float(forward_returns.mean()), 8),
        "forward_return_std": round(float(forward_returns.std(ddof=0)) if len(forward_returns) > 1 else 0.0, 8),
        "target_mean": round(float(target_values.mean()), 8),
        "target_std": round(float(target_values.std(ddof=0)) if len(target_values) > 1 else 0.0, 8),
        "residual_return_mean": round(float(residual_returns.mean()), 8) if not residual_returns.empty else 0.0,
        "residual_return_std": round(float(residual_returns.std(ddof=0)) if len(residual_returns) > 1 else 0.0, 8) if not residual_returns.empty else 0.0,
        "residual_target_mean": round(float(residual_targets.mean()), 8) if not residual_targets.empty else 0.0,
        "residual_target_std": round(float(residual_targets.std(ddof=0)) if len(residual_targets) > 1 else 0.0, 8) if not residual_targets.empty else 0.0,
        **dict(residual_meta),
    }


def resolve_or_build_cross_sectional_rank_label_artifact(
    *,
    feature_artifact: Artifact,
    spec: CrossSectionalRankLabelSpec | None = None,
    output_basename: str = "cross_sectional_rank_labels",
    start_date: str | None = None,
    end_date: str | None = None,
) -> Artifact:
    if feature_artifact is None:
        raise ValueError("feature_artifact is required.")
    resolved_spec = spec or CrossSectionalRankLabelSpec()
    cache_key = stable_payload_hash(
        {
            "source_features_artifact_id": int(feature_artifact.id),
            "feature_uri": str(feature_artifact.uri or ""),
            "horizon_days": int(resolved_spec.horizon_days),
            "rebalance_freq": str(resolved_spec.rebalance_freq),
            "start_offset_days": int(resolved_spec.start_offset_days),
            "minimum_cross_section": int(resolved_spec.minimum_cross_section),
            "label_variant": str(resolved_spec.label_variant or "raw"),
            "target_col": str(resolved_spec.target_col),
            "forward_return_col": str(resolved_spec.forward_return_col),
            "residualize_targets": bool(resolved_spec.residualize_targets),
            "residual_target_col": str(resolved_spec.residual_target_col),
            "residual_return_col": str(resolved_spec.residual_return_col),
            "fitted_return_col": str(resolved_spec.fitted_return_col),
            "price_col_candidates": list(resolved_spec.price_col_candidates),
            "size_col_candidates": list(resolved_spec.size_col_candidates),
            "momentum_long_col_candidates": list(resolved_spec.momentum_long_col_candidates),
            "momentum_short_col_candidates": list(resolved_spec.momentum_short_col_candidates),
            "momentum_fallback_col_candidates": list(resolved_spec.momentum_fallback_col_candidates),
            "volatility_col_candidates": list(resolved_spec.volatility_col_candidates),
            "start_date": str(start_date or ""),
            "end_date": str(end_date or ""),
        }
    )
    cached_qs = Artifact.objects.filter(
        artifact_type="LABELS",
        pipeline_run__status=PipelineRun.Status.SUCCEEDED,
        metadata__rank_label_cache_key=str(cache_key),
        metadata__source_features_artifact_id=int(feature_artifact.id),
    ).select_related("pipeline_run").order_by("-created_at", "-id")
    for cached in cached_qs:
        uri = Path(str(cached.uri or "").strip())
        if uri.exists():
            return cached

    label_df, meta = build_cross_sectional_rank_label_frame(
        feature_artifact,
        spec=resolved_spec,
        start_date=start_date,
        end_date=end_date,
    )
    if label_df.empty:
        raise ValueError("Cross-sectional rank label build produced no rows.")

    key = f"{output_basename}_{uuid.uuid4().hex}"
    stored = write_frame_artifact(
        key,
        frame=label_df,
        fieldnames=list(label_df.columns),
    )
    now = timezone.now()
    pipeline_run = PipelineRun.objects.create(
        name=f"{output_basename}-artifact",
        requested_job="cross_sectional_rank_labels",
        mode=PipelineRun.Mode.STRICT,
        status=PipelineRun.Status.SUCCEEDED,
        config={
            "source_features_artifact_id": int(feature_artifact.id),
            "horizon_days": int(resolved_spec.horizon_days),
            "rebalance_freq": str(resolved_spec.rebalance_freq),
            "start_offset_days": int(resolved_spec.start_offset_days),
            "minimum_cross_section": int(resolved_spec.minimum_cross_section),
            "label_variant": str(resolved_spec.label_variant or "raw"),
            "target_col": str(resolved_spec.target_col),
            "forward_return_col": str(resolved_spec.forward_return_col),
            "residualize_targets": bool(resolved_spec.residualize_targets),
            "residual_target_col": str(resolved_spec.residual_target_col),
            "residual_return_col": str(resolved_spec.residual_return_col),
            "fitted_return_col": str(resolved_spec.fitted_return_col),
            "start_date": str(start_date or ""),
            "end_date": str(end_date or ""),
        },
        started_at=now,
        finished_at=now,
    )
    content = {
        "rows": int(meta.get("rows") or 0),
        "symbols": int(meta.get("symbols") or 0),
        "rebalance_dates": int(meta.get("rebalance_dates") or 0),
        "target_col": str(meta.get("active_target_col") or resolved_spec.target_col),
        "forward_return_col": str(resolved_spec.forward_return_col),
        "horizon_days": int(resolved_spec.horizon_days),
        "rebalance_freq": str(resolved_spec.rebalance_freq).strip().upper() or "M",
        "label_variant": str(resolved_spec.label_variant or "raw").strip().lower() or "raw",
    }
    metadata = {
        "schema_version": RANK_LABELS_SCHEMA_VERSION,
        "source_features_artifact_id": int(feature_artifact.id),
        "rank_label_cache_key": str(cache_key),
        **dict(meta),
        **stored.storage_metadata(),
    }
    payload_hash = artifact_payload_hash(content, stored.uri)
    return Artifact.objects.create(
        pipeline_run=pipeline_run,
        producer_job=None,
        artifact_type="LABELS",
        key=f"rank_labels_{uuid.uuid4().hex}",
        uri=stored.uri,
        content=content,
        metadata=metadata,
        payload_hash=payload_hash,
    )


__all__ = [
    "CrossSectionalRankLabelSpec",
    "DEFAULT_PRICE_COL_CANDIDATES",
    "RANK_LABELS_SCHEMA_VERSION",
    "build_cross_sectional_rank_label_frame",
    "first_available_column",
    "rebalance_dates",
    "resolve_or_build_cross_sectional_rank_label_artifact",
]
