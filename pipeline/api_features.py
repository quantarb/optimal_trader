from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from data import build_technical_panel
from features.fundamentals import broadcast_fundamentals_to_daily, fetch_fundamentals_data
from features.macro import MacroFeatureConfig, broadcast_macro_to_daily, fetch_macro_series
from features.time_features import TimeFeatureConfig, build_time_features
from pipeline.api_common import (
    _camelize_dataframe_columns,
    _compact_dataframe_columns,
    _drop_source_duplicate_columns,
    _extract_date_bounds,
    _filter_df_by_date_bounds,
    _summarize_dataset_for_llm,
)


def _resolve_fundamental_limit(
    *,
    start_date: Optional[str],
    end_date: Optional[str],
    period: str,
) -> int:
    """
    Ensure fundamentals fetch limit can cover requested date span.
    FMP fundamentals are period/limit-based; they are not date-ranged at source.
    """
    base = 160
    if start_date is None or end_date is None:
        return base

    s = pd.Timestamp(start_date)
    e = pd.Timestamp(end_date)
    if s > e:
        return base

    years = max(0.0, float((e - s).days) / 365.25)
    p = str(period).lower()
    if p.startswith("q"):
        needed = int(math.ceil(years * 4.0)) + 8
    elif p.startswith("a") or p.startswith("y"):
        needed = int(math.ceil(years)) + 2
    else:
        return base

    return max(base, needed)


_DAILY_PRICE_LINKED_FUNDAMENTAL_COLUMNS: tuple[str, ...] = (
    "km__marketcap",
    "km__enterprisevalue",
    "km__evtosales",
    "km__evtooperatingcashflow",
    "km__evtofreecashflow",
    "km__evtoebitda",
    "rt__enterprisevaluemultiple",
    "rt__debttomarketcap",
    "km__earningsyield",
    "km__freecashflowyield",
    "rt__dividendyield",
    "rt__dividendyieldpercentage",
    "rt__pricetoearningsratio",
    "rt__pricetoearningsgrowthratio",
    "rt__forwardpricetoearningsgrowthratio",
    "rt__pricetobookratio",
    "rt__pricetosalesratio",
    "rt__pricetofreecashflowratio",
    "rt__pricetooperatingcashflowratio",
    "rt__pricetofairvalue",
)


def _safe_divide(numer: pd.Series, denom: pd.Series) -> pd.Series:
    out = pd.to_numeric(numer, errors="coerce") / pd.to_numeric(denom, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def _add_fundamental_update_timing_features(
    dense_fund_df: pd.DataFrame,
    *,
    sparse_effective_index: pd.MultiIndex,
) -> pd.DataFrame:
    """
    Add timing features based on sparse fundamental update dates.
    """
    if dense_fund_df.empty or not isinstance(dense_fund_df.index, pd.MultiIndex):
        return dense_fund_df

    out = dense_fund_df.copy()
    idx = out.index

    if "date" not in (idx.names or []) or "symbol" not in (idx.names or []):
        return out

    event = pd.Series(0.0, index=idx, dtype="float64")
    if sparse_effective_index is not None and len(sparse_effective_index) > 0:
        sparse_idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp(d), str(s)) for d, s in sparse_effective_index.to_list()],
            names=["date", "symbol"],
        )
        aligned = idx.intersection(sparse_idx)
        if len(aligned) > 0:
            event.loc[aligned] = 1.0

    work = pd.DataFrame(
        {
            "_date": pd.to_datetime(idx.get_level_values("date")),
            "_symbol": idx.get_level_values("symbol").astype(str),
            "_event": event.values,
        },
        index=idx,
    )
    work = work.sort_values(["_symbol", "_date"])
    work["_last_update_date"] = work["_date"].where(work["_event"] > 0.0)
    work["_last_update_date"] = work.groupby("_symbol", sort=False)["_last_update_date"].ffill()
    days = (work["_date"] - work["_last_update_date"]).dt.days.astype("float64")

    out["km__dayssincelastfundamentalupdate"] = days
    out["km__isnewfundamentalupdate"] = work["_event"].astype("float64")
    return out


def _extract_daily_close_series(
    daily_prices: Optional[Union[pd.DataFrame, pd.Series]],
    *,
    target_index: Optional[pd.Index] = None,
    preferred_col: str = "Close",
) -> Optional[pd.Series]:
    if daily_prices is None:
        return None

    if isinstance(daily_prices, pd.Series):
        out = pd.to_numeric(daily_prices, errors="coerce")
    else:
        df = daily_prices.copy()
        candidates = [
            str(preferred_col),
            str(preferred_col).lower(),
            "Close",
            "close",
            "AdjClose",
            "adjclose",
            "Adj_Close",
            "adj_close",
        ]
        col = next((c for c in candidates if c in df.columns), None)
        if col is None:
            return None
        out = pd.to_numeric(df[col], errors="coerce")

    if isinstance(out.index, pd.MultiIndex):
        names = list(out.index.names or [])
        if "date" not in names or "symbol" not in names:
            return None
    else:
        return None

    out = out.rename("close_daily")
    if target_index is not None:
        out = out.reindex(target_index)
    return out


def _recompute_price_linked_fundamentals_daily(
    fund_df: pd.DataFrame,
    close_daily: pd.Series,
) -> tuple[pd.DataFrame, list[str]]:
    if fund_df.empty or close_daily is None or close_daily.empty:
        return fund_df, []

    out = fund_df.copy()
    close = pd.to_numeric(close_daily, errors="coerce")
    close = close.reindex(out.index)
    base = out.copy()

    def has(col: str) -> bool:
        return col in out.columns

    def upd(col: str, values: pd.Series, updated_cols: list[str]) -> None:
        if not has(col):
            return
        out[col] = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if col not in updated_cols:
            updated_cols.append(col)

    updated: list[str] = []

    if has("rt__netincomepershare"):
        eps = pd.to_numeric(out["rt__netincomepershare"], errors="coerce")
        upd("rt__pricetoearningsratio", _safe_divide(close, eps), updated)
        upd("km__earningsyield", _safe_divide(eps, close), updated)

    if has("rt__bookvaluepershare"):
        bvps = pd.to_numeric(out["rt__bookvaluepershare"], errors="coerce")
        pb = _safe_divide(close, bvps)
        upd("rt__pricetobookratio", pb, updated)
        upd("rt__pricetofairvalue", pb, updated)

    if has("rt__revenuepershare"):
        rev_ps = pd.to_numeric(out["rt__revenuepershare"], errors="coerce")
        upd("rt__pricetosalesratio", _safe_divide(close, rev_ps), updated)

    if has("rt__freecashflowpershare"):
        fcf_ps = pd.to_numeric(out["rt__freecashflowpershare"], errors="coerce")
        upd("rt__pricetofreecashflowratio", _safe_divide(close, fcf_ps), updated)
        upd("km__freecashflowyield", _safe_divide(fcf_ps, close), updated)

    if has("rt__operatingcashflowpershare"):
        ocf_ps = pd.to_numeric(out["rt__operatingcashflowpershare"], errors="coerce")
        upd("rt__pricetooperatingcashflowratio", _safe_divide(close, ocf_ps), updated)

    if has("rt__dividendpershare"):
        div_ps = pd.to_numeric(out["rt__dividendpershare"], errors="coerce")
        div_yld = _safe_divide(div_ps, close)
        upd("rt__dividendyield", div_yld, updated)
        upd("rt__dividendyieldpercentage", 100.0 * div_yld, updated)

    ref_candidates: list[pd.Series] = []
    if has("rt__pricetoearningsratio") and has("rt__netincomepershare"):
        ref_candidates.append(
            pd.to_numeric(base["rt__pricetoearningsratio"], errors="coerce")
            * pd.to_numeric(base["rt__netincomepershare"], errors="coerce")
        )
    if has("rt__pricetobookratio") and has("rt__bookvaluepershare"):
        ref_candidates.append(
            pd.to_numeric(base["rt__pricetobookratio"], errors="coerce")
            * pd.to_numeric(base["rt__bookvaluepershare"], errors="coerce")
        )
    if has("rt__pricetosalesratio") and has("rt__revenuepershare"):
        ref_candidates.append(
            pd.to_numeric(base["rt__pricetosalesratio"], errors="coerce")
            * pd.to_numeric(base["rt__revenuepershare"], errors="coerce")
        )
    if has("rt__pricetofreecashflowratio") and has("rt__freecashflowpershare"):
        ref_candidates.append(
            pd.to_numeric(base["rt__pricetofreecashflowratio"], errors="coerce")
            * pd.to_numeric(base["rt__freecashflowpershare"], errors="coerce")
        )
    if has("rt__pricetooperatingcashflowratio") and has("rt__operatingcashflowpershare"):
        ref_candidates.append(
            pd.to_numeric(base["rt__pricetooperatingcashflowratio"], errors="coerce")
            * pd.to_numeric(base["rt__operatingcashflowpershare"], errors="coerce")
        )
    if has("rt__dividendpershare") and has("rt__dividendyield"):
        ref_candidates.append(
            _safe_divide(
                pd.to_numeric(base["rt__dividendpershare"], errors="coerce"),
                pd.to_numeric(base["rt__dividendyield"], errors="coerce"),
            )
        )

    if ref_candidates:
        ref_px = pd.concat(ref_candidates, axis=1).median(axis=1, skipna=True)
    else:
        ref_px = pd.Series(np.nan, index=out.index, dtype="float64")

    if has("km__marketcap"):
        stale_mcap = pd.to_numeric(base["km__marketcap"], errors="coerce")
        implied_shares = _safe_divide(stale_mcap, ref_px)
        daily_mcap = close * implied_shares
        upd("km__marketcap", daily_mcap, updated)
    else:
        stale_mcap = pd.Series(np.nan, index=out.index, dtype="float64")
        daily_mcap = pd.Series(np.nan, index=out.index, dtype="float64")

    stale_ev = (
        pd.to_numeric(base["km__enterprisevalue"], errors="coerce")
        if has("km__enterprisevalue")
        else pd.Series(np.nan, index=out.index, dtype="float64")
    )
    net_debt_amt = stale_ev - stale_mcap
    daily_ev = daily_mcap + net_debt_amt
    upd("km__enterprisevalue", daily_ev, updated)

    if has("rt__debttomarketcap"):
        debt_amt = pd.to_numeric(base["rt__debttomarketcap"], errors="coerce") * stale_mcap
        upd("rt__debttomarketcap", _safe_divide(debt_amt, daily_mcap), updated)

    ev_ratio = _safe_divide(daily_ev, stale_ev)
    for c in (
        "km__evtosales",
        "km__evtooperatingcashflow",
        "km__evtofreecashflow",
        "km__evtoebitda",
        "rt__enterprisevaluemultiple",
    ):
        if has(c):
            upd(c, pd.to_numeric(base[c], errors="coerce") * ev_ratio, updated)

    px_ratio = _safe_divide(close, ref_px)
    for c in ("rt__pricetoearningsgrowthratio", "rt__forwardpricetoearningsgrowthratio"):
        if has(c):
            upd(c, pd.to_numeric(base[c], errors="coerce") * px_ratio, updated)

    return out, updated


def build_technical_dataframe(
    *,
    ctx,
    symbols: Sequence[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    execution_params: Optional[dict[str, object]] = None,
    debug_data_quality: bool = False,
    data_quality_overrides: Optional[dict[str, object]] = None,
    skip_on_error: bool = True,
    verbose_data: bool = True,
    compact_feature_names: bool = False,
) -> Tuple[pd.DataFrame, list[str]]:
    data_dir = str(getattr(ctx, "data_dir", ".") or ".")

    panel, feature_cols, skipped = build_technical_panel(
        universe=list(symbols),
        api_key=ctx.api_key,
        data_dir=data_dir,
        execution_params=dict(execution_params) if execution_params else None,
        db_name=str(getattr(ctx, "db_name", "quant.db") or "quant.db"),
        sleep_s=ctx.sleep_s,
        skip_on_error=skip_on_error,
        verbose_data=verbose_data,
        debug_data_quality=debug_data_quality,
        data_quality_overrides=data_quality_overrides,
    )

    if skipped:
        print(f"[WARNING] Skipped {len(skipped)} symbols during build.")

    panel = _filter_df_by_date_bounds(panel, start_date=start_date, end_date=end_date)
    panel, feature_cols = _camelize_dataframe_columns(panel, feature_cols)
    if compact_feature_names:
        panel, feature_cols = _compact_dataframe_columns(panel, feature_cols)

    if verbose_data and not panel.empty:
        _summarize_dataset_for_llm(panel, feature_cols)

    return panel, feature_cols


def build_fundamental_dataframe(
    *,
    ctx,
    symbols: Sequence[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "quarter",
    verbose: bool = True,
    use_filing_lag: bool = True,
    filing_lag_days: int = 45,
    compact_feature_names: bool = False,
    target_index: Optional[pd.Index] = None,
    daily_prices: Optional[Union[pd.DataFrame, pd.Series]] = None,
    daily_price_col: str = "Close",
) -> Tuple[pd.DataFrame, list[str]]:
    effective_limit = _resolve_fundamental_limit(
        start_date=start_date,
        end_date=end_date,
        period=period,
    )
    if verbose:
        print(f"[fundamentals] Using computed limit={effective_limit} for date range.")

    df = fetch_fundamentals_data(
        symbols=list(symbols),
        api_key=ctx.api_key,
        period=period,
        limit=effective_limit,
        verbose=verbose,
        use_filing_lag=use_filing_lag,
        filing_lag_days=filing_lag_days,
    )
    src_start, src_end = _extract_date_bounds(df)

    df = _filter_df_by_date_bounds(df, start_date=start_date, end_date=end_date)
    feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if target_index is not None and not df.empty:
        sparse_effective_index = df.index if isinstance(df.index, pd.MultiIndex) else None
        if verbose:
            print("[fundamentals] Broadcasting to daily frequency (smearing forward)...")
        df = broadcast_fundamentals_to_daily(df, target_index)
        df = _filter_df_by_date_bounds(df, start_date=start_date, end_date=end_date)
        if sparse_effective_index is not None:
            df = _add_fundamental_update_timing_features(
                df,
                sparse_effective_index=sparse_effective_index,
            )

        feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        close_daily = _extract_daily_close_series(
            daily_prices,
            target_index=target_index,
            preferred_col=daily_price_col,
        )
        if close_daily is not None:
            df, updated = _recompute_price_linked_fundamentals_daily(df, close_daily)
            if verbose and updated:
                names = ", ".join(sorted(updated))
                print(f"[fundamentals] Recomputed daily price-linked features ({len(updated)}): {names}")
            feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        elif verbose and daily_prices is not None:
            print(
                f"[fundamentals] Skipped daily valuation recompute: could not find "
                f"'{daily_price_col}'/close column on a MultiIndex (date, symbol)."
            )

    df, feature_cols = _camelize_dataframe_columns(df, feature_cols)
    df, feature_cols = _drop_source_duplicate_columns(df, feature_cols)
    if compact_feature_names:
        df, feature_cols = _compact_dataframe_columns(df, feature_cols)
    if src_start is not None and src_end is not None:
        df.attrs["source_start_date"] = src_start
        df.attrs["source_end_date"] = src_end

    if verbose and not df.empty:
        _summarize_dataset_for_llm(df, feature_cols)

    return df, feature_cols


def build_macro_dataframe(
    *,
    ctx,
    start_date: str,
    end_date: str,
    config: Optional[MacroFeatureConfig] = None,
    target_index: Optional[pd.Index] = None,
    verbose: bool = True,
    compact_feature_names: bool = False,
) -> Tuple[pd.DataFrame, list[str]]:
    df = fetch_macro_series(
        api_key=ctx.api_key,
        start_date=start_date,
        end_date=end_date,
        config=config,
        verbose=verbose,
    )
    src_start, src_end = _extract_date_bounds(df)

    if target_index is not None and not df.empty:
        if verbose:
            print("[macro] Broadcasting to daily frequency...")
        df = broadcast_macro_to_daily(df, target_index)

    feature_cols = list(df.columns)
    df, feature_cols = _camelize_dataframe_columns(df, feature_cols)
    if compact_feature_names:
        df, feature_cols = _compact_dataframe_columns(df, feature_cols)
    if src_start is not None and src_end is not None:
        df.attrs["source_start_date"] = src_start
        df.attrs["source_end_date"] = src_end

    if verbose and not df.empty:
        _summarize_dataset_for_llm(df, feature_cols)

    return df, feature_cols


def build_time_dataframe(
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    target_index: Optional[pd.Index] = None,
    config: Optional[TimeFeatureConfig] = None,
    verbose: bool = True,
    compact_feature_names: bool = False,
) -> Tuple[pd.DataFrame, list[str]]:
    df = build_time_features(
        start_date=start_date,
        end_date=end_date,
        target_index=target_index,
        config=config,
    )

    feature_cols = list(df.columns)
    df, feature_cols = _camelize_dataframe_columns(df, feature_cols)
    if compact_feature_names:
        df, feature_cols = _compact_dataframe_columns(df, feature_cols)

    if verbose and not df.empty:
        _summarize_dataset_for_llm(df, feature_cols)

    return df, feature_cols


__all__ = [
    "build_fundamental_dataframe",
    "build_macro_dataframe",
    "build_technical_dataframe",
    "build_time_dataframe",
]
