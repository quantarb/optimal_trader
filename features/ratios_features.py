from __future__ import annotations

import pandas as pd

from fmp.models import Symbol
from features.section_utils import BuiltFeatureSet, broadcast_sparse, load_section_payload, safe_ratio


def build_ratios_features(
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    df_prices: pd.DataFrame | None = None,
    filing_lag_days: int = 45,
) -> BuiltFeatureSet:
    sparse = load_section_payload(symbol_obj, "ratios", prefix="rt__", keep_fields=None, filing_lag_days=filing_lag_days)
    if sparse.empty:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    work = sparse.reset_index().sort_values(["symbol", "date"])
    value_cols = [c for c in sparse.columns if c.startswith("rt__") and pd.api.types.is_numeric_dtype(sparse[c])]
    if not value_cols:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    daily = broadcast_sparse(sparse[value_cols].sort_index(), target_index)

    daily_price = _daily_price_series(df_prices, target_index)
    if daily_price is not None:
        price_on_sparse = _price_on_sparse_dates(df_prices, work["date"]) if df_prices is not None and not df_prices.empty else None
        if price_on_sparse is not None:
            eps_per_share = _broadcast_inferred_per_share(work, price_on_sparse, "rt__pricetoearningsratio", target_index)
            book_value_per_share = _broadcast_inferred_per_share(work, price_on_sparse, "rt__pricetobookratio", target_index)
            revenue_per_share = _broadcast_inferred_per_share(work, price_on_sparse, "rt__pricetosalesratio", target_index)
            free_cash_flow_per_share = _broadcast_inferred_per_share(work, price_on_sparse, "rt__pricetofreecashflowratio", target_index)
            operating_cash_flow_per_share = _broadcast_inferred_per_share(work, price_on_sparse, "rt__pricetooperatingcashflowratio", target_index)

            if eps_per_share is not None:
                daily["rt__pricetoearningsratio"] = safe_ratio(daily_price, eps_per_share)
            if book_value_per_share is not None:
                daily["rt__pricetobookratio"] = safe_ratio(daily_price, book_value_per_share)
            if revenue_per_share is not None:
                daily["rt__pricetosalesratio"] = safe_ratio(daily_price, revenue_per_share)
            if free_cash_flow_per_share is not None:
                daily["rt__pricetofreecashflowratio"] = safe_ratio(daily_price, free_cash_flow_per_share)
            if operating_cash_flow_per_share is not None:
                daily["rt__pricetooperatingcashflowratio"] = safe_ratio(daily_price, operating_cash_flow_per_share)

        dividend_per_share = _broadcast_existing_series(work, "rt__dividendpershare", target_index)
        if dividend_per_share is not None:
            dividend_yield = safe_ratio(dividend_per_share, daily_price)
            daily["rt__dividendyield"] = dividend_yield
            daily["rt__dividendyieldpercentage"] = dividend_yield * 100.0

    return BuiltFeatureSet(df=daily, feature_cols=[c for c in daily.columns if c.startswith("rt__")])


def _daily_price_series(df_prices: pd.DataFrame | None, target_index: pd.MultiIndex) -> pd.Series | None:
    if df_prices is None or df_prices.empty or "close" not in df_prices.columns:
        return None
    target_dates = pd.DatetimeIndex(pd.to_datetime(target_index.get_level_values("date"))).normalize()
    close = pd.to_numeric(df_prices["close"], errors="coerce").sort_index()
    aligned = close.reindex(target_dates, method="ffill")
    return pd.Series(aligned.to_numpy(), index=target_index)


def _price_on_sparse_dates(df_prices: pd.DataFrame, sparse_dates: pd.Series) -> pd.Series | None:
    if df_prices is None or df_prices.empty or "close" not in df_prices.columns:
        return None
    close = pd.to_numeric(df_prices["close"], errors="coerce").sort_index()
    dates = pd.DatetimeIndex(pd.to_datetime(sparse_dates)).normalize()
    return pd.Series(close.reindex(dates, method="ffill").to_numpy(), index=sparse_dates.index)


def _broadcast_inferred_per_share(
    work: pd.DataFrame,
    price_on_sparse: pd.Series,
    ratio_col: str,
    target_index: pd.MultiIndex,
) -> pd.Series | None:
    if ratio_col not in work.columns:
        return None
    ratio = pd.to_numeric(work[ratio_col], errors="coerce")
    inferred = safe_ratio(price_on_sparse, ratio)
    sparse = pd.DataFrame({"date": work["date"], "symbol": work["symbol"], "value": inferred}).dropna(subset=["value"])
    if sparse.empty:
        return None
    daily = broadcast_sparse(sparse.set_index(["date", "symbol"]).sort_index(), target_index)
    return pd.to_numeric(daily.get("value"), errors="coerce")


def _broadcast_existing_series(work: pd.DataFrame, value_col: str, target_index: pd.MultiIndex) -> pd.Series | None:
    if value_col not in work.columns:
        return None
    sparse = pd.DataFrame({"date": work["date"], "symbol": work["symbol"], "value": pd.to_numeric(work[value_col], errors="coerce")}).dropna(subset=["value"])
    if sparse.empty:
        return None
    daily = broadcast_sparse(sparse.set_index(["date", "symbol"]).sort_index(), target_index)
    return pd.to_numeric(daily.get("value"), errors="coerce")
