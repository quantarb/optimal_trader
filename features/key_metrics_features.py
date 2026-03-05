from __future__ import annotations

import pandas as pd

from fmp.models import Symbol
from features.section_utils import BuiltFeatureSet, broadcast_sparse, load_section_payload, safe_ratio


def build_key_metrics_features(
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    df_prices: pd.DataFrame | None = None,
    filing_lag_days: int = 45,
) -> BuiltFeatureSet:
    sparse = load_section_payload(symbol_obj, "key_metrics", prefix="km__", keep_fields=None, filing_lag_days=filing_lag_days)
    if sparse.empty:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    work = sparse.reset_index().sort_values(["symbol", "date"])
    value_cols = [c for c in sparse.columns if c.startswith("km__") and pd.api.types.is_numeric_dtype(sparse[c])]
    if not value_cols:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    daily = broadcast_sparse(sparse[value_cols].sort_index(), target_index)

    daily_market_cap = _infer_daily_market_cap_from_sparse(work, target_index, df_prices)
    if daily_market_cap is not None:
        daily["km__marketcap"] = daily_market_cap

    if daily_market_cap is not None:
        free_cf_yield_base = _broadcast_inferred_yield_base(work, "km__marketcap", "km__freecashflowyield", target_index)
        if free_cf_yield_base is not None:
            daily["km__freecashflowyield"] = safe_ratio(free_cf_yield_base, daily_market_cap)

    return BuiltFeatureSet(df=daily, feature_cols=[c for c in daily.columns if c.startswith("km__")])


def _broadcast_inferred_denominator(
    work: pd.DataFrame,
    numerator_col: str,
    ratio_col: str,
    target_index: pd.MultiIndex,
) -> pd.Series | None:
    if numerator_col not in work.columns or ratio_col not in work.columns:
        return None
    inferred = safe_ratio(pd.to_numeric(work[numerator_col], errors="coerce"), pd.to_numeric(work[ratio_col], errors="coerce"))
    sparse = pd.DataFrame({"date": work["date"], "symbol": work["symbol"], "value": inferred}).dropna(subset=["value"])
    if sparse.empty:
        return None
    daily = broadcast_sparse(sparse.set_index(["date", "symbol"]).sort_index(), target_index)
    return pd.to_numeric(daily.get("value"), errors="coerce")


def _broadcast_inferred_yield_base(
    work: pd.DataFrame,
    market_cap_col: str,
    yield_col: str,
    target_index: pd.MultiIndex,
) -> pd.Series | None:
    if market_cap_col not in work.columns or yield_col not in work.columns:
        return None
    market_cap = pd.to_numeric(work[market_cap_col], errors="coerce")
    yield_value = pd.to_numeric(work[yield_col], errors="coerce")
    inferred = market_cap * yield_value
    sparse = pd.DataFrame({"date": work["date"], "symbol": work["symbol"], "value": inferred}).dropna(subset=["value"])
    if sparse.empty:
        return None
    daily = broadcast_sparse(sparse.set_index(["date", "symbol"]).sort_index(), target_index)
    return pd.to_numeric(daily.get("value"), errors="coerce")


def _infer_daily_market_cap_from_sparse(
    work: pd.DataFrame,
    target_index: pd.MultiIndex,
    df_prices: pd.DataFrame | None,
) -> pd.Series | None:
    if df_prices is None or df_prices.empty or "close" not in df_prices.columns:
        return None
    shares_col = None
    for candidate in ("km__sharesoutstanding", "km__weightedaverageshsout", "km__weightedaverageshsoutdil"):
        if candidate in work.columns:
            shares_col = candidate
            break
    if shares_col is None:
        return None
    sparse = pd.DataFrame(
        {
            "date": work["date"],
            "symbol": work["symbol"],
            "shares": pd.to_numeric(work[shares_col], errors="coerce"),
        }
    ).dropna(subset=["shares"])
    if sparse.empty:
        return None
    daily_shares = broadcast_sparse(sparse.set_index(["date", "symbol"]).sort_index(), target_index)
    shares = pd.to_numeric(daily_shares.get("shares"), errors="coerce")
    if shares is None:
        return None
    target_dates = pd.DatetimeIndex(pd.to_datetime(target_index.get_level_values("date"))).normalize()
    close = pd.to_numeric(df_prices["close"], errors="coerce").sort_index().reindex(target_dates, method="ffill")
    daily_price = pd.Series(close.to_numpy(), index=target_index)
    return shares * daily_price
