from __future__ import annotations

import numpy as np
import pandas as pd

from fmp.models import Symbol
from features.section_utils import BuiltFeatureSet, days_since_last_event, load_section_payload, safe_ratio, target_dates


def build_insider_trading_features(symbol_obj: Symbol, target_index: pd.MultiIndex) -> BuiltFeatureSet:
    sparse = load_section_payload(symbol_obj, "insider_trading", prefix="insider__", filing_lag_days=0)
    if sparse.empty:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    work = sparse.reset_index().sort_values(["symbol", "date"])
    trade_date = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    price = pd.to_numeric(work.get("insider__price"), errors="coerce")
    shares = pd.to_numeric(work.get("insider__securitiestransacted"), errors="coerce")
    disposition = work.get("insider__acquisitionordisposition")
    sign = disposition.astype(str).str.upper().map({"A": 1.0, "D": -1.0}).fillna(0.0)
    trade_value = sign * price.fillna(0.0) * shares.fillna(0.0)
    event_df = pd.DataFrame({
        "date": trade_date,
        "signed_value": trade_value,
        "buy_count": (sign > 0).astype(float),
        "sell_count": (sign < 0).astype(float),
    }).dropna(subset=["date"])
    event_df = event_df.groupby("date", as_index=False).sum().sort_values("date")
    td = target_dates(target_index)
    daily = pd.DataFrame(index=td)
    daily = daily.join(event_df.set_index("date"), how="left").fillna(0.0)
    daily["own__insider_net_buy_value_90d"] = daily["signed_value"].rolling(90, min_periods=1).sum()
    daily["own__insider_buy_count_90d"] = daily["buy_count"].rolling(90, min_periods=1).sum()
    daily["own__insider_sell_count_90d"] = daily["sell_count"].rolling(90, min_periods=1).sum()
    daily["own__insider_buy_sell_ratio_90d"] = safe_ratio(daily["own__insider_buy_count_90d"], daily["own__insider_sell_count_90d"].replace(0.0, np.nan))
    daily["own__insider_days_since"] = days_since_last_event(td, event_df["date"])
    daily["symbol"] = str(symbol_obj.symbol)
    daily = daily.drop(columns=["signed_value", "buy_count", "sell_count"]).reset_index()
    daily = daily.rename(columns={"index": "date"}).set_index(["date", "symbol"]).sort_index()
    cols = [
        "own__insider_net_buy_value_90d",
        "own__insider_buy_count_90d",
        "own__insider_sell_count_90d",
        "own__insider_buy_sell_ratio_90d",
        "own__insider_days_since",
    ]
    return BuiltFeatureSet(df=daily.replace([np.inf, -np.inf], np.nan), feature_cols=cols)
