from __future__ import annotations

import pandas as pd

from fmp.models import Symbol
from features.section_utils import BuiltFeatureSet, add_daily_price_linked_features, build_passthrough_section_features


def build_cash_flow_features(symbol_obj: Symbol, target_index: pd.MultiIndex, df_prices: pd.DataFrame | None = None, market_cap: pd.Series | None = None, filing_lag_days: int = 45) -> BuiltFeatureSet:
    return _build_cash_flow_features(symbol_obj, target_index, section_key="cash_flow", prefix="cf__", df_prices=df_prices, market_cap=market_cap, filing_lag_days=filing_lag_days)


def build_cash_flow_ttm_features(symbol_obj: Symbol, target_index: pd.MultiIndex, df_prices: pd.DataFrame | None = None, market_cap: pd.Series | None = None, filing_lag_days: int = 45) -> BuiltFeatureSet:
    return _build_cash_flow_features(symbol_obj, target_index, section_key="cash_flow_ttm", prefix="cf_ttm__", df_prices=df_prices, market_cap=market_cap, filing_lag_days=filing_lag_days)


def _build_cash_flow_features(symbol_obj: Symbol, target_index: pd.MultiIndex, *, section_key: str, prefix: str, df_prices: pd.DataFrame | None, market_cap: pd.Series | None, filing_lag_days: int) -> BuiltFeatureSet:
    built = build_passthrough_section_features(symbol_obj, target_index, section_key=section_key, prefix=prefix, filing_lag_days=filing_lag_days)
    if built.df.empty:
        return built
    enriched, linked_cols = add_daily_price_linked_features(
        built.df,
        target_index,
        df_prices=df_prices,
        market_cap=market_cap,
        market_cap_denominated=(
            ((f"{prefix}operatingcashflow", f"{prefix}netcashprovidedbyoperatingactivities"), f"{prefix}operatingcashflow_to_mcap_daily"),
            ((f"{prefix}freecashflow",), f"{prefix}freecashflow_to_mcap_daily"),
            ((f"{prefix}capitalexpenditure", f"{prefix}capitalexpenditures"), f"{prefix}capex_to_mcap_daily"),
        ),
        negate_market_cap_sources=(f"{prefix}capex_to_mcap_daily",),
    )
    return BuiltFeatureSet(df=enriched, feature_cols=[*built.feature_cols, *linked_cols])
