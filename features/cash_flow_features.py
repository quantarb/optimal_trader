from __future__ import annotations

import pandas as pd

from fmp.models import Symbol
from features.section_utils import BuiltFeatureSet, add_daily_price_linked_features, build_passthrough_section_features


def build_cash_flow_features(
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    df_prices: pd.DataFrame | None = None,
    market_cap: pd.Series | None = None,
    filing_lag_days: int = 45,
) -> BuiltFeatureSet:
    built = build_passthrough_section_features(
        symbol_obj,
        target_index,
        section_key="cash_flow",
        prefix="cf__",
        filing_lag_days=filing_lag_days,
    )
    if built.df.empty:
        return built
    enriched, linked_cols = add_daily_price_linked_features(
        built.df,
        target_index,
        df_prices=df_prices,
        market_cap=market_cap,
        market_cap_denominated=(
            (("cf__operatingcashflow", "cf__netcashprovidedbyoperatingactivities"), "cf__operatingcashflow_to_mcap_daily"),
            (("cf__freecashflow",), "cf__freecashflow_to_mcap_daily"),
            (("cf__capitalexpenditure", "cf__capitalexpenditures"), "cf__capex_to_mcap_daily"),
        ),
        negate_market_cap_sources=("cf__capex_to_mcap_daily",),
    )
    return BuiltFeatureSet(df=enriched, feature_cols=[*built.feature_cols, *linked_cols])
