from __future__ import annotations

import pandas as pd

from fmp.models import Symbol
from features.section_utils import BuiltFeatureSet, add_daily_price_linked_features, build_passthrough_section_features


def build_balance_sheet_features(
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    df_prices: pd.DataFrame | None = None,
    market_cap: pd.Series | None = None,
    filing_lag_days: int = 45,
) -> BuiltFeatureSet:
    built = build_passthrough_section_features(
        symbol_obj,
        target_index,
        section_key="balance_sheet",
        prefix="bs__",
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
            (("bs__cashandcashequivalents", "bs__cashandshortterminvestments"), "bs__cash_to_mcap_daily"),
            (("bs__totaldebt", "bs__shorttermdebt", "bs__longtermdebt"), "bs__debt_to_mcap_daily"),
            (("bs__netdebt",), "bs__netdebt_to_mcap_daily"),
            (("bs__totalstockholdersequity", "bs__totalequity"), "bs__equity_to_mcap_daily"),
            (("bs__totalassets",), "bs__assets_to_mcap_daily"),
        ),
    )
    return BuiltFeatureSet(df=enriched, feature_cols=[*built.feature_cols, *linked_cols])
