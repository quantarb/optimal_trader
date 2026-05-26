from __future__ import annotations

import pandas as pd

from fmp.models import Symbol
from features.section_utils import BuiltFeatureSet, add_growth_adjusted_valuation_features, build_passthrough_section_features


def build_balance_sheet_growth_features(
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    valuation_frame: pd.DataFrame | None = None,
    filing_lag_days: int = 45,
) -> BuiltFeatureSet:
    built = build_passthrough_section_features(
        symbol_obj,
        target_index,
        section_key="balance_sheet_growth",
        prefix="bsg__",
        filing_lag_days=filing_lag_days,
    )
    if built.df.empty:
        return built
    enriched, peg_cols = add_growth_adjusted_valuation_features(
        built.df,
        valuation_frame=valuation_frame,
        specs=(
            (("bsg__totalstockholdersequitygrowth", "bsg__totalequitygrowth", "bsg__bookvaluepersharegrowth"), ("rt__pricetobookratio", "bs__equity_to_mcap_daily"), "bsg__book_growth_valuation_daily"),
            (("bsg__totalassetsgrowth",), ("bs__assets_to_mcap_daily",), "bsg__assets_growth_valuation_daily"),
            (("bsg__cashandcashequivalentsgrowth", "bsg__cashandshortterminvestmentsgrowth"), ("bs__cash_to_mcap_daily",), "bsg__cash_growth_valuation_daily"),
            (("bsg__totaldebtgrowth", "bsg__netdebtgrowth"), ("bs__debt_to_mcap_daily", "bs__netdebt_to_mcap_daily"), "bsg__debt_growth_valuation_daily"),
        ),
    )
    return BuiltFeatureSet(df=enriched, feature_cols=[*built.feature_cols, *peg_cols])
