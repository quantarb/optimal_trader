from __future__ import annotations

import pandas as pd

from fmp.models import Symbol
from features.section_utils import BuiltFeatureSet, add_growth_adjusted_valuation_features, build_passthrough_section_features


def build_financial_growth_features(
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    valuation_frame: pd.DataFrame | None = None,
    filing_lag_days: int = 45,
) -> BuiltFeatureSet:
    built = build_passthrough_section_features(
        symbol_obj,
        target_index,
        section_key="financial_growth",
        prefix="fg__",
        filing_lag_days=filing_lag_days,
    )
    if built.df.empty:
        return built
    enriched, peg_cols = add_growth_adjusted_valuation_features(
        built.df,
        valuation_frame=valuation_frame,
        specs=(
            (("fg__epsgrowth", "fg__epsdilutedgrowth", "fg__netincomegrowth"), ("rt__pricetoearningsratio",), "fg__earnings_peg_daily"),
            (("fg__revenuegrowth",), ("rt__pricetosalesratio", "km__evtosales"), "fg__sales_growth_valuation_daily"),
            (("fg__ebitdagrowth",), ("km__evtoebitda", "is__ebitda_to_mcap_daily"), "fg__ebitda_growth_valuation_daily"),
            (("fg__freecashflowgrowth",), ("rt__pricetofreecashflowratio", "km__evtofreecashflow"), "fg__freecashflow_growth_valuation_daily"),
            (("fg__operatingcashflowgrowth",), ("rt__pricetooperatingcashflowratio", "km__evtooperatingcashflow"), "fg__operatingcashflow_growth_valuation_daily"),
        ),
    )
    return BuiltFeatureSet(df=enriched, feature_cols=[*built.feature_cols, *peg_cols])
