from __future__ import annotations

import pandas as pd

from fmp.models import Symbol
from features.section_utils import BuiltFeatureSet, add_growth_adjusted_valuation_features, build_passthrough_section_features


def build_cash_flow_growth_features(
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    valuation_frame: pd.DataFrame | None = None,
    filing_lag_days: int = 45,
) -> BuiltFeatureSet:
    built = build_passthrough_section_features(
        symbol_obj,
        target_index,
        section_key="cash_flow_growth",
        prefix="cfg__",
        filing_lag_days=filing_lag_days,
    )
    if built.df.empty:
        return built
    enriched, peg_cols = add_growth_adjusted_valuation_features(
        built.df,
        valuation_frame=valuation_frame,
        specs=(
            (("cfg__operatingcashflowgrowth", "cfg__netcashprovidedbyoperatingactivitiesgrowth"), ("rt__pricetooperatingcashflowratio", "km__evtooperatingcashflow"), "cfg__operatingcashflow_growth_valuation_daily"),
            (("cfg__freecashflowgrowth",), ("rt__pricetofreecashflowratio", "km__evtofreecashflow"), "cfg__freecashflow_growth_valuation_daily"),
            (("cfg__capitalexpendituregrowth", "cfg__capitalexpendituresgrowth"), ("cf__capex_to_mcap_daily",), "cfg__capex_growth_valuation_daily"),
        ),
    )
    return BuiltFeatureSet(df=enriched, feature_cols=[*built.feature_cols, *peg_cols])
