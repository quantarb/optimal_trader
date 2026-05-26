from __future__ import annotations

import pandas as pd

from fmp.models import Symbol
from features.section_utils import BuiltFeatureSet, add_growth_adjusted_valuation_features, build_passthrough_section_features


def build_income_statement_growth_features(
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    valuation_frame: pd.DataFrame | None = None,
    filing_lag_days: int = 45,
) -> BuiltFeatureSet:
    built = build_passthrough_section_features(
        symbol_obj,
        target_index,
        section_key="income_statement_growth",
        prefix="isg__",
        filing_lag_days=filing_lag_days,
    )
    if built.df.empty:
        return built
    enriched, peg_cols = add_growth_adjusted_valuation_features(
        built.df,
        valuation_frame=valuation_frame,
        specs=(
            (("isg__epsgrowth", "isg__epsdilutedgrowth", "isg__netincomegrowth"), ("rt__pricetoearningsratio",), "isg__earnings_peg_daily"),
            (("isg__revenuegrowth",), ("rt__pricetosalesratio", "km__evtosales"), "isg__sales_growth_valuation_daily"),
            (("isg__grossprofitgrowth",), ("is__grossprofit_to_mcap_daily",), "isg__grossprofit_growth_valuation_daily"),
            (("isg__ebitdagrowth",), ("km__evtoebitda", "is__ebitda_to_mcap_daily"), "isg__ebitda_growth_valuation_daily"),
            (("isg__operatingincomegrowth",), ("is__operatingincome_to_mcap_daily",), "isg__operatingincome_growth_valuation_daily"),
        ),
    )
    return BuiltFeatureSet(df=enriched, feature_cols=[*built.feature_cols, *peg_cols])
