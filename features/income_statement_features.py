from __future__ import annotations

import pandas as pd

from fmp.models import Symbol
from features.section_utils import BuiltFeatureSet, add_daily_price_linked_features, build_passthrough_section_features


def build_income_statement_features(
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    df_prices: pd.DataFrame | None = None,
    filing_lag_days: int = 45,
) -> BuiltFeatureSet:
    built = build_passthrough_section_features(
        symbol_obj,
        target_index,
        section_key="income_statement",
        prefix="is__",
        filing_lag_days=filing_lag_days,
    )
    if built.df.empty:
        return built
    enriched, linked_cols = add_daily_price_linked_features(
        built.df,
        target_index,
        df_prices=df_prices,
        share_count_candidates=("is__weightedaverageshsoutdil", "is__weightedaverageshsout"),
        price_denominated=(
            (("is__eps", "is__epsdiluted"), "is__eps_to_price_daily"),
        ),
        market_cap_denominated=(
            (("is__revenue",), "is__revenue_to_mcap_daily"),
            (("is__grossprofit",), "is__grossprofit_to_mcap_daily"),
            (("is__ebitda",), "is__ebitda_to_mcap_daily"),
            (("is__operatingincome",), "is__operatingincome_to_mcap_daily"),
            (("is__netincome",), "is__netincome_to_mcap_daily"),
        ),
    )
    return BuiltFeatureSet(df=enriched, feature_cols=[*built.feature_cols, *linked_cols])
