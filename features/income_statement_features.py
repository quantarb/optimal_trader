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
    return _build_income_statement_features(
        symbol_obj,
        target_index,
        section_key="income_statement",
        prefix="is__",
        df_prices=df_prices,
        filing_lag_days=filing_lag_days,
    )


def build_income_statement_ttm_features(
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    df_prices: pd.DataFrame | None = None,
    filing_lag_days: int = 45,
) -> BuiltFeatureSet:
    return _build_income_statement_features(
        symbol_obj,
        target_index,
        section_key="income_statement_ttm",
        prefix="is_ttm__",
        df_prices=df_prices,
        filing_lag_days=filing_lag_days,
    )


def _build_income_statement_features(
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    *,
    section_key: str,
    prefix: str,
    df_prices: pd.DataFrame | None,
    filing_lag_days: int,
) -> BuiltFeatureSet:
    built = build_passthrough_section_features(
        symbol_obj,
        target_index,
        section_key=section_key,
        prefix=prefix,
        filing_lag_days=filing_lag_days,
    )
    if built.df.empty:
        return built
    enriched, linked_cols = add_daily_price_linked_features(
        built.df,
        target_index,
        df_prices=df_prices,
        share_count_candidates=(f"{prefix}weightedaverageshsoutdil", f"{prefix}weightedaverageshsout"),
        price_denominated=(
            ((f"{prefix}eps", f"{prefix}epsdiluted"), f"{prefix}eps_to_price_daily"),
        ),
        market_cap_denominated=(
            ((f"{prefix}revenue",), f"{prefix}revenue_to_mcap_daily"),
            ((f"{prefix}grossprofit",), f"{prefix}grossprofit_to_mcap_daily"),
            ((f"{prefix}ebitda",), f"{prefix}ebitda_to_mcap_daily"),
            ((f"{prefix}operatingincome",), f"{prefix}operatingincome_to_mcap_daily"),
            ((f"{prefix}netincome",), f"{prefix}netincome_to_mcap_daily"),
        ),
    )
    return BuiltFeatureSet(df=enriched, feature_cols=[*built.feature_cols, *linked_cols])
