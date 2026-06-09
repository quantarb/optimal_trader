from __future__ import annotations

import pandas as pd

from fmp.models import Symbol
from features.section_utils import BuiltFeatureSet, add_daily_price_linked_features, build_passthrough_section_features


def build_balance_sheet_features(symbol_obj: Symbol, target_index: pd.MultiIndex, df_prices: pd.DataFrame | None = None, market_cap: pd.Series | None = None, filing_lag_days: int = 45) -> BuiltFeatureSet:
    return _build_balance_sheet_features(symbol_obj, target_index, section_key="balance_sheet", prefix="bs__", df_prices=df_prices, market_cap=market_cap, filing_lag_days=filing_lag_days)


def build_balance_sheet_ttm_features(symbol_obj: Symbol, target_index: pd.MultiIndex, df_prices: pd.DataFrame | None = None, market_cap: pd.Series | None = None, filing_lag_days: int = 45) -> BuiltFeatureSet:
    return _build_balance_sheet_features(symbol_obj, target_index, section_key="balance_sheet_ttm", prefix="bs_ttm__", df_prices=df_prices, market_cap=market_cap, filing_lag_days=filing_lag_days)


def _build_balance_sheet_features(symbol_obj: Symbol, target_index: pd.MultiIndex, *, section_key: str, prefix: str, df_prices: pd.DataFrame | None, market_cap: pd.Series | None, filing_lag_days: int) -> BuiltFeatureSet:
    built = build_passthrough_section_features(symbol_obj, target_index, section_key=section_key, prefix=prefix, filing_lag_days=filing_lag_days)
    if built.df.empty:
        return built
    enriched, linked_cols = add_daily_price_linked_features(
        built.df,
        target_index,
        df_prices=df_prices,
        market_cap=market_cap,
        market_cap_denominated=(
            ((f"{prefix}cashandcashequivalents", f"{prefix}cashandshortterminvestments"), f"{prefix}cash_to_mcap_daily"),
            ((f"{prefix}totaldebt", f"{prefix}shorttermdebt", f"{prefix}longtermdebt"), f"{prefix}debt_to_mcap_daily"),
            ((f"{prefix}netdebt",), f"{prefix}netdebt_to_mcap_daily"),
            ((f"{prefix}totalstockholdersequity", f"{prefix}totalequity"), f"{prefix}equity_to_mcap_daily"),
            ((f"{prefix}totalassets",), f"{prefix}assets_to_mcap_daily"),
        ),
    )
    return BuiltFeatureSet(df=enriched, feature_cols=[*built.feature_cols, *linked_cols])
