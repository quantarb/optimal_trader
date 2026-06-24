from __future__ import annotations

import pandas as pd

from fmp.models import Symbol
from fmp.positions_summary import load_positions_summary_frame
from quant_warehouse.feature_engineering import (
    build_event_features as qw_build_event_features,
    build_fundamental_change_features as qw_build_fundamental_change_features,
    build_ownership_features as qw_build_ownership_features,
    build_price_ta_classic_feature_families,
    build_price_technical_features as build_price_technical_feature_family,
    build_statement_quality_features as qw_build_statement_quality_features,
)
from features.classification_performance_features import (
    build_industry_performance_features,
    build_sector_performance_features,
)
from features.classification_pe_features import build_industry_pe_features, build_sector_pe_features
from features.section_utils import BuiltFeatureSet, load_section_payload
from features.time_features import build_time_calendar_features


__all__ = [
    "build_event_features",
    "build_fundamental_change_features",
    "build_industry_performance_features",
    "build_industry_pe_features",
    "build_ownership_features",
    "build_price_technical_features",
    "build_sector_performance_features",
    "build_sector_pe_features",
    "build_statement_quality_features",
    "build_ta_classic_technical_features",
    "build_time_calendar_feature_family",
]


def build_price_technical_features(symbol: str, df_prices: pd.DataFrame) -> BuiltFeatureSet:
    return build_price_technical_feature_family(symbol, df_prices)


def build_ta_classic_technical_features(symbol: str, df_prices: pd.DataFrame) -> dict[str, BuiltFeatureSet]:
    return build_price_ta_classic_feature_families(symbol, df_prices)


def build_fundamental_change_features(
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    df_prices: pd.DataFrame | None = None,
    filing_lag_days: int = 45,
) -> BuiltFeatureSet:
    return qw_build_fundamental_change_features(
        symbol_obj,
        target_index,
        df_prices=df_prices,
        filing_lag_days=filing_lag_days,
        sparse_loader=load_section_payload,
    )


def build_statement_quality_features(
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    df_prices: pd.DataFrame | None = None,
    filing_lag_days: int = 45,
) -> BuiltFeatureSet:
    return qw_build_statement_quality_features(
        symbol_obj,
        target_index,
        df_prices=df_prices,
        filing_lag_days=filing_lag_days,
        sparse_loader=load_section_payload,
    )


def build_event_features(
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    df_prices: pd.DataFrame | None = None,
) -> BuiltFeatureSet:
    return qw_build_event_features(
        symbol_obj,
        target_index,
        df_prices=df_prices,
        sparse_loader=load_section_payload,
    )


def build_ownership_features(symbol_obj: Symbol, target_index: pd.MultiIndex) -> BuiltFeatureSet:
    return qw_build_ownership_features(
        symbol_obj,
        target_index,
        sparse_loader=load_section_payload,
        positions_source_loader=load_positions_summary_frame,
    )


def build_time_calendar_feature_family(symbol_obj: Symbol, target_index: pd.MultiIndex) -> BuiltFeatureSet:
    return build_time_calendar_features(symbol_obj, target_index)
