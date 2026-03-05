from __future__ import annotations

import pandas as pd

from fmp.models import Symbol
from features.analyst_estimates_features import build_analyst_estimates_features
from features.balance_sheet_features import build_balance_sheet_features
from features.balance_sheet_growth_features import build_balance_sheet_growth_features
from features.cash_flow_features import build_cash_flow_features
from features.cash_flow_growth_features import build_cash_flow_growth_features
from features.earnings_features import build_earnings_features
from features.financial_growth_features import build_financial_growth_features
from features.grades_historical_features import build_grades_historical_features
from features.income_statement_features import build_income_statement_features
from features.income_statement_growth_features import build_income_statement_growth_features
from features.insider_trading_features import build_insider_trading_features
from features.key_metrics_features import build_key_metrics_features
from features.prices_div_adj_features import build_prices_div_adj_features
from features.ratios_features import build_ratios_features
from features.ratings_historical_features import build_ratings_historical_features
from features.section_utils import BuiltFeatureSet


def build_price_technical_features(symbol: str, df_prices: pd.DataFrame) -> BuiltFeatureSet:
    return build_prices_div_adj_features(symbol, df_prices)


def build_fundamental_change_features(
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    df_prices: pd.DataFrame | None = None,
    filing_lag_days: int = 45,
) -> BuiltFeatureSet:
    parts = [
        build_key_metrics_features(symbol_obj, target_index, df_prices=df_prices, filing_lag_days=filing_lag_days),
        build_ratios_features(symbol_obj, target_index, df_prices=df_prices, filing_lag_days=filing_lag_days),
    ]
    return _merge_feature_sets(parts, target_index)


def build_statement_quality_features(
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    filing_lag_days: int = 45,
) -> BuiltFeatureSet:
    parts = [
        build_income_statement_features(symbol_obj, target_index, filing_lag_days=filing_lag_days),
        build_income_statement_growth_features(symbol_obj, target_index, filing_lag_days=filing_lag_days),
        build_cash_flow_features(symbol_obj, target_index, filing_lag_days=filing_lag_days),
        build_cash_flow_growth_features(symbol_obj, target_index, filing_lag_days=filing_lag_days),
        build_balance_sheet_features(symbol_obj, target_index, filing_lag_days=filing_lag_days),
        build_balance_sheet_growth_features(symbol_obj, target_index, filing_lag_days=filing_lag_days),
        build_financial_growth_features(symbol_obj, target_index, filing_lag_days=filing_lag_days),
    ]
    return _merge_feature_sets(parts, target_index)


def build_event_features(symbol_obj: Symbol, target_index: pd.MultiIndex) -> BuiltFeatureSet:
    parts = [
        build_earnings_features(symbol_obj, target_index),
        build_analyst_estimates_features(symbol_obj, target_index),
        build_ratings_historical_features(symbol_obj, target_index),
        build_grades_historical_features(symbol_obj, target_index),
    ]
    return _merge_feature_sets(parts, target_index)


def build_ownership_features(symbol_obj: Symbol, target_index: pd.MultiIndex) -> BuiltFeatureSet:
    parts = [
        build_insider_trading_features(symbol_obj, target_index),
    ]
    return _merge_feature_sets(parts, target_index)


def _merge_feature_sets(parts: list[BuiltFeatureSet], target_index: pd.MultiIndex) -> BuiltFeatureSet:
    merged = pd.DataFrame(index=target_index)
    feature_cols: list[str] = []
    for part in parts:
        if not part.df.empty and part.feature_cols:
            merged = merged.join(part.df[part.feature_cols], how="left")
            feature_cols.extend(part.feature_cols)
    feature_cols = list(dict.fromkeys(feature_cols))
    return BuiltFeatureSet(df=merged, feature_cols=feature_cols)
