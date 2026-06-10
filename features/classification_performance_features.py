from __future__ import annotations

import numpy as np
import pandas as pd

from fmp.classification_performance import (
    INDUSTRY_PERFORMANCE_CATEGORY,
    SECTOR_PERFORMANCE_CATEGORY,
    classification_performance_series_code,
)
from fmp.models import MacroObservation, Symbol
from features.section_utils import BuiltFeatureSet, daily_price_series


def _empty(target_index: pd.MultiIndex) -> BuiltFeatureSet:
    return BuiltFeatureSet(pd.DataFrame(index=target_index), [])


def _classification_name(symbol_obj: Symbol, category: str) -> str:
    if category == SECTOR_PERFORMANCE_CATEGORY:
        return str(symbol_obj.sector or "").strip()
    return str(symbol_obj.industry or "").strip()


def build_classification_performance_features(
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    *,
    category: str,
    df_prices: pd.DataFrame | None = None,
) -> BuiltFeatureSet:
    classification = _classification_name(symbol_obj, category)
    exchange = str(symbol_obj.exchange or "").strip().upper()
    if not classification or not exchange or target_index.empty:
        return _empty(target_index)
    code = classification_performance_series_code(category, classification, exchange)
    date_index = pd.DatetimeIndex(pd.to_datetime(target_index.get_level_values("date")))
    observations = MacroObservation.objects.filter(
        series__code=code,
        observation_date__gte=date_index.min().date(),
        observation_date__lte=date_index.max().date(),
    ).order_by("observation_date").values_list("observation_date", "value")
    values = list(observations)
    if not values:
        return _empty(target_index)

    prefix = "sector_perf__" if category == SECTOR_PERFORMANCE_CATEGORY else "industry_perf__"
    daily_return = pd.Series(
        [float(value) for _, value in values],
        index=pd.DatetimeIndex([day for day, _ in values]),
        dtype="float64",
    ).reindex(date_index)
    frame = pd.DataFrame(index=target_index)
    frame[f"{prefix}return_1d"] = daily_return.to_numpy()
    for window in (5, 21):
        compounded = (1.0 + daily_return).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1.0
        frame[f"{prefix}return_{window}d"] = compounded.to_numpy()
    frame[f"{prefix}volatility_21d"] = daily_return.rolling(21, min_periods=5).std().to_numpy()

    stock_return = daily_price_series(df_prices, target_index)
    if stock_return is not None:
        stock_return = stock_return.pct_change()
        frame[f"{prefix}stock_excess_1d"] = (stock_return - frame[f"{prefix}return_1d"]).to_numpy()
        stock_21d = (1.0 + stock_return).rolling(21, min_periods=21).apply(np.prod, raw=True) - 1.0
        frame[f"{prefix}stock_excess_21d"] = (stock_21d - frame[f"{prefix}return_21d"]).to_numpy()
    return BuiltFeatureSet(frame, list(frame.columns))


def build_sector_performance_features(
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    df_prices: pd.DataFrame | None = None,
) -> BuiltFeatureSet:
    return build_classification_performance_features(
        symbol_obj,
        target_index,
        category=SECTOR_PERFORMANCE_CATEGORY,
        df_prices=df_prices,
    )


def build_industry_performance_features(
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    df_prices: pd.DataFrame | None = None,
) -> BuiltFeatureSet:
    return build_classification_performance_features(
        symbol_obj,
        target_index,
        category=INDUSTRY_PERFORMANCE_CATEGORY,
        df_prices=df_prices,
    )
