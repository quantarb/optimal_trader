from __future__ import annotations

import numpy as np
import pandas as pd

from features.section_utils import BuiltFeatureSet
from fmp.classification_pe import (
    INDUSTRY_PE_CATEGORY,
    SECTOR_PE_CATEGORY,
    classification_pe_series_code,
)
from fmp.models import MacroObservation, Symbol


def _empty(target_index: pd.MultiIndex) -> BuiltFeatureSet:
    return BuiltFeatureSet(pd.DataFrame(index=target_index), [])


def build_classification_pe_features(
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    *,
    category: str,
) -> BuiltFeatureSet:
    classification = str(symbol_obj.sector if category == SECTOR_PE_CATEGORY else symbol_obj.industry).strip()
    exchange = str(symbol_obj.exchange or "").strip().upper()
    if not classification or not exchange or target_index.empty:
        return _empty(target_index)
    dates = pd.DatetimeIndex(pd.to_datetime(target_index.get_level_values("date")))
    code = classification_pe_series_code(category, classification, exchange)
    values = list(
        MacroObservation.objects.filter(
            series__code=code,
            observation_date__gte=dates.min().date(),
            observation_date__lte=dates.max().date(),
        ).order_by("observation_date").values_list("observation_date", "value")
    )
    if not values:
        return _empty(target_index)
    pe = pd.Series(
        [float(value) for _, value in values],
        index=pd.DatetimeIndex([day for day, _ in values]),
        dtype="float64",
    ).reindex(dates)
    prefix = "sector_pe__" if category == SECTOR_PE_CATEGORY else "industry_pe__"
    frame = pd.DataFrame(index=target_index)
    frame[f"{prefix}level"] = pe.to_numpy()
    frame[f"{prefix}change_1d"] = pe.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).to_numpy()
    frame[f"{prefix}change_21d"] = pe.pct_change(21, fill_method=None).replace([np.inf, -np.inf], np.nan).to_numpy()
    median_63 = pe.rolling(63, min_periods=21).median()
    frame[f"{prefix}vs_median_63d"] = (pe / median_63 - 1.0).replace([np.inf, -np.inf], np.nan).to_numpy()
    for window, min_periods in ((63, 21), (252, 63)):
        mean = pe.rolling(window, min_periods=min_periods).mean()
        std = pe.rolling(window, min_periods=min_periods).std().replace(0.0, np.nan)
        frame[f"{prefix}zscore_{window}d"] = ((pe - mean) / std).to_numpy()
    return BuiltFeatureSet(frame, list(frame.columns))


def build_sector_pe_features(symbol_obj: Symbol, target_index: pd.MultiIndex) -> BuiltFeatureSet:
    return build_classification_pe_features(symbol_obj, target_index, category=SECTOR_PE_CATEGORY)


def build_industry_pe_features(symbol_obj: Symbol, target_index: pd.MultiIndex) -> BuiltFeatureSet:
    return build_classification_pe_features(symbol_obj, target_index, category=INDUSTRY_PE_CATEGORY)
