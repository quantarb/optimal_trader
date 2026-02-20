from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeFeatureConfig:
    include_day_of_week_one_hot: bool = True
    include_month_one_hot: bool = True


def _extract_dates(target_index: pd.Index) -> pd.DatetimeIndex:
    if isinstance(target_index, pd.MultiIndex):
        names = target_index.names or []
        if "date" not in names:
            raise ValueError("target_index MultiIndex must contain a 'date' level.")
        dts = pd.to_datetime(target_index.get_level_values("date"), errors="coerce")
        return pd.DatetimeIndex(dts)

    if isinstance(target_index, pd.DatetimeIndex):
        return pd.DatetimeIndex(pd.to_datetime(target_index, errors="coerce"))

    dts = pd.to_datetime(target_index, errors="coerce")
    return pd.DatetimeIndex(dts)


def build_time_features(
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    target_index: Optional[pd.Index] = None,
    config: Optional[TimeFeatureConfig] = None,
) -> pd.DataFrame:
    """
    Build numeric calendar features from daily dates.

    If target_index is provided, output index matches target_index
    (supports DatetimeIndex or MultiIndex with a 'date' level).
    Otherwise, output is a daily DatetimeIndex from start_date..end_date.
    """
    cfg = config or TimeFeatureConfig()

    if target_index is None:
        if start_date is None or end_date is None:
            raise ValueError("Provide both start_date and end_date when target_index is not set.")
        out_index = pd.date_range(start=pd.Timestamp(start_date), end=pd.Timestamp(end_date), freq="D")
        dti = pd.DatetimeIndex(out_index)
    else:
        out_index = target_index
        dti = _extract_dates(target_index)

    if dti.isna().any():
        raise ValueError("Date index contains invalid/NaT values; cannot build time features.")

    out = pd.DataFrame(index=out_index)
    out["day_of_week"] = np.asarray(dti.dayofweek, dtype=np.int8)
    out["day_of_month"] = np.asarray(dti.day, dtype=np.int8)
    out["day_of_year"] = np.asarray(dti.dayofyear, dtype=np.int16)
    out["week_of_year"] = np.asarray(dti.isocalendar().week, dtype=np.int16)
    out["month"] = np.asarray(dti.month, dtype=np.int8)
    out["quarter"] = np.asarray(dti.quarter, dtype=np.int8)
    out["year"] = np.asarray(dti.year, dtype=np.int16)

    if cfg.include_day_of_week_one_hot:
        names = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
        for i, name in enumerate(names):
            out[f"is_{name}"] = (out["day_of_week"] == i).astype(np.int8)

    if cfg.include_month_one_hot:
        for m in range(1, 13):
            out[f"is_month_{m}"] = (out["month"] == m).astype(np.int8)

    return out
