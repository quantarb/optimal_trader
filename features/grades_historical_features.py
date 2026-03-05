from __future__ import annotations

import numpy as np
import pandas as pd

from fmp.models import Symbol
from features.section_utils import BuiltFeatureSet, broadcast_sparse, days_since_for_target, days_since_last_event, load_section_payload, safe_ratio, target_dates


GRADE_COLS = [
    "grade__analystratingsstrongbuy",
    "grade__analystratingsbuy",
    "grade__analystratingshold",
    "grade__analystratingssell",
    "grade__analystratingsstrongsell",
]


def build_grades_historical_features(symbol_obj: Symbol, target_index: pd.MultiIndex) -> BuiltFeatureSet:
    sparse = load_section_payload(symbol_obj, "grades_historical", prefix="grade__", filing_lag_days=0)
    if sparse.empty:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    work = sparse.reset_index().sort_values(["symbol", "date"])
    for col in GRADE_COLS:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    total = sum((work[col].fillna(0.0) for col in GRADE_COLS))
    bullish = work["grade__analystratingsstrongbuy"].fillna(0.0) + work["grade__analystratingsbuy"].fillna(0.0)
    bearish = work["grade__analystratingssell"].fillna(0.0) + work["grade__analystratingsstrongsell"].fillna(0.0)
    out = work[["date", "symbol"]].copy()
    out["evt__grade_bullish_ratio"] = safe_ratio(bullish, total.replace(0.0, np.nan))
    out["evt__grade_bearish_ratio"] = safe_ratio(bearish, total.replace(0.0, np.nan))
    out["evt__grade_net_bullish"] = safe_ratio(bullish - bearish, total.replace(0.0, np.nan))
    daily = broadcast_sparse(out.set_index(["date", "symbol"]).sort_index(), target_index)
    daily["evt__grade_days_since"] = days_since_for_target(target_index, days_since_last_event(target_dates(target_index), work["date"]))
    cols = ["evt__grade_bullish_ratio", "evt__grade_bearish_ratio", "evt__grade_net_bullish", "evt__grade_days_since"]
    return BuiltFeatureSet(df=daily, feature_cols=cols)
