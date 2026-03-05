from __future__ import annotations

import numpy as np
import pandas as pd

from fmp.models import Symbol
from features.section_utils import BuiltFeatureSet, broadcast_sparse, days_since_for_target, days_since_last_event, load_section_payload, safe_ratio, target_dates


def build_earnings_features(symbol_obj: Symbol, target_index: pd.MultiIndex) -> BuiltFeatureSet:
    sparse = load_section_payload(symbol_obj, "earnings", prefix="earn__", filing_lag_days=0)
    if sparse.empty:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    work = sparse.reset_index().sort_values(["symbol", "date"])
    eps_actual = pd.to_numeric(work.get("earn__epsactual"), errors="coerce")
    eps_estimated = pd.to_numeric(work.get("earn__epsestimated"), errors="coerce")
    rev_actual = pd.to_numeric(work.get("earn__revenueactual"), errors="coerce")
    rev_estimated = pd.to_numeric(work.get("earn__revenueestimated"), errors="coerce")
    out = work[["date", "symbol"]].copy()
    out["evt__earn_eps_surprise"] = safe_ratio(eps_actual - eps_estimated, eps_estimated.abs())
    out["evt__earn_rev_surprise"] = safe_ratio(rev_actual - rev_estimated, rev_estimated.abs())
    out["evt__earn_beat_flag"] = ((eps_actual >= eps_estimated) & eps_actual.notna() & eps_estimated.notna()).astype(float)
    out["evt__earn_beat_streak_4"] = out.groupby("symbol")["evt__earn_beat_flag"].transform(lambda s: s.rolling(4, min_periods=1).sum())
    daily = broadcast_sparse(out.set_index(["date", "symbol"]).sort_index(), target_index)
    daily["evt__earn_days_since"] = days_since_for_target(target_index, days_since_last_event(target_dates(target_index), work["date"]))
    daily = daily.replace([np.inf, -np.inf], np.nan)
    cols = ["evt__earn_eps_surprise", "evt__earn_rev_surprise", "evt__earn_beat_flag", "evt__earn_beat_streak_4", "evt__earn_days_since"]
    return BuiltFeatureSet(df=daily, feature_cols=cols)
