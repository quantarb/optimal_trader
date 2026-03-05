from __future__ import annotations

import pandas as pd

from fmp.models import Symbol
from features.section_utils import BuiltFeatureSet, broadcast_sparse, days_since_for_target, days_since_last_event, load_section_payload, target_dates


def build_ratings_historical_features(symbol_obj: Symbol, target_index: pd.MultiIndex) -> BuiltFeatureSet:
    sparse = load_section_payload(symbol_obj, "ratings_historical", prefix="rating__", filing_lag_days=0)
    if sparse.empty:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    work = sparse.reset_index().sort_values(["symbol", "date"])
    work["rating__overallscore"] = pd.to_numeric(work.get("rating__overallscore"), errors="coerce")
    out = work[["date", "symbol"]].copy()
    out["evt__rating_score"] = work["rating__overallscore"]
    out["evt__rating_score_change"] = work.groupby("symbol")["rating__overallscore"].diff()
    daily = broadcast_sparse(out.set_index(["date", "symbol"]).sort_index(), target_index)
    daily["evt__rating_days_since"] = days_since_for_target(target_index, days_since_last_event(target_dates(target_index), work["date"]))
    cols = ["evt__rating_score", "evt__rating_score_change", "evt__rating_days_since"]
    return BuiltFeatureSet(df=daily, feature_cols=cols)
