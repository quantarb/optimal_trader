from __future__ import annotations

import pandas as pd

from fmp.models import Symbol
from features.section_utils import BuiltFeatureSet, broadcast_sparse, days_since_for_target, days_since_last_event, load_section_payload, target_dates


def build_analyst_estimates_features(symbol_obj: Symbol, target_index: pd.MultiIndex) -> BuiltFeatureSet:
    sparse = load_section_payload(symbol_obj, "analyst_estimates", prefix="ae__", filing_lag_days=0)
    if sparse.empty:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    work = sparse.reset_index().sort_values(["symbol", "date"])
    work["ae__epsavg"] = pd.to_numeric(work.get("ae__epsavg"), errors="coerce")
    work["ae__revenueavg"] = pd.to_numeric(work.get("ae__revenueavg"), errors="coerce")
    out = work[["date", "symbol"]].copy()
    out["evt__ae_eps_avg"] = work["ae__epsavg"]
    out["evt__ae_eps_rev_qoq"] = work.groupby("symbol")["ae__epsavg"].pct_change()
    out["evt__ae_revenue_rev_qoq"] = work.groupby("symbol")["ae__revenueavg"].pct_change()
    daily = broadcast_sparse(out.set_index(["date", "symbol"]).sort_index(), target_index)
    daily["evt__ae_days_since"] = days_since_for_target(target_index, days_since_last_event(target_dates(target_index), work["date"]))
    cols = ["evt__ae_eps_avg", "evt__ae_eps_rev_qoq", "evt__ae_revenue_rev_qoq", "evt__ae_days_since"]
    return BuiltFeatureSet(df=daily, feature_cols=cols)
