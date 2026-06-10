from __future__ import annotations

import numpy as np
import pandas as pd

from fmp.models import Symbol
from fmp.positions_summary import POSITIONS_SUMMARY_SECTION_KEY, load_positions_summary_frame
from features.section_utils import (
    BuiltFeatureSet,
    broadcast_sparse,
    days_since_for_target,
    days_since_last_event,
    load_section_payload,
    safe_ratio,
    target_dates,
)


POSITION_PREFIX = "ps__"

_CANONICAL_FIELDS: dict[str, tuple[str, ...]] = {
    "investor_count": (
        "investor_count",
        f"{POSITION_PREFIX}investor_count",
        f"{POSITION_PREFIX}investorcount",
        f"{POSITION_PREFIX}investorscount",
        f"{POSITION_PREFIX}holdercount",
        f"{POSITION_PREFIX}holderscount",
        f"{POSITION_PREFIX}institutionalholders",
        f"{POSITION_PREFIX}numberofinvestors",
        f"{POSITION_PREFIX}numberofholders",
    ),
    "shares_held": (
        "shares_held",
        f"{POSITION_PREFIX}shares_held",
        f"{POSITION_PREFIX}sharesheld",
        f"{POSITION_PREFIX}totalshares",
        f"{POSITION_PREFIX}sharecount",
        f"{POSITION_PREFIX}shares",
        f"{POSITION_PREFIX}institutionalshares",
    ),
    "investment_value": (
        "investment_value",
        f"{POSITION_PREFIX}investment_value",
        f"{POSITION_PREFIX}investmentvalue",
        f"{POSITION_PREFIX}totalinvestmentvalue",
        f"{POSITION_PREFIX}marketvalue",
        f"{POSITION_PREFIX}positionvalue",
    ),
    "ownership_pct": (
        "ownership_pct",
        f"{POSITION_PREFIX}ownership_pct",
        f"{POSITION_PREFIX}ownershippct",
        f"{POSITION_PREFIX}ownershippercentage",
        f"{POSITION_PREFIX}ownershippercent",
    ),
    "shares_change": (
        "shares_change",
        f"{POSITION_PREFIX}shares_change",
        f"{POSITION_PREFIX}shareschange",
        f"{POSITION_PREFIX}changeinshares",
        f"{POSITION_PREFIX}sharechange",
    ),
    "investment_change": (
        "investment_change",
        f"{POSITION_PREFIX}investment_change",
        f"{POSITION_PREFIX}investmentchange",
        f"{POSITION_PREFIX}changeininvestment",
        f"{POSITION_PREFIX}valuechange",
    ),
    "ownership_pct_change": (
        "ownership_pct_change",
        f"{POSITION_PREFIX}ownership_pct_change",
        f"{POSITION_PREFIX}ownershippctchange",
        f"{POSITION_PREFIX}ownershippercentagechange",
        f"{POSITION_PREFIX}changeinownership",
    ),
    "put_call_ratio": (
        "put_call_ratio",
        f"{POSITION_PREFIX}put_call_ratio",
        f"{POSITION_PREFIX}putcallratio",
        f"{POSITION_PREFIX}putcall",
    ),
    "call_count": (
        "call_count",
        f"{POSITION_PREFIX}call_count",
        f"{POSITION_PREFIX}callcount",
        f"{POSITION_PREFIX}calls",
        f"{POSITION_PREFIX}callscount",
    ),
    "put_count": (
        "put_count",
        f"{POSITION_PREFIX}put_count",
        f"{POSITION_PREFIX}putcount",
        f"{POSITION_PREFIX}puts",
        f"{POSITION_PREFIX}putscount",
    ),
}


def _resolve_numeric_series(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series | None:
    for column in candidates:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return None


def _prepare_source_frame(symbol_obj: Symbol) -> pd.DataFrame:
    source = load_positions_summary_frame(symbol_obj)
    if not source.empty:
        return source

    fallback = load_section_payload(
        symbol_obj,
        POSITIONS_SUMMARY_SECTION_KEY,
        prefix=POSITION_PREFIX,
        filing_lag_days=0,
    )
    return fallback


def build_positions_summary_features(symbol_obj: Symbol, target_index: pd.MultiIndex) -> BuiltFeatureSet:
    sparse = _prepare_source_frame(symbol_obj)
    if sparse.empty:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])

    work = sparse.reset_index().sort_values(["symbol", "date"]).copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"])
    if work.empty:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])

    work["symbol"] = str(symbol_obj.symbol).strip().upper()
    feature_frame = work[["date", "symbol"]].copy()

    numeric_sources: dict[str, pd.Series] = {}
    for canonical_name, candidates in _CANONICAL_FIELDS.items():
        series = _resolve_numeric_series(work, candidates)
        if series is None:
            continue
        numeric_sources[canonical_name] = series
        feature_frame[f"{POSITION_PREFIX}{canonical_name}"] = series

    if not numeric_sources:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])

    for canonical_name in ("investor_count", "shares_held", "investment_value", "ownership_pct", "put_call_ratio"):
        series = numeric_sources.get(canonical_name)
        if series is None:
            continue
        feature_frame[f"{POSITION_PREFIX}{canonical_name}_change"] = series.groupby(work["symbol"]).diff()
        feature_frame[f"{POSITION_PREFIX}{canonical_name}_pct_change"] = series.groupby(work["symbol"]).pct_change().replace([np.inf, -np.inf], np.nan)

    if "investor_count" in numeric_sources and "shares_held" in numeric_sources:
        feature_frame["ps__shares_per_investor"] = safe_ratio(numeric_sources["shares_held"], numeric_sources["investor_count"].replace(0.0, np.nan))
    if "investor_count" in numeric_sources and "investment_value" in numeric_sources:
        feature_frame["ps__investment_per_investor"] = safe_ratio(numeric_sources["investment_value"], numeric_sources["investor_count"].replace(0.0, np.nan))
    if "shares_held" in numeric_sources and "ownership_pct" in numeric_sources:
        feature_frame["ps__shares_ownership_ratio"] = safe_ratio(numeric_sources["shares_held"], numeric_sources["ownership_pct"].replace(0.0, np.nan))

    daily = broadcast_sparse(feature_frame.set_index(["date", "symbol"]).sort_index(), target_index)
    report_days = days_since_last_event(target_dates(target_index), work["date"])
    daily["ps__days_since_report"] = days_since_for_target(target_index, report_days)

    feature_cols = [col for col in daily.columns if str(col).startswith(POSITION_PREFIX)]
    daily = daily.replace([np.inf, -np.inf], np.nan)
    return BuiltFeatureSet(df=daily, feature_cols=feature_cols)


__all__ = ["build_positions_summary_features"]
