from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from fmp.models import (
    EconomicIndicatorObservation,
    EconomicIndicatorSeries,
    TreasuryRateObservation,
    TreasuryRateSeries,
)


@dataclass(frozen=True)
class EconomicDataConfig:
    economic_indicator_series: Tuple[str, ...] = ("GDP", "CPI", "unemploymentRate", "inflationRate", "federalFunds")
    include_treasury_rates: bool = True
    winsorize_p: Optional[float] = None
    fill_method: str = "none"
    verbose_debug: bool = False


MacroFeatureConfig = EconomicDataConfig


ECON_NAME_CANDIDATES: Dict[str, List[str]] = {
    "GDP": ["GDP", "realGDP", "nominalGDP"],
    "CPI": ["CPI", "cpi"],
    "UNEMPLOYMENT": ["unemploymentRate", "unemployment", "Unemployment Rate"],
    "INFLATION": ["inflationRate", "inflation", "Inflation Rate"],
    "FEDERAL_FUNDS_RATE": ["federalFunds", "federalFundsRate", "Federal Funds Rate"],
}


def _resolve_requested_series_codes(cfg: EconomicDataConfig) -> list[str]:
    economic_available = set(EconomicIndicatorSeries.objects.values_list("code", flat=True))
    treasury_available = [str(code) for code in TreasuryRateSeries.objects.order_by("code").values_list("code", flat=True)]
    treasury_available_set = set(treasury_available)
    requested: list[str] = []
    for raw in cfg.economic_indicator_series:
        alias = str(raw)
        if alias in economic_available or alias in treasury_available_set:
            chosen = alias
        else:
            candidates = ECON_NAME_CANDIDATES.get(alias, [alias])
            expanded_candidates: list[str] = []
            for cand in candidates:
                expanded_candidates.append(cand)
                if not str(cand).startswith("macro__"):
                    expanded_candidates.append(f"macro__{cand}")
            chosen = next((cand for cand in expanded_candidates if cand in economic_available), expanded_candidates[0])
        if chosen not in requested:
            requested.append(chosen)
    if cfg.include_treasury_rates:
        for code in treasury_available:
            if code not in requested:
                requested.append(code)
    return requested


def fetch_economic_data_series(
    api_key: str,
    start_date: str,
    end_date: str,
    config: Optional[EconomicDataConfig] = None,
    verbose: bool = False,
    lookback_days: int = 0,
) -> pd.DataFrame:
    """
    Compatibility entrypoint.
    Loads sparse economic indicator and treasury-rate series from the Django DB.
    """
    del api_key, verbose, lookback_days
    cfg = config or EconomicDataConfig()
    series_codes = _resolve_requested_series_codes(cfg)
    if not series_codes:
        return pd.DataFrame()

    rows: dict[str, dict[str, float]] = {}
    economic_codes = [code for code in series_codes if not str(code).startswith("macro__ust_")]
    treasury_codes = [code for code in series_codes if str(code).startswith("macro__ust_")]

    economic_obs_qs = (
        EconomicIndicatorObservation.objects.filter(
            series__code__in=economic_codes,
            observation_date__gte=pd.to_datetime(start_date).date(),
            observation_date__lte=pd.to_datetime(end_date).date(),
        )
        .select_related("series")
        .order_by("observation_date")
    )
    for obs in economic_obs_qs.iterator():
        row = rows.setdefault(obs.observation_date.isoformat(), {})
        row[obs.series.code] = float(obs.value)

    treasury_obs_qs = (
        TreasuryRateObservation.objects.filter(
            series__code__in=treasury_codes,
            observation_date__gte=pd.to_datetime(start_date).date(),
            observation_date__lte=pd.to_datetime(end_date).date(),
        )
        .select_related("series")
        .order_by("observation_date")
    )
    for obs in treasury_obs_qs.iterator():
        row = rows.setdefault(obs.observation_date.isoformat(), {})
        row[obs.series.code] = float(obs.value)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index = pd.to_datetime(df.index)
    for code in series_codes:
        if code not in df.columns:
            df[code] = pd.NA
    return df[series_codes]


def broadcast_series_to_daily(
    series_df: pd.DataFrame,
    target_daily_index: pd.Index,
) -> pd.DataFrame:
    if series_df.empty:
        return pd.DataFrame(index=target_daily_index)
    sparse = series_df.copy().sort_index()
    if not isinstance(sparse.index, pd.DatetimeIndex):
        sparse.index = pd.to_datetime(sparse.index, errors="coerce")
        sparse = sparse[~sparse.index.isna()].sort_index()

    if isinstance(target_daily_index, pd.MultiIndex):
        target_dates = pd.DatetimeIndex(pd.to_datetime(target_daily_index.get_level_values("date"))).normalize()
        unique_dates = pd.DatetimeIndex(sorted(target_dates.unique()))
        dense = sparse.reindex(unique_dates).ffill()
        expanded = dense.reindex(target_dates)
        expanded.index = target_daily_index
        return expanded

    target_dates = pd.DatetimeIndex(pd.to_datetime(target_daily_index)).normalize()
    dense = sparse.reindex(target_dates).ffill()
    dense.index = target_daily_index
    return dense


def fetch_macro_series(
    api_key: str,
    start_date: str,
    end_date: str,
    config: Optional[EconomicDataConfig] = None,
    verbose: bool = False,
    lookback_days: int = 0,
) -> pd.DataFrame:
    return fetch_economic_data_series(
        api_key=api_key,
        start_date=start_date,
        end_date=end_date,
        config=config,
        verbose=verbose,
        lookback_days=lookback_days,
    )


def broadcast_macro_to_daily(
    macro_df: pd.DataFrame,
    target_daily_index: pd.Index,
) -> pd.DataFrame:
    return broadcast_series_to_daily(macro_df, target_daily_index)
