# modules/features/macro_fmp.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Sequence
import pandas as pd
import numpy as np
from datetime import timedelta

from modules.data.fmp_client import FMPClient, FMPInvalidNameError
from modules.data.pit import broadcast_asof_to_target_index


@dataclass(frozen=True)
class MacroFeatureConfig:
    # We use keys here that map to our candidate list below
    economic_series: Tuple[str, ...] = ("GDP", "CPI", "UNEMPLOYMENT", "INFLATION", "FEDERAL_FUNDS_RATE")
    include_treasury_rates: bool = True
    winsorize_p: Optional[float] = None
    fill_method: str = "ffill"
    verbose_debug: bool = False


# ------------------------------------------------------------
# CORRECTED FMP NAMES (Case Sensitive)
# ------------------------------------------------------------
ECON_NAME_CANDIDATES: Dict[str, List[str]] = {
    "GDP": ["GDP", "realGDP", "nominalGDP"],
    "CPI": ["CPI", "cpi"],
    # FMP expects 'unemploymentRate', not 'UNEMPLOYMENT'
    "UNEMPLOYMENT": ["unemploymentRate", "unemployment", "Unemployment Rate"],
    # FMP expects 'inflationRate'
    "INFLATION": ["inflationRate", "inflation", "Inflation Rate"],
    # FMP expects 'federalFunds'
    "FEDERAL_FUNDS_RATE": ["federalFunds", "federalFundsRate", "Federal Funds Rate"],
}


# ------------------------------------------------------------
# 1. Core Fetching Logic
# ------------------------------------------------------------

def fetch_macro_series(
        api_key: str,
        start_date: str,
        end_date: str,
        config: Optional[MacroFeatureConfig] = None,
        verbose: bool = False,
        lookback_days: int = 365  # Buffer to ensure we have data for the first day
) -> pd.DataFrame:
    """
    Fetches Economic Indicators + Treasury Rates and combines them into a single
    DataFrame indexed by Date.
    """
    cfg = config or MacroFeatureConfig()
    fmp = FMPClient(api_key=api_key)

    # Add buffer to start_date to prevent leading NaNs
    s_dt = pd.to_datetime(start_date) - timedelta(days=lookback_days)
    fetch_start = s_dt.strftime("%Y-%m-%d")

    if verbose:
        print(f"[macro] Fetching range: {fetch_start} -> {end_date} (includes buffer)")

    frames = []

    # A) Economic Indicators (CPI, GDP, etc.)
    for alias in cfg.economic_series:
        try:
            # Resolve name
            resolved_name, raw_df = _economic_indicator_fetch_resolved(
                fmp, alias, from_date=fetch_start, to_date=end_date
            )

            # Clean
            clean_df = _prep_econ_df(alias, resolved_name, raw_df)
            if not clean_df.empty:
                clean_df = clean_df.set_index("date")
                frames.append(clean_df)
                if verbose:
                    print(f"[macro] Fetched {alias} (as '{resolved_name}'): {len(clean_df)} rows")
            elif verbose:
                print(f"[macro] {alias} returned empty. (Tried: {ECON_NAME_CANDIDATES.get(alias)})")

        except Exception as e:
            if verbose: print(f"[macro] Failed to fetch {alias}: {e}")

    # B) Treasury Rates
    if cfg.include_treasury_rates:
        try:
            # Treasury rates usually update daily
            raw_tr = fmp.treasury_rates(from_date=fetch_start, to_date=end_date)
            tr_df = _prep_treasury_df(raw_tr)
            if not tr_df.empty:
                tr_df = tr_df.set_index("date")
                frames.append(tr_df)
                if verbose: print(f"[macro] Fetched Treasury Rates: {len(tr_df)} rows")
        except Exception as e:
            if verbose: print(f"[macro] Failed to fetch Treasury Rates: {e}")

    if not frames:
        return pd.DataFrame()

    # C) Merge All (Outer Join on Date)
    full_macro = pd.concat(frames, axis=1)
    full_macro = full_macro.sort_index()

    # D) Fill Logic (Forward Fill Sparse Data)
    if cfg.fill_method == "ffill":
        full_macro = full_macro.ffill()
    elif cfg.fill_method == "bfill":
        full_macro = full_macro.bfill()

    # E) Trim back to requested range (remove buffer)
    # We slice ONLY AFTER filling, ensuring Day 1 has data
    full_macro = full_macro[full_macro.index >= pd.to_datetime(start_date)]
    full_macro = full_macro[full_macro.index <= pd.to_datetime(end_date)]

    return full_macro


def broadcast_macro_to_daily(
        macro_df: pd.DataFrame,
        target_daily_index: pd.Index,
) -> pd.DataFrame:
    """
    Broadcasts macro data (Date index) to a target daily index (MultiIndex Date, Symbol).
    """
    if macro_df.empty:
        return pd.DataFrame(index=target_daily_index)
    return broadcast_asof_to_target_index(
        sparse_df=macro_df,
        target_index=target_daily_index,
        on="date",
        by=None,
    )


# ------------------------------------------------------------
# 2. Helpers
# ------------------------------------------------------------

def _coerce_date_col(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    if "date" not in df.columns: return pd.DataFrame()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    return out


def _economic_indicator_fetch_resolved(fmp: FMPClient, alias: str, from_date: str, to_date: str) -> Tuple[
    str, pd.DataFrame]:
    # Try candidates in order
    candidates = ECON_NAME_CANDIDATES.get(alias, [alias])

    for cand in candidates:
        try:
            # Call FMP
            df = fmp.economic_indicators(cand, from_date=from_date, to_date=to_date)
            df = _coerce_date_col(df)

            # FMP sometimes returns a DF with "Error Message" in it
            if not df.empty and "Error Message" not in df.columns:
                return cand, df
        except Exception:
            continue

    return alias, pd.DataFrame()


def _prep_econ_df(alias: str, resolved_name: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()

    # Find value column
    val_col = "value"
    if "value" not in df.columns:
        # heuristic: first numeric column that isn't date
        nums = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
        if nums: val_col = nums[0]

    if val_col not in df.columns:
        return pd.DataFrame()

    out = df[["date", val_col]].rename(columns={val_col: f"macro__{alias}"})
    # Econ data often has duplicates or revisions; take last
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return out


def _prep_treasury_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    df = _coerce_date_col(df).sort_values("date").drop_duplicates(subset=["date"], keep="last")

    # Prefix columns
    rename_map = {c: f"macro__ust_{c}" for c in df.columns if c != "date"}
    return df.rename(columns=rename_map)
