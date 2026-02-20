# modules/features/fundamentals_fmp.py
from __future__ import annotations

from typing import Any, Sequence, Optional, List, Union
import pandas as pd
import numpy as np
import re

from modules.data.fmp_client import FMPClient
from modules.data.pit import asof_join_pit, broadcast_asof_to_target_index


# from modules.utils.normalize import normalize_cols # We use a local robust version below

# ------------------------------------------------------------
# 1. Core Data Fetching
# ------------------------------------------------------------

def fetch_fundamentals_data(
        symbols: Sequence[str],
        api_key: str,
        period: str = "quarter",
        limit: int = 160,
        verbose: bool = True,
        use_filing_lag: bool = True,
        filing_lag_days: int = 45,
) -> pd.DataFrame:
    """
    Fetches Key Metrics and Ratios from FMP, joins them properly on Date/Symbol.
    Returns a SPARSE DataFrame (rows exist only on filing dates).
    """
    fmp = FMPClient(api_key=api_key)
    dfs_per_symbol: List[pd.DataFrame] = []

    if verbose:
        print(f"[fundamentals] Starting fetch for {len(symbols)} symbols. (Period={period}, Limit={limit})")

    for i, sym in enumerate(symbols, 1):
        try:
            # A) Fetch Raw Data
            res_km = fmp.key_metrics(str(sym), period=period, limit=limit)
            res_rt = fmp.ratios(str(sym), period=period, limit=limit)

            # B) Convert to DataFrame
            raw_km = _to_df(res_km)
            raw_rt = _to_df(res_rt)

            # C) Normalize & Prefix Key Metrics
            if not raw_km.empty:
                df_km = _clean_fmp_response(raw_km, sym, "km__", verbose=False)
            else:
                df_km = pd.DataFrame()

            # D) Normalize & Prefix Ratios
            if not raw_rt.empty:
                df_rt = _clean_fmp_response(raw_rt, sym, "rt__", verbose=False)
            else:
                df_rt = pd.DataFrame()

            # E) Merge logic
            if df_km.empty and df_rt.empty:
                if verbose:
                    print(f"[fundamentals][{sym}] No valid data after cleaning.")
                continue

            # Ensure merge keys exist
            merge_keys = ["date", "symbol", "period"]
            for d in [df_km, df_rt]:
                for k in merge_keys:
                    if k not in d.columns:
                        d[k] = np.nan

            # Ensure type consistency for merge keys
            if not df_km.empty: df_km["symbol"] = df_km["symbol"].astype(str)
            if not df_rt.empty: df_rt["symbol"] = df_rt["symbol"].astype(str)

            # Merge
            df_merged = pd.merge(df_km, df_rt, on=merge_keys, how="outer")

            # Restore dates
            if "date" in df_merged.columns:
                df_merged["date"] = pd.to_datetime(df_merged["date"], errors="coerce")

                # Apply Filing Lag
                if use_filing_lag:
                    df_merged["date"] = df_merged["date"] + pd.to_timedelta(filing_lag_days, unit="D")

            dfs_per_symbol.append(df_merged)

        except Exception as e:
            if verbose:
                print(f"[fundamentals] Error processing {sym}: {e}")
                if i == 1:
                    import traceback
                    traceback.print_exc()

        if verbose and (i % 25 == 0 or i == len(symbols)):
            print(f"[fundamentals] processed {i}/{len(symbols)} symbols")

    if not dfs_per_symbol:
        if verbose: print("[fundamentals] WARN: No data collected for any symbol.")
        return pd.DataFrame()

    # Consolidate
    fund_df = pd.concat(dfs_per_symbol, ignore_index=True)

    # Sort and Index
    if "date" in fund_df.columns and "symbol" in fund_df.columns:
        fund_df = fund_df.sort_values(["symbol", "date"])
        fund_df = fund_df.set_index(["date", "symbol"])

    # Feature Cleanup
    fund_df = _enforce_numeric_features(fund_df)

    return fund_df


# ------------------------------------------------------------
# 2. Broadcasting (Sparse -> Dense)
# ------------------------------------------------------------

def broadcast_fundamentals_to_daily(
        fund_df: pd.DataFrame,
        target_daily_index: pd.Index,
) -> pd.DataFrame:
    """
    Broadcasts sparse quarterly fundamentals forward to every day in the target index.

    Args:
        fund_df: Sparse DataFrame (MultiIndex: date, symbol)
        target_daily_index: The target daily index (MultiIndex: date, symbol)

    Returns:
        Dense DataFrame matching target_daily_index row-for-row.
    """
    return broadcast_asof_to_target_index(
        sparse_df=fund_df,
        target_index=target_daily_index,
        on="date",
        by=("symbol",),
    )


# ------------------------------------------------------------
# 3. Robust Helpers
# ------------------------------------------------------------

def _to_df(raw: Any) -> pd.DataFrame:
    if raw is None: return pd.DataFrame()
    if isinstance(raw, pd.DataFrame): return raw
    if isinstance(raw, list): return pd.DataFrame(raw)
    if isinstance(raw, dict): return pd.DataFrame([raw])
    return pd.DataFrame()


def _clean_fmp_response(df: pd.DataFrame, symbol: str, prefix: str, verbose: bool = False) -> pd.DataFrame:
    """
    Robust cleaning that prints reasons for failure.
    """
    out = df.copy()

    # 1. Force lowercase columns (safe normalization)
    out.columns = out.columns.str.lower().str.strip()

    # 2. Add Symbol
    out["symbol"] = str(symbol)

    # 3. Identify Date Column
    date_col = None
    if "date" in out.columns:
        date_col = "date"
    elif "fillingdate" in out.columns:
        date_col = "fillingdate"
    elif "accepteddate" in out.columns:
        date_col = "accepteddate"
    elif "calendar_date" in out.columns:
        date_col = "calendar_date"

    if not date_col:
        if verbose:
            print(f"[DEBUG] Failed to find date column. Available: {out.columns.tolist()}")
        return pd.DataFrame()

    # 4. Standardize Date
    if date_col != "date":
        out = out.rename(columns={date_col: "date"})

    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # Check for NaT
    n_before = len(out)
    out = out.dropna(subset=["date"])
    n_after = len(out)

    if n_after == 0:
        if verbose:
            print(f"[DEBUG] All rows dropped because 'date' conversion failed. (Before: {n_before}, After: 0)")
        return pd.DataFrame()

    # 5. Prefix Features
    # Exclude metadata from prefixing
    exclude = {
        "date",
        "symbol",
        "period",
        "fiscalyear",
        "fiscalyearended",
        "calendaryear",
        "calendar_year",
        # metadata/non-model fields that should never become features
        "reportedcurrency",
        "currency",
        "link",
        "finallink",
        "accepteddate",
        "fillingdate",
        "filingdate",
        "cik",
    }

    # Create map
    rename_map = {}
    for c in out.columns:
        if c in exclude:
            continue
        rename_map[c] = f"{prefix}{c}"

    out = out.rename(columns=rename_map)

    return out


def _enforce_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.startswith("km__") or col.startswith("rt__"):
            # Force numeric, coerce bad strings to NaN
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


# ------------------------------------------------------------
# 4. Legacy Support (Called by build.py pipelines)
# ------------------------------------------------------------

def _asof_join_fundamentals(target_df: pd.DataFrame, fund_df: pd.DataFrame) -> pd.DataFrame:
    """Match daily target with sparse fundamentals."""
    t = target_df.reset_index() if isinstance(target_df.index, pd.MultiIndex) else target_df.copy()
    f = fund_df.reset_index() if isinstance(fund_df.index, pd.MultiIndex) else fund_df.copy()

    # Ensure types
    t["date"] = pd.to_datetime(t["date"])
    f["date"] = pd.to_datetime(f["date"])
    t["symbol"] = t["symbol"].astype(str)
    f["symbol"] = f["symbol"].astype(str)

    merged = asof_join_pit(
        left=t,
        right=f,
        on="date",
        by=("symbol",),
        direction="backward",
    )
    return merged.sort_values(["symbol", "date"])


def add_fmp_fundamentals_to_dataset(
        dataset: Any,
        *,
        symbols: Sequence[str],
        api_key: str,
        period: str = "quarter",
        limit: int = 160,
        verbose_debug: bool = True,
        include_inference: bool = False,
        use_filing_lag: bool = True,
        filing_lag_days: int = 45,
) -> None:
    fund_df = fetch_fundamentals_data(
        symbols=symbols,
        api_key=api_key,
        period=period,
        limit=limit,
        verbose=verbose_debug,
        use_filing_lag=use_filing_lag,
        filing_lag_days=filing_lag_days,
    )

    if fund_df.empty:
        print("[fundamentals] No data to merge.")
        return

    # Update training
    if hasattr(dataset, "training_df"):
        merged = _asof_join_fundamentals(dataset.training_df, fund_df)

        # Restore index
        if "symbol" in dataset.training_df.index.names:
            merged = merged.set_index(["date", "symbol"])
        elif "date" in dataset.training_df.index.names:
            merged = merged.set_index("date")

        # Add new features to list
        new_cols = [c for c in fund_df.columns if c.startswith("km__") or c.startswith("rt__")]
        dataset.feature_cols = sorted(list(set(dataset.feature_cols + new_cols)))
        dataset.training_df = merged

        if verbose_debug:
            print(f"[fundamentals] Merged {len(new_cols)} features into training_df")

    # Update inference
    if include_inference and hasattr(dataset, "inference_panel"):
        merged_inf = _asof_join_fundamentals(dataset.inference_panel, fund_df)
        if "symbol" in dataset.inference_panel.index.names:
            merged_inf = merged_inf.set_index(["date", "symbol"])
        dataset.inference_panel = merged_inf
