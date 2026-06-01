from __future__ import annotations

import pandas as pd


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common panel/label frames to have a proper datetime index.
    Supports:
      - DatetimeIndex
      - MultiIndex with a 'date' level (e.g., ['date','symbol'])
      - A 'date' column (optionally with 'symbol' column)

    Returns a sorted dataframe with invalid dates removed.
    """
    if df is None or len(df) == 0:
        return df.copy()

    # --- Fast path: already a clean DatetimeIndex, no copy needed ---
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.is_monotonic_increasing and df.index.tz is None:
            return df

    # --- Fast path: clean MultiIndex with a proper DatetimeIndex date level ---
    if isinstance(df.index, pd.MultiIndex):
        names = list(df.index.names)
        if "date" in names:
            date_vals = df.index.get_level_values("date")
            if isinstance(date_vals, pd.DatetimeIndex) and date_vals.tz is None:
                # Already clean — check if sorted
                if df.index.is_monotonic_increasing:
                    return df

    out = df.copy()

    # ------------------------------------------------------------
    # Case 1: MultiIndex (e.g. (date, symbol))
    # ------------------------------------------------------------
    if isinstance(out.index, pd.MultiIndex):
        names = list(out.index.names)

        # If date exists as a level, coerce just that level to datetime
        if "date" in names:
            date_vals = out.index.get_level_values("date")
            date_dt = pd.to_datetime(date_vals, errors="coerce").tz_localize(None)

            # Rebuild MultiIndex with coerced date level (preserve other levels)
            arrays = []
            for n in names:
                if n == "date":
                    arrays.append(date_dt)
                else:
                    arrays.append(out.index.get_level_values(n))
            out.index = pd.MultiIndex.from_arrays(arrays, names=names)

            # Drop rows where date is NaT
            good = ~out.index.get_level_values("date").isna()
            out = out.loc[good]
            return out.sort_index()

        # If no 'date' level, but a 'date' column exists, fall through to column logic below
        # Otherwise, return as-is (still sort for determinism)
        return out.sort_index()

    # ------------------------------------------------------------
    # Case 2: 'date' column exists
    # ------------------------------------------------------------
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
        out = out.dropna(subset=["date"])

        # Preserve symbol if present
        if "symbol" in out.columns:
            out = out.set_index(["date", "symbol"]).sort_index()
        else:
            out = out.set_index("date").sort_index()
        return out

    # ------------------------------------------------------------
    # Case 3: Single-level index (try to coerce to datetime)
    # ------------------------------------------------------------
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce").tz_localize(None)

    out = out[~out.index.isna()]
    return out.sort_index()
