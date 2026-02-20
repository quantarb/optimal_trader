# ============================================================
# modules/data/prices_sqlite.py
#
# Purpose:
#   - Correct 30y daily OHLCV ingestion from FMP "stable" API
#   - Uses dividend-adjusted endpoint payload:
#       ['adjOpen','adjHigh','adjLow','adjClose','volume']
#     and maps to canonical columns:
#       ['open','high','low','close','volume']
#   - Chunked fetching to avoid surprises
#   - SQLite upsert + coverage prints
#
# Fixes:
#   - volume-only DB rows (caused by not mapping adj* fields)
#   - NameError: Optional, expected_latest_trading_day
# ============================================================
from __future__ import annotations

import time
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from typing import Optional, Sequence, Tuple, Dict, Any, List

import pandas as pd
import requests

from modules.utils.normalize import normalize_cols
from modules.data.context import DataContext
from modules.data.fmp_client import FMPClient  # optional; we also provide requests fallback


# Stable endpoints (you already confirmed payload)
FMP_STABLE_DIV_ADJ_URL = "https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted"
_ET = ZoneInfo("America/New_York")
_DEFAULT_PROVIDER_READY_ET = dtime(18, 30)  # conservative


# ============================================================
# Time helpers
# ============================================================
def _prev_weekday(dt: datetime) -> datetime:
    dt = dt - timedelta(days=1)
    while dt.weekday() >= 5:  # 5=Sat, 6=Sun
        dt = dt - timedelta(days=1)
    return dt


def expected_latest_trading_day(
    now: datetime | None = None,
    *,
    provider_ready_et: dtime = _DEFAULT_PROVIDER_READY_ET,
) -> pd.Timestamp:
    """
    Latest date for which daily bars are likely available (weekend-aware, holiday-blind).
    """
    now = now or datetime.now(tz=_ET)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_ET)
    else:
        now = now.astimezone(_ET)

    if now.weekday() >= 5:
        last = now
        while last.weekday() >= 5:
            last -= timedelta(days=1)
        return pd.Timestamp(last.date()).normalize()

    if now.time() < provider_ready_et:
        return pd.Timestamp(_prev_weekday(now).date()).normalize()

    return pd.Timestamp(now.date()).normalize()


def should_check_fmp(last_dt: pd.Timestamp, *, provider_ready_et: dtime = _DEFAULT_PROVIDER_READY_ET) -> bool:
    last_dt = pd.Timestamp(last_dt).normalize()
    latest_possible = expected_latest_trading_day(provider_ready_et=provider_ready_et)
    return last_dt < latest_possible


def _desired_window(ctx: DataContext) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    (desired_start, desired_end) based on ctx.history_years and latest possible trading day.
    """
    years = int(getattr(ctx, "history_years", 30) or 30)
    desired_end = expected_latest_trading_day()
    desired_start = (pd.Timestamp(desired_end) - pd.Timedelta(days=years * 365)).normalize()
    return desired_start, pd.Timestamp(desired_end).normalize()


# ============================================================
# Coverage logging
# ============================================================
def _df_coverage(df: Optional[pd.DataFrame]) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], int]:
    if df is None or df.empty:
        return None, None, 0
    d0 = pd.to_datetime(df.index.min(), errors="coerce")
    d1 = pd.to_datetime(df.index.max(), errors="coerce")
    if pd.isna(d0) or pd.isna(d1):
        return None, None, int(len(df))
    return pd.Timestamp(d0).normalize(), pd.Timestamp(d1).normalize(), int(len(df))


def _print_coverage(symbol: str, df: Optional[pd.DataFrame], *, prefix: str = "") -> None:
    d0, d1, n = _df_coverage(df)
    if n == 0 or d0 is None or d1 is None:
        print(f"{prefix}[{symbol}] coverage: rows=0")
    else:
        print(f"{prefix}[{symbol}] coverage: rows={n:,} start={d0.date()} end={d1.date()}")


# ============================================================
# Core: map stable dividend-adjusted payload -> canonical OHLCV
# ============================================================
def _normalize_fmp_div_adj_payload(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Input example row:
      {'symbol':'AAPL','date':'1996-02-20','adjOpen':0.2,'adjHigh':...,'adjClose':...,'volume':...}

    Output indexed by date with columns:
      open, high, low, close, volume  (all numeric)
    """
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Map adjusted fields to canonical
    rename_map = {
        "adjOpen": "open",
        "adjHigh": "high",
        "adjLow": "low",
        "adjClose": "close",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    # Coerce numeric
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop unusable rows (solver needs close)
    df = df.dropna(subset=["close"])

    # Index + sort + de-dup
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Keep only canonical cols
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep]


# ============================================================
# Fetch: chunked 30y stable dividend-adjusted
# ============================================================
def _fetch_fmp_div_adj_chunk(
    symbol: str,
    *,
    api_key: str,
    date_from: pd.Timestamp,
    date_to: pd.Timestamp,
    timeout_s: int = 60,
) -> pd.DataFrame:
    params = {
        "symbol": symbol,
        "from": pd.Timestamp(date_from).strftime("%Y-%m-%d"),
        "to": pd.Timestamp(date_to).strftime("%Y-%m-%d"),
        "apikey": api_key,
    }
    r = requests.get(FMP_STABLE_DIV_ADJ_URL, params=params, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        return pd.DataFrame()
    return _normalize_fmp_div_adj_payload(data)


def fetch_prices_daily_fmp_30y_div_adj(
    symbol: str,
    *,
    api_key: str,
    years: int = 30,
    chunk_years: int = 10,
    sleep_s: float = 0.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Fetch up to 'years' years of dividend-adjusted daily bars from stable API,
    chunked in 'chunk_years' windows.
    """
    end = expected_latest_trading_day()
    start = (end - pd.Timedelta(days=int(years) * 365)).normalize()

    all_chunks: List[pd.DataFrame] = []
    cur = start

    while cur <= end:
        nxt = min(end, (cur + pd.Timedelta(days=int(chunk_years) * 365)) )
        dfc = _fetch_fmp_div_adj_chunk(symbol, api_key=api_key, date_from=cur, date_to=nxt)
        if not dfc.empty:
            if verbose:
                d0, d1, n = _df_coverage(dfc)
                print(f"[{symbol}] chunk {cur.date()} → {nxt.date()}: rows={n:,} min={d0.date() if d0 else None} max={d1.date() if d1 else None}")
            all_chunks.append(dfc)
        else:
            if verbose:
                print(f"[{symbol}] empty chunk {cur.date()} → {nxt.date()}")
        if sleep_s and sleep_s > 0:
            time.sleep(float(sleep_s))
        cur = (nxt + pd.Timedelta(days=1)).normalize()

    if not all_chunks:
        return pd.DataFrame()

    out = pd.concat(all_chunks, axis=0).sort_index()
    out = out[~out.index.duplicated(keep="last")].sort_index()

    # final safety: drop missing close
    if "close" in out.columns:
        out = out.dropna(subset=["close"])

    return out


# ============================================================
# Canonical loader used by dataset builder
# Behavior:
#   - Ensures DB has at least desired window (30y) worth of bars when possible
#   - Repairs "volume-only" bad rows by re-seeding symbol if prices are missing
# ============================================================
def _has_valid_prices(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    if "close" not in df.columns:
        return False
    # must have some non-null numeric close values
    s = pd.to_numeric(df["close"], errors="coerce")
    return bool(s.notna().any())


def load_or_fetch_prices_daily(symbol: str, *, ctx: DataContext) -> pd.DataFrame:
    store = ctx.store
    api_key = ctx.api_key
    sleep_s = float(getattr(ctx, "sleep_s", 0.0) or 0.0)
    verbose = bool(getattr(ctx, "verbose", False))

    store.init_schema()

    desired_start, desired_end = _desired_window(ctx)

    df_db = store.load_prices_daily(symbol=symbol)
    df_db = normalize_cols(df_db) if df_db is not None else pd.DataFrame()
    df_db = df_db.sort_index() if not df_db.empty else df_db

    # If DB exists but prices are invalid (your current volume-only issue), force reseed
    if df_db is not None and not df_db.empty and not _has_valid_prices(df_db):
        if verbose:
            print(f"[{symbol}] WARNING: DB has invalid prices (volume-only or non-numeric). Reseeding from FMP...")
        df_api = fetch_prices_daily_fmp_30y_div_adj(
            symbol,
            api_key=api_key,
            years=int(getattr(ctx, "history_years", 30) or 30),
            chunk_years=10,
            sleep_s=sleep_s,
            verbose=verbose,
        )
        store.upsert_prices_daily(df_api.reset_index(), symbol=symbol)
        out = normalize_cols(store.load_prices_daily(symbol=symbol)).sort_index()
        if verbose:
            _print_coverage(symbol, out)
        return out

    # If DB miss, seed
    if df_db is None or df_db.empty:
        if verbose:
            print(f"[{symbol}] prices fetched from FMP (sqlite miss)")
        df_api = fetch_prices_daily_fmp_30y_div_adj(
            symbol,
            api_key=api_key,
            years=int(getattr(ctx, "history_years", 30) or 30),
            chunk_years=10,
            sleep_s=sleep_s,
            verbose=verbose,
        )
        store.upsert_prices_daily(df_api.reset_index(), symbol=symbol)
        out = normalize_cols(store.load_prices_daily(symbol=symbol)).sort_index()
        if verbose:
            _print_coverage(symbol, out)
        return out

    # DB hit: check if we need to extend LEFT (older history) to meet desired_start
    db_start = pd.Timestamp(df_db.index.min()).normalize()
    db_end = pd.Timestamp(df_db.index.max()).normalize()

    if verbose:
        _print_coverage(symbol, df_db)

    needs_left = db_start > desired_start
    needs_right = should_check_fmp(db_end)

    # Extend left if needed
    if needs_left:
        if verbose:
            print(f"[{symbol}] backfilling older history (sqlite min {db_start.date()} > desired {desired_start.date()})")
        # Fetch the missing left side only: desired_start -> day before db_start
        df_left = fetch_prices_daily_fmp_30y_div_adj(
            symbol,
            api_key=api_key,
            years=int(getattr(ctx, "history_years", 30) or 30),
            chunk_years=10,
            sleep_s=sleep_s,
            verbose=verbose,
        )
        # Upsert full (simple + safe). If you want efficiency, slice to < db_start.
        if not df_left.empty:
            store.upsert_prices_daily(df_left.reset_index(), symbol=symbol)
            df_db = normalize_cols(store.load_prices_daily(symbol=symbol)).sort_index()
            db_start = pd.Timestamp(df_db.index.min()).normalize()
            db_end = pd.Timestamp(df_db.index.max()).normalize()

    # Extend right (new rows) if needed
    if needs_right:
        if verbose:
            print(f"[{symbol}] checking FMP for new rows after {db_end.date()}...")
        df_full = fetch_prices_daily_fmp_30y_div_adj(
            symbol,
            api_key=api_key,
            years=int(getattr(ctx, "history_years", 30) or 30),
            chunk_years=10,
            sleep_s=sleep_s,
            verbose=False,
        )
        df_new = df_full[df_full.index > db_end] if not df_full.empty else pd.DataFrame()
        if df_new.empty:
            if verbose:
                print(f"[{symbol}] up to date (no new rows)")
            out = df_db
            if verbose:
                _print_coverage(symbol, out)
            return out
        if verbose:
            print(f"[{symbol}] appending {len(df_new):,} new rows ({df_new.index.min().date()} → {df_new.index.max().date()})")
        store.upsert_prices_daily(df_new.reset_index(), symbol=symbol)
        out = normalize_cols(store.load_prices_daily(symbol=symbol)).sort_index()
        if verbose:
            _print_coverage(symbol, out)
        return out

    # Otherwise return DB
    out = df_db.sort_index()
    if verbose:
        latest_possible = expected_latest_trading_day().date()
        print(f"[{symbol}] skip FMP check (sqlite max {db_end.date()} >= latest_possible {latest_possible})")
        _print_coverage(symbol, out)
    return out


def load_or_fetch_prices_daily_fast(
    symbol: str,
    *,
    ctx: DataContext,
    last_dt_hint: Optional[pd.Timestamp],
) -> pd.DataFrame:
    """
    Fast loader:
      - if hint missing: seeds full window (30y)
      - if hint present but DB is volume-only: reseed
      - otherwise uses hint only for 'should_check_fmp' gating
    """
    store = ctx.store
    api_key = ctx.api_key
    sleep_s = float(getattr(ctx, "sleep_s", 0.0) or 0.0)
    verbose = bool(getattr(ctx, "verbose", False))

    store.init_schema()

    df_db = store.load_prices_daily(symbol=symbol)
    df_db = normalize_cols(df_db) if df_db is not None else pd.DataFrame()
    df_db = df_db.sort_index() if not df_db.empty else df_db

    # Fix corrupted rows
    if df_db is not None and not df_db.empty and not _has_valid_prices(df_db):
        if verbose:
            print(f"[{symbol}] WARNING: DB has invalid prices (volume-only). Reseeding from FMP...")
        df_api = fetch_prices_daily_fmp_30y_div_adj(
            symbol,
            api_key=api_key,
            years=int(getattr(ctx, "history_years", 30) or 30),
            chunk_years=10,
            sleep_s=sleep_s,
            verbose=verbose,
        )
        store.upsert_prices_daily(df_api.reset_index(), symbol=symbol)
        out = normalize_cols(store.load_prices_daily(symbol=symbol)).sort_index()
        if verbose:
            _print_coverage(symbol, out)
        return out

    if last_dt_hint is None or df_db is None or df_db.empty:
        if verbose:
            print(f"[{symbol}] prices fetched from FMP (sqlite miss; bulk last_dt missing)")
        df_api = fetch_prices_daily_fmp_30y_div_adj(
            symbol,
            api_key=api_key,
            years=int(getattr(ctx, "history_years", 30) or 30),
            chunk_years=10,
            sleep_s=sleep_s,
            verbose=verbose,
        )
        store.upsert_prices_daily(df_api.reset_index(), symbol=symbol)
        out = normalize_cols(store.load_prices_daily(symbol=symbol)).sort_index()
        if verbose:
            _print_coverage(symbol, out)
        return out

    last_dt = pd.Timestamp(last_dt_hint).normalize()

    if not should_check_fmp(last_dt):
        out = df_db
        if verbose:
            latest_possible = expected_latest_trading_day().date()
            print(f"[{symbol}] skip FMP check (sqlite max {last_dt.date()} >= latest_possible {latest_possible})")
            _print_coverage(symbol, out)
        return out

    if verbose:
        print(f"[{symbol}] checking FMP for new rows after {last_dt.date()}...")

    df_full = fetch_prices_daily_fmp_30y_div_adj(
        symbol,
        api_key=api_key,
        years=int(getattr(ctx, "history_years", 30) or 30),
        chunk_years=10,
        sleep_s=sleep_s,
        verbose=False,
    )
    df_new = df_full[df_full.index > last_dt] if not df_full.empty else pd.DataFrame()

    if df_new.empty:
        out = df_db
        if verbose:
            print(f"[{symbol}] up to date (no new rows)")
            _print_coverage(symbol, out)
        return out

    if verbose:
        print(f"[{symbol}] appending {len(df_new):,} new rows ({df_new.index.min().date()} → {df_new.index.max().date()})")

    store.upsert_prices_daily(df_new.reset_index(), symbol=symbol)
    out = normalize_cols(store.load_prices_daily(symbol=symbol)).sort_index()
    if verbose:
        _print_coverage(symbol, out)
    return out
