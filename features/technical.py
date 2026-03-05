from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from modules.utils.normalize import normalize_cols


# Base columns that must exist in prices
BASE_PRICE_COLS = ("open", "high", "low", "close", "volume")


# ============================================================
# Result container
# ============================================================
@dataclass(frozen=True)
class FeaturesResult:
    df_daily: pd.DataFrame
    feature_cols: List[str]


# ============================================================
# Helpers
# ============================================================
def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns and not isinstance(out.index, pd.DatetimeIndex):
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.set_index("date")
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    out = out.sort_index()
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]
    return out


def _pick_feature_cols(df_daily: pd.DataFrame) -> List[str]:
    cols = []
    for c in df_daily.columns:
        if c in BASE_PRICE_COLS or c == "symbol":
            continue
        if pd.api.types.is_numeric_dtype(df_daily[c]):
            cols.append(c)
    return sorted(cols)


def _sanitize_features(
    df_daily: pd.DataFrame,
    feature_cols: List[str],
    *,
    fill_method: str = "ffill_bfill_zero",  # "zero" | "ffill_bfill_zero" | "drop_rows"
) -> pd.DataFrame:
    """
    Make feature matrix finite for ML.

    - Keeps OHLCV as-is.
    - Replaces inf -> nan.
    - Then either fills or drops rows that contain NaNs in feature columns.

    fill_method:
      - "ffill_bfill_zero" (default): forward fill, then backfill, then 0
      - "zero": fill NaNs with 0 only
      - "drop_rows": drop rows with any NaN in feature columns (aggressive, but clean)
    """
    out = df_daily.copy()
    if not feature_cols:
        return out

    X = out[feature_cols].replace([np.inf, -np.inf], np.nan)

    if fill_method == "drop_rows":
        mask = X.notna().all(axis=1)
        out = out.loc[mask].copy()
        return out

    if fill_method == "zero":
        X = X.fillna(0.0)
    else:
        X = X.ffill().bfill().fillna(0.0)

    out[feature_cols] = X
    return out


# ============================================================
# World-class quant feature set (price + volume only)
# ============================================================
def compute_features_worldclass(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        df indexed by date with columns:
        open, high, low, close, volume

    Output:
        df with OHLCV + derived features
        (NO lookahead, NaNs at window warmup)

    NOTE:
        This implementation avoids pandas "highly fragmented" warnings by:
        - computing derived features into a dict
        - concatenating once at the end
    """
    out = df.copy()
    eps = 1e-12

    def _safe_div(a, b):
        # Works for Series; for scalars it also behaves fine.
        if hasattr(b, "replace"):
            b = b.replace(0, np.nan)
        return a / (b + eps)

    # Ensure numeric OHLCV
    for c in BASE_PRICE_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    open_ = out["open"]
    high = out["high"]
    low = out["low"]
    close = out["close"]
    vol = out["volume"]

    feats: dict[str, pd.Series] = {}

    # ========================================================
    # 1) Returns & momentum (core Simons-style signals)
    # ========================================================
    ret_1d = close.pct_change()
    # Keep simple returns; drop log-returns to avoid redundant encoding.
    feats["Ret1d"] = ret_1d

    for n in [2, 3, 5, 10, 20, 63, 126, 252]:
        feats[f"Ret{n}d"] = close.pct_change(n)

    for n in [5, 10, 20, 63]:
        feats[f"CumRet{n}d"] = (1.0 + ret_1d).rolling(n).apply(np.prod, raw=True) - 1.0

    # ========================================================
    # 2) Trend (moving averages, slopes, MACD)
    # ========================================================
    for n in [5, 10, 20, 50, 100, 200]:
        sma = close.rolling(n).mean()
        feats[f"DistSMA{n}"] = _safe_div(close - sma, sma)
        feats[f"SMASlope{n}"] = sma.diff()

    for n in [12, 26, 50]:
        ema = close.ewm(span=n, adjust=False).mean()
        feats[f"DistEMA{n}"] = _safe_div(close - ema, ema)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    feats["MACD"] = macd
    feats["MACDSignal"] = signal
    feats["MACDHist"] = macd - signal

    # ========================================================
    # 3) Mean reversion & normalization
    # ========================================================
    for n in [10, 20, 63]:
        m = close.rolling(n).mean()
        s = close.rolling(n).std()
        feats[f"ZClose{n}"] = _safe_div(close - m, s + eps)

        upper = m + 2 * s
        lower = m - 2 * s
        feats[f"BBPos{n}"] = _safe_div(close - lower, (upper - lower) + eps)

    # ========================================================
    # 4) Volatility & range (risk regime)
    # ========================================================
    feats["HlRange"] = _safe_div(high - low, close)
    feats["OcChange"] = _safe_div(close - open_, open_)
    feats["Gap"] = _safe_div(open_ - close.shift(1), close.shift(1))

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    feats["TrueRange"] = tr

    for n in [14, 20]:
        atr = tr.rolling(n).mean()
        feats[f"ATRPct{n}"] = _safe_div(atr, close)

    for n in [5, 10, 20, 63]:
        vol_n = ret_1d.rolling(n).std()
        feats[f"Vol{n}"] = vol_n

        base_mean = vol_n.rolling(252).mean()
        base_std = vol_n.rolling(252).std()
        feats[f"VolRegimeZ{n}"] = _safe_div(vol_n - base_mean, base_std + eps)

    # ========================================================
    # 5) Breakouts & channels (Donchian)
    # ========================================================
    for n in [10, 20, 55]:
        hh = high.rolling(n).max()
        ll = low.rolling(n).min()

        # shift(1) avoids same-bar lookahead
        feats[f"BreakoutUp{n}"] = (close > hh.shift(1)).astype(float)
        feats[f"BreakoutDn{n}"] = (close < ll.shift(1)).astype(float)

        feats[f"PosInChannel{n}"] = _safe_div(close - ll, (hh - ll) + eps)
        feats[f"DistHh{n}"] = _safe_div(close - hh, hh)
        feats[f"DistLl{n}"] = _safe_div(close - ll, ll)

    # ========================================================
    # 6) Volume & flow proxies
    # ========================================================
    for n in [5, 20, 63]:
        vmean = vol.rolling(n).mean()
        vstd = vol.rolling(n).std()
        feats[f"VolZ{n}"] = _safe_div(vol - vmean, vstd + eps)

    direction = np.sign(close.diff()).fillna(0.0)
    feats["OBV"] = (direction * vol.fillna(0.0)).cumsum()

    dollar_vol = close * vol
    feats["DollarVol"] = dollar_vol
    feats["DollarVolZ20"] = _safe_div(
        dollar_vol - dollar_vol.rolling(20).mean(),
        dollar_vol.rolling(20).std() + eps,
    )

    # ========================================================
    # 7) Microstructure-ish daily signal
    # ========================================================
    feats["CLV"] = _safe_div((close - low) - (high - close), (high - low) + eps)

    # ========================================================
    # Final assembly (single concat = no fragmentation)
    # ========================================================
    feats_df = pd.DataFrame(feats, index=out.index)
    out = pd.concat([out, feats_df], axis=1)

    # Cleanup: convert inf -> nan (sanitizer will handle remaining NaNs)
    out = out.replace([np.inf, -np.inf], np.nan)

    return out


# ============================================================
# Public API: always compute features
# ============================================================
def load_or_compute_features_daily(
    symbol: str,
    *,
    df_prices: pd.DataFrame,
    compute_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    compute_features_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> FeaturesResult:
    """
    Always recompute features from df_prices.
    No caching. No feature store. Deterministic.

    Returns:
        FeaturesResult(df_daily, feature_cols)
    """
    # Resolve compute function
    if compute_fn is not None and compute_features_fn is not None:
        raise ValueError("Pass only one of compute_fn or compute_features_fn.")
    if compute_fn is None:
        compute_fn = compute_features_fn
    if compute_fn is None:
        compute_fn = compute_features_worldclass

    df_prices_n = normalize_cols(df_prices)
    df_prices_n = _ensure_dt_index(df_prices_n)

    missing = [c for c in BASE_PRICE_COLS if c not in df_prices_n.columns]
    if missing:
        raise ValueError(f"df_prices missing required columns: {missing}")

    df_daily = compute_fn(df_prices_n.copy())
    df_daily = normalize_cols(df_daily)
    df_daily = _ensure_dt_index(df_daily)

    # Ensure OHLCV still present
    for c in BASE_PRICE_COLS:
        if c not in df_daily.columns:
            df_daily[c] = df_prices_n[c]

    # Pick feature cols BEFORE sanitize
    feature_cols = _pick_feature_cols(df_daily)

    # --- NEW: sanitize features for ML consumers (torch, sklearn, etc.) ---
    # Options:
    #   - "ffill_bfill_zero": keeps all rows, removes warmup NaNs
    #   - "drop_rows": drops warmup period where rolling stats are undefined
    #   - "zero": simplest
    fill_method = "drop_rows"
    df_daily = _sanitize_features(df_daily, feature_cols, fill_method=fill_method)

    # Re-pick to be safe (especially if drop_rows is used)
    feature_cols = _pick_feature_cols(df_daily)

    return FeaturesResult(df_daily=df_daily, feature_cols=feature_cols)
