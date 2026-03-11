# modules/data/quality.py
# ============================================================
# Data quality checks + cleaning for daily OHLCV price bars.
#
# Why:
# - Your backtests were exploding due to a handful of symbols having
#   insane price scale / split artifacts (e.g. BMNR/RCAT showing 1800 -> 9000).
# - Even if only ~0.05% of bars are bad, if they land in top-k selection,
#   they can dominate portfolio returns.
#
# Design:
# - Run *before* feature computation + labeling so garbage prices don't leak.
# - Work on OBSERVED rows only (no NaN grid artifacts).
# - Two-level handling:
#   (1) Bar-level: drop invalid prices and insane 1d moves.
#   (2) Symbol-level: drop symbols whose bad-bar fraction exceeds thresholds.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataQualityConfig:
    # Bar-level checks
    price_floor: float = 0.01               # prices <= this are considered invalid
    max_abs_ret_1d: float = 2.0             # abs(1d return) > 2.0 (200%) is considered insane

    # Symbol-level rejection thresholds (fractions over VALID rows)
    max_bad_price_frac: float = 0.01        # drop symbol if >1% bars have invalid price
    max_bad_ret_frac: float = 0.01          # drop symbol if >1% bars have insane returns

    # If you have both close and adj_close and want to use adj_close as the trading price
    prefer_adj_close: bool = False

    # Behavior
    drop_bad_bars: bool = True              # drop bad bars from the price df


def _as_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    if not isinstance(out.index, pd.DatetimeIndex):
        if "date" in out.columns:
            out = out.copy()
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out = out.set_index("date")
        else:
            raise ValueError("prices df must have a DatetimeIndex or a 'date' column")
    out = out.sort_index()
    return out


def _choose_price_col(df: pd.DataFrame, price_col: str, prefer_adj_close: bool) -> str:
    if prefer_adj_close and price_col == "close" and "adj_close" in df.columns:
        return "adj_close"
    return price_col


def assess_and_clean_prices_daily(
    df_prices: pd.DataFrame,
    *,
    symbol: str,
    price_col: str = "close",
    cfg: Optional[DataQualityConfig] = None,
    debug: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return (clean_df, stats). Raises RuntimeError if symbol should be dropped."""
    cfg = cfg or DataQualityConfig()
    df = _as_dt_index(df_prices)

    px_col = _choose_price_col(df, price_col, cfg.prefer_adj_close)
    if px_col not in df.columns:
        raise RuntimeError(f"missing price column '{px_col}'")

    px = pd.to_numeric(df[px_col], errors="coerce").astype(float)

    # VALID rows: where we actually have a price observation (not NaN)
    valid_mask = px.notna() & np.isfinite(px.to_numpy(dtype=float, copy=False))
    if valid_mask.sum() < 50:
        raise RuntimeError("too few valid price bars")

    # Bad price bars
    bad_price = valid_mask & ((px <= float(cfg.price_floor)) | (~np.isfinite(px)))

    # 1d returns (only meaningful where both t and t-1 exist)
    ret_1d = px.pct_change()
    ret_valid = valid_mask & ret_1d.notna() & np.isfinite(ret_1d.to_numpy(dtype=float, copy=False))
    bad_ret = ret_valid & (ret_1d.abs() > float(cfg.max_abs_ret_1d))

    # Fractions over VALID rows only
    valid_n = int(valid_mask.sum())
    bad_price_n = int(bad_price.sum())
    bad_ret_n = int(bad_ret.sum())

    bad_price_frac = bad_price_n / max(valid_n, 1)
    bad_ret_frac = bad_ret_n / max(valid_n, 1)

    stats: Dict[str, Any] = dict(
        symbol=symbol,
        price_col=px_col,
        valid_bars=valid_n,
        bad_price_bars=bad_price_n,
        bad_ret_bars=bad_ret_n,
        bad_price_frac=bad_price_frac,
        bad_ret_frac=bad_ret_frac,
    )

    # Symbol rejection
    if bad_price_frac > float(cfg.max_bad_price_frac):
        raise RuntimeError(
            f"symbol rejected: bad_price_frac={bad_price_frac:.6f} > {cfg.max_bad_price_frac}"
        )
    if bad_ret_frac > float(cfg.max_bad_ret_frac):
        raise RuntimeError(
            f"symbol rejected: bad_ret_frac={bad_ret_frac:.6f} > {cfg.max_bad_ret_frac}"
        )

    # Bar-level cleaning
    if cfg.drop_bad_bars:
        bad_any = (bad_price | bad_ret)
        if bad_any.any():
            df = df.loc[~bad_any].copy()

    if debug:
        print("\n" + "=" * 60)
        print(f"[DATA QUALITY] {symbol} ({px_col})")
        print("valid_bars:", stats["valid_bars"])
        print("bad_price_bars:", stats["bad_price_bars"], f"({stats['bad_price_frac']:.6f})")
        print("bad_ret_bars:", stats["bad_ret_bars"], f"({stats['bad_ret_frac']:.6f})")

        if bad_ret.any():
            worst = pd.DataFrame(
                {"price": px, "ret_1d": ret_1d, "abs_ret": ret_1d.abs()}
            ).loc[bad_ret].sort_values("abs_ret", ascending=False).head(10)
            print("\nTop insane 1d returns (sample):")
            print(worst)

    return df, stats
