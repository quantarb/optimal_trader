# ============================================================
# modules/labels/strategy_solver.py
#
# Lightweight optimal trade solver (NO ortools dependency).
#
# Purpose:
#   - Generate "optimal" (oracle) trades for labeling / research.
#   - Used by data build pipeline to create events & labels.
#
# Approach:
#   - Classic "at most k transactions" dynamic programming (O(k*n)).
#   - Produces non-overlapping completed trades (entry < exit).
#   - Keeps your existing API: solve_longs_by_frequency / solve_shorts_by_frequency
#
# Note:
#   - Deterministic and fast.
#   - Does NOT attempt to handle constraints beyond non-overlap.
#
# NEW:
#   - Adds min_profit_pct filtering (default 2%) to reduce label count.
# ============================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd

Side = Literal["long", "short"]


@dataclass
class Trade:
    side: Side
    entry_row: pd.Series
    exit_row: pd.Series
    entry_price: float
    exit_price: float
    profit: float
    period_label: Optional[str] = None


def _resolve_freq(freq: str) -> Tuple[str, str]:
    """
    - freq_resolved: pandas Grouper freq
    - label_freq: Period freq label for display (W/M/Q/Y)
    """
    freq_map = {
        "W": "W",
        "M": "ME",
        "ME": "ME",
        "QE": "QE",
        "YE": "YE",
    }
    freq_resolved = freq_map.get(freq, freq)

    label_freq_map = {
        "W": "W",
        "M": "M",
        "ME": "M",
        "QE": "Q",
        "YE": "Y",
    }
    label_freq = label_freq_map.get(freq, "M")
    return freq_resolved, label_freq


def _pick_price_cols(side: Side, entry_price_col: Optional[str], exit_price_col: Optional[str]) -> Tuple[str, str]:
    if side == "long":
        return (entry_price_col or "high", exit_price_col or "low")
    return (entry_price_col or "low", exit_price_col or "high")


def _profit_pct(side: Side, entry: float, exit: float) -> float:
    """
    Profit percentage relative to entry price.
      - long : (exit - entry) / entry
      - short: (entry - exit) / entry
    """
    if entry <= 0:
        return 0.0
    if side == "long":
        return (exit - entry) / entry
    return (entry - exit) / entry


def solve_optimal_trades_generic(
    df: pd.DataFrame,
    k: int,
    side: Side = "long",
    entry_price_col: Optional[str] = None,
    exit_price_col: Optional[str] = None,
    min_profit_pct: float = 0.05,  # ✅ default 2% minimum profit threshold
) -> List[Trade]:
    """
    Solve up to k completed trades maximizing total profit.

    Implements classic DP for max profit with at most k transactions.
    Returns the actual trade list (entry/exit indices).

    For "short", we transform prices so that profit behaves like long.

    NEW: Filters out reconstructed trades whose profit_pct < min_profit_pct.
    """
    if k <= 0 or df is None or len(df) < 2:
        return []

    entry_col, exit_col = _pick_price_cols(side, entry_price_col, exit_price_col)
    col_map = {str(c).lower(): c for c in df.columns}

    def _resolve_col(col: str) -> str:
        key = str(col).lower()
        if key in col_map:
            return col_map[key]
        raise ValueError(f"Missing column '{col}' (needed by solver)")

    entry_col = _resolve_col(entry_col)
    exit_col = _resolve_col(exit_col)

    # We use entry prices and exit prices separately (matches earlier intent).
    entry_prices = df[entry_col].astype(float).values
    exit_prices = df[exit_col].astype(float).values
    n = len(df)

    # For shorting: profit = entry - exit. Convert to long equivalent by negating both.
    if side == "short":
        entry_prices = -entry_prices
        exit_prices = -exit_prices

    # DP arrays:
    # hold[t] = best value after entering t-th trade (holding) up to i
    # cash[t] = best profit after completing t trades up to i
    cash = [0.0] * (k + 1)
    hold = [float("-inf")] * (k + 1)

    # Snapshots to reconstruct
    cash_val = [[0.0] * (k + 1) for _ in range(n)]
    hold_val = [[float("-inf")] * (k + 1) for _ in range(n)]

    for i in range(n):
        e = float(entry_prices[i])
        x = float(exit_prices[i])

        for t in range(1, k + 1):
            # hold[t] = max(hold[t], cash[t-1] - entry_price[i])
            cand_hold = cash[t - 1] - e
            if cand_hold > hold[t]:
                hold[t] = cand_hold

            # cash[t] = max(cash[t], hold[t] + exit_price[i])
            cand_cash = hold[t] + x
            if cand_cash > cash[t]:
                cash[t] = cand_cash

        # snapshot values
        for t in range(k + 1):
            cash_val[i][t] = cash[t]
            hold_val[i][t] = hold[t]

    # Reconstruct trades by walking backwards on cash[k]
    trades_idx: List[Tuple[int, int]] = []
    t = k
    i = n - 1
    while t > 0 and i >= 1:
        if i > 0 and cash_val[i][t] == cash_val[i - 1][t]:
            i -= 1
            continue

        sell_price = float(exit_prices[i])
        target_hold = cash_val[i][t] - sell_price

        j = i - 1
        entry_idx = -1
        while j >= 0:
            if abs(hold_val[j][t] - target_hold) < 1e-9:
                # confirm hold achievable via buy at j:
                if abs((cash_val[j][t - 1] - float(entry_prices[j])) - hold_val[j][t]) < 1e-9:
                    entry_idx = j
                    break
            j -= 1

        if entry_idx < 0:
            break

        if entry_idx < i:
            trades_idx.append((entry_idx, i))
        i = entry_idx - 1
        t -= 1

    trades_idx.reverse()

    # Convert to Trade objects (with min_profit_pct filter applied)
    out: List[Trade] = []
    for entry_i, exit_i in trades_idx:
        raw_entry = float(df[entry_col].iloc[entry_i])
        raw_exit = float(df[exit_col].iloc[exit_i])

        profit_pct = _profit_pct(side, raw_entry, raw_exit)
        if profit_pct < float(min_profit_pct):
            continue  # ✅ drop small-profit trades to reduce label count

        entry_row = df.iloc[entry_i]
        exit_row = df.iloc[exit_i]

        if side == "long":
            profit = raw_exit - raw_entry
            entry_price = raw_entry
            exit_price = raw_exit
        else:
            # short profit: entry - exit
            profit = raw_entry - raw_exit
            entry_price = raw_entry
            exit_price = raw_exit

        out.append(
            Trade(
                side=side,
                entry_row=entry_row,
                exit_row=exit_row,
                entry_price=entry_price,
                exit_price=exit_price,
                profit=profit,
            )
        )

    return out


def solve_trades_by_frequency(
    df: pd.DataFrame,
    k: int,
    freq: str = "QE",
    side: Side = "long",
    min_profit_pct: float = 0.02,  # ✅ default 2% here too
) -> List[Dict]:
    """
    Split the dataframe into periods and solve up to k trades per period.
    Returns list[dict] to match your existing downstream usage.

    NEW: forwards min_profit_pct into solve_optimal_trades_generic.
    """
    if df is None or df.empty:
        return []

    freq_resolved, label_freq = _resolve_freq(freq)

    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            dfi = df.copy()
            dfi["date"] = pd.to_datetime(dfi["date"], errors="coerce")
            dfi = dfi.set_index("date")
        else:
            raise ValueError("solve_trades_by_frequency requires a DatetimeIndex or a 'date' column")
    else:
        dfi = df

    dfi = dfi.sort_index()
    all_trades: List[Dict] = []

    for period, g in dfi.groupby(pd.Grouper(freq=freq_resolved)):
        if g is None or len(g) < 2:
            continue

        try:
            if hasattr(period, "to_timestamp"):
                ts = period.to_timestamp()
            else:
                ts = pd.to_datetime(period)
            period_label = f"{label_freq}:{ts.date()}"
        except Exception:
            period_label = str(period)

        trades = solve_optimal_trades_generic(
            g,
            k=k,
            side=side,
            min_profit_pct=min_profit_pct,
        )

        for tr in trades:
            all_trades.append(
                {
                    "side": tr.side,
                    "entry_row": tr.entry_row,
                    "exit_row": tr.exit_row,
                    "entry_price": tr.entry_price,
                    "exit_price": tr.exit_price,
                    "profit": tr.profit,
                    "period_label": period_label,
                }
            )

    return all_trades


def solve_longs_by_frequency(
    df: pd.DataFrame,
    k: int,
    freq: str = "QE",
    min_profit_pct: float = 0.01,
) -> List[Dict]:
    return solve_trades_by_frequency(df, k=k, freq=freq, side="long", min_profit_pct=min_profit_pct)


def solve_shorts_by_frequency(
    df: pd.DataFrame,
    k: int,
    freq: str = "QE",
    min_profit_pct: float = 0.01,
) -> List[Dict]:
    return solve_trades_by_frequency(df, k=k, freq=freq, side="short", min_profit_pct=min_profit_pct)
