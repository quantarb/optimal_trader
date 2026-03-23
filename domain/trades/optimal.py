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


@dataclass
class _TradeCandidate:
    side: Side
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    profit: float


@dataclass(frozen=True)
class _TradePathState:
    value: float
    trades: tuple[_TradeCandidate, ...]


@dataclass(frozen=True)
class _HoldState:
    value: float
    entry_idx: int
    entry_price: float
    base_trades: tuple[_TradeCandidate, ...]


def _resolve_freq(freq: str) -> Tuple[str, str]:
    freq_map = {"W": "W", "M": "ME", "ME": "ME", "QE": "QE", "YE": "YE"}
    label_freq_map = {"W": "W", "M": "M", "ME": "M", "QE": "Q", "YE": "Y"}
    return freq_map.get(freq, freq), label_freq_map.get(freq, "M")


def _pick_price_cols(side: Side, entry_price_col: Optional[str], exit_price_col: Optional[str]) -> Tuple[str, str]:
    if side == "long":
        return (entry_price_col or "high", exit_price_col or "low")
    return (entry_price_col or "low", exit_price_col or "high")


def _profit_pct(side: Side, entry: float, exit: float) -> float:
    if entry <= 0:
        return 0.0
    if side == "long":
        return (exit - entry) / entry
    return (entry - exit) / entry


def _resolve_required_col(df: pd.DataFrame, col: str) -> str:
    col_map = {str(c).lower(): c for c in df.columns}
    key = str(col).lower()
    if key in col_map:
        return col_map[key]
    raise ValueError(f"Missing column '{col}' (needed by solver)")


def solve_optimal_joint_trades_generic(
    df: pd.DataFrame,
    k: int,
    *,
    long_entry_price_col: Optional[str] = None,
    long_exit_price_col: Optional[str] = None,
    short_entry_price_col: Optional[str] = None,
    short_exit_price_col: Optional[str] = None,
    min_profit_pct: float = 0.01,
) -> List[Trade]:
    if k <= 0 or df is None or len(df) < 2:
        return []

    le_col = _resolve_required_col(df, long_entry_price_col or "high")
    lx_col = _resolve_required_col(df, long_exit_price_col or "low")
    se_col = _resolve_required_col(df, short_entry_price_col or "low")
    sx_col = _resolve_required_col(df, short_exit_price_col or "high")

    long_entry = df[le_col].astype(float).to_numpy()
    long_exit = df[lx_col].astype(float).to_numpy()
    short_entry = df[se_col].astype(float).to_numpy()
    short_exit = df[sx_col].astype(float).to_numpy()
    n = len(df)

    min_profit = float(min_profit_pct)
    cash: List[_TradePathState] = [_TradePathState(0.0, ()) for _ in range(k + 1)]
    long_hold: List[_HoldState | None] = [None] * (k + 1)
    short_hold: List[_HoldState | None] = [None] * (k + 1)

    for i in range(n):
        le = float(long_entry[i])
        lx = float(long_exit[i])
        se = float(short_entry[i])
        sx = float(short_exit[i])

        prev_cash = list(cash)
        prev_long_hold = list(long_hold)
        prev_short_hold = list(short_hold)

        for trade_count in range(1, k + 1):
            best_cash = prev_cash[trade_count]

            long_state = prev_long_hold[trade_count]
            if long_state is not None and long_state.entry_idx < i:
                long_pct = _profit_pct("long", long_state.entry_price, lx)
                if long_pct >= min_profit:
                    candidate_value = float(long_state.value + lx)
                    if candidate_value > float(best_cash.value) + 1e-12:
                        best_cash = _TradePathState(
                            candidate_value,
                            long_state.base_trades
                            + (
                                _TradeCandidate(
                                    side="long",
                                    entry_idx=int(long_state.entry_idx),
                                    exit_idx=i,
                                    entry_price=float(long_state.entry_price),
                                    exit_price=float(lx),
                                    profit=float(lx - long_state.entry_price),
                                ),
                            ),
                        )

            short_state = prev_short_hold[trade_count]
            if short_state is not None and short_state.entry_idx < i:
                short_pct = _profit_pct("short", short_state.entry_price, sx)
                if short_pct >= min_profit:
                    candidate_value = float(short_state.value - sx)
                    if candidate_value > float(best_cash.value) + 1e-12:
                        best_cash = _TradePathState(
                            candidate_value,
                            short_state.base_trades
                            + (
                                _TradeCandidate(
                                    side="short",
                                    entry_idx=int(short_state.entry_idx),
                                    exit_idx=i,
                                    entry_price=float(short_state.entry_price),
                                    exit_price=float(sx),
                                    profit=float(short_state.entry_price - sx),
                                ),
                            ),
                        )

            cash[trade_count] = best_cash

        for trade_count in range(1, k + 1):
            base_state = prev_cash[trade_count - 1]

            if le > 0:
                candidate_hold_value = float(base_state.value - le)
                current_long = prev_long_hold[trade_count]
                if current_long is None or candidate_hold_value > float(current_long.value) + 1e-12:
                    long_hold[trade_count] = _HoldState(
                        value=candidate_hold_value,
                        entry_idx=i,
                        entry_price=float(le),
                        base_trades=base_state.trades,
                    )
                else:
                    long_hold[trade_count] = current_long

            if se > 0:
                candidate_hold_value = float(base_state.value + se)
                current_short = prev_short_hold[trade_count]
                if current_short is None or candidate_hold_value > float(current_short.value) + 1e-12:
                    short_hold[trade_count] = _HoldState(
                        value=candidate_hold_value,
                        entry_idx=i,
                        entry_price=float(se),
                        base_trades=base_state.trades,
                    )
                else:
                    short_hold[trade_count] = current_short

    chosen = list(max(cash, key=lambda state: float(state.value)).trades)
    if not chosen:
        return []

    out: List[Trade] = []
    for candidate in chosen:
        out.append(
            Trade(
                side=candidate.side,
                entry_row=df.iloc[candidate.entry_idx],
                exit_row=df.iloc[candidate.exit_idx],
                entry_price=float(candidate.entry_price),
                exit_price=float(candidate.exit_price),
                profit=float(candidate.profit),
            )
        )
    return out


def solve_optimal_joint_trade_sequence_generic(
    df: pd.DataFrame,
    *,
    long_entry_price_col: Optional[str] = None,
    long_exit_price_col: Optional[str] = None,
    short_entry_price_col: Optional[str] = None,
    short_exit_price_col: Optional[str] = None,
    min_profit_pct: float = 0.01,
) -> List[Trade]:
    """Find the best long/short action sequence through time without a per-period k cap."""

    if df is None or len(df) < 2:
        return []

    le_col = _resolve_required_col(df, long_entry_price_col or "high")
    lx_col = _resolve_required_col(df, long_exit_price_col or "low")
    se_col = _resolve_required_col(df, short_entry_price_col or "low")
    sx_col = _resolve_required_col(df, short_exit_price_col or "high")

    long_entry = df[le_col].astype(float).to_numpy()
    long_exit = df[lx_col].astype(float).to_numpy()
    short_entry = df[se_col].astype(float).to_numpy()
    short_exit = df[sx_col].astype(float).to_numpy()
    n = len(df)

    min_profit = float(min_profit_pct)
    cash = _TradePathState(0.0, ())
    long_hold: _HoldState | None = None
    short_hold: _HoldState | None = None

    for i in range(n):
        le = float(long_entry[i])
        lx = float(long_exit[i])
        se = float(short_entry[i])
        sx = float(short_exit[i])

        prev_cash = cash
        prev_long_hold = long_hold
        prev_short_hold = short_hold

        best_cash = prev_cash

        if prev_long_hold is not None and prev_long_hold.entry_idx < i:
            long_pct = _profit_pct("long", prev_long_hold.entry_price, lx)
            if long_pct >= min_profit:
                candidate_value = float(prev_long_hold.value + lx)
                if candidate_value > float(best_cash.value) + 1e-12:
                    best_cash = _TradePathState(
                        candidate_value,
                        prev_long_hold.base_trades
                        + (
                            _TradeCandidate(
                                side="long",
                                entry_idx=int(prev_long_hold.entry_idx),
                                exit_idx=i,
                                entry_price=float(prev_long_hold.entry_price),
                                exit_price=float(lx),
                                profit=float(lx - prev_long_hold.entry_price),
                            ),
                        ),
                    )

        if prev_short_hold is not None and prev_short_hold.entry_idx < i:
            short_pct = _profit_pct("short", prev_short_hold.entry_price, sx)
            if short_pct >= min_profit:
                candidate_value = float(prev_short_hold.value - sx)
                if candidate_value > float(best_cash.value) + 1e-12:
                    best_cash = _TradePathState(
                        candidate_value,
                        prev_short_hold.base_trades
                        + (
                            _TradeCandidate(
                                side="short",
                                entry_idx=int(prev_short_hold.entry_idx),
                                exit_idx=i,
                                entry_price=float(prev_short_hold.entry_price),
                                exit_price=float(sx),
                                profit=float(prev_short_hold.entry_price - sx),
                            ),
                        ),
                    )

        cash = best_cash

        if le > 0:
            candidate_hold_value = float(prev_cash.value - le)
            if prev_long_hold is None or candidate_hold_value > float(prev_long_hold.value) + 1e-12:
                long_hold = _HoldState(
                    value=candidate_hold_value,
                    entry_idx=i,
                    entry_price=float(le),
                    base_trades=prev_cash.trades,
                )
            else:
                long_hold = prev_long_hold

        if se > 0:
            candidate_hold_value = float(prev_cash.value + se)
            if prev_short_hold is None or candidate_hold_value > float(prev_short_hold.value) + 1e-12:
                short_hold = _HoldState(
                    value=candidate_hold_value,
                    entry_idx=i,
                    entry_price=float(se),
                    base_trades=prev_cash.trades,
                )
            else:
                short_hold = prev_short_hold

    chosen = list(cash.trades)
    if not chosen:
        return []

    out: List[Trade] = []
    for candidate in chosen:
        out.append(
            Trade(
                side=candidate.side,
                entry_row=df.iloc[candidate.entry_idx],
                exit_row=df.iloc[candidate.exit_idx],
                entry_price=float(candidate.entry_price),
                exit_price=float(candidate.exit_price),
                profit=float(candidate.profit),
            )
        )
    return out


def solve_optimal_trades_generic(
    df: pd.DataFrame,
    k: int,
    side: Side = "long",
    entry_price_col: Optional[str] = None,
    exit_price_col: Optional[str] = None,
    min_profit_pct: float = 0.05,
) -> List[Trade]:
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

    entry_prices = df[entry_col].astype(float).values
    exit_prices = df[exit_col].astype(float).values
    n = len(df)
    if side == "short":
        entry_prices = -entry_prices
        exit_prices = -exit_prices

    cash = [0.0] * (k + 1)
    hold = [float("-inf")] * (k + 1)
    cash_val = [[0.0] * (k + 1) for _ in range(n)]
    hold_val = [[float("-inf")] * (k + 1) for _ in range(n)]

    for i in range(n):
        entry_price = float(entry_prices[i])
        exit_price = float(exit_prices[i])
        for trade_count in range(1, k + 1):
            cand_hold = cash[trade_count - 1] - entry_price
            if cand_hold > hold[trade_count]:
                hold[trade_count] = cand_hold
            cand_cash = hold[trade_count] + exit_price
            if cand_cash > cash[trade_count]:
                cash[trade_count] = cand_cash
        for trade_count in range(k + 1):
            cash_val[i][trade_count] = cash[trade_count]
            hold_val[i][trade_count] = hold[trade_count]

    trades_idx: List[Tuple[int, int]] = []
    trade_count = k
    i = n - 1
    while trade_count > 0 and i >= 1:
        if i > 0 and cash_val[i][trade_count] == cash_val[i - 1][trade_count]:
            i -= 1
            continue

        sell_price = float(exit_prices[i])
        target_hold = cash_val[i][trade_count] - sell_price

        j = i - 1
        entry_idx = -1
        while j >= 0:
            if abs(hold_val[j][trade_count] - target_hold) < 1e-9:
                if abs((cash_val[j][trade_count - 1] - float(entry_prices[j])) - hold_val[j][trade_count]) < 1e-9:
                    entry_idx = j
                    break
            j -= 1

        if entry_idx < 0:
            break
        if entry_idx < i:
            trades_idx.append((entry_idx, i))
        i = entry_idx - 1
        trade_count -= 1

    trades_idx.reverse()
    out: List[Trade] = []
    for entry_i, exit_i in trades_idx:
        raw_entry = float(df[entry_col].iloc[entry_i])
        raw_exit = float(df[exit_col].iloc[exit_i])
        profit_pct = _profit_pct(side, raw_entry, raw_exit)
        if profit_pct < float(min_profit_pct):
            continue
        if side == "long":
            profit = raw_exit - raw_entry
            entry_price = raw_entry
            exit_price = raw_exit
        else:
            profit = raw_entry - raw_exit
            entry_price = raw_entry
            exit_price = raw_exit
        out.append(
            Trade(
                side=side,
                entry_row=df.iloc[entry_i],
                exit_row=df.iloc[exit_i],
                entry_price=entry_price,
                exit_price=exit_price,
                profit=profit,
            )
        )
    return out


def solve_joint_trades_by_frequency(
    df: pd.DataFrame,
    k: int,
    freq: str = "QE",
    min_profit_pct: float = 0.01,
    long_entry_price_col: Optional[str] = None,
    long_exit_price_col: Optional[str] = None,
    short_entry_price_col: Optional[str] = None,
    short_exit_price_col: Optional[str] = None,
) -> List[Dict]:
    if df is None or df.empty:
        return []
    freq_resolved, label_freq = _resolve_freq(freq)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            dfi = df.copy()
            dfi["date"] = pd.to_datetime(dfi["date"], errors="coerce")
            dfi = dfi.set_index("date")
        else:
            raise ValueError("solve_joint_trades_by_frequency requires a DatetimeIndex or a 'date' column")
    else:
        dfi = df

    dfi = dfi.sort_index()
    all_trades: List[Dict] = []
    for period, group in dfi.groupby(pd.Grouper(freq=freq_resolved)):
        if group is None or len(group) < 2:
            continue
        try:
            ts = period.to_timestamp() if hasattr(period, "to_timestamp") else pd.to_datetime(period)
            period_label = f"{label_freq}:{ts.date()}"
        except Exception:
            period_label = str(period)
        trades = solve_optimal_joint_trades_generic(
            group,
            k=k,
            min_profit_pct=min_profit_pct,
            long_entry_price_col=long_entry_price_col,
            long_exit_price_col=long_exit_price_col,
            short_entry_price_col=short_entry_price_col,
            short_exit_price_col=short_exit_price_col,
        )
        for trade in trades:
            all_trades.append(
                {
                    "side": trade.side,
                    "entry_row": trade.entry_row,
                    "exit_row": trade.exit_row,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "profit": trade.profit,
                    "period_label": period_label,
                }
            )
    return all_trades


def solve_joint_trade_sequence_by_frequency(
    df: pd.DataFrame,
    freq: str = "QE",
    min_profit_pct: float = 0.01,
    long_entry_price_col: Optional[str] = None,
    long_exit_price_col: Optional[str] = None,
    short_entry_price_col: Optional[str] = None,
    short_exit_price_col: Optional[str] = None,
) -> List[Dict]:
    if df is None or df.empty:
        return []
    freq_resolved, label_freq = _resolve_freq(freq)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            dfi = df.copy()
            dfi["date"] = pd.to_datetime(dfi["date"], errors="coerce")
            dfi = dfi.set_index("date")
        else:
            raise ValueError("solve_joint_trade_sequence_by_frequency requires a DatetimeIndex or a 'date' column")
    else:
        dfi = df

    dfi = dfi.sort_index()
    all_trades: List[Dict] = []
    for period, group in dfi.groupby(pd.Grouper(freq=freq_resolved)):
        if group is None or len(group) < 2:
            continue
        try:
            ts = period.to_timestamp() if hasattr(period, "to_timestamp") else pd.to_datetime(period)
            period_label = f"{label_freq}:{ts.date()}"
        except Exception:
            period_label = str(period)
        trades = solve_optimal_joint_trade_sequence_generic(
            group,
            min_profit_pct=min_profit_pct,
            long_entry_price_col=long_entry_price_col,
            long_exit_price_col=long_exit_price_col,
            short_entry_price_col=short_entry_price_col,
            short_exit_price_col=short_exit_price_col,
        )
        for trade in trades:
            all_trades.append(
                {
                    "side": trade.side,
                    "entry_row": trade.entry_row,
                    "exit_row": trade.exit_row,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "profit": trade.profit,
                    "period_label": period_label,
                }
            )
    return all_trades


def solve_trades_by_frequency(
    df: pd.DataFrame,
    k: int,
    freq: str = "QE",
    side: Side = "long",
    min_profit_pct: float = 0.02,
    entry_price_col: Optional[str] = None,
    exit_price_col: Optional[str] = None,
) -> List[Dict]:
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
    for period, group in dfi.groupby(pd.Grouper(freq=freq_resolved)):
        if group is None or len(group) < 2:
            continue
        try:
            ts = period.to_timestamp() if hasattr(period, "to_timestamp") else pd.to_datetime(period)
            period_label = f"{label_freq}:{ts.date()}"
        except Exception:
            period_label = str(period)
        trades = solve_optimal_trades_generic(
            group,
            k=k,
            side=side,
            min_profit_pct=min_profit_pct,
            entry_price_col=entry_price_col,
            exit_price_col=exit_price_col,
        )
        for trade in trades:
            all_trades.append(
                {
                    "side": trade.side,
                    "entry_row": trade.entry_row,
                    "exit_row": trade.exit_row,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "profit": trade.profit,
                    "period_label": period_label,
                }
            )
    return all_trades


def solve_longs_by_frequency(
    df: pd.DataFrame,
    k: int,
    freq: str = "QE",
    min_profit_pct: float = 0.01,
    entry_price_col: Optional[str] = None,
    exit_price_col: Optional[str] = None,
) -> List[Dict]:
    return solve_trades_by_frequency(
        df,
        k=k,
        freq=freq,
        side="long",
        min_profit_pct=min_profit_pct,
        entry_price_col=entry_price_col,
        exit_price_col=exit_price_col,
    )


def solve_shorts_by_frequency(
    df: pd.DataFrame,
    k: int,
    freq: str = "QE",
    min_profit_pct: float = 0.01,
    entry_price_col: Optional[str] = None,
    exit_price_col: Optional[str] = None,
) -> List[Dict]:
    return solve_trades_by_frequency(
        df,
        k=k,
        freq=freq,
        side="short",
        min_profit_pct=min_profit_pct,
        entry_price_col=entry_price_col,
        exit_price_col=exit_price_col,
    )
