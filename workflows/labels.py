from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from data.historical_prices import load_adjusted_price_frames
from domain.labels.specs import LabelBuildSpec
from domain.trades import (
    TradeGenerationResult,
    apply_trade_deduplication,
    build_label_rows_from_completed_trades,
    build_label_statistics,
    solve_joint_trades_by_frequency,
    trade_return_pct,
)
from fmp.models import Symbol


@dataclass(frozen=True)
class OracleLabelWorkflowResult:
    """Canonical label rows and diagnostics from oracle trade generation."""

    label_rows: list[dict[str, Any]]
    statistics: dict[str, Any]
    completed_trades: list[dict[str, Any]]


def build_trade_results(
    symbols: list[str],
    *,
    spec: LabelBuildSpec,
    progress_callback=None,
    price_frames: dict[str, pd.DataFrame] | None = None,
) -> TradeGenerationResult:
    """Build raw oracle trade candidates for a symbol set."""

    trade_rows: list[dict[str, Any]] = []
    completed_trades: list[dict[str, Any]] = []
    normalized_symbols = [str(sym).strip().upper() for sym in list(symbols or []) if str(sym).strip()]
    symbol_map = {
        str(symbol.symbol).strip().upper(): symbol
        for symbol in Symbol.objects.filter(symbol__in=normalized_symbols).only("id", "symbol")
    }
    if price_frames is None:
        price_frames = load_adjusted_price_frames(normalized_symbols, start_date=spec.start_date, end_date=spec.end_date)
    total_symbols = len(normalized_symbols)
    if callable(progress_callback):
        progress_callback(completed=0, total=total_symbols, current_symbol="")

    for idx, symbol_code in enumerate(normalized_symbols, start=1):
        if callable(progress_callback):
            progress_callback(completed=max(0, idx - 1), total=total_symbols, current_symbol=symbol_code)
        symbol_obj = symbol_map.get(symbol_code)
        if not symbol_obj:
            symbol_obj = Symbol.objects.create(symbol=symbol_code)
        daily_prices = _load_adjusted_daily(
            symbol_obj,
            price_frames=price_frames,
            start_date=spec.start_date,
            end_date=spec.end_date,
        )
        if daily_prices.empty:
            if callable(progress_callback):
                progress_callback(completed=idx, total=total_symbols, current_symbol=symbol_code)
            continue

        for freq, ks in spec.k_params.items():
            for k in ks:
                joint_trades = solve_joint_trades_by_frequency(
                    daily_prices,
                    k=int(k),
                    freq=freq,
                    min_profit_pct=spec.min_profit_pct,
                    long_entry_price_col=spec.buy_execution,
                    long_exit_price_col=spec.sell_execution,
                    short_entry_price_col=spec.short_execution,
                    short_exit_price_col=spec.cover_execution,
                )
                for trade in joint_trades:
                    entry_dt = pd.Timestamp(trade["entry_row"].name)
                    exit_dt = pd.Timestamp(trade["exit_row"].name)
                    entry_px = float(trade["entry_price"])
                    exit_px = float(trade["exit_price"])
                    side = str(trade.get("side") or "").strip().lower()
                    if side not in {"long", "short"}:
                        continue
                    ret_dec = trade_return_pct(side, entry_px, exit_px)
                    trade_rows.append(
                        {
                            "symbol": symbol_code,
                            "side": side,
                            "freq": freq,
                            "k": int(k),
                            "entry_date": entry_dt.strftime("%Y-%m-%d"),
                            "exit_date": exit_dt.strftime("%Y-%m-%d"),
                            "entry_px": f"{entry_px:,.4f}",
                            "exit_px": f"{exit_px:,.4f}",
                            "ret_pct": f"{ret_dec * 100:.2f}%",
                        }
                    )
                    completed_trades.append(
                        {
                            "symbol": symbol_code,
                            "side": side,
                            "freq": freq,
                            "k": int(k),
                            "entry_date": entry_dt.strftime("%Y-%m-%d"),
                            "exit_date": exit_dt.strftime("%Y-%m-%d"),
                            "entry_px": f"{entry_px:,.4f}",
                            "exit_px": f"{exit_px:,.4f}",
                            "ret_dec": ret_dec,
                            "hold_days": int((exit_dt - entry_dt).days),
                        }
                    )
        if callable(progress_callback):
            progress_callback(completed=idx, total=total_symbols, current_symbol=symbol_code)
    return TradeGenerationResult(trade_rows=trade_rows, completed_trades=completed_trades)


def build_oracle_labels(
    symbols: list[str],
    *,
    spec: LabelBuildSpec,
    progress_callback=None,
    price_frames: dict[str, pd.DataFrame] | None = None,
) -> OracleLabelWorkflowResult:
    """Build canonical label rows and summary statistics from oracle trades."""

    generated = build_trade_results(symbols, spec=spec, progress_callback=progress_callback, price_frames=price_frames)
    _, completed = apply_trade_deduplication(generated.trade_rows, generated.completed_trades, mode=spec.trade_dedup_mode)
    label_rows = build_label_rows_from_completed_trades(completed)
    return OracleLabelWorkflowResult(
        label_rows=label_rows,
        statistics=build_label_statistics(label_rows),
        completed_trades=completed,
    )

def _load_adjusted_daily(
    symbol_obj: Symbol,
    *,
    price_frames: dict[str, pd.DataFrame] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    symbol_code = str(symbol_obj.symbol).strip().upper()
    if price_frames is not None:
        cached = price_frames.get(symbol_code)
        if cached is not None:
            return cached
    frames = load_adjusted_price_frames([symbol_code], start_date=start_date, end_date=end_date)
    frame = frames.get(symbol_code)
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    return out[~out.index.duplicated(keep="last")]
