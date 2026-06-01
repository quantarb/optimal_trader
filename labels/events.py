# modules/labels/events.py
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, Optional, List, Any, Union
import pandas as pd
import numpy as np

from utils.normalize import normalize_cols
from domain.labels.directional import add_binary_classification_labels, add_action_labels
from labels.ranking import add_rank_regression_labels


def _get_price_series(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """Extracts a valid price series from the dataframe for return calculations."""
    df_n = normalize_cols(df)
    col_map = {str(c).lower(): c for c in df_n.columns}

    requested = str(price_col).lower()
    if requested in col_map:
        return df_n[col_map[requested]]

    for c in ["close", "adj_close", "adjclose", "price"]:
        key = c.lower()
        if key in col_map:
            return df_n[col_map[key]]
    raise ValueError(f"Could not find a usable price column. Available: {list(df_n.columns)}")


def _net_return_from_gross(gross_return: float, fee_bps: float, slippage_bps: float) -> float:
    """Calculates net return after subtracting round-trip transaction costs."""
    total_cost = 2.0 * (float(fee_bps) + float(slippage_bps)) / 10000.0
    return float(gross_return) - total_cost


def deduplicate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes deduplication to ensure exactly one signal per date/symbol/side.
    Keeps the signal with the highest trade_return.
    """
    if df.empty:
        return df

    idx_names = list(df.index.names)
    tmp = df.reset_index()

    subset = ["date", "symbol", "side"]
    if "label" in tmp.columns:
        subset.append("label")

    if "trade_return" in tmp.columns:
        # Sort by return descending so drop_duplicates keeps the highest-return row
        tmp = tmp.sort_values("trade_return", ascending=False)

    unique = tmp.drop_duplicates(subset=subset, keep="first")
    return unique.set_index(idx_names).sort_index()


def generate_optimal_events(
        df_daily: pd.DataFrame,
        solve_longs_by_frequency_fn: Callable,
        k_params: Dict[str, Union[int, List[int]]],
        solve_shorts_by_frequency_fn: Optional[Callable] = None,
        solve_joint_by_frequency_fn: Optional[Callable] = None,
        *,
        price_col: str = "close",
        fee_bps: float = 0.0,
        slippage_bps: float = 0.0,
) -> pd.DataFrame:
    """Generates oracle events supporting multiple k-values per frequency."""
    df = normalize_cols(df_daily)
    px = _get_price_series(df, price_col=price_col)

    if not px.index.is_unique:
        px = px.groupby(level=0).last()

    rows: List[dict] = []
    trade_counter = 0

    def _safe_loc_price(ts: pd.Timestamp) -> float:
        if ts not in px.index:
            prev = px.index[px.index <= ts]
            if len(prev) == 0:
                raise KeyError(f"No price available on or before {ts}")
            ts = prev[-1]
        return float(px.loc[ts])

    def _process_side(side: str, solve_fn: Callable):
        nonlocal trade_counter
        for freq, k_val in k_params.items():
            # Support both single integers and lists for k_params
            ks = [k_val] if isinstance(k_val, int) else k_val
            for k in ks:
                trades = solve_fn(df, k=k, freq=freq)
                for t in trades:
                    trade_counter += 1
                    trade_id = f"{side}:{freq}:k{k}:{trade_counter}"
                    entry_dt = pd.Timestamp(t["entry_row"].name)
                    exit_dt = pd.Timestamp(t["exit_row"].name)
                    entry_px = _safe_loc_price(entry_dt)
                    exit_px = _safe_loc_price(exit_dt)

                    # Calculate side-appropriate returns
                    gross_r = (exit_px - entry_px) / entry_px if side == "long" else (entry_px - exit_px) / entry_px
                    net_r = _net_return_from_gross(gross_r, fee_bps=fee_bps, slippage_bps=slippage_bps)

                    payload = dict(
                        side=side,
                        horizon=f"{freq}_k{k}",
                        trade_id=trade_id,
                        entry_date=entry_dt,
                        exit_date=exit_dt,
                        entry_px=float(entry_px),
                        exit_px=float(exit_px),
                        trade_duration_days=int((exit_dt - entry_dt).days),
                        trade_return=float(net_r),
                    )
                    rows.append({"date": entry_dt, "event": "entry", **payload})
                    rows.append({"date": exit_dt, "event": "exit", **payload})

    if solve_joint_by_frequency_fn is not None:
        for freq, k_val in k_params.items():
            ks = [k_val] if isinstance(k_val, int) else k_val
            for k in ks:
                trades = solve_joint_by_frequency_fn(df, k=k, freq=freq)
                for t in trades:
                    side = str(t.get("side") or "").strip().lower()
                    if side not in {"long", "short"}:
                        continue
                    trade_counter += 1
                    trade_id = f"{side}:{freq}:k{k}:{trade_counter}"
                    entry_dt = pd.Timestamp(t["entry_row"].name)
                    exit_dt = pd.Timestamp(t["exit_row"].name)
                    entry_px = _safe_loc_price(entry_dt)
                    exit_px = _safe_loc_price(exit_dt)

                    gross_r = (exit_px - entry_px) / entry_px if side == "long" else (entry_px - exit_px) / entry_px
                    net_r = _net_return_from_gross(gross_r, fee_bps=fee_bps, slippage_bps=slippage_bps)

                    payload = dict(
                        side=side,
                        horizon=f"{freq}_k{k}",
                        trade_id=trade_id,
                        entry_date=entry_dt,
                        exit_date=exit_dt,
                        entry_px=float(entry_px),
                        exit_px=float(exit_px),
                        trade_duration_days=int((exit_dt - entry_dt).days),
                        trade_return=float(net_r),
                    )
                    rows.append({"date": entry_dt, "event": "entry", **payload})
                    rows.append({"date": exit_dt, "event": "exit", **payload})
    else:
        _process_side("long", solve_longs_by_frequency_fn)
        if solve_shorts_by_frequency_fn:
            _process_side("short", solve_shorts_by_frequency_fn)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("date").sort_index()


def _build_one_symbol_labels(args: tuple) -> tuple:
    """Process a single symbol: events → actions → labels. Top-level for picklability."""
    (symbol, df_daily, k_params, execution_params, weighting,
     price_col, fee_bps, slippage_bps) = args

    from labels.strategy_solver import (
        solve_joint_trades_by_frequency,
        solve_longs_by_frequency,
        solve_shorts_by_frequency,
    )

    try:
        events = generate_optimal_events(
            df_daily=df_daily,
            solve_longs_by_frequency_fn=solve_longs_by_frequency,
            solve_shorts_by_frequency_fn=solve_shorts_by_frequency,
            solve_joint_by_frequency_fn=solve_joint_trades_by_frequency,
            k_params=k_params,
            price_col=price_col,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )

        if events.empty:
            return (symbol, None, "no events produced")

        actions = add_action_labels(events)
        labels = add_binary_classification_labels(events, **weighting)

        labels["label"] = actions["label"]
        labels["market_position"] = actions["market_position"]
        labels["symbol"] = symbol

        passthrough_columns = [
            "event", "trade_id", "entry_date", "exit_date",
            "entry_px", "exit_px", "trade_duration_days",
        ]
        for column in passthrough_columns:
            if column in events.columns:
                labels[column] = events[column]

        if "trade_duration_days" in labels.columns and "hold_days" not in labels.columns:
            labels["hold_days"] = labels["trade_duration_days"]

        return (symbol, labels.reset_index(), None)
    except Exception as e:
        return (symbol, None, f"{type(e).__name__}: {e}")


def build_label_panel(
        daily_by_symbol: Dict[str, pd.DataFrame],
        solve_longs_by_frequency_fn: Callable,
        solve_shorts_by_frequency_fn: Optional[Callable],
        k_params: Dict[str, Union[int, List[int]]],
        execution_params: Dict[str, Any],
        weighting: Dict[str, Any],
        solve_joint_by_frequency_fn: Optional[Callable] = None,
        add_rank_labels: bool = True,
        deduplicate: bool = True,
        max_workers: int = 1,
) -> pd.DataFrame:
    """
    Builds a single label dataframe.
    If deduplicate is True, it keeps only one signal per date/symbol/side/action.
    Set max_workers > 1 to process symbols in parallel via ProcessPoolExecutor.
    """
    price_col = execution_params.get("price_col", "close")
    fee_bps = execution_params.get("fee_bps", 0.0)
    slippage_bps = execution_params.get("slippage_bps", 0.0)

    all_label_frames = []

    if max_workers > 1:
        # --- Parallel path ---
        tasks = [
            (symbol, df_daily, k_params, execution_params, weighting,
             price_col, fee_bps, slippage_bps)
            for symbol, df_daily in daily_by_symbol.items()
            if df_daily is not None and not df_daily.empty
        ]
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_build_one_symbol_labels, t): t[0] for t in tasks}
            for future in as_completed(futures):
                symbol, result, error = future.result()
                if error:
                    if error != "no events produced":
                        print(f"[build_label_panel] {symbol}: {error}")
                else:
                    all_label_frames.append(result)
    else:
        # --- Sequential path (original) ---
        for symbol, df_daily in daily_by_symbol.items():
            if df_daily is None or df_daily.empty:
                continue

            _, result, error = _build_one_symbol_labels(
                (symbol, df_daily, k_params, execution_params, weighting,
                 price_col, fee_bps, slippage_bps))
            if error:
                if error != "no events produced":
                    print(f"[build_label_panel] {symbol}: {error}")
                continue
            all_label_frames.append(result)

    if not all_label_frames:
        return pd.DataFrame()

    # Combine all symbols into a single MultiIndex frame
    full_labels = pd.concat(all_label_frames, ignore_index=True).set_index(["date", "symbol"]).sort_index()

    # Apply deduplication logic if requested
    if deduplicate:
        full_labels = deduplicate_labels(full_labels)

    # Add cross-sectional rank labels
    if add_rank_labels:
        full_labels = add_rank_regression_labels(full_labels)

    return full_labels
