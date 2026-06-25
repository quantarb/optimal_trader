from __future__ import annotations

import numpy as np
import pandas as pd

def summarize_curve(returns, years, mode):
    returns = pd.Series(returns).fillna(0.0)
    equity = (1.0 + returns).cumprod()
    total_return_pct = float((equity.iloc[-1] - 1.0) * 100.0) if len(equity) else np.nan
    sharpe = float((returns.mean() / returns.std(ddof=0)) * np.sqrt(252.0)) if len(returns) and returns.std(ddof=0) > 1e-12 else np.nan
    max_drawdown_pct = float((((equity / equity.cummax()) - 1.0).min()) * 100.0) if len(equity) else np.nan
    yearly_rows = []
    for yr in years:
        yret = returns.loc[(returns.index >= pd.Timestamp(f"{yr}-01-01")) & (returns.index <= pd.Timestamp(f"{yr}-12-31"))]
        yeq = (1.0 + yret).cumprod()
        yearly_rows.append(
            {
                "mode": str(mode),
                "test_year": int(yr),
                "total_return_pct": float((yeq.iloc[-1] - 1.0) * 100.0) if len(yeq) else np.nan,
                "sharpe": float((yret.mean() / yret.std(ddof=0)) * np.sqrt(252.0)) if len(yret) and yret.std(ddof=0) > 1e-12 else np.nan,
                "max_drawdown_pct": float((((yeq / yeq.cummax()) - 1.0).min()) * 100.0) if len(yeq) else np.nan,
            }
        )
    return {
        "total_return_pct": total_return_pct,
        "sharpe": sharpe,
        "max_drawdown_pct": max_drawdown_pct,
        "equity_curve": equity,
        "yearly_df": pd.DataFrame(yearly_rows),
    }

def _pivot_rule_panel(panel, col, *, symbols=None):
    working_symbols = sorted(panel.index.get_level_values("symbol").unique()) if symbols is None else list(symbols)
    work = panel[[col]].reset_index()
    if work.duplicated(subset=["date", "symbol"]).any():
        work = (
            work.sort_values(["date", "symbol"])
            .groupby(["date", "symbol"], as_index=False, sort=False)
            .last()
        )
    return (
        work
        .pivot(index="date", columns="symbol", values=col)
        .reindex(columns=working_symbols)
        .sort_index()
    )

def prepare_capacity_rule_inputs(panel, score_col, component_cols, price_col):
    if panel.index.has_duplicates:
        panel = panel[~panel.index.duplicated(keep="last")]
    symbols = sorted(panel.index.get_level_values("symbol").unique())
    score = _pivot_rule_panel(panel, score_col, symbols=symbols).shift(1)
    prob_buy = _pivot_rule_panel(panel, "prob_buy", symbols=symbols).shift(1)
    prob_short = _pivot_rule_panel(panel, "prob_short", symbols=symbols).shift(1)
    close = _pivot_rule_panel(panel, price_col, symbols=symbols)
    common_dates = score.index.intersection(prob_buy.index).intersection(prob_short.index).intersection(close.index)
    score = score.loc[common_dates].replace([np.inf, -np.inf], np.nan)
    prob_buy = prob_buy.loc[common_dates].replace([np.inf, -np.inf], np.nan)
    prob_short = prob_short.loc[common_dates].replace([np.inf, -np.inf], np.nan)
    close = close.loc[common_dates].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    component_frames = {}
    for col in component_cols:
        component = _pivot_rule_panel(panel, col, symbols=symbols).shift(1).reindex(index=common_dates, columns=symbols)
        component_frames[str(col)] = component.replace([np.inf, -np.inf], np.nan)
    return {
        "symbols": symbols,
        "common_dates": common_dates,
        "score": score,
        "prob_buy": prob_buy,
        "prob_short": prob_short,
        "close": close,
        "component_cols": [str(col) for col in component_cols],
        "component_frames": component_frames,
    }


# Backward compatibility for callers that imported the original private helper.
_prepare_capacity_rule_inputs = prepare_capacity_rule_inputs

def _build_entry_ok_matrix(inputs, component_threshold):
    score = inputs["score"]
    close = inputs["close"]
    entry_ok = (score.notna() & np.isfinite(score) & close.gt(0.0)).fillna(False)
    for col in inputs["component_cols"]:
        component = inputs["component_frames"][col]
        component_valid = component.notna() & np.isfinite(component)
        entry_ok &= (component.gt(float(component_threshold)) & component_valid).fillna(False)
    return entry_ok

def _run_capacity_limited_long_only_rule(*, panel, score_col, component_cols, component_threshold, price_col, top_k=None):
    inputs = prepare_capacity_rule_inputs(panel, score_col, component_cols, price_col)
    symbols = inputs["symbols"]
    common_dates = inputs["common_dates"]
    score = inputs["score"]
    prob_buy = inputs["prob_buy"]
    prob_short = inputs["prob_short"]
    close = inputs["close"]

    entry_ok = _build_entry_ok_matrix(inputs, component_threshold)

    held_idx = set()
    symbol_to_idx = {sym: idx for idx, sym in enumerate(symbols)}
    position_by_day = pd.DataFrame(0, index=common_dates, columns=symbols, dtype=int)

    for dt in common_dates:
        prob_buy_t = prob_buy.loc[dt]
        prob_short_t = prob_short.loc[dt]
        classifier_short = (prob_short_t > prob_buy_t).fillna(False)

        exit_idx = sorted(
            idx
            for idx in held_idx
            if bool(classifier_short.iloc[idx])
        )
        if exit_idx:
            held_idx -= set(exit_idx)

        slots_left = None if top_k is None else max(0, int(top_k) - len(held_idx))
        if slots_left != 0:
            price_ok_t = close.loc[dt].gt(0.0)
            candidate_mask = entry_ok.loc[dt] & price_ok_t & (~classifier_short)
            ranked = score.loc[dt][candidate_mask].sort_values(ascending=False, kind="stable")
            enter_idx = []
            for sym in ranked.index:
                idx = symbol_to_idx[str(sym)]
                if idx in held_idx:
                    continue
                enter_idx.append(idx)
                if slots_left is not None and len(enter_idx) >= slots_left:
                    break
            if enter_idx:
                held_idx |= set(enter_idx)

        if held_idx:
            position_by_day.loc[dt, [symbols[idx] for idx in sorted(held_idx)]] = 1

    return {
        "positions": position_by_day,
        "score": score,
        "close": close,
    }

def run_top_k_long_only_score_rule(*, panel, score_col, component_cols, component_threshold, price_col, top_k, rebalance_freq=None):
    _ = rebalance_freq
    return _run_capacity_limited_long_only_rule(
        panel=panel,
        score_col=score_col,
        component_cols=component_cols,
        component_threshold=component_threshold,
        price_col=price_col,
        top_k=int(top_k),
    )

def _run_capacity_limited_long_short_rule(
    *,
    panel,
    long_score_col,
    short_score_col,
    long_component_cols,
    short_component_cols,
    component_threshold,
    price_col,
    top_k=None,
):
    long_inputs = prepare_capacity_rule_inputs(panel, long_score_col, long_component_cols, price_col)
    short_inputs = prepare_capacity_rule_inputs(panel, short_score_col, short_component_cols, price_col)
    symbols = long_inputs["symbols"]
    common_dates = long_inputs["common_dates"]
    close = long_inputs["close"]
    long_score = long_inputs["score"]
    short_score = short_inputs["score"]
    prob_buy = long_inputs["prob_buy"]
    prob_short = long_inputs["prob_short"]

    long_entry_ok = _build_entry_ok_matrix(long_inputs, component_threshold)
    short_entry_ok = _build_entry_ok_matrix(short_inputs, component_threshold)

    held_side_by_idx = {}
    symbol_to_idx = {sym: idx for idx, sym in enumerate(symbols)}
    position_by_day = pd.DataFrame(0, index=common_dates, columns=symbols, dtype=int)

    for dt in common_dates:
        price_ok_t = close.loc[dt].gt(0.0)
        prob_buy_t = prob_buy.loc[dt]
        prob_short_t = prob_short.loc[dt]
        long_score_t = long_score.loc[dt]
        short_score_t = short_score.loc[dt]

        next_held = {}
        for idx, side in sorted(held_side_by_idx.items()):
            if side > 0:
                if bool(prob_short_t.iloc[idx] > prob_buy_t.iloc[idx]):
                    continue
            else:
                if bool(prob_buy_t.iloc[idx] > prob_short_t.iloc[idx]):
                    continue
            next_held[idx] = side
        held_side_by_idx = next_held

        capacity = None if top_k is None else max(0, int(top_k))
        slots_left = None if capacity is None else max(0, capacity - len(held_side_by_idx))
        if slots_left != 0:
            candidates = []
            for sym in symbols:
                idx = symbol_to_idx[str(sym)]
                if idx in held_side_by_idx or (not bool(price_ok_t.iloc[idx])):
                    continue
                long_ok = bool(long_entry_ok.loc[dt].iloc[idx]) and np.isfinite(long_score_t.iloc[idx])
                short_ok = bool(short_entry_ok.loc[dt].iloc[idx]) and np.isfinite(short_score_t.iloc[idx])
                if not long_ok and not short_ok:
                    continue
                if long_ok and short_ok:
                    long_value = float(long_score_t.iloc[idx])
                    short_value = float(short_score_t.iloc[idx])
                    if long_value >= short_value:
                        best_side, best_score = 1, long_value
                    else:
                        best_side, best_score = -1, short_value
                elif long_ok:
                    best_side, best_score = 1, float(long_score_t.iloc[idx])
                else:
                    best_side, best_score = -1, float(short_score_t.iloc[idx])
                candidates.append((best_score, str(sym), idx, best_side))

            for _score_value, _sym, idx, side in sorted(candidates, key=lambda row: (row[0], row[1]), reverse=True):
                held_side_by_idx[idx] = int(side)
                if slots_left is not None and len(held_side_by_idx) >= int(capacity):
                    break

        for idx, side in sorted(held_side_by_idx.items()):
            position_by_day.loc[dt, symbols[idx]] = int(side)

    return {
        "positions": position_by_day,
        "long_score": long_score,
        "short_score": short_score,
        "close": close,
    }

def run_top_k_long_short_score_rule(
    *,
    panel,
    long_score_col,
    short_score_col,
    long_component_cols,
    short_component_cols,
    component_threshold,
    price_col,
    top_k,
    rebalance_freq=None,
):
    _ = rebalance_freq
    return _run_capacity_limited_long_short_rule(
        panel=panel,
        long_score_col=long_score_col,
        short_score_col=short_score_col,
        long_component_cols=long_component_cols,
        short_component_cols=short_component_cols,
        component_threshold=component_threshold,
        price_col=price_col,
        top_k=int(top_k),
    )

def run_top_k_momentum_baseline(*, panel, price_col, top_k, lookback_days=21, rebalance_freq=None):
    _ = rebalance_freq
    symbols = sorted(panel.index.get_level_values("symbol").unique())
    close = _pivot_rule_panel(panel, price_col, symbols=symbols).replace([np.inf, -np.inf], np.nan).ffill()
    score = close.pct_change(int(lookback_days)).shift(1).replace([np.inf, -np.inf], np.nan)
    common_dates = score.index.intersection(close.index)
    score = score.loc[common_dates]
    close = close.loc[common_dates].fillna(0.0)
    held_side_by_idx = {}
    symbol_to_idx = {sym: idx for idx, sym in enumerate(symbols)}
    position_by_day = pd.DataFrame(0, index=common_dates, columns=symbols, dtype=int)

    for dt in common_dates:
        score_t = score.loc[dt]
        price_ok_t = close.loc[dt].gt(0.0)
        next_held = {}
        for idx, side in sorted(held_side_by_idx.items()):
            if not bool(price_ok_t.iloc[idx]) or (not np.isfinite(score_t.iloc[idx])):
                continue
            current_score = float(score_t.iloc[idx])
            if side > 0 and current_score <= 0.0:
                continue
            if side < 0 and current_score >= 0.0:
                continue
            next_held[idx] = side
        held_side_by_idx = next_held

        capacity = max(0, int(top_k))
        slots_left = max(0, capacity - len(held_side_by_idx))
        if slots_left != 0:
            candidates = []
            for sym in symbols:
                idx = symbol_to_idx[str(sym)]
                if idx in held_side_by_idx or (not bool(price_ok_t.iloc[idx])) or (not np.isfinite(score_t.iloc[idx])):
                    continue
                momentum_value = float(score_t.iloc[idx])
                if momentum_value > 0.0:
                    candidates.append((abs(momentum_value), str(sym), idx, 1))
                elif momentum_value < 0.0:
                    candidates.append((abs(momentum_value), str(sym), idx, -1))
            for _score_value, _sym, idx, side in sorted(candidates, key=lambda row: (row[0], row[1]), reverse=True):
                held_side_by_idx[idx] = int(side)
                if len(held_side_by_idx) >= capacity:
                    break

        for idx, side in sorted(held_side_by_idx.items()):
            position_by_day.loc[dt, symbols[idx]] = int(side)

    return {
        "positions": position_by_day,
        "score": score,
        "close": close,
    }

def resolve_component_cols(score_col):
    mapping = {
        "prob_buy": ["prob_buy"],
        "prob_short": ["prob_short"],
        "buy_score_mean_raw3": ["prob_buy", "pred_rf_reg", "ae_familiarity"],
        "buy_score_mean_raw_pct6": [
            "prob_buy",
            "pred_rf_reg",
            "ae_familiarity",
            "prob_buy_pct",
            "pred_rf_reg_pct",
            "ae_familiarity_pct",
        ],
        "buy_score_pct_mean": ["prob_buy_pct", "pred_rf_reg_pct", "ae_familiarity_pct"],
        "buy_score_pct_product": ["prob_buy_pct", "pred_rf_reg_pct", "ae_familiarity_pct"],
        "buy_score_raw": ["prob_buy", "pred_rf_reg", "ae_familiarity"],
        "buy_score": ["prob_buy", "pred_rf_reg", "ae_familiarity"],
        "short_score_mean_raw3": ["prob_short", "pred_rf_reg", "ae_familiarity"],
        "short_score_mean_raw_pct6": [
            "prob_short",
            "pred_rf_reg",
            "ae_familiarity",
            "prob_short_pct",
            "pred_rf_reg_pct",
            "ae_familiarity_pct",
        ],
        "short_score_pct_mean": ["prob_short_pct", "pred_rf_reg_pct", "ae_familiarity_pct"],
        "short_score_pct_product": ["prob_short_pct", "pred_rf_reg_pct", "ae_familiarity_pct"],
        "short_score_raw": ["prob_short", "pred_rf_reg", "ae_familiarity"],
        "short_score": ["prob_short", "pred_rf_reg", "ae_familiarity"],
        "__momentum_21d__": [],
    }
    return list(mapping.get(str(score_col), [str(score_col)]))

def resolve_short_score_col(score_col):
    mapping = {
        "prob_buy": "prob_short",
        "buy_score_mean_raw3": "short_score_mean_raw3",
        "buy_score_mean_raw_pct6": "short_score_mean_raw_pct6",
        "buy_score_pct_mean": "short_score_pct_mean",
        "buy_score_pct_product": "short_score_pct_product",
        "buy_score_raw": "short_score_raw",
        "buy_score": "short_score",
    }
    key = str(score_col)
    if key in mapping:
        return str(mapping[key])
    if key.startswith("buy_"):
        return "short_" + key[len("buy_"):]
    raise KeyError(f"No short-score mapping configured for: {score_col}")

__all__ = ['summarize_curve', 'prepare_capacity_rule_inputs', 'run_top_k_long_only_score_rule', 'run_top_k_long_short_score_rule', 'run_top_k_momentum_baseline', 'resolve_component_cols', 'resolve_short_score_col']
