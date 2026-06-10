from __future__ import annotations

import os
from pathlib import Path
from datetime import timedelta
from typing import Any, Sequence

import numpy as np
import pandas as pd
from django.utils import timezone

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

from data import FMPClient
from fmp.models import Symbol
from fmp.refresh import (
    expected_latest_price_date_from_market_clock,
    refresh_symbol_price_history,
    refresh_universe_price_history_from_fmp,
    refresh_universe_symbol_sections_from_fmp,
    refresh_macro_series_from_fmp,
    plan_symbol_fundamental_refresh_from_fmp,
    plan_symbol_price_refresh_from_fmp,
    plan_symbol_section_refresh_from_fmp,
    resolve_fmp_api_key,
    symbol_needs_fundamental_refresh,
    symbol_needs_price_refresh,
    symbol_needs_required_refresh,
    historical_symbol_refresh_needed,
)
from fmp.sections import (
    REQUIRED_FUNDAMENTAL_SECTION_KEYS,
    REQUIRED_SCORING_HISTORICAL_SECTIONS,
)
from features.feature_builders import build_price_technical_features
from features.macro import MacroFeatureConfig
from features.views import _load_adjusted_prices

# Note: REQUIRED_*, resolve_fmp_api_key, the plan_* , refresh_universe_* , symbol_needs_*,
# historical_symbol_refresh_needed, refresh_macro etc. are implemented in fmp.refresh
# (and constants in fmp.sections). They are imported at the top of this file.
# Local re-exports below keep old "from trading.live_trade import ..." call sites working.


def build_technical_dataframe_from_django(
    *,
    symbols: Sequence[str],
    start_date=None,
    end_date=None,
) -> tuple[pd.DataFrame, list[str]]:
    start_ts = pd.Timestamp(start_date) if start_date is not None else None
    end_ts = pd.Timestamp(end_date) if end_date is not None else None
    frames: list[pd.DataFrame] = []
    feature_cols: list[str] = []

    for sym in symbols:
        code = str(sym).strip().upper()
        if not code:
            continue

        symbol_obj = Symbol.objects.filter(symbol__iexact=code).only("id", "symbol").first()
        if symbol_obj is None:
            continue

        df_prices = _load_adjusted_prices(
            symbol_obj,
            start_ts.date() if start_ts is not None else None,
            end_ts.date() if end_ts is not None else None,
        )
        if df_prices.empty:
            continue

        built = build_price_technical_features(code, df_prices)
        if built.df.empty:
            continue

        px = df_prices[["open", "high", "low", "close", "volume"]].copy()
        px["symbol"] = code
        px = px.reset_index().set_index(["date", "symbol"]).sort_index()

        panel = px.join(built.df[built.feature_cols], how="left")
        frames.append(panel)
        for col in built.feature_cols:
            if col not in feature_cols:
                feature_cols.append(col)

    if not frames:
        empty_index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["date", "symbol"])
        return pd.DataFrame(index=empty_index), feature_cols

    technical_df = pd.concat(frames, axis=0).sort_index()
    if technical_df.index.has_duplicates:
        technical_df = technical_df[~technical_df.index.duplicated(keep="last")]
    return technical_df, feature_cols


# (refresh planning/orchestration, needs, plans, universe refreshers and macro refresh
#  now live in fmp.refresh -- imported above and re-exported for compat)


def normalize_holdings(raw_holdings: Sequence[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in list(raw_holdings or []):
        code = str(raw).strip().upper()
        if not code or code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


def component_cols_for_score(score_col: str) -> list[str]:
    mapping = {
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
    }
    if score_col not in mapping:
        raise KeyError(f"No component mapping configured for score column: {score_col}")
    return list(mapping[score_col])


def build_live_trade_plan(
    *,
    latest_scored_df: pd.DataFrame,
    current_holdings: Sequence[str] | None,
    top_k: int,
    score_col: str,
    component_cols: Sequence[str],
    component_threshold: float,
    price_col: str = "close",
) -> dict[str, Any]:
    work = latest_scored_df.copy()
    work.index = pd.Index([str(idx).strip().upper() for idx in work.index], name="symbol")

    required_cols = [score_col, price_col, "prob_buy", "prob_short", "pred_rf_reg", "ae_familiarity", *component_cols]
    missing_cols = [col for col in required_cols if col not in work.columns]
    if missing_cols:
        raise KeyError(f"Missing required latest-score columns: {missing_cols}")

    numeric_cols = list(dict.fromkeys(required_cols))
    work.loc[:, numeric_cols] = work[numeric_cols].apply(pd.to_numeric, errors="coerce")
    work["entry_ok"] = work[score_col].notna() & np.isfinite(work[score_col]) & work[price_col].gt(0.0)
    for col in component_cols:
        work["entry_ok"] &= work[col].notna() & np.isfinite(work[col]) & work[col].gt(float(component_threshold))

    work["classifier_long"] = (work["prob_buy"] > work["prob_short"]).fillna(False)
    work["classifier_short"] = (work["prob_short"] > work["prob_buy"]).fillna(False)
    work["component_min"] = work[list(component_cols)].min(axis=1, skipna=True)
    work["score_rank"] = work[score_col].rank(ascending=False, method="first")

    current = normalize_holdings(current_holdings)
    current_set = set(current)
    exits: list[dict[str, Any]] = []
    retained: list[str] = []

    for sym in current:
        if sym not in work.index:
            exits.append({"symbol": sym, "action": "sell", "reason": "missing_from_latest_panel"})
            continue

        row = work.loc[sym]
        if (not np.isfinite(row[price_col])) or float(row[price_col]) <= 0.0:
            exits.append({"symbol": sym, "action": "sell", "reason": "invalid_price"})
        elif (not np.isfinite(row["prob_buy"])) or (not np.isfinite(row["prob_short"])):
            exits.append({"symbol": sym, "action": "sell", "reason": "invalid_probability"})
        elif bool(row["classifier_short"]):
            exits.append({"symbol": sym, "action": "sell", "reason": "classifier_flipped_short"})
        else:
            retained.append(sym)

    slots_left = max(0, int(top_k) - len(retained))
    candidates = work.loc[work["entry_ok"]].copy()
    if current_set:
        candidates = candidates.drop(index=[sym for sym in current_set if sym in candidates.index], errors="ignore")
    candidates = candidates.sort_values(
        [score_col, "prob_buy", "pred_rf_reg", "ae_familiarity"],
        ascending=[False, False, False, False],
        kind="stable",
    )
    buys = candidates.head(slots_left).index.tolist()
    target_symbols = retained + buys
    target_weight = (1.0 / float(top_k)) if int(top_k) > 0 and target_symbols else 0.0
    cash_weight = max(0.0, 1.0 - (float(len(target_symbols)) * target_weight))

    portfolio_cols = [score_col, "score_rank", price_col, "prob_buy", "prob_short", "pred_rf_reg", "ae_familiarity", "component_min"]
    if target_symbols:
        target_portfolio = work.loc[target_symbols, portfolio_cols].copy()
        target_portfolio.insert(0, "target_weight", target_weight)
        target_portfolio.insert(1, "status", ["hold" if sym in retained else "buy" for sym in target_portfolio.index])
        target_portfolio = target_portfolio.sort_values(["status", score_col], ascending=[True, False], kind="stable")
    else:
        target_portfolio = pd.DataFrame(columns=["target_weight", "status", *portfolio_cols])

    action_rows: list[dict[str, Any]] = []
    for row in exits:
        sym = row["symbol"]
        if sym in work.index:
            live_row = work.loc[sym]
            action_rows.append(
                {
                    "symbol": sym,
                    "action": "sell",
                    "reason": row["reason"],
                    "target_weight": 0.0,
                    price_col: live_row.get(price_col, np.nan),
                    score_col: live_row.get(score_col, np.nan),
                    "prob_buy": live_row.get("prob_buy", np.nan),
                    "prob_short": live_row.get("prob_short", np.nan),
                }
            )
        else:
            action_rows.append(
                {
                    "symbol": sym,
                    "action": "sell",
                    "reason": row["reason"],
                    "target_weight": 0.0,
                    price_col: np.nan,
                    score_col: np.nan,
                    "prob_buy": np.nan,
                    "prob_short": np.nan,
                }
            )

    for sym in retained:
        row = work.loc[sym]
        action_rows.append(
            {
                "symbol": sym,
                "action": "hold",
                "reason": "still_held_not_exited",
                "target_weight": target_weight,
                price_col: row[price_col],
                score_col: row[score_col],
                "prob_buy": row["prob_buy"],
                "prob_short": row["prob_short"],
            }
        )

    for sym in buys:
        row = work.loc[sym]
        action_rows.append(
            {
                "symbol": sym,
                "action": "buy",
                "reason": "eligible_with_open_slot",
                "target_weight": target_weight,
                price_col: row[price_col],
                score_col: row[score_col],
                "prob_buy": row["prob_buy"],
                "prob_short": row["prob_short"],
            }
        )

    actions = pd.DataFrame(action_rows)
    if not actions.empty:
        action_order = {"sell": 0, "buy": 1, "hold": 2}
        actions["_action_order"] = actions["action"].map(action_order).fillna(99)
        actions = actions.sort_values(["_action_order", score_col], ascending=[True, False], kind="stable").drop(columns=["_action_order"])

    watchlist_cols = [score_col, "score_rank", price_col, "prob_buy", "prob_short", "pred_rf_reg", "ae_familiarity", "component_min"]
    watchlist = candidates.loc[:, watchlist_cols].head(max(20, int(top_k) * 3)).copy()

    summary = pd.DataFrame(
        [
            {
                "current_holdings": len(current),
                "positions_kept": len(retained),
                "positions_sold": len(exits),
                "slots_open_after_sells": slots_left,
                "new_buys": len(buys),
                "target_positions": len(target_symbols),
                "target_weight_per_position": target_weight,
                "target_cash_weight": cash_weight,
                "component_threshold": float(component_threshold),
                "top_k": int(top_k),
                "score_col": score_col,
            }
        ]
    )

    return {
        "summary": summary,
        "target_portfolio": target_portfolio,
        "actions": actions,
        "watchlist": watchlist,
        "latest_scored": work.sort_values(score_col, ascending=False, kind="stable"),
        "retained": retained,
        "buys": buys,
        "exits": exits,
    }
