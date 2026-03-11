from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


DEFAULT_HORIZONS = (30, 60, 90, 180)


def _price_path_lookup(frame: pd.DataFrame, price_col: str) -> dict[str, pd.DataFrame]:
    lookup: dict[str, pd.DataFrame] = {}
    for symbol, group in frame.groupby("symbol", observed=True):
        work = group[["date", price_col]].copy()
        work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
        work = work.dropna(subset=["date", price_col]).sort_values("date").reset_index(drop=True)
        if work.empty:
            continue
        lookup[str(symbol)] = work
    return lookup


def _outcome_for_row(price_rows: pd.DataFrame, row_date: pd.Timestamp, horizon: int, price_col: str) -> dict[str, float | None]:
    if price_rows.empty:
        return {"return": None, "drawdown": None, "volatility": None}
    idx_matches = price_rows.index[price_rows["date"] == row_date].tolist()
    if not idx_matches:
        return {"return": None, "drawdown": None, "volatility": None}
    idx = int(idx_matches[-1])
    end_idx = idx + int(horizon)
    if end_idx >= len(price_rows):
        return {"return": None, "drawdown": None, "volatility": None}
    current_price = float(price_rows.at[idx, price_col])
    future_slice = price_rows.iloc[idx + 1 : end_idx + 1].copy()
    if future_slice.empty or current_price == 0.0:
        return {"return": None, "drawdown": None, "volatility": None}
    final_price = float(future_slice.iloc[-1][price_col])
    rel_prices = pd.to_numeric(future_slice[price_col], errors="coerce") / current_price
    daily_rets = pd.to_numeric(future_slice[price_col], errors="coerce").pct_change().dropna()
    return {
        "return": float(final_price / current_price - 1.0),
        "drawdown": float(rel_prices.min() - 1.0) if not rel_prices.empty else None,
        "volatility": float(daily_rets.std(ddof=0)) if not daily_rets.empty else 0.0,
    }


def enrich_similarity_matches_with_outcomes(
    matches: Sequence[dict[str, Any]],
    frame: pd.DataFrame,
    *,
    price_col: str,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> list[dict[str, Any]]:
    price_lookup = _price_path_lookup(frame, price_col)
    out: list[dict[str, Any]] = []
    for match in list(matches or []):
        symbol = str(match.get("symbol") or "").strip().upper()
        match_date = pd.Timestamp(str(match.get("date") or ""))
        price_rows = price_lookup.get(symbol)
        item = dict(match)
        for horizon in horizons:
            outcome = _outcome_for_row(price_rows, match_date, int(horizon), price_col) if price_rows is not None else {"return": None, "drawdown": None, "volatility": None}
            item[f"return_{int(horizon)}d"] = outcome["return"]
            item[f"drawdown_{int(horizon)}d"] = outcome["drawdown"]
            item[f"volatility_{int(horizon)}d"] = outcome["volatility"]
        out.append(item)
    return out


def aggregate_outcome_distribution(
    matches: Sequence[dict[str, Any]],
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    horizon_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        key = f"return_{int(horizon)}d"
        drawdown_key = f"drawdown_{int(horizon)}d"
        vol_key = f"volatility_{int(horizon)}d"
        returns = pd.Series([row.get(key) for row in list(matches or [])], dtype="float64").dropna()
        drawdowns = pd.Series([row.get(drawdown_key) for row in list(matches or [])], dtype="float64").dropna()
        vols = pd.Series([row.get(vol_key) for row in list(matches or [])], dtype="float64").dropna()
        if returns.empty:
            horizon_rows.append(
                {
                    "horizon_days": int(horizon),
                    "median_return": None,
                    "mean_return": None,
                    "win_rate": None,
                    "worst_case": None,
                    "best_case": None,
                    "tail_risk": None,
                    "avg_drawdown": None,
                    "avg_volatility": None,
                    "sample_size": 0,
                }
            )
            continue
        horizon_rows.append(
            {
                "horizon_days": int(horizon),
                "median_return": round(float(returns.median()), 6),
                "mean_return": round(float(returns.mean()), 6),
                "win_rate": round(float((returns > 0).mean()), 6),
                "worst_case": round(float(returns.min()), 6),
                "best_case": round(float(returns.max()), 6),
                "tail_risk": round(float(returns.quantile(0.1)), 6),
                "avg_drawdown": round(float(drawdowns.mean()), 6) if not drawdowns.empty else None,
                "avg_volatility": round(float(vols.mean()), 6) if not vols.empty else None,
                "sample_size": int(len(returns)),
            }
        )
    primary = next((row for row in horizon_rows if int(row["horizon_days"]) == 60 and row["sample_size"] > 0), None)
    if primary is None:
        primary = next((row for row in horizon_rows if row["sample_size"] > 0), horizon_rows[0] if horizon_rows else {})
    return {
        "horizon_rows": horizon_rows,
        "primary_horizon_days": int(primary.get("horizon_days") or 0) if primary else 0,
        "median_return": primary.get("median_return") if primary else None,
        "mean_return": primary.get("mean_return") if primary else None,
        "win_rate": primary.get("win_rate") if primary else None,
        "worst_case": primary.get("worst_case") if primary else None,
        "best_case": primary.get("best_case") if primary else None,
        "tail_risk": primary.get("tail_risk") if primary else None,
        "avg_drawdown": primary.get("avg_drawdown") if primary else None,
        "avg_volatility": primary.get("avg_volatility") if primary else None,
        "sample_size": int(primary.get("sample_size") or 0) if primary else 0,
    }
