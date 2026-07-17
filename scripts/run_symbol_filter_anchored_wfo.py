"""Anchored WFO with a one-year per-symbol profitability filter.

For forward year Y:
  * fit RF models on all data through Y-1;
  * predict/backtest each symbol independently on Y-1 with backtesting.py;
  * retain symbols with a profitable prior-year strategy;
  * predict/backtest only those symbols in Y.

Oracle and sparse-HITS labels are evaluated independently for long-only and
short-only strategies.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import run_oracle_hits_anchored_wfo as wfo

OUT = REPO_ROOT / "artifacts" / "oracle_hits_symbol_filter_anchored_wfo"
OUT.mkdir(parents=True, exist_ok=True)
FIRST_FORWARD_YEAR = int(os.getenv("SYMBOL_FILTER_FIRST_YEAR", "2021"))
LAST_FORWARD_YEAR = int(os.getenv("SYMBOL_FILTER_LAST_YEAR", "2025"))
RF_ESTIMATORS = int(os.getenv("SYMBOL_FILTER_RF_ESTIMATORS", "40"))
COMMISSION_BPS = float(os.getenv("SYMBOL_FILTER_COMMISSION_BPS", "0.5"))
SPREAD_BPS = float(os.getenv("SYMBOL_FILTER_SPREAD_BPS", "5.0"))
ORACLE_THRESHOLD = float(os.getenv("SYMBOL_FILTER_ORACLE_THRESHOLD", "0.5"))
HITS_THRESHOLD = float(os.getenv("SYMBOL_FILTER_HITS_THRESHOLD", "0.5"))
# The shared WFO scoring helpers read their estimator count from their module
# global, so keep that setting synchronized with this experiment.
wfo.RF_ESTIMATORS = RF_ESTIMATORS


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return wfo.base.SEED + int.from_bytes(digest[:4], "little") % 10000


def directional_backtest(prices: pd.DataFrame, scores: pd.DataFrame, side: str, threshold: float, symbol: str, stage: str) -> dict:
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values("date").drop_duplicates("date")
    frame = frame.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    numeric_cols = ["Open", "High", "Low", "Close"]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_cols)
    frame = frame.loc[(frame[numeric_cols] > 0).all(axis=1) & (frame[numeric_cols] < 1e9).all(axis=1)].copy()
    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    frame = frame.rename(columns={"volume": "Volume"})
    score = scores[["date", "entry_score", "exit_score"]].copy()
    score["date"] = pd.to_datetime(score["date"]).dt.normalize()
    score = score.drop_duplicates("date").set_index("date").reindex(pd.DatetimeIndex(frame.date))
    data = frame.set_index("date")[["Open", "High", "Low", "Close", "Volume"]].copy()
    data["entry_signal"] = (score["entry_score"].to_numpy() >= threshold).astype(float)
    data["exit_signal"] = (score["exit_score"].to_numpy() >= threshold).astype(float)
    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    if len(data) < 3:
        return {"symbol": symbol, "stage": stage, "status": "skipped_short_history", "trades": 0, "total_return": np.nan}

    class DirectionalStrategy(Strategy):
        def init(self):
            pass

        def next(self):
            entry = bool(self.data.entry_signal[-1])
            exit_ = bool(self.data.exit_signal[-1])
            if self.position:
                if exit_:
                    self.position.close()
            elif entry:
                self.buy(size=0.99) if side == "long" else self.sell(size=0.99)

    try:
        stats = Backtest(
            data,
            DirectionalStrategy,
            cash=100_000.0,
            commission=COMMISSION_BPS / 10_000.0,
            spread=SPREAD_BPS / 10_000.0,
            trade_on_close=False,
            exclusive_orders=True,
            finalize_trades=True,
        ).run()
    except (ArithmeticError, OverflowError, ValueError) as exc:
        return {"symbol": symbol, "stage": stage, "status": f"skipped_backtest_{type(exc).__name__}", "trades": 0, "total_return": np.nan}
    return {
        "symbol": symbol,
        "stage": stage,
        "status": "ok",
        "trades": int(stats.get("# Trades", 0)),
        "total_return": float(stats.get("Return [%]", 0.0)) / 100.0,
        "sharpe": float(stats.get("Sharpe Ratio", np.nan)),
        "max_drawdown": float(stats.get("Max. Drawdown [%]", np.nan)) / 100.0,
        "win_rate": float(stats.get("Win Rate [%]", np.nan)) / 100.0,
    }


def prepare_frame(meta: pd.Series, prices: pd.DataFrame, labels: pd.DataFrame, train_end: pd.Timestamp, selection_start: pd.Timestamp, eval_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, list[str]] | None:
    panel = pd.read_parquet(meta.panel_path)
    metadata = pd.read_parquet(meta.metadata_path)
    features = [c for c in metadata.feature.astype(str) if c in panel.columns]
    if not features:
        return None
    frame = panel[["symbol", "date", *features]].copy()
    frame["symbol"] = frame.symbol.astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame.date).dt.normalize()
    frame = frame.merge(labels, on=["symbol", "date"], how="inner")
    frame = frame.merge(prices[["symbol", "date", "open", "high", "low", "close"]], on=["symbol", "date"], how="inner")
    frame = frame.loc[frame.date <= eval_end].reset_index(drop=True)
    train = frame.loc[frame.date <= train_end].copy()
    # The selection year is intentionally part of both the fit and evaluation
    # windows: the model has seen it, and we use its per-symbol profitability
    # as the symbol-selection signal for the following year.
    evaluate = frame.loc[frame.date >= selection_start].copy()
    if train.empty or evaluate.empty:
        return None
    med = train[features].apply(pd.to_numeric, errors="coerce").median().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    all_x = frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    mean = all_x.loc[frame.date <= train_end].mean()
    std = all_x.loc[frame.date <= train_end].std().replace(0, 1).fillna(1.0)
    frame[features] = ((all_x - mean) / std).clip(-8, 8).astype("float32")
    return frame.loc[frame.date <= train_end].copy(), frame.loc[frame.date >= selection_start].copy(), features


def get_scores(train: pd.DataFrame, evaluate: pd.DataFrame, features: list[str], family: str, year: int, model_kind: str, side: str) -> tuple[np.ndarray, np.ndarray] | None:
    if model_kind == "oracle":
        return wfo.oracle_scores(train, evaluate, features, side)
    roles = ("hub", "authority")
    values = []
    for role in roles:
        target = f"{side}_{role}"
        value = wfo.hits_score(train, evaluate, features, target, family, year)
        if value is None:
            return None
        values.append(value)
    return values[0], values[1]


def run_family(meta: pd.Series, prices: pd.DataFrame, labels: pd.DataFrame, close_year: int, tier: str) -> list[dict]:
    selection_year = close_year - 1
    train_end = pd.Timestamp(f"{selection_year}-12-31")
    selection_start = pd.Timestamp(f"{selection_year}-01-01")
    forward_start = pd.Timestamp(f"{close_year}-01-01")
    forward_end = pd.Timestamp(f"{close_year}-12-31")
    prepared = prepare_frame(meta, prices, labels, train_end, selection_start, forward_end)
    if prepared is None:
        return []
    train, evaluate, features = prepared
    family = str(meta.family)
    results = []
    for model_kind, threshold in (("oracle", ORACLE_THRESHOLD), ("hits", HITS_THRESHOLD)):
        for side in ("long", "short"):
            scores = get_scores(train, evaluate, features, family, close_year, model_kind, side)
            if scores is None:
                continue
            entry, exit_ = scores
            eval_scores = evaluate[["symbol", "date"]].copy()
            eval_scores["entry_score"] = entry
            eval_scores["exit_score"] = exit_
            selection_results = []
            forward_results = []
            for symbol, symbol_scores in eval_scores.groupby("symbol", sort=True):
                symbol = str(symbol).upper()
                symbol_prices = prices.loc[prices.symbol.eq(symbol)].copy()
                selection_prices = symbol_prices.loc[symbol_prices.date.between(selection_start, forward_start - pd.Timedelta(days=1))]
                forward_prices = symbol_prices.loc[symbol_prices.date.between(forward_start, forward_end)]
                selection_scores = symbol_scores.loc[symbol_scores.date.between(selection_start, forward_start - pd.Timedelta(days=1))]
                forward_scores = symbol_scores.loc[symbol_scores.date.between(forward_start, forward_end)]
                selection_results.append(directional_backtest(selection_prices, selection_scores, side, threshold, symbol, "selection"))
                forward_results.append((symbol, forward_prices, forward_scores))
            selection_df = pd.DataFrame(selection_results)
            profitable = set(selection_df.loc[(selection_df.status == "ok") & (selection_df.trades > 0) & (selection_df.total_return > 0), "symbol"].astype(str))
            selected_forward = []
            for symbol, symbol_prices, symbol_scores in forward_results:
                if symbol in profitable:
                    selected_forward.append(directional_backtest(symbol_prices, symbol_scores, side, threshold, symbol, "forward"))
            forward_df = pd.DataFrame(selected_forward)
            results.append({
                "tier": tier, "forward_year": close_year, "selection_year": selection_year,
                "model": model_kind, "variant": side, "family": family,
                "eligible_symbols": int(len(selection_df)), "profitable_symbols": int(len(profitable)),
                "selection_mean_return": float(selection_df.total_return.mean()) if not selection_df.empty else np.nan,
                "forward_mean_return": float(forward_df.total_return.mean()) if not forward_df.empty else 0.0,
                "forward_median_return": float(forward_df.total_return.median()) if not forward_df.empty else 0.0,
                "forward_min_return": float(forward_df.total_return.min()) if not forward_df.empty else 0.0,
                "forward_max_return": float(forward_df.total_return.max()) if not forward_df.empty else 0.0,
                "forward_mean_sharpe": float(forward_df.sharpe.mean()) if not forward_df.empty else np.nan,
            })
    return results


def run_tier(tier: str) -> pd.DataFrame:
    started = perf_counter()
    index, prices, labels = wfo.load_data(tier)
    index["tier"] = tier
    requested = tuple(x.strip() for x in os.getenv("SYMBOL_FILTER_FAMILIES", "").split(",") if x.strip())
    if requested:
        index = index.loc[index.family.astype(str).isin(requested)].copy()
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    rows = []
    for year in range(FIRST_FORWARD_YEAR, LAST_FORWARD_YEAR + 1):
        for _, meta in index.iterrows():
            rows.extend(run_family(meta, prices, labels, year, tier))
        print({"tier": tier, "forward_year": year, "rows": len(rows)}, flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(OUT / f"{tier.lower()}_results.parquet", index=False)
    return out


def main() -> None:
    tiers = tuple(x.strip().upper() for x in os.getenv("SYMBOL_FILTER_TIERS", "1T,100B,10B").split(",") if x.strip())
    all_rows = [run_tier(tier) for tier in tiers]
    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    out.to_csv(OUT / "all_results.csv", index=False)
    if not out.empty:
        out.groupby(["tier", "forward_year", "model", "variant"], as_index=False).agg(
            families=("family", "nunique"), mean_forward_return=("forward_mean_return", "mean"),
            median_forward_return=("forward_mean_return", "median"), mean_selected=("profitable_symbols", "mean"),
        ).to_csv(OUT / "summary.csv", index=False)


if __name__ == "__main__":
    main()
