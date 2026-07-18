"""Feature-family oracle versus top/bottom HITS entry/exit experiment.

Train on 2024 and backtest on 2025. Oracle models see only oracle event dates.
HITS regressors train only on the top and bottom tails of their own score.
Long and short books are evaluated separately.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter

import cupy as cp
import cudf
import numpy as np
import pandas as pd
from cuml.ensemble import RandomForestRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
for repo in (REPO_ROOT, WORKSPACE_ROOT / "quant-warehouse", WORKSPACE_ROOT / "quant-orchestrator"):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

from quant_warehouse.platforms.data_providers.fmp.target_engineering import LabelBuildSpec, build_trade_results
from quant_warehouse.warehouse.api import Warehouse
from quant_orchestrator.platforms.backtesting_frameworks.shared_book import (
    SharedBookCostModel,
    build_shared_book_weights,
    run_shared_book_backtest,
    shared_book_performance_metrics,
)
from quant_orchestrator.platforms.ml_frameworks.rapids.random_forest import RapidsRandomForestClassifier

TRAIN_START, TRAIN_END = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")
TEST_START, TEST_END = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")
MIN_RETURN = float(os.getenv("ORACLE_HITS_MIN_RETURN", "0.01"))
MAX_HOLD = int(os.getenv("ORACLE_HITS_MAX_HOLD", "120"))
HITS_ITERS = int(os.getenv("ORACLE_HITS_ITERS", "50"))
RF_ESTIMATORS = int(os.getenv("ORACLE_HITS_RF_ESTIMATORS", "40"))
HITS_THRESHOLD = float(os.getenv("ORACLE_HITS_THRESHOLD", "0.80"))
HITS_TAIL_QUANTILE = float(os.getenv("ORACLE_HITS_TAIL_QUANTILE", "0.20"))
ORACLE_THRESHOLD = float(os.getenv("ORACLE_THRESHOLD", "0.50"))
SEED = 20260716

TIER_CONFIGS = {
    "1T": (1_000_000_000_000, "equity_meta_model_1t"),
    "100B": (100_000_000_000, "equity_meta_model_100b"),
    "10B": (10_000_000_000, "equity_meta_model_10b"),
}
OUT = REPO_ROOT / "artifacts" / "oracle_vs_hits_top_bottom_entry_exit_2024_2025"
OUT.mkdir(parents=True, exist_ok=True)


def feature_dir(tier: str) -> Path:
    cap, cache = TIER_CONFIGS[tier]
    return REPO_ROOT / "artifacts" / "trading_app_v2" / cache / f"mcap_{cap}_train_2020-12-31_seed_20260707" / "feature_family_panels"


def normalize_prices(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy().reset_index() if "date" not in raw.columns else raw.copy()
    frame.columns = [str(c).lower() for c in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    return frame[["date", "open", "high", "low", "close"]].dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _hits_scores(frame: pd.DataFrame, weighting: str) -> pd.DataFrame:
    frame = frame.sort_values("date").reset_index(drop=True)
    n = len(frame)
    high, low = frame.high.to_numpy(float), frame.low.to_numpy(float)
    valid = np.triu(np.ones((n, n), dtype=bool), 1)
    horizon = np.arange(n)[None, :] - np.arange(n)[:, None]
    valid &= horizon <= MAX_HOLD
    outputs = {"date": frame.date.to_numpy()}
    for side, returns in {
        "long": low[None, :] / high[:, None] - 1.0,
        "short": low[:, None] / high[None, :] - 1.0,
    }.items():
        if weighting == "clip":
            weights = np.maximum(returns, 0.0)
        elif weighting == "rank":
            raw = pd.DataFrame(returns).where(valid)
            weights = raw.rank(axis=1, pct=True).to_numpy(float)
            weights = np.nan_to_num(weights, nan=0.0)
        else:
            raise ValueError(f"unknown weighting {weighting}")
        weights = np.where(valid, weights, 0.0)
        hub, authority = np.ones(n, dtype=float), np.ones(n, dtype=float)
        for _ in range(HITS_ITERS):
            authority = weights.T @ hub
            authority /= np.linalg.norm(authority) or 1.0
            hub = weights @ authority
            hub /= np.linalg.norm(hub) or 1.0
        outputs[f"{side}_hub"] = hub / (hub.max() or 1.0)
        outputs[f"{side}_authority"] = authority / (authority.max() or 1.0)
    return pd.DataFrame(outputs)


def _oracle_events(frame: pd.DataFrame) -> pd.DataFrame:
    spec = LabelBuildSpec(k_params={"YE": [3]}, min_profit_pct=MIN_RETURN, buy_execution="high", sell_execution="low", short_execution="low", cover_execution="high")
    result = build_trade_results(["S"], spec=spec, price_frames={"S": frame})
    out = pd.DataFrame({"date": frame.date})
    for label in ("buy", "sell", "short", "cover"):
        out[label] = 0
    trades = pd.DataFrame(result.completed_trades)
    if trades.empty:
        return out
    for side, entry_label, exit_label in (("long", "buy", "sell"), ("short", "short", "cover")):
        part = trades.loc[trades.side.astype(str).str.lower().eq(side)]
        if part.empty:
            continue
        entries = set(pd.to_datetime(part.entry_date, errors="coerce"))
        exits = set(pd.to_datetime(part.exit_date, errors="coerce"))
        out.loc[out.date.isin(entries), entry_label] = 1
        out.loc[out.date.isin(exits), exit_label] = 1
    return out


def load_data(tier: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.read_csv(feature_dir(tier) / "index.csv")
    first = pd.read_parquet(index.iloc[0].panel_path)
    symbols = sorted(first.symbol.astype(str).str.upper().unique())
    warehouse = Warehouse()
    price_rows, label_rows = [], []
    for i, symbol in enumerate(symbols, 1):
        raw = warehouse.read_prices(symbol, provider="fmp", start="2024-01-01", end="2025-12-31")
        if raw is None or raw.empty:
            continue
        try:
            prices = normalize_prices(raw)
        except Exception:
            continue
        prices = prices.loc[prices.date.between(TRAIN_START, TEST_END)].copy()
        if len(prices) < 30:
            continue
        p = prices.copy(); p.insert(0, "symbol", symbol); price_rows.append(p)
        for _, year_frame in p.groupby(p.date.dt.year):
            if year_frame.date.dt.year.iloc[0] not in (2024, 2025):
                continue
            h = _hits_scores(year_frame.drop(columns="symbol"), "clip")
            r = _hits_scores(year_frame.drop(columns="symbol"), "rank").drop(columns="date")
            events = _oracle_events(year_frame.drop(columns="symbol"))
            row = pd.concat([year_frame[["symbol", "date"]].reset_index(drop=True), h.drop(columns="date"), r.reset_index(drop=True).add_suffix("_rank"), events.drop(columns="date")], axis=1)
            label_rows.append(row)
        if i % 100 == 0:
            print({"tier": tier, "prices_loaded": i, "usable_symbols": len(price_rows)}, flush=True)
    prices = pd.concat(price_rows, ignore_index=True) if price_rows else pd.DataFrame()
    labels = pd.concat(label_rows, ignore_index=True) if label_rows else pd.DataFrame()
    return index, prices, labels


def reg_predict(train: pd.DataFrame, pred: pd.DataFrame, features: list[str], target: str, seed: int) -> np.ndarray:
    x_train = train[features].astype("float32")
    x_pred = pred[features].astype("float32")
    model = RandomForestRegressor(n_estimators=RF_ESTIMATORS, max_depth=16, max_features=1.0, n_streams=8, random_state=seed)
    model.fit(cudf.from_pandas(x_train), cudf.Series(train[target].astype("float32").to_numpy()))
    values = model.predict(cudf.from_pandas(x_pred))
    return cp.asnumpy(values) if hasattr(values, "__cuda_array_interface__") else np.asarray(values)


def clean_features(frame: pd.DataFrame, features: list[str], medians: pd.Series) -> pd.DataFrame:
    out = frame.copy()
    numeric = out[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out[features] = numeric.fillna(medians).fillna(0.0).astype("float32")
    return out


def select_top_bottom_rows(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Keep only top/bottom score tails within each symbol-year."""
    out = frame.copy()
    year = out.date.dt.year
    rank = out.groupby([out.symbol, year], sort=False)[target].rank(pct=True, method="first")
    mask = rank.le(HITS_TAIL_QUANTILE) | rank.ge(1.0 - HITS_TAIL_QUANTILE)
    return out.loc[mask].copy()


def backtest_scores(scores: pd.DataFrame, close: pd.DataFrame, variant: str, dates: pd.DatetimeIndex, threshold: float, top_k: int = 20, cost_bps: float = 5.5) -> dict:
    symbols = tuple(close.columns)
    if variant == "long":
        fields = ["symbol", "date", "long_score", "short_score", "long_exit_score", "short_exit_score"]
    else:
        fields = ["symbol", "date", "long_score", "short_score", "long_exit_score", "short_exit_score"]
    weights, trades = build_shared_book_weights(scores[fields], symbols, dates, top_k=top_k, variant="long_only" if variant == "long" else "short_only", entry_threshold=threshold, exit_threshold=threshold, planner="threshold")
    next_returns = close.pct_change().shift(-1)
    returns, equity, _ = run_shared_book_backtest(weights, next_returns, cost_bps=cost_bps)
    row = shared_book_performance_metrics(returns, equity, weights, trades, framework="shared_book", variant=variant, top_k=top_k, cost_bps=cost_bps)
    row["trades"] = int(len(trades))
    return row


def run_family(index_row: pd.Series, prices: pd.DataFrame, labels: pd.DataFrame, close: pd.DataFrame, test_dates: pd.DatetimeIndex, weighting: str) -> list[dict]:
    panel = pd.read_parquet(index_row.panel_path)
    metadata = pd.read_parquet(index_row.metadata_path)
    source, family = str(index_row.source), str(index_row.family)
    features = [c for c in metadata.feature.astype(str) if c in panel.columns]
    if not features:
        return []
    base = panel[["symbol", "date", *features]].copy()
    base.symbol = base.symbol.astype(str).str.upper(); base.date = pd.to_datetime(base.date).dt.normalize()
    base = base.merge(labels, on=["symbol", "date"], how="inner")
    base = base.loc[base.date.between(TRAIN_START, TEST_END)].reset_index(drop=True)
    if base.empty:
        return []
    medians = base.loc[base.date.between(TRAIN_START, TRAIN_END), features].apply(pd.to_numeric, errors="coerce").median().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    base = clean_features(base, features, medians)
    train = base.loc[base.date.between(TRAIN_START, TRAIN_END)].copy()
    test = base.loc[base.date.between(TEST_START, TEST_END)].copy()
    if len(train) < 50 or test.empty:
        return []
    result: list[dict] = []
    # Oracle: event rows only, no non-event/hold class.
    for side, entry_label, exit_label in (("long", "buy", "sell"), ("short", "short", "cover")):
        event_rows = train.loc[train[[entry_label, exit_label]].sum(axis=1).gt(0)].copy()
        event_rows["target"] = np.where(event_rows[entry_label].eq(1), entry_label, exit_label)
        event_rows = event_rows.loc[event_rows.target.isin([entry_label, exit_label])]
        if len(event_rows) < 10 or event_rows.target.nunique() < 2:
            continue
        model = RapidsRandomForestClassifier.fit(event_rows, features=features, target_col="target", random_state=SEED, params={"n_estimators": RF_ESTIMATORS, "max_depth": 16, "max_features": "sqrt", "n_bins": 128, "n_streams": 8})
        proba = model.predict_proba_frame(test, features)
        pred = test[["symbol", "date"]].copy()
        entry_prob = proba.get(f"prob__{entry_label}", pd.Series(0.0, index=test.index)).to_numpy()
        exit_prob = proba.get(f"prob__{exit_label}", pd.Series(0.0, index=test.index)).to_numpy()
        if side == "long":
            pred["long_score"], pred["short_score"] = entry_prob, 0.0
            pred["long_exit_score"], pred["short_exit_score"] = 0.0, exit_prob
        else:
            pred["long_score"], pred["short_score"] = 0.0, entry_prob
            pred["long_exit_score"], pred["short_exit_score"] = exit_prob, 0.0
        metrics = backtest_scores(pred, close, side, test_dates, ORACLE_THRESHOLD)
        metrics.update({"tier": str(index_row.get("tier", "")), "model": "oracle", "label_variant": "dp", "weighting": "oracle_events", "family": family, "source": source, "train_rows": len(event_rows), "entry_events": int(event_rows[entry_label].sum()), "exit_events": int(event_rows[exit_label].sum())})
        result.append(metrics)
    # HITS: every date, one regressor for each node score.
    for side in ("long", "short"):
        for role in ("hub", "authority"):
            target = f"{side}_{role}" if weighting == "clip" else f"{side}_{role}_rank"
            if target not in train.columns:
                continue
            sparse_train = select_top_bottom_rows(train, target)
            if len(sparse_train) < max(20, len(features) * 2):
                continue
            pred_values = reg_predict(sparse_train, test, features, target, SEED + hash((family, side, role, weighting)) % 10000)
            pred = test[["symbol", "date"]].copy()
            pred["score"] = pred_values
            pred["score"] = pred.groupby("date")["score"].rank(pct=True, method="average")
            if side == "long":
                if role == "hub":
                    pred["long_score"], pred["short_score"] = pred.score, 0.0
                else:
                    pred["long_exit_score"], pred["short_exit_score"] = 0.0, pred.score
            else:
                if role == "hub":
                    pred["long_score"], pred["short_score"] = 0.0, pred.score
                else:
                    pred["long_exit_score"], pred["short_exit_score"] = pred.score, 0.0
            for col in ("long_score", "short_score", "long_exit_score", "short_exit_score"):
                if col not in pred:
                    pred[col] = 0.0
            # Do not backtest a single HITS head; combine hub and authority later.
            result.append({"tier": str(index_row.get("tier", "")), "model": "hits_component", "label_variant": weighting, "weighting": weighting, "family": family, "source": source, "role": f"{side}_{role}", "predictions": pred})
    # Replace component records with the actual two HITS strategies.
    components = [r for r in result if r.get("model") == "hits_component"]
    result = [r for r in result if r.get("model") != "hits_component"]
    for side in ("long", "short"):
        hub = next((r["predictions"] for r in components if r["role"] == f"{side}_hub"), None)
        authority = next((r["predictions"] for r in components if r["role"] == f"{side}_authority"), None)
        if hub is None or authority is None:
            continue
        pred = hub[["symbol", "date"]].copy()
        if side == "long":
            pred["long_score"], pred["short_score"] = hub["long_score"], 0.0
            pred["long_exit_score"], pred["short_exit_score"] = 0.0, authority["short_exit_score"]
        else:
            pred["long_score"], pred["short_score"] = 0.0, hub["short_score"]
            pred["long_exit_score"], pred["short_exit_score"] = authority["long_exit_score"], 0.0
        metrics = backtest_scores(pred, close, side, test_dates, HITS_THRESHOLD)
        metrics.update({"tier": str(index_row.get("tier", "")), "model": "hits_top_bottom", "label_variant": weighting, "weighting": weighting, "family": family, "source": source, "tail_quantile": HITS_TAIL_QUANTILE})
        result.append(metrics)
    return result


def run_tier(tier: str) -> pd.DataFrame:
    started = perf_counter()
    index, prices, labels = load_data(tier)
    index["tier"] = tier
    requested_families = tuple(x.strip() for x in os.getenv("ORACLE_HITS_FAMILIES", "").split(",") if x.strip())
    if requested_families:
        index = index.loc[index.family.astype(str).isin(requested_families)].copy()
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    test_dates = pd.DatetimeIndex(close.index[(close.index >= TEST_START) & (close.index <= TEST_END)])
    all_rows: list[dict] = []
    for weighting in ("clip", "rank"):
        for _, row in index.iterrows():
            rows = run_family(row, prices, labels, close, test_dates, weighting)
            for item in rows:
                if "predictions" not in item:
                    all_rows.append(item)
            print({"tier": tier, "weighting": weighting, "family": str(row.family), "rows": len(all_rows)}, flush=True)
    out = pd.DataFrame(all_rows)
    out.to_parquet(OUT / f"{tier.lower()}_results.parquet", index=False)
    print(out.groupby(["model", "label_variant", "variant"], as_index=False).agg(families=("family", "nunique"), mean_return=("total_return", "mean"), median_return=("total_return", "median"), min_return=("total_return", "min"), max_return=("total_return", "max"), mean_sharpe=("sharpe", "mean")).round(4).to_string(index=False))
    print({"tier": tier, "symbols": len(close.columns), "rows": len(out), "seconds": round(perf_counter() - started, 1)}, flush=True)
    return out


def main() -> None:
    tiers = tuple(x.strip().upper() for x in os.getenv("ORACLE_HITS_TIERS", "1T,100B,10B").split(",") if x.strip())
    outputs = [run_tier(tier) for tier in tiers]
    combined = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    combined.to_csv(OUT / "all_results.csv", index=False)
    if not combined.empty:
        summary = combined.groupby(["tier", "model", "label_variant", "variant"], as_index=False).agg(families=("family", "nunique"), mean_return=("total_return", "mean"), median_return=("total_return", "median"), min_return=("total_return", "min"), max_return=("total_return", "max"), mean_sharpe=("sharpe", "mean")).round(4)
        summary.to_csv(OUT / "summary.csv", index=False)
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
