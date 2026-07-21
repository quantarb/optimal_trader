"""Compare a congressional event graph with regular HITS and Oracle models.

The three models are trained on historical years and score every daily date in
each forward test year:

* ``congress_graph``: event nodes are dates on which Congress/Senate traded a
  symbol; buy->sell and sell->buy event combinations are weighted by returns
  and converted into entry/exit HITS node targets.
* ``hits``: the regular price graph uses every daily market date.
* ``oracle``: historical future-price trade labels train a directional model.

The future prices used to create labels are confined to the historical
training window.  They are never used to create test-period predictions.
Market-cap tiers are configuration, not separate workflows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path[:0] = [str(ROOT), str(WORKSPACE / "quant-warehouse"), str(WORKSPACE / "quant-orchestrator")]

from quant_warehouse.platforms.data_providers.fmp.target_engineering.event_pairs.store import (  # noqa: E402
    build_event_pairs_from_historical_data,
)
from quant_warehouse.warehouse.api import Warehouse  # noqa: E402

import run_oracle_vs_hits_entry_exit_2024_2025 as base  # noqa: E402

FIRST_YEAR = int(os.getenv("COMPARE_FIRST_YEAR", "2021"))
LAST_YEAR = int(os.getenv("COMPARE_LAST_YEAR", "2025"))
DATA_START = pd.Timestamp(os.getenv("COMPARE_DATA_START", "2015-01-01"))
DATA_END = pd.Timestamp(os.getenv("COMPARE_DATA_END", f"{LAST_YEAR}-12-31"))
# Zero means every later event node in the same symbol/year.  This is the
# requested all-combinations event graph; set a positive value only for an
# explicitly bounded sensitivity run.
EVENT_HOLD_SESSIONS = int(os.getenv("COMPARE_EVENT_HOLD_SESSIONS", "0"))
MIN_EDGE_RETURN = float(os.getenv("COMPARE_MIN_EDGE_RETURN", "0.0"))
RF_ESTIMATORS = int(os.getenv("COMPARE_RF_ESTIMATORS", "80"))
MAX_TRAIN_ROWS = int(os.getenv("COMPARE_MAX_TRAIN_ROWS", "200000"))
RF_JOBS = int(os.getenv("COMPARE_RF_JOBS", "-1"))
TOP_K = int(os.getenv("COMPARE_TOP_K", "10"))
COST_BPS = float(os.getenv("COMPARE_COST_BPS", "5.5"))
CONGRESS_THRESHOLD = float(os.getenv("COMPARE_CONGRESS_THRESHOLD", "0.5"))
HITS_THRESHOLD = float(os.getenv("COMPARE_HITS_THRESHOLD", "0.8"))
ORACLE_THRESHOLD = float(os.getenv("COMPARE_ORACLE_THRESHOLD", "0.5"))
OUT = Path(os.getenv("COMPARE_OUT", str(ROOT / "artifacts" / "congress_oracle_hits_comparison")))
OUT.mkdir(parents=True, exist_ok=True)
FAMILY_FILTER = {x.strip() for x in os.getenv("COMPARE_FAMILIES", "").split(",") if x.strip()}
MODEL_FILTER = tuple(x.strip() for x in os.getenv("COMPARE_MODELS", "congress_graph,hits,oracle").split(",") if x.strip())

TIER_CONFIG = {
    "1T": (1_000_000_000_000, "equity_meta_model_1t_congress_buy_only"),
    "100B": (100_000_000_000, "equity_meta_model_100b_congress_buy_only"),
    "10B": (10_000_000_000, "equity_meta_model_10b_congress_buy_only"),
}


def feature_dir(tier: str) -> Path:
    cap, cache = TIER_CONFIG[tier]
    return ROOT / "artifacts" / "trading_app_v2" / cache / f"mcap_{cap}_train_2020-12-31_seed_20260707" / "feature_family_panels"


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return 20260718 + int.from_bytes(digest[:4], "little") % 10000


def normalize_event_rows(symbol: str, pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs is None or pairs.empty:
        return pd.DataFrame(columns=["symbol", "date", "buy", "sell", "buy_count", "sell_count"])
    pair = pairs.copy()
    chamber = pair.get("actor_chamber", pd.Series("unknown", index=pair.index)).astype(str).str.lower()
    pair = pair.loc[chamber.isin({"house", "senate"})].copy()
    if pair.empty:
        return pd.DataFrame(columns=["symbol", "date", "buy", "sell", "buy_count", "sell_count"])
    pair["date"] = pd.to_datetime(pair["event_date"], errors="coerce").dt.normalize()
    pair["side"] = pair["event_type"].map({"congress_buy": "buy", "congress_sell": "sell"})
    pair = pair.dropna(subset=["date", "side"])
    if pair.empty:
        return pd.DataFrame(columns=["symbol", "date", "buy", "sell", "buy_count", "sell_count"])
    counts = pair.groupby(["date", "side"], as_index=False).size().pivot(index="date", columns="side", values="size").fillna(0.0)
    counts = counts.rename_axis(None, axis=1).reset_index()
    for col in ("buy", "sell"):
        if col not in counts:
            counts[col] = 0.0
    counts["symbol"] = symbol.upper()
    counts["buy_count"] = counts["buy"].astype(float)
    counts["sell_count"] = counts["sell"].astype(float)
    counts["buy"] = counts["buy"].gt(0).astype(float)
    counts["sell"] = counts["sell"].gt(0).astype(float)
    return counts[["symbol", "date", "buy", "sell", "buy_count", "sell_count"]]


def _node_hits(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(weights)
    if n == 0 or not np.any(weights > 0):
        return np.zeros(n), np.zeros(n)
    hub = np.ones(n, dtype=float)
    authority = np.ones(n, dtype=float)
    for _ in range(50):
        authority = weights.T @ hub
        authority /= np.linalg.norm(authority) or 1.0
        hub = weights @ authority
        hub /= np.linalg.norm(hub) or 1.0
    return hub / (hub.max() or 1.0), authority / (authority.max() or 1.0)


def event_graph_targets(events: pd.DataFrame, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create side-specific HITS targets from congressional event-date nodes."""
    outputs: list[pd.DataFrame] = []
    for (symbol, year), group in events.assign(year=events.date.dt.year).groupby(["symbol", "year"], sort=False):
        if symbol not in prices:
            continue
        nodes = group.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        price = prices[symbol]
        close = price.set_index("date").close
        node_prices = close.reindex(nodes.date).to_numpy(float)
        if len(nodes) < 2 or not np.isfinite(node_prices).all():
            continue
        positions = price.date.reset_index(drop=True)
        node_pos = np.searchsorted(positions.to_numpy(), nodes.date.to_numpy())
        long_w = np.zeros((len(nodes), len(nodes)), dtype=float)
        short_w = np.zeros_like(long_w)
        for i in range(len(nodes) - 1):
            end = len(nodes) if EVENT_HOLD_SESSIONS <= 0 else min(len(nodes), i + 1 + EVENT_HOLD_SESSIONS)
            for j in range(i + 1, end):
                if EVENT_HOLD_SESSIONS > 0 and node_pos[j] - node_pos[i] > EVENT_HOLD_SESSIONS:
                    break
                if nodes.buy.iloc[i] and nodes.sell.iloc[j]:
                    ret = node_prices[j] / node_prices[i] - 1.0
                    long_w[i, j] = max(float(ret), MIN_EDGE_RETURN)
                if nodes.sell.iloc[i] and nodes.buy.iloc[j]:
                    ret = node_prices[i] / node_prices[j] - 1.0
                    short_w[i, j] = max(float(ret), MIN_EDGE_RETURN)
        long_hub, long_authority = _node_hits(long_w)
        short_hub, short_authority = _node_hits(short_w)
        out = nodes[["symbol", "date", "buy", "sell", "buy_count", "sell_count"]].copy()
        out["congress_long_entry"] = long_hub
        out["congress_long_exit"] = long_authority
        out["congress_short_entry"] = short_hub
        out["congress_short_exit"] = short_authority
        outputs.append(out)
    if not outputs:
        return pd.DataFrame(columns=["symbol", "date", "buy", "sell", "buy_count", "sell_count", "congress_long_entry", "congress_long_exit", "congress_short_entry", "congress_short_exit"])
    return pd.concat(outputs, ignore_index=True).drop_duplicates(["symbol", "date"])


def price_targets(prices: dict[str, pd.DataFrame], events: pd.DataFrame) -> pd.DataFrame:
    """Build all-date Oracle and regular all-date HITS labels for training."""
    rows: list[pd.DataFrame] = []
    for symbol, frame in prices.items():
        for year, year_frame in frame.groupby(frame.date.dt.year, sort=False):
            bare = year_frame[["date", "open", "high", "low", "close"]].reset_index(drop=True)
            hits = base._hits_scores(bare, "clip")
            oracle = base._oracle_events(bare)
            out = year_frame[["symbol", "date"]].reset_index(drop=True)
            for col in ("long_hub", "long_authority", "short_hub", "short_authority"):
                out[col] = hits[col].to_numpy()
            for col in ("buy", "sell", "short", "cover"):
                out[f"oracle_{col}"] = oracle[col].to_numpy()
            rows.append(out)
    daily = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    graph = event_graph_targets(events, prices)
    if daily.empty:
        return daily
    return daily.merge(graph, on=["symbol", "date"], how="left")


def clean(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    med = train[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).median().fillna(0.0)

    def one(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out[features] = out[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0).astype("float32")
        return out

    return one(train), one(test)


def fit_regressor(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str, sparse: bool = False) -> np.ndarray:
    rows = train.loc[train[target].notna()].copy()
    if sparse:
        rank = rows.groupby(["symbol", rows.date.dt.year], sort=False)[target].rank(pct=True, method="first")
        rows = rows.loc[rank.le(0.2) | rank.ge(0.8)]
    if len(rows) < max(20, len(features) * 2) or rows[target].nunique() < 2:
        return np.full(len(test), 0.5)
    if MAX_TRAIN_ROWS > 0 and len(rows) > MAX_TRAIN_ROWS:
        rows = rows.sample(MAX_TRAIN_ROWS, random_state=stable_seed(target, "sample"))
    model = RandomForestRegressor(n_estimators=RF_ESTIMATORS, max_depth=16, min_samples_leaf=3, random_state=stable_seed(target), n_jobs=RF_JOBS)
    model.fit(rows[features], rows[target].astype("float32"))
    return np.clip(model.predict(test[features]), 0.0, 1.0)


def oracle_scores(train: pd.DataFrame, test: pd.DataFrame, features: list[str], side: str) -> tuple[np.ndarray, np.ndarray]:
    entry_label, exit_label = ("oracle_buy", "oracle_sell") if side == "long" else ("oracle_short", "oracle_cover")
    rows = train.loc[train[[entry_label, exit_label]].sum(axis=1).gt(0)].copy()
    rows["target"] = np.where(rows[entry_label].eq(1), entry_label, exit_label)
    rows = rows.loc[rows.target.isin([entry_label, exit_label])]
    if len(rows) < 20 or rows.target.nunique() < 2:
        return np.zeros(len(test)), np.zeros(len(test))
    if MAX_TRAIN_ROWS > 0 and len(rows) > MAX_TRAIN_ROWS:
        rows = rows.sample(MAX_TRAIN_ROWS, random_state=stable_seed("oracle", side, "sample"))
    model = RandomForestClassifier(n_estimators=RF_ESTIMATORS, max_depth=16, min_samples_leaf=2, class_weight="balanced", random_state=stable_seed("oracle", side), n_jobs=RF_JOBS)
    model.fit(rows[features], rows.target)
    proba = model.predict_proba(test[features])
    classes = list(model.classes_)
    entry = proba[:, classes.index(entry_label)] if entry_label in classes else np.zeros(len(test))
    exit_ = proba[:, classes.index(exit_label)] if exit_label in classes else np.zeros(len(test))
    return entry, exit_


def model_scores(train: pd.DataFrame, test: pd.DataFrame, features: list[str], model: str, side: str) -> tuple[np.ndarray, np.ndarray]:
    if model == "congress_graph":
        entry = fit_regressor(train, test, features, f"congress_{side}_entry")
        exit_ = fit_regressor(train, test, features, f"congress_{side}_exit")
        return entry, exit_
    if model == "hits":
        entry = fit_regressor(train, test, features, f"{side}_hub", sparse=True)
        exit_ = fit_regressor(train, test, features, f"{side}_authority", sparse=True)
        # HITS entry/exit scores are selected by cross-sectional percentile,
        # matching the regular cross-rank Oracle/HITS workflow.  Raw forest
        # regression values are not calibrated to the [0, 1] HITS threshold.
        entry = pd.Series(entry, index=test.index).groupby(test.date).rank(pct=True, method="average").to_numpy()
        exit_ = pd.Series(exit_, index=test.index).groupby(test.date).rank(pct=True, method="average").to_numpy()
        return entry, exit_
    if model == "oracle":
        return oracle_scores(train, test, features, side)
    raise ValueError(model)


def run_family(meta: pd.Series, labels: pd.DataFrame, close: pd.DataFrame, year: int, tier: str) -> list[dict]:
    if FAMILY_FILTER and str(meta.family) not in FAMILY_FILTER:
        return []
    panel, metadata = pd.read_parquet(meta.panel_path), pd.read_parquet(meta.metadata_path)
    features = [str(c) for c in metadata.feature if str(c) in panel.columns]
    if not features:
        return []
    frame = panel[["symbol", "date", *features]].copy()
    frame.symbol = frame.symbol.astype(str).str.upper()
    frame.date = pd.to_datetime(frame.date).dt.normalize()
    frame = frame.merge(labels, on=["symbol", "date"], how="inner")
    train = frame.loc[frame.date.dt.year < year].copy()
    test = frame.loc[frame.date.dt.year.eq(year)].copy()
    if train.empty or test.empty:
        return []
    train, test = clean(train, test, features)
    dates = pd.DatetimeIndex(close.index[(close.index >= f"{year}-01-01") & (close.index <= f"{year}-12-31")])
    rows: list[dict] = []
    thresholds = {"congress_graph": CONGRESS_THRESHOLD, "hits": HITS_THRESHOLD, "oracle": ORACLE_THRESHOLD}
    for model in MODEL_FILTER:
        threshold = thresholds[model]
        for side in ("long", "short"):
            entry, exit_ = model_scores(train, test, features, model, side)
            pred = test[["symbol", "date"]].copy()
            if side == "long":
                pred["long_score"], pred["short_score"] = entry, 0.0
                pred["long_exit_score"], pred["short_exit_score"] = 0.0, exit_
            else:
                pred["long_score"], pred["short_score"] = 0.0, entry
                pred["long_exit_score"], pred["short_exit_score"] = exit_, 0.0
            eligible = sorted(pred.symbol.astype(str).str.upper().unique())
            metrics = base.backtest_scores(pred, close.loc[:, close.columns.intersection(eligible)], side, dates, threshold, TOP_K, COST_BPS)
            metrics.update({"tier": tier, "year": year, "model": model, "variant": side, "family": str(meta.family), "source": str(meta.source), "top_k": TOP_K, "train_rows": len(train), "test_rows": len(test), "train_end": pd.Timestamp(f"{year - 1}-12-31")})
            rows.append(metrics)
    return rows


def run_tier(tier: str) -> pd.DataFrame:
    index = pd.read_csv(feature_dir(tier) / "index.csv")
    first = pd.read_parquet(index.iloc[0].panel_path)
    symbols = sorted(first.symbol.astype(str).str.upper().unique())
    warehouse = Warehouse()
    prices: dict[str, pd.DataFrame] = {}
    event_frames: list[pd.DataFrame] = []
    for i, symbol in enumerate(symbols, 1):
        raw = warehouse.read_prices(symbol, provider="fmp", start=str(DATA_START.date()), end=str(DATA_END.date()))
        if raw is None or raw.empty:
            continue
        frame = base.normalize_prices(raw)
        frame = frame.loc[frame.date.between(DATA_START, DATA_END)].copy()
        if len(frame) < 30:
            continue
        frame.insert(0, "symbol", symbol)
        prices[symbol] = frame
        pairs = build_event_pairs_from_historical_data(symbol, fundamentals=warehouse.fundamentals, event_families=("congress",), provider="fmp", start_date=str(DATA_START.date()), end_date=str(DATA_END.date()))
        events = normalize_event_rows(symbol, pairs)
        if not events.empty:
            event_frames.append(events)
        if i % 100 == 0:
            print({"tier": tier, "prices_loaded": i, "usable_symbols": len(prices)}, flush=True)
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame(columns=["symbol", "date", "buy", "sell", "buy_count", "sell_count"])
    labels = price_targets(prices, events)
    close = pd.DataFrame({s: f.set_index("date").close for s, f in prices.items()}).sort_index().ffill()
    print({"tier": tier, "symbols": len(symbols), "price_symbols": len(prices), "event_rows": len(events), "daily_rows": len(labels)}, flush=True)
    rows: list[dict] = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        year_rows: list[dict] = []
        for _, meta in index.iterrows():
            year_rows.extend(run_family(meta, labels, close, year, tier))
        rows.extend(year_rows)
        pd.DataFrame(rows).to_parquet(OUT / f"{tier.lower()}_through_{year}.parquet", index=False)
        print({"tier": tier, "year": year, "rows": len(year_rows)}, flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default=os.getenv("COMPARE_TIERS", "1T,100B,10B"), help="Comma-separated configured tiers")
    args = parser.parse_args()
    tiers = tuple(x.strip().upper() for x in args.universe.split(",") if x.strip())
    invalid = sorted(set(tiers).difference(TIER_CONFIG))
    if invalid:
        parser.error(f"unknown tier(s): {', '.join(invalid)}")
    frames = [run_tier(tier) for tier in tiers]
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result.to_csv(OUT / "all_results.csv", index=False)
    if not result.empty:
        summary = result.groupby(["tier", "year", "model", "variant", "top_k"], as_index=False).agg(
            families=("family", "nunique"), mean_return=("total_return", "mean"), median_return=("total_return", "median"), mean_sharpe=("sharpe", "mean"), total_trades=("trades", "sum")
        )
        summary.to_csv(OUT / "summary.csv", index=False)
        print(summary.round(4).to_string(index=False))
    (OUT / "run_config.json").write_text(json.dumps({"tiers": tiers, "first_year": FIRST_YEAR, "last_year": LAST_YEAR, "data_start": str(DATA_START.date()), "data_end": str(DATA_END.date()), "top_k": TOP_K, "cost_bps": COST_BPS}, indent=2))


if __name__ == "__main__":
    main()
