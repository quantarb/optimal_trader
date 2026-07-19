"""Congress/Senate trade-date classifier + ranker experiment.

This is a small, event-date-only companion to the existing Oracle/HITS runs.
The first stage predicts buy versus sell on public House/Senate trade rows.
The second stage ranks candidates within each trade date and side by their
forward realized return.  Existing Oracle and HITS scores are evaluated on
the same event-date universe for an apples-to-apples comparison.

The ranker uses LightGBM LambdaMART when installed and otherwise falls back to
an sklearn random-forest regressor trained on within-date return percentiles.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path[:0] = [str(ROOT), str(WORKSPACE / "quant-warehouse"), str(WORKSPACE / "quant-orchestrator")]

from quant_warehouse.platforms.data_providers.fmp.target_engineering import (  # noqa: E402
    LabelBuildSpec,
    build_trade_results,
)
from quant_warehouse.platforms.data_providers.fmp.target_engineering.event_pairs.store import (  # noqa: E402
    build_event_pairs_from_historical_data,
)
from quant_warehouse.warehouse.api import Warehouse  # noqa: E402

import run_oracle_vs_hits_entry_exit_2024_2025 as base  # noqa: E402

FIRST_YEAR = int(os.getenv("CS_FIRST_YEAR", "2021"))
LAST_YEAR = int(os.getenv("CS_LAST_YEAR", "2025"))
DATA_START = pd.Timestamp(os.getenv("CS_DATA_START", "1900-01-01"))
DATA_END = pd.Timestamp(os.getenv("CS_DATA_END", f"{LAST_YEAR}-12-31"))
_TOP_KS_ENV = os.getenv("CS_TOP_KS", "")
TOP_KS = tuple(int(x) for x in (_TOP_KS_ENV.split(",") if _TOP_KS_ENV else [os.getenv("CS_TOP_K", "10")]) if x.strip())
MAX_HOLD = int(os.getenv("CS_MAX_HOLD", "20"))
MIN_RETURN = float(os.getenv("CS_MIN_RETURN", "0.01"))
OUT = Path(os.getenv("CS_OUT", str(ROOT / "artifacts" / "congress_senate_classifier_ranker_hits_anchored_wfo")))
OUT.mkdir(parents=True, exist_ok=True)
FAMILY_FILTER = {x.strip() for x in os.getenv("CS_FAMILIES", "").split(",") if x.strip()}
_ORACLE_CACHE: dict[tuple[str, int], pd.DataFrame] = {}
_HITS_CACHE: dict[tuple[str, int], pd.DataFrame] = {}

TIER_CONFIG = {
    "1T": (1_000_000_000_000, "equity_meta_model_1t_congress_buy_only"),
    "100B": (100_000_000_000, "equity_meta_model_100b_congress_buy_only"),
    "10B": (10_000_000_000, "equity_meta_model_10b_congress_buy_only"),
}


def feature_dir(tier: str) -> Path:
    cap, cache = TIER_CONFIG[tier]
    return ROOT / "artifacts" / "trading_app_v2" / cache / f"mcap_{cap}_train_2020-12-31_seed_20260707" / "feature_family_panels"


def prices_for(symbols: list[str]) -> dict[str, pd.DataFrame]:
    wh = Warehouse()
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        raw = wh.read_prices(symbol, provider="fmp", start=str(DATA_START.date()), end=str(DATA_END.date()))
        if raw is None or raw.empty:
            continue
        frame = base.normalize_prices(raw)
        frame = frame.loc[frame.date.between(DATA_START, DATA_END)].copy()
        if len(frame) >= MAX_HOLD + 5:
            out[symbol] = frame
    return out


def congress_labels(symbols: list[str], wh: Warehouse, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for symbol in symbols:
        if symbol not in prices:
            continue
        pairs = build_event_pairs_from_historical_data(
            symbol,
            fundamentals=wh.fundamentals,
            event_families=("congress",),
            provider="fmp",
            start_date=str(DATA_START.date()),
            end_date=str(DATA_END.date()),
        )
        if pairs is None or pairs.empty:
            continue
        pair = pairs.copy()
        chamber = pair.get("actor_chamber", pd.Series("unknown", index=pair.index)).astype(str).str.lower()
        # Keep both public chambers; exclude malformed/unknown rows explicitly.
        pair = pair.loc[chamber.isin({"house", "senate"})].copy()
        if pair.empty:
            continue
        pair["symbol"] = symbol
        pair["date"] = pd.to_datetime(pair["event_date"], errors="coerce").dt.normalize()
        pair["label"] = pair["event_type"].map({"congress_buy": "buy", "congress_sell": "sell"})
        pair = pair.dropna(subset=["date", "label"])
        rows.append(pair[["symbol", "date", "label", "actor_chamber"]])
    if not rows:
        return pd.DataFrame(columns=["symbol", "date", "label", "actor_chamber"])
    return pd.concat(rows, ignore_index=True).drop_duplicates(["symbol", "date", "label", "actor_chamber"])


def forward_return(frame: pd.DataFrame, date: pd.Timestamp, side: str) -> float:
    dates = frame.date.to_numpy()
    idx = int(np.searchsorted(dates, np.datetime64(date)))
    end = min(len(frame), idx + MAX_HOLD + 1)
    if idx >= len(frame) or end <= idx + 1:
        return np.nan
    entry = float(frame.iloc[idx].close)
    future = frame.iloc[idx + 1:end]
    if side == "buy":
        return float(future.close.max() / entry - 1.0)
    return float(entry / future.close.min() - 1.0)


def event_targets(labels: pd.DataFrame, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = labels.copy()
    out["forward_return"] = [forward_return(prices[s], d, side) for s, d, side in zip(out.symbol, out.date, out.label)]
    out = out.dropna(subset=["forward_return"])
    out["rank_target"] = out.groupby(["date", "label"], sort=False)["forward_return"].rank(pct=True, method="average")
    return out


def clean(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    med = train[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).median().fillna(0.0)
    def one(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out[features] = out[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0).astype("float32")
        return out
    return one(train), one(test)


def fit_ranker(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, str]:
    usable = train.loc[train.rank_target.notna()].copy()
    if len(usable) < 20 or usable.rank_target.nunique() < 2:
        return np.full(len(test), 0.5), "none"
    try:
        from lightgbm import LGBMRanker
        groups = usable.groupby(["date", "label"], sort=False).size().to_numpy()
        model = LGBMRanker(objective="lambdarank", metric="ndcg", n_estimators=80, learning_rate=0.04, num_leaves=15, random_state=20260718, verbosity=-1)
        model.fit(usable[features], usable.rank_target, group=groups)
        return np.clip(model.predict(test[features]), 0.0, 1.0), "lightgbm_lambdarank"
    except ImportError:
        model = RandomForestRegressor(n_estimators=80, max_depth=12, min_samples_leaf=3, random_state=20260718, n_jobs=-1)
        model.fit(usable[features], usable.rank_target)
        return np.clip(model.predict(test[features]), 0.0, 1.0), "sklearn_random_forest_regressor"


def event_oracle_scores(test: pd.DataFrame, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    scores = np.zeros((len(test), 4), dtype=float)
    for i, row in enumerate(test.itertuples(index=False)):
        frame = prices[row.symbol]
        year = pd.Timestamp(row.date).year
        one = frame.loc[frame.date.dt.year.eq(year)].reset_index(drop=True)
        key = (row.symbol, year)
        events = _ORACLE_CACHE.get(key)
        if events is None:
            events = base._oracle_events(one) if not one.empty else pd.DataFrame({"date": [], "buy": [], "sell": [], "short": [], "cover": []})
            _ORACLE_CACHE[key] = events
        if (events.date == row.date).any():
            event = events.loc[events.date.eq(row.date)].iloc[0]
            scores[i] = [event.buy, event.short, event.sell, event.cover]
    return pd.DataFrame(scores, columns=["long_entry", "short_entry", "long_exit", "short_exit"])


def hits_scores(test: pd.DataFrame, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    scores = np.zeros((len(test), 4), dtype=float)
    for i, row in enumerate(test.itertuples(index=False)):
        frame = prices[row.symbol]
        year_number = pd.Timestamp(row.date).year
        year = frame.loc[frame.date.dt.year.eq(year_number)].reset_index(drop=True)
        key = (row.symbol, year_number)
        graph = _HITS_CACHE.get(key)
        if graph is None:
            graph = base._hits_scores(year, "clip") if not year.empty else pd.DataFrame({"date": [], "long_hub": [], "long_authority": [], "short_hub": [], "short_authority": []})
            _HITS_CACHE[key] = graph
        if year.empty or not len(graph):
            continue
        pos = int(np.searchsorted(year.date.to_numpy(), np.datetime64(row.date)))
        scores[i] = [
            pd.Series(graph["long_hub"]).rank(pct=True).iloc[pos],
            pd.Series(graph["short_hub"]).rank(pct=True).iloc[pos],
            pd.Series(graph["long_authority"]).rank(pct=True).iloc[pos],
            pd.Series(graph["short_authority"]).rank(pct=True).iloc[pos],
        ]
    return pd.DataFrame(scores, columns=["long_entry", "short_entry", "long_exit", "short_exit"])


def add_baseline_scores(labels: pd.DataFrame, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute price-only baselines once per tier, not once per feature family."""
    out = labels.copy()
    oracle = event_oracle_scores(out, prices)
    hits = hits_scores(out, prices)
    for col in oracle.columns:
        out[f"oracle_{col}"] = oracle[col].to_numpy()
        out[f"hits_{col}"] = hits[col].to_numpy()
    return out


def run_family(meta: pd.Series, labels: pd.DataFrame, prices: dict[str, pd.DataFrame], close: pd.DataFrame, year: int, tier: str, score_cache_dir: Path) -> list[dict]:
    if FAMILY_FILTER and str(meta.family) not in FAMILY_FILTER:
        return []
    panel, metadata = pd.read_parquet(meta.panel_path), pd.read_parquet(meta.metadata_path)
    cache_path = score_cache_dir / f"{str(meta.family)}_{year}.parquet"
    cache_meta_path = cache_path.with_suffix(".json")
    score_cache_dir.mkdir(parents=True, exist_ok=True)
    features = [str(c) for c in metadata.feature if str(c) in panel.columns]
    if not features:
        return []
    frame = panel[["symbol", "date", *features]].copy()
    frame.symbol = frame.symbol.astype(str).str.upper(); frame.date = pd.to_datetime(frame.date).dt.normalize()
    frame = frame.merge(labels, on=["symbol", "date"], how="inner")
    train = frame.loc[frame.date.dt.year < year].copy(); test = frame.loc[frame.date.dt.year.eq(year)].copy()
    if len(train) < 40 or test.empty or train.label.nunique() < 2:
        return []
    train_rows = len(train)
    train_start = train.date.min()
    if cache_path.exists():
        out = pd.read_parquet(cache_path)
        cache_meta = json.loads(cache_meta_path.read_text()) if cache_meta_path.exists() else {}
        backend = str(cache_meta.get("ranker_backend", "cached"))
    else:
        train, test = clean(train, test, features)
        clf = RandomForestClassifier(n_estimators=120, max_depth=14, min_samples_leaf=2, class_weight="balanced", random_state=20260718, n_jobs=-1)
        clf.fit(train[features], train.label)
        buy_idx = list(clf.classes_).index("buy")
        clf_score = clf.predict_proba(test[features])[:, buy_idx]
        rank_score, backend = fit_ranker(train, test, features)
        out = test[["symbol", "date", "label", "actor_chamber"]].copy()
        out["classifier_long_entry"] = clf_score
        out["classifier_short_entry"] = 1.0 - clf_score
        out["classifier_long_exit"] = 1.0 - clf_score
        out["classifier_short_exit"] = clf_score
        out["rank_score"] = rank_score
        out["ranker_long_entry"] = out.classifier_long_entry * out.rank_score
        out["ranker_short_entry"] = out.classifier_short_entry * out.rank_score
        baseline = labels.loc[labels.date.dt.year.eq(year)].drop_duplicates(["symbol", "date"]).set_index(["symbol", "date"])
        keys = pd.MultiIndex.from_frame(out[["symbol", "date"]])
        for col in ("long_entry", "short_entry", "long_exit", "short_exit"):
            out[f"oracle_{col}"] = baseline[f"oracle_{col}"].reindex(keys).to_numpy()
            out[f"hits_{col}"] = baseline[f"hits_{col}"].reindex(keys).to_numpy()
        out.to_parquet(cache_path, index=False)
        cache_meta_path.write_text(json.dumps({"train_rows": train_rows, "train_start": str(train_start), "ranker_backend": backend}, indent=2))
    rows = []
    score_sets = {
        "classifier": ("classifier_long_entry", "classifier_short_entry", "classifier_long_exit", "classifier_short_exit"),
        "classifier_ranker": ("ranker_long_entry", "ranker_short_entry", "classifier_long_exit", "classifier_short_exit"),
        "oracle": ("oracle_long_entry", "oracle_short_entry", "oracle_long_exit", "oracle_short_exit"),
        "hits": ("hits_long_entry", "hits_short_entry", "hits_long_exit", "hits_short_exit"),
    }
    for model, cols in score_sets.items():
        for top_k in TOP_KS:
          for side, label in (("long", "buy"), ("short", "sell")):
            pred = out[["symbol", "date"]].copy()
            pred["long_score"], pred["short_score"] = out[cols[0]], out[cols[1]]
            pred["long_exit_score"], pred["short_exit_score"] = out[cols[2]], out[cols[3]]
            dates = pd.DatetimeIndex(sorted(pred.date.unique()))
            eligible = sorted(pred.symbol.astype(str).str.upper().unique())
            effective_top_k = min(top_k, len(eligible))
            metrics = base.backtest_scores(pred, close.loc[:, close.columns.intersection(eligible)], side, dates, 0.5, effective_top_k, 5.5)
            metrics.update({"tier": tier, "year": year, "top_k": top_k, "train_start": train_start, "train_end": pd.Timestamp(f"{year - 1}-12-31"), "model": model, "variant": side, "family": str(meta.family), "source": str(meta.source), "train_rows": train_rows, "event_rows": len(out), "ranker_backend": backend})
            rows.append(metrics)
    return rows


def run_tier(tier: str) -> pd.DataFrame:
    index = pd.read_csv(feature_dir(tier) / "index.csv")
    first = pd.read_parquet(index.iloc[0].panel_path)
    symbols = sorted(first.symbol.astype(str).str.upper().unique())
    prices = prices_for(symbols)
    labels = event_targets(congress_labels(symbols, Warehouse(), prices), prices)
    labels = add_baseline_scores(labels, prices)
    print({"tier": tier, "symbols": len(symbols), "price_symbols": len(prices), "trade_rows": len(labels), "chambers": labels.actor_chamber.value_counts().to_dict()}, flush=True)
    rows = []
    close = pd.DataFrame({s: f.set_index("date").close for s, f in prices.items()}).sort_index().ffill()
    score_cache_dir = OUT / "score_cache" / tier.lower()
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        for _, meta in index.iterrows():
            rows.extend(run_family(meta, labels, prices, close, year, tier, score_cache_dir))
        pd.DataFrame(rows).to_parquet(OUT / f"{tier.lower()}_through_{year}.parquet", index=False)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run congressional/senate classifier-ranker WFO by market-cap universe.")
    parser.add_argument("--universe", "--tier", dest="universe", default=os.getenv("CS_TIERS", "1T"), help="Universe: 1T, 100B, 10B, or comma-separated values.")
    args = parser.parse_args()
    tiers = tuple(x.strip().upper() for x in args.universe.split(",") if x.strip())
    invalid = sorted(set(tiers).difference(TIER_CONFIG))
    if invalid:
        parser.error(f"unknown universe(s): {', '.join(invalid)}; choose from 1T, 100B, 10B")
    frames = [run_tier(tier) for tier in tiers]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(OUT / "all_results.csv", index=False)
    if not out.empty:
        summary = out.groupby(["tier", "year", "top_k", "model", "variant"], as_index=False).agg(
            families=("family", "nunique"), mean_return=("total_return", "mean"), median_return=("total_return", "median"), mean_sharpe=("sharpe", "mean"), trades=("trades", "sum")
        )
        summary.to_csv(OUT / "summary.csv", index=False)
        print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
