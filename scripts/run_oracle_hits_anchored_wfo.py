"""Anchored walk-forward Oracle versus sparse-HITS experiment.

Initial training data ends in 2020.  Each subsequent calendar year is a
separate backtest, and models are retrained on every prior year.  Oracle
models are trained only on oracle event dates; HITS uses sparse top/bottom
hub/authority tails.  Long and short books remain independent.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import run_oracle_vs_hits_entry_exit_2024_2025 as base
from quant_orchestrator.platforms.ml_frameworks.rapids.random_forest import RapidsRandomForestClassifier

INITIAL_TRAIN_END = pd.Timestamp(os.getenv("WFO_INITIAL_TRAIN_END", "2020-12-31"))
FIRST_TEST_YEAR = int(os.getenv("WFO_FIRST_TEST_YEAR", "2021"))
LAST_TEST_YEAR = int(os.getenv("WFO_LAST_TEST_YEAR", "2025"))
DATA_START = pd.Timestamp(os.getenv("WFO_DATA_START", "2015-01-01"))
DATA_END = pd.Timestamp(os.getenv("WFO_DATA_END", "2025-12-31"))
RF_ESTIMATORS = int(os.getenv("WFO_RF_ESTIMATORS", "40"))
HITS_WEIGHTING = os.getenv("WFO_HITS_WEIGHTING", "clip")
VARIANTS = tuple(x.strip() for x in os.getenv("WFO_VARIANTS", "long,short").split(",") if x.strip())
ORACLE_THRESHOLDS = tuple(float(x) for x in os.getenv("WFO_ORACLE_THRESHOLDS", str(base.ORACLE_THRESHOLD)).split(",") if x.strip())
HITS_THRESHOLDS = tuple(float(x) for x in os.getenv("WFO_HITS_THRESHOLDS", str(base.HITS_THRESHOLD)).split(",") if x.strip())
TOP_KS = tuple(int(x) for x in os.getenv("WFO_TOP_KS", "20").split(",") if x.strip())
PORTFOLIO_COST_BPS = float(os.getenv("WFO_PORTFOLIO_COST_BPS", "5.5"))
OUT = Path(os.getenv("WFO_OUT", str(REPO_ROOT / "artifacts" / "oracle_hits_anchored_wfo")))
OUT.mkdir(parents=True, exist_ok=True)


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return base.SEED + int.from_bytes(digest[:4], "little") % 10000


def load_data(tier: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.read_csv(base.feature_dir(tier) / "index.csv")
    first = pd.read_parquet(index.iloc[0].panel_path)
    symbols = sorted(first.symbol.astype(str).str.upper().unique())
    warehouse = base.Warehouse()
    price_rows, label_rows = [], []
    years = set(range(DATA_START.year, DATA_END.year + 1))
    for i, symbol in enumerate(symbols, 1):
        raw = warehouse.read_prices(symbol, provider="fmp", start=str(DATA_START.date()), end=str(DATA_END.date()))
        if raw is None or raw.empty:
            continue
        try:
            prices = base.normalize_prices(raw)
        except Exception:
            continue
        prices = prices.loc[prices.date.between(DATA_START, DATA_END)].copy()
        if len(prices) < 30:
            continue
        p = prices.copy(); p.insert(0, "symbol", symbol); price_rows.append(p)
        for year, year_frame in p.groupby(p.date.dt.year):
            if year not in years or len(year_frame) < 10:
                continue
            bare = year_frame.drop(columns="symbol")
            h = base._hits_scores(bare, "clip")
            r = base._hits_scores(bare, "rank").drop(columns="date")
            events = base._oracle_events(bare)
            row = pd.concat([
                year_frame[["symbol", "date"]].reset_index(drop=True),
                h.drop(columns="date"),
                r.reset_index(drop=True).add_suffix("_rank"),
                events.drop(columns="date"),
            ], axis=1)
            label_rows.append(row)
        if i % 100 == 0:
            print({"tier": tier, "prices_loaded": i, "usable_symbols": len(price_rows)}, flush=True)
    prices = pd.concat(price_rows, ignore_index=True) if price_rows else pd.DataFrame()
    labels = pd.concat(label_rows, ignore_index=True) if label_rows else pd.DataFrame()
    return index, prices, labels


def oracle_scores(train: pd.DataFrame, test: pd.DataFrame, features: list[str], side: str) -> tuple[np.ndarray, np.ndarray] | None:
    entry_label, exit_label = ("buy", "sell") if side == "long" else ("short", "cover")
    rows = train.loc[train[[entry_label, exit_label]].sum(axis=1).gt(0)].copy()
    rows["target"] = np.where(rows[entry_label].eq(1), entry_label, exit_label)
    rows = rows.loc[rows.target.isin([entry_label, exit_label])]
    if len(rows) < 10 or rows.target.nunique() < 2:
        return None
    model = RapidsRandomForestClassifier.fit(
        rows, features=features, target_col="target", random_state=base.SEED,
        params={"n_estimators": RF_ESTIMATORS, "max_depth": 16, "max_features": "sqrt", "n_bins": 128, "n_streams": 8},
    )
    proba = model.predict_proba_frame(test, features)
    entry = proba.get(f"prob__{entry_label}", pd.Series(0.0, index=test.index)).to_numpy()
    exit_ = proba.get(f"prob__{exit_label}", pd.Series(0.0, index=test.index)).to_numpy()
    return entry, exit_


def hits_score(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str, family: str, train_year: int) -> np.ndarray | None:
    sparse = base.select_top_bottom_rows(train, target)
    if len(sparse) < max(20, len(features) * 2):
        return None
    values = base.reg_predict(sparse, test, features, target, stable_seed(family, target, str(train_year)))
    return pd.Series(values, index=test.index).groupby(test.date).rank(pct=True, method="average").to_numpy()


def run_family(row: pd.Series, prices: pd.DataFrame, labels: pd.DataFrame, close: pd.DataFrame, year: int, tier: str) -> list[dict]:
    panel = pd.read_parquet(row.panel_path)
    metadata = pd.read_parquet(row.metadata_path)
    family, source = str(row.family), str(row.source)
    features = [c for c in metadata.feature.astype(str) if c in panel.columns]
    if not features:
        return []
    frame = panel[["symbol", "date", *features]].copy()
    frame.symbol = frame.symbol.astype(str).str.upper()
    frame.date = pd.to_datetime(frame.date).dt.normalize()
    frame = frame.merge(labels, on=["symbol", "date"], how="inner")
    train_end = pd.Timestamp(f"{year - 1}-12-31")
    test_start, test_end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
    frame = frame.loc[frame.date.between(DATA_START, test_end)].reset_index(drop=True)
    train = frame.loc[frame.date.between(DATA_START, train_end)].copy()
    test = frame.loc[frame.date.between(test_start, test_end)].copy()
    if train.empty or test.empty:
        return []
    medians = train[features].apply(pd.to_numeric, errors="coerce").median().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train = base.clean_features(train, features, medians)
    test = base.clean_features(test, features, medians)
    dates = pd.DatetimeIndex(close.index[(close.index >= test_start) & (close.index <= test_end)])
    result = []

    for side in VARIANTS:
        scores = oracle_scores(train, test, features, side)
        if scores is None:
            continue
        entry, exit_ = scores
        pred = test[["symbol", "date"]].copy()
        pred["long_score"], pred["short_score"] = (entry, 0.0) if side == "long" else (0.0, entry)
        pred["long_exit_score"], pred["short_exit_score"] = (0.0, exit_) if side == "long" else (exit_, 0.0)
        for threshold in ORACLE_THRESHOLDS:
            for top_k in TOP_KS:
                metrics = base.backtest_scores(pred, close, side, dates, threshold, top_k, PORTFOLIO_COST_BPS)
                metrics.update({"tier": tier, "year": year, "train_end": train_end, "model": "oracle_separate", "variant": side, "family": family, "source": source, "train_rows": len(train), "entry_threshold": threshold, "exit_threshold": threshold, "portfolio_top_k": top_k})
                result.append(metrics)

    components = {}
    for side in VARIANTS:
        for role in ("hub", "authority"):
            target = f"{side}_{role}" if HITS_WEIGHTING == "clip" else f"{side}_{role}_rank"
            values = hits_score(train, test, features, target, family, year)
            if values is not None:
                components[(side, role)] = values
    for side in ("long", "short"):
        if (side, "hub") not in components or (side, "authority") not in components:
            continue
        pred = test[["symbol", "date"]].copy()
        hub, authority = components[(side, "hub")], components[(side, "authority")]
        pred["long_score"], pred["short_score"] = (hub, 0.0) if side == "long" else (0.0, hub)
        pred["long_exit_score"], pred["short_exit_score"] = (0.0, authority) if side == "long" else (authority, 0.0)
        for threshold in HITS_THRESHOLDS:
            for top_k in TOP_KS:
                metrics = base.backtest_scores(pred, close, side, dates, threshold, top_k, PORTFOLIO_COST_BPS)
                metrics.update({"tier": tier, "year": year, "train_end": train_end, "model": "hits_two_per_side", "variant": side, "family": family, "source": source, "train_rows": len(train), "tail_quantile": base.HITS_TAIL_QUANTILE, "entry_threshold": threshold, "exit_threshold": threshold, "portfolio_top_k": top_k})
                result.append(metrics)
    return result


def run_tier(tier: str) -> pd.DataFrame:
    index, prices, labels = load_data(tier)
    index["tier"] = tier
    requested = tuple(x.strip() for x in os.getenv("WFO_FAMILIES", "").split(",") if x.strip())
    if requested:
        index = index.loc[index.family.astype(str).isin(requested)]
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    rows = []
    for year in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1):
        t0 = perf_counter()
        for _, row in index.iterrows():
            rows.extend(run_family(row, prices, labels, close, year, tier))
        print({"tier": tier, "year": year, "rows": len(rows), "seconds": round(perf_counter() - t0, 1)}, flush=True)
        pd.DataFrame(rows).to_parquet(OUT / f"{tier.lower()}_results.parquet", index=False)
    return pd.DataFrame(rows)


def main() -> None:
    tiers = tuple(x.strip().upper() for x in os.getenv("WFO_TIERS", "1T,100B,10B").split(",") if x.strip())
    all_rows = [run_tier(tier) for tier in tiers]
    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    out.to_csv(OUT / "all_results.csv", index=False)
    summary = out.groupby(["tier", "year", "model", "variant", "entry_threshold", "portfolio_top_k"], as_index=False).agg(
        families=("family", "nunique"), mean_return=("total_return", "mean"), median_return=("total_return", "median"),
        min_return=("total_return", "min"), max_return=("total_return", "max"), mean_sharpe=("sharpe", "mean"),
    ) if not out.empty else out
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
