"""Directional RF baseline using the same sparse HITS labels as the GNN."""
from __future__ import annotations

import hashlib
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from quant_orchestrator.platforms.backtesting_frameworks.shared_book import (
    SharedBookCostModel,
    run_shared_book_framework_comparison,
)
from run_feature_family_gnn_smoke import (
    HITS_ITERATIONS,
    HITS_TAIL_QUANTILE,
    MAX_HOLD,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    TIER_CONFIGS,
    build_price_and_labels,
    feature_dir,
)

RF_ESTIMATORS = int(os.getenv("RF_HITS_ESTIMATORS", "40"))
RF_VARIANTS = tuple(x.strip() for x in os.getenv("RF_HITS_VARIANTS", "long_only,short_only").split(",") if x.strip())
OUT = REPO_ROOT / "artifacts" / "graph_oracle_feature_family_rf_sparse_hits"
OUT.mkdir(parents=True, exist_ok=True)


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return 20260716 + int.from_bytes(digest[:4], "little") % 10000


def reg_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str, seed: int) -> np.ndarray | None:
    sparse = train.loc[train[f"{target}_tail"]].copy()
    if len(sparse) < max(20, len(features) * 2):
        return None
    model = RandomForestRegressor(
        n_estimators=RF_ESTIMATORS,
        max_depth=16,
        max_features=1.0,
        n_streams=8,
        random_state=seed,
    )
    model.fit(
        cudf.from_pandas(sparse[features].astype("float32")),
        cudf.Series(sparse[target].astype("float32").to_numpy()),
    )
    values = model.predict(cudf.from_pandas(test[features].astype("float32")))
    return cp.asnumpy(values) if hasattr(values, "__cuda_array_interface__") else np.asarray(values)


def run_family(meta: pd.Series, price_map: pd.DataFrame, labels: pd.DataFrame, close: pd.DataFrame, dates: pd.DatetimeIndex, tier: str) -> list[pd.DataFrame]:
    panel = pd.read_parquet(meta.panel_path)
    metadata = pd.read_parquet(meta.metadata_path)
    family = str(meta.family)
    features = [c for c in metadata.feature.astype(str) if c in panel.columns]
    if not features:
        return []
    frame = panel[["symbol", "date", *features]].copy()
    frame["symbol"] = frame.symbol.astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame.date).dt.normalize()
    frame = frame.merge(price_map, on=["symbol", "date"], how="inner").sort_values(["symbol", "date"]).reset_index(drop=True)
    frame = frame.loc[frame.date.between(TRAIN_START, TEST_END)].reset_index(drop=True)
    if frame.empty:
        return []
    y = labels.copy()
    y["symbol"] = y.symbol.astype(str).str.upper()
    y["date"] = pd.to_datetime(y.date).dt.normalize()
    frame = frame.merge(y, on=["symbol", "date"], how="left")
    targets = ["long_hub", "long_authority", "short_hub", "short_authority"]
    tails = [f"{target}_tail" for target in targets]
    frame[targets] = frame[targets].fillna(0.0).astype("float32")
    frame[tails] = frame[tails].fillna(False).astype(bool)
    train_mask = frame.date.between(TRAIN_START, TRAIN_END)
    train = frame.loc[train_mask].copy()
    test = frame.loc[frame.date.between(TEST_START, TEST_END)].copy()
    if train.empty or test.empty:
        return []
    med = train[features].apply(pd.to_numeric, errors="coerce").median().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    mean = x.loc[train_mask].mean()
    std = x.loc[train_mask].std().replace(0, 1).fillna(1.0)
    frame[features] = ((x - mean) / std).clip(-8, 8).astype("float32")
    train = frame.loc[train_mask].copy()
    test = frame.loc[frame.date.between(TEST_START, TEST_END)].copy()
    predictions: dict[str, np.ndarray] = {}
    for target in targets:
        values = reg_predict(train, test, features, target, stable_seed(tier, family, target))
        if values is not None:
            predictions[target] = pd.Series(values, index=test.index).groupby(test.date).rank(pct=True, method="average").to_numpy()
    if not all(target in predictions for target in targets):
        return []
    pred = test[["symbol", "date"]].copy()
    for target in targets:
        pred[target] = predictions[target]
    pred["long_score"] = pred.long_hub
    pred["long_exit_score"] = pred.long_authority
    pred["short_score"] = pred.short_hub
    pred["short_exit_score"] = pred.short_authority
    pred["long_agree_count"] = 1
    pred["short_agree_count"] = 1
    pred["model_count"] = 1
    summaries = []
    next_returns = close.pct_change().shift(-1)
    for variant in RF_VARIANTS:
        if variant not in {"long_only", "short_only"}:
            raise ValueError("RF_HITS_VARIANTS may only contain long_only or short_only")
        summary, _, _ = run_shared_book_framework_comparison(
            scores=pred[["symbol", "date", "long_score", "short_score", "long_exit_score", "short_exit_score", "long_agree_count", "short_agree_count", "model_count"]],
            next_returns=next_returns,
            symbols=tuple(close.columns),
            dates=dates,
            variants=(variant,),
            top_k_values=(20,),
            entry_threshold=0.5,
            exit_threshold=0.5,
            cost_models={"family_common": SharedBookCostModel(0.5, 5.0)},
        )
        if not summary.empty:
            summary["tier"] = tier
            summary["family"] = family
            summary["label_source"] = "rf_sparse_hits"
            summaries.append(summary)
    return summaries


def run_tier(tier: str) -> pd.DataFrame:
    started = perf_counter()
    index = pd.read_csv(feature_dir(tier) / "index.csv")
    requested = tuple(x.strip() for x in os.getenv("RF_HITS_FAMILIES", "").split(",") if x.strip())
    if requested:
        index = index.loc[index.family.astype(str).isin(requested)].copy()
    first = pd.read_parquet(index.iloc[0].panel_path)
    symbols = sorted(first.symbol.astype(str).str.upper().unique())
    price_map, labels = build_price_and_labels(symbols)
    close = price_map.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    next_returns = close.pct_change().shift(-1)
    dates = pd.DatetimeIndex(next_returns.index[(next_returns.index >= TEST_START) & (next_returns.index <= TEST_END)])
    summaries = []
    for _, meta in index.iterrows():
        summaries.extend(run_family(meta, price_map, labels, close, dates, tier))
    result = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    result.to_csv(OUT / f"{tier.lower()}_train_2024_test_2025_results.csv", index=False)
    print(result.groupby(["variant", "label_source"], as_index=False).agg(families=("family", "nunique"), mean_return=("total_return", "mean"), median_return=("total_return", "median"), min_return=("total_return", "min"), max_return=("total_return", "max"), mean_sharpe=("sharpe", "mean")).round(4).to_string(index=False) if not result.empty else result)
    print({"tier": tier, "seconds": round(perf_counter() - started, 1), "result_rows": len(result)}, flush=True)
    return result


def main() -> None:
    tiers = tuple(x.strip().upper() for x in os.getenv("RF_HITS_TIERS", "1T,100B,10B").split(",") if x.strip())
    all_results = [run_tier(tier) for tier in tiers]
    combined = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    combined.to_csv(OUT / "all_train_2024_test_2025_results.csv", index=False)
    if not combined.empty:
        print(combined.groupby(["tier", "variant"], as_index=False).agg(families=("family", "nunique"), mean_return=("total_return", "mean"), median_return=("total_return", "median"), min_return=("total_return", "min"), max_return=("total_return", "max"), mean_sharpe=("sharpe", "mean")).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
