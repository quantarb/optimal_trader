"""Diagnostics for the official transformer baseline.

This script is intentionally inference/label-analysis only.  It does not
train a model or create synthetic issuer/instrument pairs.  It produces:

* learned task-to-family routing from the saved WFO checkpoints;
* family coverage and feature-count diagnostics;
* holding-time/speed HITS statistics for 5/20/60/120-day horizons; and
* stability of speed labels relative to the current 120-day labels.

Run with ``DIAGNOSTIC_TIER=100B`` (the default) after the baseline WFO.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
for repo in (WORKSPACE_ROOT / "quant-warehouse", WORKSPACE_ROOT / "quant-orchestrator"):
    if repo.exists() and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

TRANSFORMER_PATH = REPO_ROOT / "scripts" / "run_symbol_year_transformer_mtl.py"
spec = importlib.util.spec_from_file_location("baseline_transformer", TRANSFORMER_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load {TRANSFORMER_PATH}")
transformer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transformer)

from quant_warehouse.platforms.data_providers.fmp.target_engineering.hits import (  # noqa: E402
    HitsLabelSpec,
    _score_weight_matrix,
    _build_edge_channels,
)

TIER = os.getenv("DIAGNOSTIC_TIER", "100B").strip().upper()
if TIER not in {"1T", "100B", "10B"}:
    raise ValueError("DIAGNOSTIC_TIER must be 1T, 100B, or 10B")
HORIZONS = tuple(int(x) for x in os.getenv("DIAGNOSTIC_HORIZONS", "5,20,60,120").split(",") if x.strip())
OUT = REPO_ROOT / "artifacts" / "graph_oracle_feature_family_gnn_improved_hits" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

CAPS = {"1T": 1_000_000_000_000, "100B": 100_000_000_000, "10B": 10_000_000_000}
CACHE_NAME = {
    "1T": "equity_meta_model_1t",
    "100B": "equity_meta_model_100b",
    "10B": "equity_meta_model_10b",
}


def _checkpoint_dir() -> Path:
    return REPO_ROOT / "artifacts" / "graph_oracle_feature_family_gnn_improved_hits" / "transformer_checkpoints"


def _load_checkpoint_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    feature_cols = list(checkpoint["feature_cols"])
    family_indices: dict[str, list[int]] = {}
    for index, column in enumerate(feature_cols):
        family_indices.setdefault(str(column).split("__", 1)[0], []).append(index)
    family_indices = {name: tuple(indices) for name, indices in family_indices.items() if indices}
    asset_indices = {
        name: tuple(index for index, column in enumerate(feature_cols) if str(column).startswith(f"{name}__"))
        for name in transformer.ASSET_CLASSES
    }
    asset_indices = {name: indices for name, indices in asset_indices.items() if indices}
    robust_mask = np.asarray(
        [transformer._is_price_volume_feature(column) for column in feature_cols], dtype=bool
    )
    position = checkpoint["model_state_dict"]["position"]
    model = transformer.TransformerMTL(
        len(feature_cols), checkpoint["aux_dims"], asset_feature_indices=asset_indices,
        family_feature_indices=family_indices, feature_mean=checkpoint["feature_mean"],
        feature_std=checkpoint["feature_std"], robust_feature_mask=robust_mask,
        max_position=int(position.shape[0]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model


def routing_diagnostics() -> None:
    rows: list[dict] = []
    summary: list[dict] = []
    checkpoints = sorted(_checkpoint_dir().glob(f"{TIER.lower()}_*_2trunk_cross_year.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No {TIER} baseline checkpoints found in {_checkpoint_dir()}")
    for path in checkpoints:
        checkpoint, model = _load_checkpoint_model(path)
        year = int(checkpoint["test_year"])
        family_names = list(model.coverage_family_names)
        task_family = model.task_family_weights()
        task_trunk = model.routing_weights()
        trunk_family = torch.softmax(model.trunk_family_router.detach(), dim=-1).numpy()
        for trunk, weights in enumerate(trunk_family):
            for family, weight in zip(family_names, weights):
                rows.append({"year": year, "scope": f"trunk_{trunk}", "task": "__trunk__", "family": family, "weight": float(weight)})
        for task, weights in task_family.items():
            values = np.asarray(weights, dtype=float)
            entropy = float(-(values * np.log(np.clip(values, 1e-12, None))).sum())
            top = int(values.argmax())
            summary.append({
                "year": year, "task": task, "top_family": family_names[top],
                "top_weight": float(values[top]), "family_entropy": entropy,
                "task_trunk_0": float(task_trunk[task][0]), "task_trunk_1": float(task_trunk[task][1]),
            })
            for family, weight in zip(family_names, values):
                rows.append({"year": year, "scope": "task_effective", "task": task, "family": family, "weight": float(weight)})
    pd.DataFrame(rows).to_csv(OUT / f"{TIER.lower()}_routing_family_weights.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / f"{TIER.lower()}_routing_summary.csv", index=False)


def coverage_diagnostics() -> None:
    cache_override = os.getenv(f"TRANSFORMER_FEATURE_CACHE_{TIER}", "").strip()
    if cache_override:
        feature_dir = Path(cache_override)
    else:
        feature_dir = REPO_ROOT / "artifacts" / "trading_app_v2" / CACHE_NAME[TIER] / f"mcap_{CAPS[TIER]}_train_2020-12-31_seed_20260707" / "feature_family_panels"
    index = pd.read_csv(feature_dir / "index.csv")
    price_files = sorted((REPO_ROOT / "artifacts" / "graph_oracle_feature_family_gnn_improved_hits" / "cache").glob(f"*_{TIER.lower()}_*_prices.parquet"))
    if not price_files:
        raise FileNotFoundError(f"No cached prices found for {TIER}")
    universe = pd.read_parquet(price_files[0], columns=["symbol", "date"])
    universe["symbol"] = universe["symbol"].astype(str).str.upper()
    universe["date"] = pd.to_datetime(universe["date"], errors="coerce").dt.normalize()
    universe_keys = pd.MultiIndex.from_frame(universe.drop_duplicates())
    checkpoint_path = sorted(_checkpoint_dir().glob(f"{TIER.lower()}_*_2trunk_cross_year.pt"))[-1]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    family_counts = pd.Series([str(c).split("__", 1)[0] for c in checkpoint["feature_cols"]]).value_counts()
    rows: list[dict] = []
    for meta in index.itertuples(index=False):
        panel = pd.read_parquet(meta.panel_path, columns=["symbol", "date"])
        panel["symbol"] = panel["symbol"].astype(str).str.upper()
        panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
        keys = pd.MultiIndex.from_frame(panel.drop_duplicates())
        overlap = universe_keys.intersection(keys)
        rows.append({
            "family": str(meta.family), "feature_count": int(family_counts.get(str(meta.family), 0)),
            "panel_rows": int(len(panel)), "covered_symbol_dates": int(len(overlap)),
            "universe_symbol_dates": int(len(universe_keys)),
            "coverage_rate": float(len(overlap) / max(1, len(universe_keys))),
        })
    pd.DataFrame(rows).sort_values("coverage_rate").to_csv(OUT / f"{TIER.lower()}_feature_family_coverage.csv", index=False)


def holding_time_diagnostics() -> None:
    cache_dir = REPO_ROOT / "artifacts" / "graph_oracle_feature_family_gnn_improved_hits" / "cache"
    price_path = sorted(cache_dir.glob(f"*_{TIER.lower()}_*_prices.parquet"))[0]
    prices = pd.read_parquet(price_path, columns=["symbol", "date", "high", "low"])
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    prices["symbol"] = prices["symbol"].astype(str).str.upper()
    metric_rows: list[dict] = []
    score_arrays: dict[tuple[int, str, str], list[np.ndarray]] = {}
    return_arrays: dict[str, list[np.ndarray]] = {"long": [], "short": []}
    speed_arrays: dict[tuple[int, str], list[np.ndarray]] = {}
    for (symbol, year), group in prices.groupby(["symbol", prices.date.dt.year], sort=False):
        frame = group.sort_values("date").dropna(subset=["high", "low"]).reset_index(drop=True)
        if len(frame) < 2:
            continue
        high, low = frame.high.to_numpy(float), frame.low.to_numpy(float)
        for horizon in HORIZONS:
            returns, valid, holding_days = _build_edge_channels(high, low, horizon)
            for side in ("long", "short"):
                positive = valid & (returns[side] > 0.0)
                positive_returns = returns[side][positive]
                positive_holds = holding_days[positive]
                weights = np.zeros_like(returns[side], dtype=float)
                np.divide(1.0, holding_days, out=weights, where=positive)
                weighted_returns = float((weights * np.maximum(returns[side], 0.0)).sum() / max(weights.sum(), 1e-12))
                hub, authority, hub_tail, authority_tail = _score_weight_matrix(
                    returns[side], valid, holding_days, 50, "inverse_holding_time", 0.20
                )
                speed_arrays.setdefault((horizon, side), []).append(hub)
                score_arrays.setdefault((horizon, side, "authority"), []).append(authority)
                metric_rows.append({
                    "symbol": symbol, "year": int(year), "horizon_days": horizon, "side": side,
                    "nodes": len(frame), "valid_edges": int(valid.sum()), "positive_edges": int(positive.sum()),
                    "positive_edge_rate": float(positive.sum() / max(1, valid.sum())),
                    "mean_positive_return": float(np.mean(positive_returns)) if len(positive_returns) else 0.0,
                    "median_positive_return": float(np.median(positive_returns)) if len(positive_returns) else 0.0,
                    "mean_positive_hold_days": float(np.mean(positive_holds)) if len(positive_holds) else 0.0,
                    "speed_weighted_return": weighted_returns,
                    "hub_tail_rate": float(hub_tail.mean()), "authority_tail_rate": float(authority_tail.mean()),
                })
        # The current label is the 120-day speed graph.  Keep return scores
        # only for the direct return-vs-speed stability comparison.
        returns, valid, holding_days = _build_edge_channels(high, low, 120)
        for side in ("long", "short"):
            hub, authority, _, _ = _score_weight_matrix(returns[side], valid, holding_days, 50, "return", 0.20)
            return_arrays[side].append(hub)
            score_arrays.setdefault((120, side, "return_authority"), []).append(authority)
    metrics = pd.DataFrame(metric_rows)
    metrics.groupby(["year", "horizon_days", "side"], as_index=False).mean(numeric_only=True).to_csv(
        OUT / f"{TIER.lower()}_holding_time_summary.csv", index=False
    )
    stability_rows: list[dict] = []
    for side in ("long", "short"):
        return_hub = np.concatenate(return_arrays[side])
        speed_hub = np.concatenate(speed_arrays[(120, side)])
        stability_rows.append({"comparison": "return_vs_speed_hub", "side": side, "horizon_days": 120, "spearman": float(pd.Series(return_hub).corr(pd.Series(speed_hub), method="spearman"))})
        for horizon in HORIZONS:
            current = np.concatenate(speed_arrays[(120, side)])
            candidate = np.concatenate(speed_arrays[(horizon, side)])
            current_tail = pd.Series(current).rank(pct=True, method="first").to_numpy() >= 0.80
            candidate_tail = pd.Series(candidate).rank(pct=True, method="first").to_numpy() >= 0.80
            union = np.logical_or(current_tail, candidate_tail).sum()
            stability_rows.append({
                "comparison": "speed_hub_top_tail_overlap_with_120d", "side": side,
                "horizon_days": horizon, "spearman": float(pd.Series(current).corr(pd.Series(candidate), method="spearman")),
                "top_tail_jaccard": float(np.logical_and(current_tail, candidate_tail).sum() / max(1, union)),
            })
    pd.DataFrame(stability_rows).to_csv(OUT / f"{TIER.lower()}_holding_time_stability.csv", index=False)


if __name__ == "__main__":
    routing_diagnostics()
    coverage_diagnostics()
    holding_time_diagnostics()
    print({"tier": TIER, "output": str(OUT), "horizons": HORIZONS}, flush=True)
