"""1T feature-family GNN smoke test: train on 2024, trade on 2025.

Each family gets an independent causal temporal GNN.  The shared encoder has
two heads: (1) an auxiliary pairwise-return edge head and (2) a hub/authority
node head.  Only the node head is used to generate live trading scores.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
for repo in (REPO_ROOT, WORKSPACE_ROOT / "quant-warehouse", WORKSPACE_ROOT / "quant-orchestrator"):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

from quant_warehouse.platforms.data_providers.fmp.target_engineering import (
    HitsLabelSpec,
    build_hits_labels,
)
from quant_warehouse.warehouse.api import Warehouse
from quant_orchestrator.platforms.backtesting_frameworks.shared_book import (
    SharedBookCostModel,
    run_shared_book_framework_comparison,
)

TRAIN_START, TRAIN_END = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")
TEST_START, TEST_END = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")
MAX_HOLD = int(os.getenv("GNN_MAX_HOLD", "120"))
HITS_ITERATIONS = int(os.getenv("GNN_HITS_ITERATIONS", "50"))
HITS_TAIL_QUANTILE = float(os.getenv("GNN_HITS_TAIL_QUANTILE", "0.20"))
GNN_VARIANT = os.getenv("GNN_VARIANT", "long_only").strip().lower()
LOOKBACK = int(os.getenv("GNN_LOOKBACK", "10"))
EPOCHS = int(os.getenv("GNN_EPOCHS", "12"))
HIDDEN = int(os.getenv("GNN_HIDDEN", "48"))
PAIR_PER_SOURCE = int(os.getenv("GNN_PAIR_PER_SOURCE", "8"))
SEED = 20260716
torch.manual_seed(SEED)
np.random.seed(SEED)

OUT = REPO_ROOT / "artifacts" / "graph_oracle_feature_family_gnn_improved_hits"
OUT.mkdir(parents=True, exist_ok=True)
TIER_CONFIGS = {
    "1T": (1_000_000_000_000, "equity_meta_model_1t"),
    "100B": (100_000_000_000, "equity_meta_model_100b"),
    "10B": (10_000_000_000, "equity_meta_model_10b"),
}


def feature_dir(tier: str) -> Path:
    cap, cache = TIER_CONFIGS[tier]
    return REPO_ROOT / "artifacts" / "trading_app_v2" / cache / f"mcap_{cap}_train_2020-12-31_seed_20260707" / "feature_family_panels"


def normalize_prices(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy().reset_index() if "date" not in raw.columns else raw.copy()
    frame.columns = [str(c).lower() for c in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    return frame[["date", "open", "high", "low", "close"]].dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)


def build_price_and_labels(symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    warehouse = Warehouse()
    node_rows: list[pd.DataFrame] = []
    price_frames: dict[str, pd.DataFrame] = {}
    for i, symbol in enumerate(symbols, 1):
        raw = warehouse.read_prices(symbol, provider="fmp", start="2023-01-01", end="2025-12-31")
        if raw is None or raw.empty:
            continue
        try:
            prices = normalize_prices(raw)
        except Exception:
            continue
        prices = prices.loc[prices.date.between(TRAIN_START, TEST_END)].copy()
        if len(prices) < 30:
            continue
        price_frames[symbol] = prices.copy()
        prices.insert(0, "symbol", symbol)
        node_rows.append(prices)
        if i % 100 == 0:
            print({"prices_loaded": i, "usable_symbols": len(node_rows)}, flush=True)
    prices = pd.concat(node_rows, ignore_index=True) if node_rows else pd.DataFrame()
    labels = build_hits_labels(
        price_frames,
        spec=HitsLabelSpec(
            max_hold=MAX_HOLD,
            iterations=HITS_ITERATIONS,
            tail_quantile=HITS_TAIL_QUANTILE,
            start_date=str(TRAIN_START.date()),
            end_date=str(TEST_END.date()),
        ),
    )
    return prices, labels


def make_temporal_edges(nodes: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Past-to-current edges only; edge return is known at current date."""
    src: list[int] = []
    dst: list[int] = []
    edge_ret: list[float] = []
    for _, group in nodes.groupby("symbol", sort=False):
        ids = group.index.to_numpy()
        close = group.close.to_numpy(float)
        for j in range(len(ids)):
            lo = max(0, j - LOOKBACK)
            for k in range(lo, j):
                if np.isfinite(close[k]) and close[k] > 0:
                    src.append(int(ids[k])); dst.append(int(ids[j])); edge_ret.append(float(np.clip(close[j] / close[k] - 1.0, -2.0, 2.0)))
    return torch.tensor([src, dst], dtype=torch.long), torch.tensor(edge_ret, dtype=torch.float32).unsqueeze(1)


def make_pair_batch(nodes: pd.DataFrame, max_pairs: int = 250_000) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample historical same-symbol entry/exit edges for auxiliary learning."""
    rng = np.random.default_rng(SEED)
    src, dst, long_y, short_y = [], [], [], []
    for _, group in nodes.groupby("symbol", sort=False):
        ids = group.index.to_numpy()
        close = group.close.to_numpy(float)
        n = len(ids)
        for i in range(n):
            hi = min(n, i + MAX_HOLD + 1)
            candidates = np.arange(i + 1, hi)
            if len(candidates) > PAIR_PER_SOURCE:
                candidates = rng.choice(candidates, PAIR_PER_SOURCE, replace=False)
            for j in candidates:
                if close[i] <= 0 or close[j] <= 0 or not np.isfinite(close[i] + close[j]):
                    continue
                src.append(int(ids[i])); dst.append(int(ids[j]))
                long_y.append(float(np.clip(close[j] / close[i] - 1.0, -2.0, 2.0)))
                short_y.append(float(np.clip(close[i] / close[j] - 1.0, -2.0, 2.0)))
    if len(src) > max_pairs:
        keep = rng.choice(len(src), max_pairs, replace=False)
        src = [src[i] for i in keep]; dst = [dst[i] for i in keep]; long_y = [long_y[i] for i in keep]; short_y = [short_y[i] for i in keep]
    return torch.tensor(src), torch.tensor(dst), torch.tensor(np.column_stack([long_y, short_y]), dtype=torch.float32)


class TemporalGNN(nn.Module):
    def __init__(self, n_features: int, hidden: int = HIDDEN):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(n_features, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.message = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.update = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.edge_head = nn.Sequential(nn.Linear(hidden * 2 + 1, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self.ha_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4), nn.Sigmoid())

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        if edge_index.numel():
            messages = self.message(torch.cat([h[edge_index[0]], edge_attr], dim=1))
            agg = torch.zeros_like(h)
            count = torch.zeros((len(h), 1), device=h.device)
            agg.index_add_(0, edge_index[1], messages)
            count.index_add_(0, edge_index[1], torch.ones((len(messages), 1), device=h.device))
            h = self.update(torch.cat([h, agg / count.clamp_min(1.0)], dim=1))
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, pair_src: torch.Tensor | None = None, pair_dst: torch.Tensor | None = None):
        z = self.encode(x, edge_index, edge_attr)
        ha = self.ha_head(z)
        pair = None
        if pair_src is not None and pair_dst is not None:
            gap = (pair_dst - pair_src).float().unsqueeze(1) / max(1.0, MAX_HOLD)
            pair = self.edge_head(torch.cat([z[pair_src], z[pair_dst], gap], dim=1))
        return z, ha, pair


def fit_family(panel: pd.DataFrame, price_map: pd.DataFrame, labels: pd.DataFrame, family: str) -> tuple[pd.DataFrame, dict]:
    metadata = pd.read_parquet(panel.attrs["metadata_path"])
    feature_cols = [c for c in metadata.feature.astype(str) if c in panel.columns]
    if not feature_cols:
        return pd.DataFrame(), {"family": family, "status": "no_features"}
    base = panel[["symbol", "date", *feature_cols]].copy()
    base["symbol"] = base.symbol.astype(str).str.upper()
    base["date"] = pd.to_datetime(base.date).dt.normalize()
    base = base.merge(price_map, on=["symbol", "date"], how="inner").sort_values(["symbol", "date"]).reset_index(drop=True)
    base = base.loc[base.date.between(TRAIN_START, TEST_END)].reset_index(drop=True)
    if base.empty:
        return pd.DataFrame(), {"family": family, "status": "no_price_overlap"}
    y = labels.copy(); y.symbol = y.symbol.astype(str).str.upper(); y.date = pd.to_datetime(y.date).dt.normalize()
    base = base.merge(y, on=["symbol", "date"], how="left")
    target_cols = ["long_hub", "long_authority", "short_hub", "short_authority"]
    tail_cols = [f"{column}_tail" for column in target_cols]
    base[target_cols] = base[target_cols].fillna(0.0).astype("float32")
    base[tail_cols] = base[tail_cols].fillna(False).astype(bool)
    train_mask = base.date.between(TRAIN_START, TRAIN_END)
    med = base.loc[train_mask, feature_cols].apply(pd.to_numeric, errors="coerce").median().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    xdf = base[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    std = xdf.loc[train_mask].std().replace(0, 1).fillna(1.0)
    xdf = ((xdf - xdf.loc[train_mask].mean()) / std).clip(-8, 8).astype("float32")
    base[feature_cols] = xdf
    # Stable row order makes node IDs usable for temporal and pair edges.
    temporal_edges, temporal_attr = make_temporal_edges(base[["symbol", "date", "close"]])
    train_nodes = base.index[train_mask].to_numpy()
    pair_src, pair_dst, pair_y = make_pair_batch(base.loc[train_mask, ["symbol", "date", "close"]].reset_index(drop=True))
    # Pair sampler returns local train IDs; translate to global node IDs.
    pair_global = train_nodes
    pair_src, pair_dst = pair_global[pair_src.numpy()], pair_global[pair_dst.numpy()]
    pair_src, pair_dst = torch.tensor(pair_src), torch.tensor(pair_dst)
    model = TemporalGNN(len(feature_cols)).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(os.getenv("GNN_LR", "0.002")), weight_decay=1e-4)
    x = torch.tensor(base[feature_cols].to_numpy(dtype=np.float32))
    ha_y = torch.tensor(base[target_cols].to_numpy(dtype=np.float32))
    ha_mask = torch.tensor(base[tail_cols].to_numpy(dtype=np.float32))
    edge_train_mask = torch.ones(len(pair_src), dtype=torch.bool)
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        _, ha_hat, edge_hat = model(x, temporal_edges, temporal_attr, pair_src, pair_dst)
        node_errors = nn.functional.smooth_l1_loss(ha_hat[train_nodes], ha_y[train_nodes], reduction="none")
        node_mask = ha_mask[train_nodes]
        node_loss = (node_errors * node_mask).sum() / node_mask.sum().clamp_min(1.0)
        edge_loss = nn.functional.smooth_l1_loss(edge_hat[edge_train_mask], pair_y[edge_train_mask]) if edge_hat is not None and len(pair_src) else torch.tensor(0.0)
        loss = node_loss + float(os.getenv("GNN_EDGE_LOSS_WEIGHT", "0.25")) * edge_loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if epoch == 0 or epoch == EPOCHS - 1:
            print({"family": family, "epoch": epoch + 1, "loss": round(float(loss.detach()), 5), "node_loss": round(float(node_loss.detach()), 5), "edge_loss": round(float(edge_loss.detach()), 5)}, flush=True)
    model.eval()
    with torch.no_grad():
        _, ha_hat, _ = model(x, temporal_edges, temporal_attr)
    pred = base[["symbol", "date"]].copy()
    pred[["long_hub", "long_authority", "short_hub", "short_authority"]] = ha_hat.numpy()
    pred = pred.loc[pred.date.between(TEST_START, TEST_END)].copy()
    # HITS is a relative score.  Calibrate predictions cross-sectionally so
    # the existing shared-book threshold has the same interpretation as the
    # prior feature-family experiments.
    for col in ["long_hub", "long_authority", "short_hub", "short_authority"]:
        pred[col] = pred.groupby("date")[col].rank(pct=True, method="average")
    pred["source"] = "gnn"
    pred["family"] = family
    pred["strategy_source"] = f"gnn.{family}"
    pred["long_score"] = pred["long_hub"]
    pred["long_exit_score"] = pred["long_authority"]
    pred["short_score"] = pred["short_hub"]
    pred["short_exit_score"] = pred["short_authority"]
    pred["long_agree_count"] = (pred.long_score >= pred.short_score).astype(int)
    pred["short_agree_count"] = (pred.short_score > pred.long_score).astype(int)
    pred["model_count"] = 1
    return pred, {"family": family, "status": "ok", "nodes": len(base), "train_nodes": int(train_mask.sum()), "pairs": len(pair_src), "features": len(feature_cols), "epochs": EPOCHS}


def run_tier(tier: str) -> pd.DataFrame:
    if GNN_VARIANT not in {"long_only", "short_only"}:
        raise ValueError("GNN_VARIANT must be 'long_only' or 'short_only'")
    started = perf_counter()
    index = pd.read_csv(feature_dir(tier) / "index.csv")
    requested = tuple(x.strip() for x in os.getenv("GNN_FAMILIES", "").split(",") if x.strip())
    if requested:
        index = index.loc[index.family.astype(str).isin(requested)].copy()
    first = pd.read_parquet(index.iloc[0].panel_path)
    symbols = sorted(first.symbol.astype(str).str.upper().unique())
    price_map, labels = build_price_and_labels(symbols)
    print({"tier": tier, "symbols": len(symbols), "price_rows": len(price_map), "label_rows": len(labels), "families": len(index)}, flush=True)
    close = price_map.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    next_returns = close.pct_change().shift(-1)
    date_mask = (next_returns.index >= TEST_START) & (next_returns.index <= TEST_END)
    dates = pd.DatetimeIndex(next_returns.index[date_mask])
    summaries, predictions = [], []
    for _, meta in index.iterrows():
        family = str(meta.family)
        panel = pd.read_parquet(meta.panel_path)
        panel.attrs["metadata_path"] = meta.metadata_path
        pred, info = fit_family(panel, price_map, labels, family)
        print(info, flush=True)
        if pred.empty:
            continue
        predictions.append(pred)
        summary, _, _ = run_shared_book_framework_comparison(
            scores=pred[["symbol", "date", "long_score", "short_score", "long_exit_score", "short_exit_score", "long_agree_count", "short_agree_count", "model_count"]],
            next_returns=next_returns,
            symbols=tuple(close.columns),
            dates=dates,
            variants=(GNN_VARIANT,),
            top_k_values=(20,),
            entry_threshold=0.5,
            exit_threshold=0.5,
            cost_models={"family_common": SharedBookCostModel(0.5, 5.0)},
        )
        if not summary.empty:
            summary["tier"] = tier; summary["family"] = family; summary["label_source"] = "gnn_sparse_hits"; summaries.append(summary)
    result = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    result.to_csv(OUT / f"{tier.lower()}_{GNN_VARIANT}_train_2024_test_2025_results.csv", index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_parquet(OUT / f"{tier.lower()}_{GNN_VARIANT}_train_2024_test_2025_predictions.parquet", index=False)
    print(result.groupby(["variant", "label_source"], as_index=False).agg(families=("family", "nunique"), mean_return=("total_return", "mean"), median_return=("total_return", "median"), mean_sharpe=("sharpe", "mean")).round(4) if not result.empty else result)
    print({"tier": tier, "seconds": round(perf_counter() - started, 1), "result_rows": len(result)}, flush=True)
    return result


def main() -> None:
    requested = tuple(x.strip().upper() for x in os.getenv("GNN_TIERS", "1T").split(",") if x.strip())
    unknown = sorted(set(requested) - set(TIER_CONFIGS))
    if unknown:
        raise ValueError(f"unknown GNN_TIERS: {unknown}")
    all_results = [run_tier(tier) for tier in requested]
    combined = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    combined.to_csv(OUT / f"all_{GNN_VARIANT}_train_2024_test_2025_results.csv", index=False)
    if not combined.empty:
        print(combined.groupby(["tier", "variant"], as_index=False).agg(families=("family", "nunique"), mean_return=("total_return", "mean"), median_return=("total_return", "median"), min_return=("total_return", "min"), max_return=("total_return", "max"), mean_sharpe=("sharpe", "mean")).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
