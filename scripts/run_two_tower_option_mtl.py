"""Cheap 1T two-tower MTL experiment for long-call versus long-put retrieval.

Option contracts are aggregated into raw call/put prototypes before entering
the option tower.  The metric objective explicitly aligns issuer states with
higher-return expressions and separates lower-return expressions.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SMOKE_PATH = ROOT / "scripts" / "run_option_instrument_mtl_smoke.py"
spec = importlib.util.spec_from_file_location("option_proto", SMOKE_PATH)
assert spec and spec.loader
option_proto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(option_proto)
gnn = option_proto.gnn
GRAPH_TARGETS = option_proto.GRAPH_FEATURE_TARGETS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 20260716
EPOCHS = 3
BATCH_SIZE = 2048
MARGIN = 0.05
TEMPERATURE = 0.10


class TwoTowerMTL(nn.Module):
    def __init__(self, issuer_dim: int, option_dim: int, aux_dims: dict[str, int], event_dim: int, macro_dim: int, metric_dim: int = 32):
        super().__init__()
        self.issuer = nn.Sequential(nn.Linear(issuer_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, metric_dim))
        self.option = nn.Sequential(nn.Linear(option_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, metric_dim))
        self.option_type = nn.Embedding(2, metric_dim)
        # The same task heads are applied to both towers.  This forces issuer
        # and option representations to organize the shared embedding space
        # around the same MTL semantics instead of supervising only issuers.
        self.aux_heads = nn.ModuleDict({name: option_proto.PrototypeHead(metric_dim, size) for name, size in aux_dims.items()})
        self.event_head = gnn.EventPrototypeHead(metric_dim, event_dim)
        self.macro_head = gnn.EventPrototypeHead(metric_dim, macro_dim)
        self.graph_head = nn.Linear(metric_dim, len(GRAPH_TARGETS))

    def forward(self, issuer_x: torch.Tensor, option_x: torch.Tensor, option_type: torch.Tensor):
        issuer_z = nn.functional.normalize(self.issuer(issuer_x), dim=-1)
        option_z = nn.functional.normalize(self.option(option_x) + self.option_type(option_type), dim=-1)
        def tasks(z: torch.Tensor):
            return ({name: head(z) for name, head in self.aux_heads.items()}, self.event_head(z), self.macro_head(z), self.graph_head(z))
        return issuer_z, option_z, tasks(issuer_z), tasks(option_z)


def task_loss(outputs, aux_target, event_target, macro_target, graph_target, aux_weight: float = 0.05) -> torch.Tensor:
    aux_hat, event_hat, macro_hat, graph_hat = outputs
    loss = 0.1 * gnn.event_loss_from_logits(event_hat, event_target)
    if macro_target.shape[1]:
        loss = loss + 0.1 * gnn.event_loss_from_logits(macro_hat, macro_target)
    loss = loss + 0.1 * nn.functional.smooth_l1_loss(graph_hat, graph_target)
    for col_idx, name in enumerate(gnn.AUX_TARGET_COLS):
        target = aux_target[:, col_idx]
        mask = target.ge(0)
        if mask.any():
            loss = loss + aux_weight * nn.functional.cross_entropy(aux_hat[name][mask], target[mask])
    return loss


def pairwise_loss(scores: torch.Tensor, returns: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Rank the higher-return call/put expression within each issuer-date."""
    losses = []
    for group in groups.unique():
        idx = torch.where(groups == group)[0]
        if len(idx) < 2:
            continue
        for left in range(len(idx)):
            for right in range(left + 1, len(idx)):
                i, j = idx[left], idx[right]
                difference = returns[i] - returns[j]
                if torch.abs(difference) > 1e-6:
                    direction = torch.sign(difference)
                    losses.append(nn.functional.relu(MARGIN - direction * (scores[i] - scores[j])))
    return torch.stack(losses).mean() if losses else scores.sum() * 0.0


def contrastive_loss(issuer_z: torch.Tensor, option_z: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Use other issuer-dates as in-batch negatives, excluding same-pair rows."""
    logits = issuer_z @ option_z.T / TEMPERATURE
    same_group = groups[:, None].eq(groups[None, :])
    diagonal = torch.eye(len(groups), dtype=torch.bool, device=groups.device)
    logits = logits.masked_fill(same_group & ~diagonal, -torch.inf)
    targets = torch.arange(len(groups), device=groups.device)
    forward = nn.functional.cross_entropy(logits, targets)
    reverse_logits = option_z @ issuer_z.T / TEMPERATURE
    reverse_logits = reverse_logits.masked_fill(same_group & ~diagonal, -torch.inf)
    return 0.5 * (forward + nn.functional.cross_entropy(reverse_logits, targets))


def prepare_issuer_features(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    prices["symbol"] = prices.symbol.astype(str).str.upper()
    prices["date"] = pd.to_datetime(prices.date).dt.normalize()
    prices = prices.sort_values(["symbol", "date"])
    prices["prev_close"] = prices.groupby("symbol")["close"].shift(1)
    for horizon in (1, 5, 20):
        returns = prices.groupby("symbol")["close"].pct_change(horizon)
        prices[f"ret_{horizon}d"] = returns.groupby(prices["symbol"]).shift(1)
    cols = ["prev_close", "ret_1d", "ret_5d", "ret_20d"]
    return prices[["symbol", "date", *cols]], cols


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    index = pd.read_csv(gnn.feature_dir("1T") / "index.csv")
    first = pd.read_parquet(index.iloc[0].panel_path)
    symbols = sorted(first.symbol.astype(str).str.upper().unique())
    option_symbols = [symbol for symbol in symbols if symbol not in {"BRK-A", "BRK-B"}]
    prices, labels = gnn.build_price_and_labels(symbols, "1T")
    labels = labels.copy()
    labels["symbol"] = labels.symbol.astype(str).str.upper()
    labels["date"] = pd.to_datetime(labels.date).dt.normalize()
    macro_panel, _ = option_proto.transformer_module._load_macro_event_panel(labels["date"])
    labels = labels.merge(macro_panel, on="date", how="left")
    rows = option_proto.load_rows(option_symbols, labels)
    # Use the same fused issuer feature panel as the full transformer MTL
    # baseline, rather than the four-feature smoke proxy.
    issuer_base, _, issuer_cols = option_proto.transformer_module._prepare_data("1T")
    issuer_frame = issuer_base[["symbol", "date", *issuer_cols]].copy()
    issuer_frame["symbol"] = issuer_frame.symbol.astype(str).str.upper()
    issuer_frame["date"] = pd.to_datetime(issuer_frame.date).dt.normalize()
    rows = rows.merge(issuer_frame, on=["symbol", "date"], how="inner")
    # Full issuer panels are intentionally sparse by feature family.  Keep
    # matched issuer-dates and impute individual feature cells below; requiring
    # every one of the 792 columns to be present would remove every row.
    rows = rows.reset_index(drop=True)
    train_mask = rows.date.dt.year < 2025
    option_features = [*option_proto.RAW_FEATURES, "prototype_contract_count"]
    option_raw = rows[option_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    option_med = option_raw.loc[train_mask].median().fillna(0.0)
    option_mean = option_raw.loc[train_mask].mean().fillna(0.0)
    option_std = option_raw.loc[train_mask].std().replace(0, 1).fillna(1.0)
    option_x = ((option_raw.fillna(option_med) - option_mean) / option_std).clip(-8, 8).to_numpy(np.float32)
    issuer_raw = rows[issuer_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    issuer_mean = issuer_raw.loc[train_mask].mean().fillna(0.0)
    issuer_std = issuer_raw.loc[train_mask].std().replace(0, 1).fillna(1.0)
    issuer_x = ((issuer_raw.fillna(issuer_mean) - issuer_mean) / issuer_std).clip(-8, 8).to_numpy(np.float32)
    option_type = rows.prototype_type.eq("put").astype(np.int64).to_numpy()
    returns = pd.to_numeric(rows.target_change_percent, errors="coerce").fillna(0.0).to_numpy(np.float32)
    groups, _ = pd.factorize(rows["symbol"].astype(str) + "|" + rows.date.astype(str))
    aux_dims = {name: max(1, int(pd.to_numeric(rows[name], errors="coerce").max()) + 1)
                if pd.to_numeric(rows[name], errors="coerce").notna().any() else 1
                for name in gnn.AUX_TARGET_COLS}
    event_y = rows[list(gnn.ALL_EVENT_TARGETS)].fillna(0.0).to_numpy(np.float32)
    macro_cols = [column for column in rows.columns if str(column).startswith("is_macro_")]
    macro_y = rows[macro_cols].fillna(0.0).to_numpy(np.float32)
    graph_y = rows[GRAPH_TARGETS].fillna(0.0).to_numpy(np.float32)
    aux_y = rows[list(gnn.AUX_TARGET_COLS)].fillna(-1).to_numpy(np.int64)
    train_idx = np.flatnonzero(train_mask.to_numpy())
    valid_idx = np.flatnonzero((~train_mask).to_numpy())
    model = TwoTowerMTL(len(issuer_cols), len(option_features), aux_dims, len(gnn.ALL_EVENT_TARGETS), len(macro_cols)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    print({"rows": len(rows), "train_rows": len(train_idx), "validation_rows": len(valid_idx), "groups": len(np.unique(groups)), "device": str(DEVICE)}, flush=True)
    for epoch in range(EPOCHS):
        model.train(); order = np.random.permutation(train_idx); losses = []
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            issuer_t = torch.from_numpy(issuer_x[idx]).to(DEVICE)
            option_t = torch.from_numpy(option_x[idx]).to(DEVICE)
            type_t = torch.from_numpy(option_type[idx]).to(DEVICE)
            group_t = torch.from_numpy(groups[idx]).to(DEVICE)
            return_t = torch.from_numpy(returns[idx]).to(DEVICE)
            issuer_z, option_z, issuer_tasks, option_tasks = model(issuer_t, option_t, type_t)
            scores = (issuer_z * option_z).sum(dim=1)
            loss = pairwise_loss(scores, return_t, group_t) + 0.25 * contrastive_loss(issuer_z, option_z, group_t)
            aux_target = torch.from_numpy(aux_y[idx]).to(DEVICE)
            event_target = torch.from_numpy(event_y[idx]).to(DEVICE)
            macro_target = torch.from_numpy(macro_y[idx]).to(DEVICE)
            graph_target = torch.from_numpy(graph_y[idx]).to(DEVICE)
            # Every MTL task supervises both towers; metric learning remains
            # the task that explicitly controls issuer-expression distance.
            loss = loss + task_loss(issuer_tasks, aux_target, event_target, macro_target, graph_target)
            loss = loss + task_loss(option_tasks, aux_target, event_target, macro_target, graph_target)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach()))
        print({"epoch": epoch + 1, "train_loss": round(float(np.mean(losses)), 5)}, flush=True)
    model.eval()
    with torch.no_grad():
        issuer_z, option_z, _, _ = model(torch.from_numpy(issuer_x[valid_idx]).to(DEVICE), torch.from_numpy(option_x[valid_idx]).to(DEVICE), torch.from_numpy(option_type[valid_idx]).to(DEVICE))
        scores = (issuer_z * option_z).sum(dim=1).cpu().numpy()
    valid_returns = returns[valid_idx]
    valid_groups = groups[valid_idx]
    pairs = []
    for group in np.unique(valid_groups):
        idx = np.flatnonzero(valid_groups == group)
        if len(idx) == 2 and valid_returns[idx[0]] != valid_returns[idx[1]]:
            pairs.append(float(np.sign(valid_returns[idx[0]] - valid_returns[idx[1]]) == np.sign(scores[idx[0]] - scores[idx[1]])))
    ranking = pd.DataFrame({"score": scores, "return": valid_returns}).sort_values("score")
    k = max(1, len(ranking) // 10)
    print({"pairwise_accuracy": round(float(np.mean(pairs)) if pairs else float("nan"), 5),
           "spearman": round(float(pd.Series(scores).corr(pd.Series(valid_returns), method="spearman")), 5),
           "bottom_decile_return": round(float(ranking.head(k)["return"].mean()), 5),
           "top_decile_return": round(float(ranking.tail(k)["return"].mean()), 5),
           "status": "complete"}, flush=True)


if __name__ == "__main__":
    main()
