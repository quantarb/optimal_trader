"""Raw option-prototype MTL smoke test for the 1T universe.

The chain is reduced in raw feature space before it reaches a neural encoder:
one call prototype and one put prototype are made for each issuer-date.  This
is intentionally a scalability test; it is not yet the final forward-return
labeling experiment.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
GNN_PATH = ROOT / "scripts" / "run_feature_family_gnn_smoke.py"
spec = importlib.util.spec_from_file_location("gnn_instrument_module", GNN_PATH)
assert spec and spec.loader
gnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gnn)
TRANSFORMER_PATH = ROOT / "scripts" / "run_symbol_year_transformer_mtl.py"
transformer_spec = importlib.util.spec_from_file_location("transformer_macro_loader", TRANSFORMER_PATH)
assert transformer_spec and transformer_spec.loader
transformer_module = importlib.util.module_from_spec(transformer_spec)
transformer_spec.loader.exec_module(transformer_module)
from quant_warehouse.platforms.data_providers.thetadata.options import read_thetadata_eod_option_chain
from quant_warehouse.platforms.data_providers.thetadata.feature_engineering import filter_option_instrument_rows

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 20260716
torch.manual_seed(SEED)
np.random.seed(SEED)
MAX_ROWS = int(os.getenv("OPTION_MTL_MAX_ROWS", "100000"))
EPOCHS = int(os.getenv("OPTION_MTL_EPOCHS", "3"))
BATCH_SIZE = int(os.getenv("OPTION_MTL_BATCH_SIZE", "2048"))
RAW_FEATURES = [
    "bid", "ask", "mid", "underlying_price", "strike", "dte", "iv",
    "delta", "gamma", "theta", "vega", "rho", "volume", "open_interest",
]
LABEL = "change_percent"
GRAPH_FEATURE_TARGETS = [
    "long_hub", "long_authority", "short_hub", "short_authority",
    "long_pagerank", "short_pagerank",
]


class PrototypeHead(nn.Module):
    def __init__(self, hidden: int, classes: int, metric_dim: int = 24):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(hidden, metric_dim), nn.LayerNorm(metric_dim))
        self.prototypes = nn.Parameter(torch.randn(classes, metric_dim) * 0.02)
        self.temperature = nn.Parameter(torch.tensor(10.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = nn.functional.normalize(self.project(x), dim=-1)
        p = nn.functional.normalize(self.prototypes, dim=-1)
        return self.temperature.clamp(1.0, 50.0) * (z @ p.T)


class InstrumentMTL(nn.Module):
    def __init__(self, feature_dim: int, aux_dims: dict[str, int], event_dim: int, macro_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 48), nn.LayerNorm(48), nn.GELU(),
        )
        self.aux_heads = nn.ModuleDict({name: PrototypeHead(48, size) for name, size in aux_dims.items()})
        self.event_head = gnn.EventPrototypeHead(48, event_dim)
        self.macro_head = gnn.EventPrototypeHead(48, macro_dim)
        self.return_head = nn.Linear(48, 1)

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        return h, {name: head(h) for name, head in self.aux_heads.items()}, self.event_head(h), self.macro_head(h), self.return_head(h).squeeze(-1)


class TwoTowerCompatibility(nn.Module):
    """Small dual encoder used only to test return-aligned geometry."""

    def __init__(self, issuer_dim: int, instrument_dim: int, metric_dim: int = 32):
        super().__init__()
        self.issuer = nn.Sequential(nn.Linear(issuer_dim, 48), nn.LayerNorm(48), nn.GELU(), nn.Linear(48, metric_dim))
        self.instrument = nn.Sequential(nn.Linear(instrument_dim, 48), nn.LayerNorm(48), nn.GELU(), nn.Linear(48, metric_dim))
        self.option_type = nn.Embedding(2, metric_dim)

    def forward(self, issuer_x: torch.Tensor, instrument_x: torch.Tensor, option_type: torch.Tensor) -> torch.Tensor:
        issuer_z = nn.functional.normalize(self.issuer(issuer_x), dim=-1)
        instrument_z = nn.functional.normalize(self.instrument(instrument_x) + self.option_type(option_type), dim=-1)
        return (issuer_z * instrument_z).sum(dim=-1)


def aggregate_raw_option_prototypes(chain: pd.DataFrame) -> pd.DataFrame:
    """Create one raw-feature prototype per option side and snapshot date.

    This deliberately averages the raw contract features first.  No contract
    row is passed through an encoder, so memory and encoder cost scale with
    issuer-dates rather than the number of contracts in each chain.
    """
    if chain.empty:
        return pd.DataFrame()
    work = chain.copy()
    option_type = work.get("option_type", pd.Series(index=work.index, dtype="object"))
    option_type = option_type.astype(str).str.strip().str.lower()
    work["prototype_type"] = np.select(
        [option_type.str.startswith("c"), option_type.str.startswith("p")],
        ["call", "put"],
        default="",
    )
    work = work.loc[work["prototype_type"].isin(["call", "put"])].copy()
    if work.empty:
        return pd.DataFrame()
    numeric = [column for column in [*RAW_FEATURES, LABEL] if column in work.columns]
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    # Equal-contract means are intentional for this first scalability test.
    # Liquidity weighting can be added later without changing the row contract.
    grouped = (
        work.groupby(["snapshot_date", "prototype_type"], as_index=False, dropna=False)[numeric]
        .mean()
        .rename(columns={LABEL: "target_change_percent"})
    )
    counts = (
        work.groupby(["snapshot_date", "prototype_type"], as_index=False)
        .size()
        .rename(columns={"size": "prototype_contract_count"})
    )
    return grouped.merge(counts, on=["snapshot_date", "prototype_type"], how="left")


def load_rows(symbols: list[str], labels: pd.DataFrame) -> pd.DataFrame:
    requested = ["snapshot_date", "underlying_symbol", "contract_symbol", "option_type", *RAW_FEATURES, LABEL]
    pieces: list[pd.DataFrame] = []
    for symbol in symbols:
        chain = read_thetadata_eod_option_chain(
            symbol, start_date="2021-01-01", end_date="2025-12-31",
            columns=requested, require_rich_columns=True,
        )
        chain = filter_option_instrument_rows(chain)
        if chain.empty:
            continue
        filtered_rows = len(chain)
        prototypes = aggregate_raw_option_prototypes(chain)
        if prototypes.empty:
            continue
        prototypes["symbol"] = symbol
        prototypes["date"] = pd.to_datetime(prototypes["snapshot_date"], errors="coerce").dt.normalize()
        pieces.append(prototypes.drop(columns=["snapshot_date"]))
        print({"symbol": symbol, "filtered_contract_rows": filtered_rows,
               "prototype_rows": len(prototypes), "dates": prototypes.date.nunique()}, flush=True)
    if not pieces:
        return pd.DataFrame()
    rows = pd.concat(pieces, ignore_index=True)
    rows = rows.merge(labels, on=["symbol", "date"], how="inner")
    macro_cols = [column for column in rows.columns if str(column).startswith("is_macro_")]
    keep = ["symbol", "date", "prototype_type", *RAW_FEATURES, "target_change_percent",
            "prototype_contract_count", *gnn.AUX_TARGET_COLS, *GRAPH_FEATURE_TARGETS,
            *gnn.ALL_EVENT_TARGETS, *macro_cols]
    rows = rows[[column for column in keep if column in rows.columns]].copy()
    if len(rows) > MAX_ROWS:
        rows = rows.sample(MAX_ROWS, random_state=SEED).reset_index(drop=True)
    return rows


def main() -> None:
    index = pd.read_csv(gnn.feature_dir("1T") / "index.csv")
    first = pd.read_parquet(index.iloc[0].panel_path)
    symbols = sorted(first.symbol.astype(str).str.upper().unique())
    price_symbols = [symbol for symbol in symbols if symbol not in {"BRK-A", "BRK-B"}]
    _, labels = gnn.build_price_and_labels(symbols, "1T")
    labels = labels.copy()
    labels["symbol"] = labels.symbol.astype(str).str.upper()
    labels["date"] = pd.to_datetime(labels.date).dt.normalize()
    macro_panel, macro_cols = transformer_module._load_macro_event_panel(labels["date"])
    labels = labels.merge(macro_panel, on="date", how="left")
    rows = load_rows(price_symbols, labels)
    if rows.empty:
        raise RuntimeError("No filtered option rows joined to issuer labels")
    macro_cols = [column for column in rows.columns if str(column).startswith("is_macro_")]
    train = rows.date.dt.year < 2025
    if not train.any() or not (~train).any():
        raise RuntimeError("Expected both pre-2025 training rows and 2025 validation rows")
    prototype_features = [*RAW_FEATURES, "prototype_contract_count"]
    raw = rows[prototype_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = raw.loc[train].median().fillna(0.0)
    raw = raw.fillna(med).fillna(0.0)
    mean = raw.loc[train].mean().fillna(0.0)
    std = raw.loc[train].std().replace(0, 1).fillna(1.0)
    x = ((raw - mean) / std).clip(-8, 8).to_numpy(np.float32)
    aux_dims = {name: int(pd.to_numeric(rows[name], errors="coerce").max()) + 1 for name in gnn.AUX_TARGET_COLS}
    model = InstrumentMTL(len(prototype_features), aux_dims, len(gnn.ALL_EVENT_TARGETS), len(macro_cols)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    train_idx = np.flatnonzero(train.to_numpy())
    valid_idx = np.flatnonzero((~train).to_numpy())
    event_y = rows[list(gnn.ALL_EVENT_TARGETS)].fillna(0.0).to_numpy(np.float32)
    macro_y = rows[macro_cols].fillna(0.0).to_numpy(np.float32)
    aux_y = rows[list(gnn.AUX_TARGET_COLS)].fillna(-1).to_numpy(np.int64)
    return_y = pd.to_numeric(rows["target_change_percent"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    print({"prototype_rows": len(rows), "train_rows": len(train_idx), "validation_rows": len(valid_idx),
           "call_prototypes": int(rows.prototype_type.eq("call").sum()),
           "put_prototypes": int(rows.prototype_type.eq("put").sum()),
           "macro_tasks": len(macro_cols), "device": str(DEVICE)}, flush=True)
    for epoch in range(EPOCHS):
        model.train()
        order = np.random.permutation(train_idx)
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            xb = torch.from_numpy(x[idx]).to(DEVICE)
            optimizer.zero_grad()
            _, aux_hat, event_hat, macro_hat, return_hat = model(xb)
            loss = nn.functional.mse_loss(return_hat, torch.from_numpy(return_y[idx]).to(DEVICE))
            loss = loss + 0.1 * gnn.event_loss_from_logits(event_hat, torch.from_numpy(event_y[idx]).to(DEVICE))
            if macro_cols:
                loss = loss + gnn.event_loss_from_logits(macro_hat, torch.from_numpy(macro_y[idx]).to(DEVICE))
            for col_idx, name in enumerate(gnn.AUX_TARGET_COLS):
                target = torch.from_numpy(aux_y[idx, col_idx]).to(DEVICE)
                mask = target.ge(0)
                if mask.any():
                    loss = loss + 0.1 * nn.functional.cross_entropy(aux_hat[name][mask], target[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        print({"epoch": epoch + 1, "train_loss": round(float(np.mean(losses)), 5)}, flush=True)
    model.eval()
    with torch.no_grad():
        vals = []
        for start in range(0, len(valid_idx), BATCH_SIZE):
            idx = valid_idx[start:start + BATCH_SIZE]
            _, aux_hat, event_hat, macro_hat, return_hat = model(torch.from_numpy(x[idx]).to(DEVICE))
            batch_loss = nn.functional.mse_loss(return_hat, torch.from_numpy(return_y[idx]).to(DEVICE))
            batch_loss = batch_loss + 0.1 * gnn.event_loss_from_logits(event_hat, torch.from_numpy(event_y[idx]).to(DEVICE))
            if macro_cols:
                batch_loss = batch_loss + gnn.event_loss_from_logits(macro_hat, torch.from_numpy(macro_y[idx]).to(DEVICE))
            vals.append(float(batch_loss))
    print({"validation_return_and_event_loss": round(float(np.mean(vals)), 5), "status": "complete"}, flush=True)

    # Cheap geometry probe: use strictly lagged OHLC state for the issuer
    # tower and the already-aggregated raw prototype for the instrument tower.
    prices = gnn.build_price_and_labels(symbols, "1T")[0].copy()
    prices["symbol"] = prices.symbol.astype(str).str.upper()
    prices["date"] = pd.to_datetime(prices.date).dt.normalize()
    prices = prices.sort_values(["symbol", "date"])
    prices["prev_close"] = prices.groupby("symbol")["close"].shift(1)
    prices["ret_1d"] = prices.groupby("symbol")["close"].pct_change().groupby(prices["symbol"]).shift(1)
    prices["ret_5d"] = prices.groupby("symbol")["close"].pct_change(5).groupby(prices["symbol"]).shift(1)
    prices["ret_20d"] = prices.groupby("symbol")["close"].pct_change(20).groupby(prices["symbol"]).shift(1)
    issuer_cols = ["prev_close", "ret_1d", "ret_5d", "ret_20d"]
    probe = rows.merge(prices[["symbol", "date", *issuer_cols]], on=["symbol", "date"], how="inner")
    probe = probe.loc[probe[issuer_cols].notna().all(axis=1)].copy()
    probe_train = probe.date.dt.year < 2025
    issuer_raw = probe[issuer_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    issuer_mean = issuer_raw.loc[probe_train].mean().fillna(0.0)
    issuer_std = issuer_raw.loc[probe_train].std().replace(0, 1).fillna(1.0)
    issuer_x = ((issuer_raw - issuer_mean) / issuer_std).clip(-8, 8).to_numpy(np.float32)
    instrument_x = probe[prototype_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    instrument_x = instrument_x.fillna(med).fillna(0.0)
    instrument_x = ((instrument_x - mean) / std).clip(-8, 8).to_numpy(np.float32)
    option_type = probe.prototype_type.eq("put").astype(np.int64).to_numpy()
    returns = pd.to_numeric(probe.target_change_percent, errors="coerce").fillna(0.0).to_numpy(np.float32)
    train_idx = np.flatnonzero(probe_train.to_numpy())
    valid_idx = np.flatnonzero((~probe_train).to_numpy())
    tower = TwoTowerCompatibility(len(issuer_cols), len(prototype_features)).to(DEVICE)
    tower_optimizer = torch.optim.AdamW(tower.parameters(), lr=0.003, weight_decay=1e-4)
    for epoch in range(2):
        tower.train()
        order = np.random.permutation(train_idx)
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            scores = tower(torch.from_numpy(issuer_x[idx]).to(DEVICE), torch.from_numpy(instrument_x[idx]).to(DEVICE), torch.from_numpy(option_type[idx]).to(DEVICE))
            target = torch.from_numpy(returns[idx]).to(DEVICE)
            target = (target - target.mean()) / target.std().clamp_min(1e-4)
            loss = nn.functional.mse_loss(scores, target)
            tower_optimizer.zero_grad(); loss.backward(); tower_optimizer.step()
            losses.append(float(loss.detach()))
        print({"compatibility_epoch": epoch + 1, "loss": round(float(np.mean(losses)), 5)}, flush=True)
    tower.eval()
    with torch.no_grad():
        scores = tower(torch.from_numpy(issuer_x[valid_idx]).to(DEVICE), torch.from_numpy(instrument_x[valid_idx]).to(DEVICE), torch.from_numpy(option_type[valid_idx]).to(DEVICE)).cpu().numpy()
    valid_returns = returns[valid_idx]
    spearman = float(pd.Series(scores).corr(pd.Series(valid_returns), method="spearman"))
    ranking = probe.iloc[valid_idx].assign(score=scores, target=valid_returns)
    pair_rows = []
    for _, group in ranking.groupby(["symbol", "date"]):
        if len(group) == 2 and group.target.nunique() > 1:
            a, b = group.iloc[0], group.iloc[1]
            pair_rows.append(float(np.sign(a.target - b.target) == np.sign(a.score - b.score)))
    ranking = ranking.sort_values("score")
    k = max(1, len(ranking) // 10)
    print({"compatibility_validation_rows": len(valid_idx),
           "spearman_score_vs_change_percent": round(spearman, 5),
           "pairwise_accuracy": round(float(np.mean(pair_rows)) if pair_rows else float("nan"), 5),
           "bottom_decile_return": round(float(ranking.head(k).target.mean()), 5),
           "top_decile_return": round(float(ranking.tail(k).target.mean()), 5),
           "status": "compatibility_probe_complete"}, flush=True)


if __name__ == "__main__":
    main()
