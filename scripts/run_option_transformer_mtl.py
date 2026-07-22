"""Daily option-chain document transformer.

An observed ``(date, option contract)`` is one token.  A document is one
``(date, underlying symbol)`` and contains every valid option token observed
in that daily chain.  The encoder may use both forward and backward attention
within that document because all tokens have the same date.  Documents are
calendar-split: 2025 trains the model and 2026 evaluates it.

The two heads are deliberately limited to the requested option tasks:
``change_percent`` regression and within-chain percentile-rank prediction.
Missing option dates create no document and are never forward-filled.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE = ROOT.parent / "quant-warehouse"
os.environ.setdefault("PYTHONPATH", str(WAREHOUSE))
sys.path.insert(0, str(WAREHOUSE))
GNN_PATH = ROOT / "scripts" / "run_feature_family_gnn_smoke.py"
gnn_spec = importlib.util.spec_from_file_location("option_gnn_targets", GNN_PATH)
assert gnn_spec and gnn_spec.loader
gnn = importlib.util.module_from_spec(gnn_spec)
gnn_spec.loader.exec_module(gnn)
from quant_warehouse.platforms.data_providers.thetadata.feature_engineering import (  # noqa: E402
    filter_option_instrument_rows,
)
from quant_warehouse.platforms.data_providers.thetadata.options import (  # noqa: E402
    read_thetadata_eod_option_chain,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIER = os.getenv("OPTION_TRANSFORMER_TIER", "1T").strip().upper()
if TIER not in {"1T", "100B", "10B"}:
    raise ValueError(f"Unknown OPTION_TRANSFORMER_TIER={TIER!r}")
SEED = 20260721
EPOCHS = int(os.getenv("OPTION_TRANSFORMER_EPOCHS", "3"))
LR = float(os.getenv("OPTION_TRANSFORMER_LR", "0.001"))
MAX_TOKENS = int(os.getenv("OPTION_TRANSFORMER_MAX_TOKENS", "0"))
MAX_DOCUMENTS = int(os.getenv("OPTION_TRANSFORMER_MAX_DOCUMENTS", "0"))
RAW_FEATURES = [
    "underlying_price", "strike", "dte", "option_type_id",
    "bid", "ask", "mid", "iv", "volume", "open_interest",
    "delta", "vega", "theta", "rho",
    "gamma", "vanna", "charm", "vomma", "volga",
    "speed", "color", "zomma", "ultima",
]
FEATURE_FAMILIES = {
    "metadata": ["underlying_price", "strike", "dte", "option_type_id"],
    "market": ["bid", "ask", "mid", "iv", "volume", "open_interest"],
    "first_order_greeks": ["delta", "vega", "theta", "rho"],
    "second_order_greeks": ["gamma", "vanna", "charm", "vomma", "volga"],
    "third_order_greeks": ["speed", "color", "zomma", "ultima"],
}
FEATURE_FAMILY = os.getenv("OPTION_TRANSFORMER_FEATURE_FAMILY", "all").strip().lower()
if FEATURE_FAMILY == "all":
    FEATURES = list(RAW_FEATURES)
elif FEATURE_FAMILY in FEATURE_FAMILIES:
    FEATURES = list(FEATURE_FAMILIES[FEATURE_FAMILY])
else:
    raise ValueError(f"Unknown OPTION_TRANSFORMER_FEATURE_FAMILY={FEATURE_FAMILY!r}; choose all or {sorted(FEATURE_FAMILIES)}")
SOURCE_FEATURES = [column for column in RAW_FEATURES if column != "option_type_id"]
LABEL = "change_percent"
GRAPH_TARGETS = ["long_hub", "long_authority", "short_hub", "short_authority", "long_pagerank", "short_pagerank"]
SPEED_TARGETS = list(gnn.SPEED_TARGET_COLS)
EVENT_TARGETS = list(gnn.ALL_EVENT_TARGETS)
AUX_TARGETS = list(gnn.AUX_TARGET_COLS)
torch.manual_seed(SEED)
np.random.seed(SEED)


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


def feature_dir() -> Path:
    path = ROOT / "artifacts" / "features" / TIER
    if path.exists():
        return path
    spec = importlib.util.spec_from_file_location("gnn_loader", ROOT / "scripts" / "run_feature_family_gnn_smoke.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.feature_dir(TIER)


def symbols_1t() -> list[str]:
    index = pd.read_csv(feature_dir() / "index.csv")
    symbols: set[str] = set()
    for path in index.panel_path.astype(str):
        symbols.update(pd.read_parquet(path, columns=["symbol"]).symbol.astype(str).str.upper())
    return sorted(symbols - {"BRK-A", "BRK-B"})


def load_documents(symbols: list[str], issuer_labels: pd.DataFrame) -> list[pd.DataFrame]:
    requested = ["snapshot_date", "underlying_symbol", "contract_symbol", "option_type", *SOURCE_FEATURES, LABEL]
    documents: list[pd.DataFrame] = []
    for symbol in symbols:
        chain = read_thetadata_eod_option_chain(
            symbol, start_date="2025-01-01", end_date="2026-12-31",
            columns=[*requested], require_rich_columns=True,
        )
        chain = filter_option_instrument_rows(chain)
        if chain.empty:
            print({"symbol": symbol, "rows": 0}, flush=True)
            continue
        chain = chain.copy()
        chain["date"] = pd.to_datetime(chain.snapshot_date, errors="coerce").dt.normalize()
        chain["symbol"] = symbol
        chain["date"] = pd.to_datetime(chain["date"]).dt.normalize()
        chain = chain.merge(issuer_labels, on=["symbol", "date"], how="left")
        chain["option_type_id"] = chain.option_type.astype(str).str.lower().str.startswith("p").astype(np.int64)
        chain["rank_target"] = chain.groupby(["date", "symbol", "option_type_id"], dropna=False)[LABEL].rank(
            method="average", pct=True,
        ).astype(np.float32)
        chain = chain.sort_values(["date", "option_type_id", "dte", "strike", "contract_symbol"], kind="stable")
        for (date, _), doc in chain.groupby(["date", "symbol"], sort=False):
            keep = ["date", "symbol", "contract_symbol", *FEATURES, LABEL, "rank_target",
                    *GRAPH_TARGETS, *SPEED_TARGETS, *EVENT_TARGETS, *AUX_TARGETS]
            doc = doc[keep].reset_index(drop=True)
            if MAX_TOKENS and len(doc) > MAX_TOKENS:
                # Explicitly opt-in only; default keeps every valid daily option token.
                doc = doc.iloc[:MAX_TOKENS].copy()
            documents.append(doc)
        print({"symbol": symbol, "daily_documents": int(chain.date.nunique()),
               "daily_option_rows": len(chain), "max_tokens": int(chain.groupby("date").size().max())}, flush=True)
    documents.sort(key=lambda d: d.date.iloc[0])
    if MAX_DOCUMENTS and len(documents) > MAX_DOCUMENTS:
        # Keep both calendar partitions when a smoke-test limit is supplied.
        train = [d for d in documents if d.date.iloc[0].year == 2025]
        valid = [d for d in documents if d.date.iloc[0].year == 2026]
        half = max(1, MAX_DOCUMENTS // 2)
        documents = train[:half] + valid[:MAX_DOCUMENTS - half]
    return documents


def normalize_documents(documents: list[pd.DataFrame], train_docs: list[bool]) -> tuple[list[dict[str, np.ndarray]], float, float, dict[str, int]]:
    train_rows = pd.concat([d.loc[:, FEATURES] for d, is_train in zip(documents, train_docs) if is_train], ignore_index=True)
    train_rows = train_rows.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = train_rows.median().fillna(0.0)
    mean = train_rows.fillna(med).mean().fillna(0.0)
    std = train_rows.fillna(med).std().replace(0, 1).fillna(1.0)
    change_train = pd.concat([d[LABEL] for d, is_train in zip(documents, train_docs) if is_train], ignore_index=True)
    change_train = pd.to_numeric(change_train, errors="coerce").fillna(0.0)
    change_mean, change_std = float(change_train.mean()), float(change_train.std()) or 1.0
    aux_dims = {
        column: int(pd.to_numeric(pd.concat([d[column] for d, flag in zip(documents, train_docs) if flag]), errors="coerce").max()) + 1
        for column in AUX_TARGETS
    }
    normalized: list[dict[str, np.ndarray]] = []
    for doc in documents:
        raw = doc[FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        x = ((raw.fillna(med).fillna(0.0) - mean) / std).clip(-8, 8).to_numpy(np.float32)
        y_change = ((pd.to_numeric(doc[LABEL], errors="coerce").fillna(0.0).to_numpy(np.float32) - change_mean) / change_std)
        normalized.append({
            "x": x, "change": y_change.astype(np.float32), "rank": doc.rank_target.to_numpy(np.float32),
            "graph": doc[GRAPH_TARGETS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32),
            "speed": doc[SPEED_TARGETS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32),
            "events": doc[EVENT_TARGETS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32),
            "aux": doc[AUX_TARGETS].apply(pd.to_numeric, errors="coerce").fillna(-1).to_numpy(np.int64),
        })
    return normalized, change_mean, change_std, aux_dims


class OptionChainTransformer(nn.Module):
    def __init__(self, feature_dim: int, aux_dims: dict[str, int], hidden: int = 64, layers: int = 2):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(feature_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        block = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=hidden * 4,
                                           dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(hidden)
        self.document_attention = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)
        )
        self.change_head = nn.Linear(hidden, 1)
        self.rank_head = nn.Linear(hidden, 1)
        self.graph_head = nn.Linear(hidden, len(GRAPH_TARGETS))
        self.speed_head = nn.Linear(hidden, len(SPEED_TARGETS))
        self.event_head = gnn.EventPrototypeHead(hidden, len(EVENT_TARGETS))
        self.aux_heads = nn.ModuleDict({name: PrototypeHead(hidden, max(1, size)) for name, size in aux_dims.items()})

    def forward(self, x: torch.Tensor, padding: torch.Tensor):
        h = self.norm(self.encoder(self.input(x), src_key_padding_mask=padding))
        valid = (~padding).unsqueeze(-1)
        attention_logits = self.document_attention(h).masked_fill(padding.unsqueeze(-1), float("-inf"))
        attention_weights = torch.softmax(attention_logits, dim=1)
        document_h = (h * attention_weights).sum(dim=1)
        return (
            self.change_head(h).squeeze(-1), self.rank_head(h).squeeze(-1),
            self.graph_head(document_h), self.speed_head(document_h), self.event_head(document_h),
            {name: head(document_h) for name, head in self.aux_heads.items()},
        )


def run_documents(model: nn.Module, docs: list[dict[str, np.ndarray]], indices: list[int], train: bool,
                  optimizer: torch.optim.Optimizer | None, change_mean: float, change_std: float) -> dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()
    total_loss: list[float] = []
    actual_change: list[np.ndarray] = []
    predicted_change: list[np.ndarray] = []
    actual_rank: list[np.ndarray] = []
    predicted_rank: list[np.ndarray] = []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for i in indices:
            item = docs[i]
            x, yc, yr = item["x"], item["change"], item["rank"]
            xb = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
            pb = torch.zeros((1, len(x)), dtype=torch.bool, device=DEVICE)
            change_hat, rank_hat, graph_hat, speed_hat, event_hat, aux_hat = model(xb, pb)
            change_hat, rank_hat = change_hat.squeeze(0), rank_hat.squeeze(0)
            graph_hat, speed_hat, event_hat = graph_hat, speed_hat, event_hat
            yc_t, yr_t = torch.from_numpy(yc).to(DEVICE), torch.from_numpy(yr).to(DEVICE)
            # Equity labels are document-level and repeated on every option
            # token in the source panel; use one row as the document label.
            graph_t = torch.from_numpy(item["graph"][0]).unsqueeze(0).to(DEVICE)
            speed_t = torch.from_numpy(item["speed"][0]).unsqueeze(0).to(DEVICE)
            event_t = torch.from_numpy(item["events"][0]).unsqueeze(0).to(DEVICE)
            aux_t = torch.from_numpy(item["aux"][0]).unsqueeze(0).to(DEVICE)
            loss_change = nn.functional.smooth_l1_loss(change_hat, yc_t)
            loss_rank = nn.functional.mse_loss(torch.sigmoid(rank_hat), yr_t)
            loss_graph = nn.functional.mse_loss(graph_hat, graph_t)
            loss_speed = nn.functional.mse_loss(speed_hat, speed_t)
            loss_event = gnn.event_loss_from_logits(event_hat, event_t)
            loss_aux = graph_hat.new_zeros(())
            for col, name in enumerate(AUX_TARGETS):
                target = aux_t[:, col]
                mask = target.ge(0) & target.lt(aux_hat[name].shape[-1])
                if mask.any():
                    loss_aux = loss_aux + nn.functional.cross_entropy(aux_hat[name][mask], target[mask])
            loss = loss_change + loss_rank + 0.1 * (loss_graph + loss_speed + loss_event + loss_aux)
            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss.append(float(loss.detach()))
            actual_change.append(yc * change_std + change_mean)
            predicted_change.append(change_hat.detach().cpu().numpy() * change_std + change_mean)
            actual_rank.append(yr)
            predicted_rank.append(torch.sigmoid(rank_hat).detach().cpu().numpy())
    ac, pc = np.concatenate(actual_change), np.concatenate(predicted_change)
    ar, pr = np.concatenate(actual_rank), np.concatenate(predicted_rank)
    return {
        "loss": float(np.mean(total_loss)),
        "change_rmse": float(np.sqrt(np.mean((ac - pc) ** 2))),
        "change_spearman": float(pd.Series(ac).corr(pd.Series(pc), method="spearman") or 0.0),
        "rank_rmse": float(np.sqrt(np.mean((ar - pr) ** 2))),
        "rank_spearman": float(pd.Series(ar).corr(pd.Series(pr), method="spearman") or 0.0),
    }


def main() -> None:
    env_symbols = os.getenv("OPTION_TRANSFORMER_SYMBOLS", "")
    symbols = [s.strip().upper() for s in env_symbols.split(",") if s.strip()] or symbols_1t()
    _, issuer_labels = gnn.build_price_and_labels(symbols, TIER)
    issuer_labels = issuer_labels.copy()
    issuer_labels["symbol"] = issuer_labels.symbol.astype(str).str.upper()
    issuer_labels["date"] = pd.to_datetime(issuer_labels.date, errors="coerce").dt.normalize()
    documents_df = load_documents(symbols, issuer_labels)
    if not documents_df:
        raise RuntimeError("No daily option documents were loaded")
    train_flags = [int(d.date.iloc[0].year) == 2025 for d in documents_df]
    valid_flags = [int(d.date.iloc[0].year) == 2026 for d in documents_df]
    if not any(train_flags) or not any(valid_flags):
        raise RuntimeError("Expected both 2025 training and 2026 evaluation documents")
    docs, change_mean, change_std, aux_dims = normalize_documents(documents_df, train_flags)
    train_indices = [i for i, flag in enumerate(train_flags) if flag]
    valid_indices = [i for i, flag in enumerate(valid_flags) if flag]
    model = OptionChainTransformer(len(FEATURES), aux_dims).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    print({
        "tier": TIER, "symbols": symbols, "train_documents_2025": len(train_indices), "eval_documents_2026": len(valid_indices),
        "train_tokens": int(sum(len(documents_df[i]) for i in train_indices)),
        "eval_tokens": int(sum(len(documents_df[i]) for i in valid_indices)),
        "features_per_option_token": len(FEATURES), "max_tokens_in_document": int(max(len(d) for d in documents_df)),
        "equity_mtl_tasks": {"change_percent": True, "rank": True, "graph": len(GRAPH_TARGETS),
                             "speed": len(SPEED_TARGETS), "company_event": len(EVENT_TARGETS),
                             "sector_industry_year": AUX_TARGETS},
        "attention": "bidirectional within each same-date option document", "device": str(DEVICE),
    }, flush=True)
    for epoch in range(EPOCHS):
        np.random.shuffle(train_indices)
        train_metrics = run_documents(model, docs, train_indices, True, optimizer, change_mean, change_std)
        valid_metrics = run_documents(model, docs, valid_indices, False, None, change_mean, change_std)
        print({"epoch": epoch + 1, "train": train_metrics, "evaluation_2026": valid_metrics}, flush=True)
    print({"status": "complete", "evaluation_2026": run_documents(model, docs, valid_indices, False, None, change_mean, change_std)}, flush=True)


if __name__ == "__main__":
    main()
