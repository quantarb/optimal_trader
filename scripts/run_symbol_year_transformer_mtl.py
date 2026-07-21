"""Causal transformer MTL baseline with one document per symbol-year."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
GNN_PATH = REPO_ROOT / "scripts" / "run_feature_family_gnn_smoke.py"
spec = importlib.util.spec_from_file_location("gnn_baseline_module", GNN_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load {GNN_PATH}")
gnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gnn)
from quant_warehouse.ingest.macro_fetch import fetch_economy_calendar_range
from quant_warehouse.platforms.data_providers.fmp.target_engineering import (
    build_macro_event_label_panel,
)

FIRST_TEST_YEAR = int(os.getenv("TRANSFORMER_FIRST_TEST_YEAR", "2021"))
LAST_TEST_YEAR = int(os.getenv("TRANSFORMER_LAST_TEST_YEAR", "2025"))
EPOCHS = int(os.getenv("TRANSFORMER_EPOCHS", "12"))
HIDDEN = int(os.getenv("TRANSFORMER_HIDDEN", "48"))
LAYERS = int(os.getenv("TRANSFORMER_LAYERS", "2"))
HEADS = int(os.getenv("TRANSFORMER_HEADS", "4"))
TRUNKS = max(1, int(os.getenv("TRANSFORMER_TRUNKS", "1")))
BATCH_SIZE = int(os.getenv("TRANSFORMER_BATCH_SIZE", "32"))
LR = float(os.getenv("TRANSFORMER_LR", "0.002"))
MACRO_ENABLED = os.getenv("TRANSFORMER_MACRO", "0") == "1"
MACRO_COUNTRIES = tuple(x.strip().upper() for x in os.getenv("TRANSFORMER_MACRO_COUNTRIES", "US").split(",") if x.strip())
DEVICE_NAME = os.getenv("TRANSFORMER_DEVICE", "auto").strip().lower()
if DEVICE_NAME == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(DEVICE_NAME)
if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")
SEED = 20260716
torch.manual_seed(SEED)
np.random.seed(SEED)

OUT = REPO_ROOT / "artifacts" / "graph_oracle_feature_family_gnn_improved_hits"
OUT.mkdir(parents=True, exist_ok=True)
TARGET_COLS = [
    "long_hub", "long_authority", "short_hub", "short_authority",
    "long_pagerank", "short_pagerank",
]
EVENT_COLS = list(gnn.ALL_EVENT_TARGETS)
AUX_COLS = list(gnn.AUX_TARGET_COLS)
MACRO_EVENT_COLS: list[str] = []


def causal_mask(length: int, device: torch.device | None = None) -> torch.Tensor:
    """Additive mask: each position can attend only to itself and its past."""
    return torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)


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


class TransformerMTL(nn.Module):
    def __init__(self, feature_dim: int, aux_dims: dict[str, int]):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(feature_dim, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU())
        layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN, nhead=HEADS, dim_feedforward=HIDDEN * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        # Trunk 0 is reserved for the graph/regression task.  When more than
        # one trunk is requested, the remaining task heads use trunk 1.  This
        # keeps the input representation and causal attention identical while
        # allowing graph and event/auxiliary gradients to specialize.
        self.encoders = nn.ModuleList(
            [nn.TransformerEncoder(layer, num_layers=LAYERS) for _ in range(TRUNKS)]
        )
        self.position = nn.Parameter(torch.randn(512, HIDDEN) * 0.01)
        self.graph_head = nn.Linear(HIDDEN, len(TARGET_COLS))
        self.event_head = gnn.EventPrototypeHead(HIDDEN, len(EVENT_COLS))
        self.aux_heads = nn.ModuleDict({
            name: PrototypeHead(HIDDEN, max(1, int(size)))
            for name, size in aux_dims.items()
        })
        self.macro_head = gnn.EventPrototypeHead(HIDDEN, len(MACRO_EVENT_COLS)) if MACRO_EVENT_COLS else None
        # Each MTL head learns how much to use every trunk.  The logits start
        # at zero, so a new multi-trunk run starts as an equal mixture rather
        # than encoding a hand-written task-to-trunk assignment.
        task_names = ["graph", "event"] + [f"aux:{name}" for name in self.aux_heads]
        if self.macro_head is not None:
            task_names.append("macro")
        self.task_router = nn.ParameterDict({
            name.replace(":", "__"): nn.Parameter(torch.zeros(TRUNKS))
            for name in task_names
        })

    def routing_weights(self) -> dict[str, list[float]]:
        return {
            name.replace("__", ":"): torch.softmax(logits.detach(), dim=0).cpu().tolist()
            for name, logits in self.task_router.items()
        }

    def _task_state(self, task: str, states: list[torch.Tensor]) -> torch.Tensor:
        key = task.replace(":", "__")
        weights = torch.softmax(self.task_router[key], dim=0)
        return torch.stack(states, dim=0).mul(weights[:, None, None, None]).sum(dim=0)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor):
        length = x.shape[1]
        if length > self.position.shape[0]:
            raise ValueError(f"document length {length} exceeds positional capacity {self.position.shape[0]}")
        h = self.input(x) + self.position[:length].unsqueeze(0)
        mask = causal_mask(length, x.device)
        trunk_states = [
            encoder(h, mask=mask, src_key_padding_mask=padding_mask)
            for encoder in self.encoders
        ]
        graph_h = self._task_state("graph", trunk_states)
        event_h = self._task_state("event", trunk_states)
        macro_h = self._task_state("macro", trunk_states) if self.macro_head is not None else None
        aux_h = {
            name: self._task_state(f"aux:{name}", trunk_states)
            for name in self.aux_heads
        }
        macro_logits = self.macro_head(macro_h) if macro_h is not None else None
        return self.graph_head(graph_h), self.event_head(event_h), {name: head(aux_h[name]) for name, head in self.aux_heads.items()}, macro_logits


def _read_fused_panel(index: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    parts: list[pd.DataFrame] = []
    feature_cols: list[str] = []
    for _, meta in index.iterrows():
        panel = pd.read_parquet(meta.panel_path)
        metadata = pd.read_parquet(meta.metadata_path)
        family = str(meta.family)
        cols = [str(c) for c in metadata.feature if str(c) in panel.columns]
        frame = panel[["symbol", "date", *cols]].copy()
        frame["symbol"] = frame.symbol.astype(str).str.upper()
        frame["date"] = pd.to_datetime(frame.date).dt.normalize()
        renamed = {column: f"{family}__{column}" for column in cols}
        frame = frame.rename(columns=renamed)
        parts.append(frame.sort_values(["symbol", "date"]).set_index(["symbol", "date"]))
        feature_cols.extend(renamed.values())
    if not parts:
        return pd.DataFrame(), []
    fused = pd.concat(parts, axis=1, join="inner").reset_index()
    return fused.sort_values(["symbol", "date"]).reset_index(drop=True), feature_cols


def _make_docs(base: pd.DataFrame, test_year: int, train: bool) -> list[dict[str, np.ndarray]]:
    year = base.date.dt.year
    selected = base.loc[year < test_year if train else year == test_year].copy()
    docs: list[dict[str, np.ndarray]] = []
    selected["_year"] = selected.date.dt.year
    for (symbol, _), frame in selected.groupby(["symbol", "_year"], sort=True):
        frame = frame.sort_values("date")
        docs.append({
            "symbol": np.array([symbol] * len(frame)),
            "date": frame.date.to_numpy(),
            "x": np.stack(frame["__x__"].to_numpy()),
            "graph": frame[TARGET_COLS].to_numpy(np.float32),
            "events": frame[EVENT_COLS].to_numpy(np.float32),
            "macro_events": frame[MACRO_EVENT_COLS].to_numpy(np.float32) if MACRO_EVENT_COLS else np.zeros((len(frame), 0), dtype=np.float32),
            "aux": frame[AUX_COLS].to_numpy(np.int64),
            "graph_mask": np.stack(frame["__graph_mask__"].to_numpy()),
        })
    return docs


def _batch(docs: list[dict[str, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    length = max(len(item["x"]) for item in docs)
    dim = docs[0]["x"].shape[1]
    x = np.zeros((len(docs), length, dim), dtype=np.float32)
    padding = np.ones((len(docs), length), dtype=bool)
    graph = np.zeros((len(docs), length, len(TARGET_COLS)), dtype=np.float32)
    events = np.zeros((len(docs), length, len(EVENT_COLS)), dtype=np.float32)
    macro_events = np.zeros((len(docs), length, len(MACRO_EVENT_COLS)), dtype=np.float32)
    aux = np.full((len(docs), length, len(AUX_COLS)), -1, dtype=np.int64)
    graph_mask = np.zeros((len(docs), length, len(TARGET_COLS)), dtype=np.float32)
    for i, item in enumerate(docs):
        n = len(item["x"])
        x[i, :n] = item["x"]
        padding[i, :n] = False
        graph[i, :n] = item["graph"]
        events[i, :n] = item["events"]
        if MACRO_EVENT_COLS:
            macro_events[i, :n] = item["macro_events"]
        aux[i, :n] = item["aux"]
        graph_mask[i, :n] = item["graph_mask"]
    return (
        torch.from_numpy(x), torch.from_numpy(padding),
        {"graph": torch.from_numpy(graph), "events": torch.from_numpy(events),
         "macro_events": torch.from_numpy(macro_events), "aux": torch.from_numpy(aux),
         "graph_mask": torch.from_numpy(graph_mask)},
    )


def _load_macro_event_panel(dates: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """Fetch FMP releases once and build labels through quant-warehouse."""
    cache = OUT / "cache" / f"transformer_macro_event_labels_v3_{gnn.DATA_END:%Y%m%d}.parquet"
    if cache.exists():
        panel = pd.read_parquet(cache)
    else:
        events = fetch_economy_calendar_range(
            start_date="2020-01-01", end_date=str(gnn.DATA_END.date())
        )
        if MACRO_COUNTRIES and not events.empty:
            events = events.loc[events.country.astype(str).str.upper().isin(MACRO_COUNTRIES)].copy()
        token_dates = pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce").dt.normalize().unique()})
        panel = build_macro_event_label_panel(token_dates, events)
        cache.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(cache, index=False)
    macro_cols = [column for column in panel.columns if str(column).startswith("is_")]
    return panel, macro_cols


def _prepare_data(tier: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    global MACRO_EVENT_COLS
    index = pd.read_csv(gnn.feature_dir(tier) / "index.csv")
    fused, feature_cols = _read_fused_panel(index)
    symbols = sorted(fused.symbol.unique())
    prices, labels = gnn.build_price_and_labels(symbols, tier)
    base = fused.merge(labels, on=["symbol", "date"], how="inner")
    if MACRO_ENABLED:
        macro_panel, MACRO_EVENT_COLS = _load_macro_event_panel(base["date"])
        base = base.merge(macro_panel, on="date", how="left")
        base[MACRO_EVENT_COLS] = base[MACRO_EVENT_COLS].fillna(0.0).astype("float32")
    else:
        MACRO_EVENT_COLS = []
    return base, prices, feature_cols


def _normalize(base: pd.DataFrame, feature_cols: list[str], test_year: int) -> pd.DataFrame:
    out = base.copy()
    train = out.date.dt.year < test_year
    raw = out[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = raw.loc[train].median().fillna(0.0)
    filled = raw.fillna(med)
    mean = filled.loc[train].mean().fillna(0.0)
    std = filled.loc[train].std().replace(0, 1).fillna(1.0)
    normalized = ((filled - mean) / std).clip(-8, 8).astype("float32")
    out["__x__"] = list(normalized.to_numpy())
    for column in TARGET_COLS + EVENT_COLS:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).astype("float32")
    graph_mask = pd.DataFrame(False, index=out.index, columns=TARGET_COLS)
    for column in ("long_pagerank", "short_pagerank"):
        ranks = out.groupby(out.date.dt.year)[column].rank(method="first", pct=True)
        graph_mask[column] = ranks.le(gnn.HITS_TAIL_QUANTILE) | ranks.ge(1.0 - gnn.HITS_TAIL_QUANTILE)
    out["__graph_mask__"] = list(graph_mask.astype("float32").to_numpy())
    return out


def _train_epoch(model: TransformerMTL, docs: list[dict[str, np.ndarray]], optimizer: torch.optim.Optimizer) -> float:
    model.train()
    losses: list[float] = []
    order = np.random.permutation(len(docs))
    for offset in range(0, len(order), BATCH_SIZE):
        selected = [docs[i] for i in order[offset:offset + BATCH_SIZE]]
        x, padding, target = _batch(selected)
        x = x.to(DEVICE); padding = padding.to(DEVICE)
        target = {name: value.to(DEVICE) for name, value in target.items()}
        optimizer.zero_grad()
        graph_hat, event_logits, aux_logits, macro_logits = model(x, padding)
        valid = ~padding
        graph_mask = target["graph_mask"] * valid.unsqueeze(-1)
        graph_error = nn.functional.smooth_l1_loss(graph_hat, target["graph"], reduction="none")
        graph_loss = (graph_error * graph_mask).sum() / graph_mask.sum().clamp_min(1.0)
        event_loss = gnn.event_loss_from_logits(event_logits[valid], target["events"][valid])
        macro_loss = torch.tensor(0.0, device=DEVICE)
        if macro_logits is not None and MACRO_EVENT_COLS:
            macro_loss = gnn.event_loss_from_logits(macro_logits[valid], target["macro_events"][valid])
        aux_loss = torch.tensor(0.0, device=DEVICE)
        for index, name in enumerate(AUX_COLS):
            mask = valid & target["aux"][:, :, index].ge(0)
            if mask.any():
                aux_loss = aux_loss + nn.functional.cross_entropy(aux_logits[name][mask], target["aux"][:, :, index][mask])
        loss = (
            graph_loss + event_loss
            + float(os.getenv("TRANSFORMER_MACRO_LOSS_WEIGHT", "1.0")) * macro_loss
            + float(os.getenv("TRANSFORMER_AUX_LOSS_WEIGHT", "0.10")) * aux_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else 0.0


def run_tier(tier: str) -> pd.DataFrame:
    started = perf_counter()
    base, prices, feature_cols = _prepare_data(tier)
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    next_returns = close.pct_change().shift(-1)
    effective_top_k = min(20, len(close.columns))
    summaries: list[pd.DataFrame] = []
    for test_year in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1):
        normalized = _normalize(base, feature_cols, test_year)
        train_docs = _make_docs(normalized, test_year, True)
        test_docs = _make_docs(normalized, test_year, False)
        if not train_docs or not test_docs:
            continue
        aux_dims = {name: int(normalized[name].max()) + 1 for name in AUX_COLS}
        model = TransformerMTL(len(feature_cols), aux_dims).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        print({"tier": tier, "year": test_year, "documents": len(train_docs), "test_documents": len(test_docs), "tokens": sum(len(d["x"]) for d in train_docs), "macro_tasks": len(MACRO_EVENT_COLS), "trunks": TRUNKS, "routing": "learned_task_softmax", "architecture": "causal_symbol_year_transformer", "device": str(DEVICE)}, flush=True)
        for epoch in range(EPOCHS):
            loss = _train_epoch(model, train_docs, optimizer)
            if epoch == 0 or epoch == EPOCHS - 1:
                print({"tier": tier, "year": test_year, "epoch": epoch + 1, "transformer_loss": round(loss, 5)}, flush=True)
        print({"tier": tier, "year": test_year, "task_trunk_weights": model.routing_weights()}, flush=True)
        model.eval()
        predictions: list[pd.DataFrame] = []
        with torch.no_grad():
            for start in range(0, len(test_docs), BATCH_SIZE):
                batch_docs = test_docs[start:start + BATCH_SIZE]
                x, padding, _ = _batch(batch_docs)
                x = x.to(DEVICE); padding = padding.to(DEVICE)
                graph_hat, _, _, _ = model(x, padding)
                values = graph_hat.cpu().numpy()
                for row, doc in enumerate(batch_docs):
                    n = len(doc["x"])
                    pred = pd.DataFrame({"symbol": doc["symbol"], "date": doc["date"]})
                    pred[TARGET_COLS] = values[row, :n]
                    predictions.append(pred)
        pred = pd.concat(predictions, ignore_index=True)
        for column in TARGET_COLS:
            pred[column] = pred.groupby("date")[column].rank(pct=True, method="average")
        pred["long_score"], pred["long_exit_score"] = pred["long_hub"], pred["long_authority"]
        pred["short_score"], pred["short_exit_score"] = pred["short_hub"], pred["short_authority"]
        pred["long_agree_count"] = (pred.long_score >= pred.short_score).astype(int)
        pred["short_agree_count"] = (pred.short_score > pred.long_score).astype(int)
        pred["model_count"] = 1
        dates = pd.DatetimeIndex(next_returns.index[(next_returns.index >= f"{test_year}-01-01") & (next_returns.index <= f"{test_year}-12-31")])
        summary, _, _ = gnn.run_shared_book_framework_comparison(
            scores=pred[["symbol", "date", "long_score", "short_score", "long_exit_score", "short_exit_score", "long_agree_count", "short_agree_count", "model_count"]],
            next_returns=next_returns, symbols=tuple(close.columns), dates=dates,
            variants=("long_only",), top_k_values=(effective_top_k,), entry_threshold=0.5,
            exit_threshold=0.5, cost_models={"family_common": gnn.SharedBookCostModel(0.5, 5.0)},
        )
        if not summary.empty:
            summary["tier"] = tier
            summary["year"] = test_year
            summary["family"] = f"causal_symbol_year_transformer_{TRUNKS}trunk_learnedrouting"
            summary["label_source"] = ("transformer_macro_mtl" if MACRO_ENABLED else "transformer_baseline_mtl") + f"_{TRUNKS}trunk_learnedrouting"
            summaries.append(summary)
        run_label = ("transformer_macro" if MACRO_ENABLED else "transformer") + f"_{TRUNKS}trunk_learnedrouting"
        pd.concat(summaries, ignore_index=True).to_parquet(OUT / f"{tier.lower()}_{run_label}_long_only_through_{test_year}.parquet", index=False)
    result = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    run_label = ("transformer_macro" if MACRO_ENABLED else "transformer") + f"_{TRUNKS}trunk_learnedrouting"
    result.to_csv(OUT / f"{tier.lower()}_{run_label}_wfo_results.csv", index=False)
    print(result.to_string(index=False) if not result.empty else result)
    print({"tier": tier, "transformer_seconds": round(perf_counter() - started, 1), "result_rows": len(result)}, flush=True)
    return result


def main() -> None:
    requested = tuple(x.strip().upper() for x in os.getenv("TRANSFORMER_TIERS", "1T").split(",") if x.strip())
    for tier in requested:
        run_tier(tier)


if __name__ == "__main__":
    main()
