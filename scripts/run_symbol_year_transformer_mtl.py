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
    build_macro_family_label_panel,
)

FIRST_TEST_YEAR = int(os.getenv("TRANSFORMER_FIRST_TEST_YEAR", "2021"))
LAST_TEST_YEAR = int(os.getenv("TRANSFORMER_LAST_TEST_YEAR", "2025"))
EPOCHS = int(os.getenv("TRANSFORMER_EPOCHS", "12"))
HIDDEN = int(os.getenv("TRANSFORMER_HIDDEN", "48"))
LAYERS = int(os.getenv("TRANSFORMER_LAYERS", "2"))
HEADS = int(os.getenv("TRANSFORMER_HEADS", "4"))
TRUNKS = max(1, int(os.getenv("TRANSFORMER_TRUNKS", "1")))
ROUTING_MODE = os.getenv("TRANSFORMER_ROUTING", "soft").strip().lower()
ROUTING_TEMPERATURE = max(0.1, float(os.getenv("TRANSFORMER_ROUTING_TEMPERATURE", "1.0")))
ROUTING_BALANCE_WEIGHT = float(os.getenv("TRANSFORMER_ROUTING_BALANCE_WEIGHT", "0.01"))
GRADNORM_ENABLED = os.getenv("TRANSFORMER_GRADNORM", "0") == "1"
GRADNORM_ALPHA = float(os.getenv("TRANSFORMER_GRADNORM_ALPHA", "0.5"))
GRADNORM_LR = float(os.getenv("TRANSFORMER_GRADNORM_LR", "0.025"))
BATCH_SIZE = int(os.getenv("TRANSFORMER_BATCH_SIZE", "32"))
LR = float(os.getenv("TRANSFORMER_LR", "0.002"))
MACRO_ENABLED = os.getenv("TRANSFORMER_MACRO", "0") == "1"
CROSS_SECTIONAL_ENABLED = os.getenv("TRANSFORMER_CROSS_SECTIONAL", "0") == "1"
CROSS_SECTIONAL_TOKEN_TASKS = os.getenv("TRANSFORMER_CROSS_TOKEN_TASKS", "1") == "1"
MACRO_COUNTRIES = tuple(x.strip().upper() for x in os.getenv("TRANSFORMER_MACRO_COUNTRIES", "US").split(",") if x.strip())
DEVICE_NAME = os.getenv("TRANSFORMER_DEVICE", "auto").strip().lower()
if DEVICE_NAME == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(DEVICE_NAME)
if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")
    if GRADNORM_ENABLED:
        # GradNorm differentiates through gradient norms. Fused CUDA
        # attention kernels do not expose the required second derivative.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
SEED = 20260716
torch.manual_seed(SEED)
np.random.seed(SEED)

OUT = REPO_ROOT / "artifacts" / "graph_oracle_feature_family_gnn_improved_hits"
OUT.mkdir(parents=True, exist_ok=True)
TARGET_COLS = [
    "long_hub", "long_authority", "short_hub", "short_authority",
    "long_pagerank", "short_pagerank",
]
SPEED_TARGET_COLS = list(gnn.SPEED_TARGET_COLS)
EVENT_COLS = list(gnn.ALL_EVENT_TARGETS)
AUX_COLS = list(gnn.AUX_TARGET_COLS)
MACRO_EVENT_COLS: list[str] = []
MACRO_FAMILY_MODE = os.getenv("TRANSFORMER_MACRO_FAMILY", "0") == "1"
MACRO_FAMILY_COLS: list[str] = []
MACRO_DIRECTION_COLS: list[str] = []
MACRO_SURPRISE_COLS: list[str] = []


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
        self.speed_head = nn.Linear(HIDDEN, len(SPEED_TARGET_COLS))
        self.event_head = gnn.EventPrototypeHead(HIDDEN, len(EVENT_COLS))
        # These labels describe the symbol/year document, not each daily
        # token.  The document representation is taken from the final valid
        # causal token below, so the prediction cannot see future tokens.
        self.document_aux_heads = nn.ModuleDict({
            name: PrototypeHead(HIDDEN, max(1, int(size)))
            for name, size in aux_dims.items()
        })
        self.macro_head = gnn.EventPrototypeHead(HIDDEN, len(MACRO_EVENT_COLS)) if MACRO_EVENT_COLS else None
        # Each MTL head gets a learned task embedding. A shared router maps
        # task identity to trunk weights, so task grouping is learned rather
        # than assigned by hand. The router starts at equal weights.
        task_names = ["graph", "speed", "event"] + [f"aux_doc:{name}" for name in self.document_aux_heads]
        if self.macro_head is not None:
            task_names.append("macro")
        self.task_names = task_names
        routing_dim = max(8, min(32, HIDDEN // 2))
        self.task_embeddings = nn.ParameterDict({
            name.replace(":", "__"): nn.Parameter(torch.randn(routing_dim) * 0.02)
            for name in task_names
        })
        self.task_router = nn.Sequential(
            nn.Linear(routing_dim, routing_dim),
            nn.GELU(),
            nn.Linear(routing_dim, TRUNKS),
        )
        nn.init.zeros_(self.task_router[-1].weight)
        nn.init.zeros_(self.task_router[-1].bias)
        self.gradnorm_task_names = ["graph", "speed", "event", "aux"]
        if self.macro_head is not None:
            self.gradnorm_task_names.append("macro")
        self.gradnorm_log_weights = nn.Parameter(torch.zeros(len(self.gradnorm_task_names)))
        self.gradnorm_initial_losses: torch.Tensor | None = None

    def routing_weights(self) -> dict[str, list[float]]:
        hard = ROUTING_MODE == "top1" and TRUNKS > 1
        return {name: self._route(name, hard=hard)[0].detach().cpu().tolist() for name in self.task_names}

    def _route(self, task: str, hard: bool | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        key = task.replace(":", "__")
        logits = self.task_router(self.task_embeddings[key])
        soft = torch.softmax(logits / ROUTING_TEMPERATURE, dim=0)
        if hard is None:
            hard = ROUTING_MODE == "top1" and TRUNKS > 1
        if not hard:
            return soft, soft
        index = soft.argmax()
        hard_weights = torch.zeros_like(soft).scatter(0, index.view(1), 1.0)
        # Straight-through estimator: hard routing in the forward pass,
        # soft routing gradients during backpropagation.
        return hard_weights + soft - soft.detach(), soft

    def routing_regularization(self) -> torch.Tensor:
        if TRUNKS <= 1 or ROUTING_BALANCE_WEIGHT == 0:
            return next(self.parameters()).new_zeros(())
        soft_weights = torch.stack([self._route(name, hard=False)[1] for name in self.task_names])
        target = soft_weights.new_full((TRUNKS,), 1.0 / TRUNKS)
        return ROUTING_BALANCE_WEIGHT * (soft_weights.mean(dim=0) - target).pow(2).sum()

    def _task_state(self, task: str, states: list[torch.Tensor]) -> torch.Tensor:
        weights, _ = self._route(task)
        return torch.stack(states, dim=0).mul(weights[:, None, None, None]).sum(dim=0)

    def gradnorm_weights(self) -> torch.Tensor:
        return len(self.gradnorm_task_names) * torch.softmax(self.gradnorm_log_weights, dim=0)

    def gradnorm_weight_values(self) -> dict[str, float]:
        values = self.gradnorm_weights().detach().cpu().tolist()
        return {name: round(float(value), 4) for name, value in zip(self.gradnorm_task_names, values)}

    def shared_parameters(self) -> list[torch.Tensor]:
        return [
            parameter
            for module in [self.input, *self.encoders]
            for parameter in module.parameters()
            if parameter.requires_grad
        ]

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor, causal: bool = True):
        length = x.shape[1]
        if length > self.position.shape[0]:
            raise ValueError(f"document length {length} exceeds positional capacity {self.position.shape[0]}")
        h = self.input(x) + self.position[:length].unsqueeze(0)
        mask = causal_mask(length, x.device) if causal else None
        trunk_states = [
            encoder(h, mask=mask, src_key_padding_mask=padding_mask)
            for encoder in self.encoders
        ]
        graph_h = self._task_state("graph", trunk_states)
        speed_h = self._task_state("speed", trunk_states)
        event_h = self._task_state("event", trunk_states)
        macro_h = self._task_state("macro", trunk_states) if self.macro_head is not None else None
        if macro_h is not None and not causal:
            valid = (~padding_mask).unsqueeze(-1)
            macro_h = (macro_h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        document_aux_h = {
            name: self._task_state(f"aux_doc:{name}", trunk_states)
            for name in self.document_aux_heads
        }
        if causal:
            lengths = (~padding_mask).sum(dim=1).clamp_min(1) - 1
            batch_index = torch.arange(x.shape[0], device=x.device)
            document_aux_h = {
                name: state[batch_index, lengths]
                for name, state in document_aux_h.items()
            }
        else:
            valid = (~padding_mask).unsqueeze(-1)
            document_aux_h = {
                name: (state * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
                for name, state in document_aux_h.items()
            }
        macro_logits = self.macro_head(macro_h) if macro_h is not None else None
        return (
            self.graph_head(graph_h), self.speed_head(speed_h), self.event_head(event_h),
            {}, {name: head(document_aux_h[name]) for name, head in self.document_aux_heads.items()},
            macro_logits,
        )


class TwoTowerTransformer(nn.Module):
    """Causal issuer/instrument encoder for hierarchical security modeling.

    ``issuer_x`` is one daily sequence per issuer. ``instrument_x`` contains
    one or more daily sequences for that issuer, such as its equity and an
    aggregated option-surface instrument. The issuer representation conditions
    each instrument sequence only at the same or earlier token positions.
    Contract-level option rows should be aggregated to issuer/date/instrument
    before entering this model; the tower is not a contract selector.
    """

    def __init__(self, issuer_dim: int, instrument_dim: int, instrument_types: int = 2):
        super().__init__()
        self.issuer_input = nn.Sequential(nn.Linear(issuer_dim, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU())
        self.instrument_input = nn.Sequential(nn.Linear(instrument_dim, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU())
        issuer_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN, nhead=HEADS, dim_feedforward=HIDDEN * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        instrument_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN, nhead=HEADS, dim_feedforward=HIDDEN * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.issuer_encoder = nn.TransformerEncoder(issuer_layer, num_layers=LAYERS)
        self.instrument_encoder = nn.TransformerEncoder(instrument_layer, num_layers=LAYERS)
        self.instrument_type = nn.Embedding(max(1, instrument_types), HIDDEN)
        self.fusion = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN), nn.LayerNorm(HIDDEN),
        )

    def forward(
        self,
        issuer_x: torch.Tensor,
        instrument_x: torch.Tensor,
        issuer_padding: torch.Tensor,
        instrument_padding: torch.Tensor,
        instrument_type_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if issuer_x.ndim != 3 or instrument_x.ndim != 4:
            raise ValueError("issuer_x must be [batch, time, features] and instrument_x [batch, instruments, time, features]")
        batch, instruments, length, _ = instrument_x.shape
        if issuer_x.shape[:2] != (batch, length):
            raise ValueError("issuer and instrument sequences must share batch and time dimensions")
        if instrument_padding.shape != (batch, instruments, length):
            raise ValueError("instrument_padding must match [batch, instruments, time]")
        mask = causal_mask(length, issuer_x.device)
        issuer_h = self.issuer_encoder(
            self.issuer_input(issuer_x), mask=mask, src_key_padding_mask=issuer_padding
        )
        flat_instrument = instrument_x.reshape(batch * instruments, length, -1)
        flat_padding = instrument_padding.reshape(batch * instruments, length)
        instrument_h = self.instrument_encoder(
            self.instrument_input(flat_instrument),
            mask=mask,
            src_key_padding_mask=flat_padding,
        ).reshape(batch, instruments, length, HIDDEN)
        type_h = self.instrument_type(instrument_type_ids).unsqueeze(2)
        instrument_h = instrument_h + type_h
        issuer_context = issuer_h.unsqueeze(1).expand(-1, instruments, -1, -1)
        fused = self.fusion(torch.cat([instrument_h, issuer_context], dim=-1)) + instrument_h
        return issuer_h, fused


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
            "speed": frame[SPEED_TARGET_COLS].to_numpy(np.float32),
            "events": frame[EVENT_COLS].to_numpy(np.float32),
            "macro_events": frame[MACRO_EVENT_COLS].to_numpy(np.float32) if MACRO_EVENT_COLS else np.zeros((len(frame), 0), dtype=np.float32),
            "aux": frame[AUX_COLS].to_numpy(np.int64),
            "graph_mask": np.stack(frame["__graph_mask__"].to_numpy()),
            "kind": "symbol_year",
            "task_mask": np.ones(5, dtype=np.float32),
        })
    return docs


def _make_cross_sectional_docs(base: pd.DataFrame, test_year: int, train: bool) -> list[dict[str, np.ndarray]]:
    """Build one document per date containing all symbols in the universe.

    These documents use bidirectional same-date attention.  Their token-level
    graph and speed targets are reused from the per-symbol temporal labels;
    no cross-sectional graph is constructed.  Macro and year remain pooled
    document-level tasks.
    """
    year = base.date.dt.year
    selected = base.loc[year < test_year if train else year == test_year].copy()
    docs: list[dict[str, np.ndarray]] = []
    for date, frame in selected.groupby("date", sort=True):
        frame = frame.sort_values("symbol")
        cross_aux = np.full((len(frame), len(AUX_COLS)), -1, dtype=np.int64)
        if "year_target" in AUX_COLS:
            year_index = AUX_COLS.index("year_target")
            cross_aux[:, year_index] = int(frame["year_target"].iloc[0])
        docs.append({
            "symbol": frame.symbol.to_numpy(),
            "date": frame.date.to_numpy(),
            "x": np.stack(frame["__x__"].to_numpy()),
            # Reuse labels generated by the temporal per-symbol graph when
            # enabled.  No graph is ever constructed over the universe on a
            # cross-sectional date.
            "graph": frame[TARGET_COLS].to_numpy(np.float32),
            "speed": frame[SPEED_TARGET_COLS].to_numpy(np.float32),
            "events": np.zeros((len(frame), len(EVENT_COLS)), dtype=np.float32),
            "macro_events": frame[MACRO_EVENT_COLS].iloc[0].to_numpy(np.float32),
            "aux": cross_aux,
            "graph_mask": np.stack(frame["__graph_mask__"].to_numpy()) if CROSS_SECTIONAL_TOKEN_TASKS else np.zeros((len(frame), len(TARGET_COLS)), dtype=np.float32),
            "kind": "cross_sectional",
            "task_mask": np.array([float(CROSS_SECTIONAL_TOKEN_TASKS), float(CROSS_SECTIONAL_TOKEN_TASKS), 0.0, 1.0, 1.0], dtype=np.float32),
        })
    return docs


def _batch(docs: list[dict[str, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    length = max(len(item["x"]) for item in docs)
    dim = docs[0]["x"].shape[1]
    x = np.zeros((len(docs), length, dim), dtype=np.float32)
    padding = np.ones((len(docs), length), dtype=bool)
    graph = np.zeros((len(docs), length, len(TARGET_COLS)), dtype=np.float32)
    speed = np.zeros((len(docs), length, len(SPEED_TARGET_COLS)), dtype=np.float32)
    events = np.zeros((len(docs), length, len(EVENT_COLS)), dtype=np.float32)
    macro_events = np.zeros((len(docs), length, len(MACRO_EVENT_COLS)), dtype=np.float32)
    macro_document_events = np.zeros((len(docs), len(MACRO_EVENT_COLS)), dtype=np.float32)
    aux = np.full((len(docs), length, len(AUX_COLS)), -1, dtype=np.int64)
    graph_mask = np.zeros((len(docs), length, len(TARGET_COLS)), dtype=np.float32)
    task_mask = np.zeros((len(docs), 5), dtype=np.float32)
    for i, item in enumerate(docs):
        n = len(item["x"])
        x[i, :n] = item["x"]
        padding[i, :n] = False
        graph[i, :n] = item["graph"]
        speed[i, :n] = item["speed"]
        events[i, :n] = item["events"]
        if MACRO_EVENT_COLS:
            if item["kind"] == "cross_sectional":
                macro_document_events[i] = item["macro_events"]
            else:
                macro_events[i, :n] = item["macro_events"]
        aux[i, :n] = item["aux"]
        graph_mask[i, :n] = item["graph_mask"]
        task_mask[i] = item["task_mask"]
    return (
        torch.from_numpy(x), torch.from_numpy(padding),
        {"graph": torch.from_numpy(graph), "speed": torch.from_numpy(speed), "events": torch.from_numpy(events),
         "macro_events": torch.from_numpy(macro_events), "macro_document_events": torch.from_numpy(macro_document_events), "aux": torch.from_numpy(aux),
         "graph_mask": torch.from_numpy(graph_mask), "task_mask": torch.from_numpy(task_mask)},
    )


def _load_macro_event_panel(dates: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """Fetch FMP releases once and build labels through quant-warehouse."""
    cache = OUT / "cache" / f"transformer_macro_event_labels_directional_no_unchanged_v6_{gnn.DATA_END:%Y%m%d}.parquet"
    if cache.exists():
        panel = pd.read_parquet(cache)
    else:
        events = fetch_economy_calendar_range(
            start_date="2020-01-01", end_date=str(gnn.DATA_END.date())
        )
        if MACRO_COUNTRIES and not events.empty:
            events = events.loc[events.country.astype(str).str.upper().isin(MACRO_COUNTRIES)].copy()
        token_dates = pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce").dt.normalize().unique()})
        panel = build_macro_event_label_panel(token_dates, events, directional_only=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(cache, index=False)
    return panel, [column for column in panel.columns if str(column).startswith("is_")]


def _prepare_data(tier: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    global MACRO_EVENT_COLS, MACRO_FAMILY_COLS, MACRO_DIRECTION_COLS, MACRO_SURPRISE_COLS
    index = pd.read_csv(gnn.feature_dir(tier) / "index.csv")
    fused, feature_cols = _read_fused_panel(index)
    symbols = sorted(fused.symbol.unique())
    prices, labels = gnn.build_price_and_labels(symbols, tier)
    base = fused.merge(labels, on=["symbol", "date"], how="inner")
    if MACRO_ENABLED:
        macro_panel, MACRO_EVENT_COLS = _load_macro_event_panel(base["date"])
        base = base.merge(macro_panel, on="date", how="left")
        MACRO_EVENT_COLS = [column for column in macro_panel.columns if str(column).startswith("is_")]
        base[MACRO_EVENT_COLS] = base[MACRO_EVENT_COLS].fillna(0.0).astype("float32")
        MACRO_FAMILY_COLS = []
        MACRO_DIRECTION_COLS = []
        MACRO_SURPRISE_COLS = []
    else:
        MACRO_EVENT_COLS = []
        MACRO_FAMILY_COLS = []
        MACRO_DIRECTION_COLS = []
        MACRO_SURPRISE_COLS = []
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
    for column in TARGET_COLS + SPEED_TARGET_COLS + EVENT_COLS:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).astype("float32")
    graph_mask = pd.DataFrame(False, index=out.index, columns=TARGET_COLS)
    for column in ("long_pagerank", "short_pagerank"):
        ranks = out.groupby(out.date.dt.year)[column].rank(method="first", pct=True)
        graph_mask[column] = ranks.le(gnn.HITS_TAIL_QUANTILE) | ranks.ge(1.0 - gnn.HITS_TAIL_QUANTILE)
    out["__graph_mask__"] = list(graph_mask.astype("float32").to_numpy())
    return out


def _train_epoch(
    model: TransformerMTL,
    docs: list[dict[str, np.ndarray]],
    optimizer: torch.optim.Optimizer,
    gradnorm_optimizer: torch.optim.Optimizer | None = None,
) -> float:
    model.train()
    losses: list[float] = []
    for kind in ("symbol_year", "cross_sectional"):
        kind_indices = [index for index, item in enumerate(docs) if item["kind"] == kind]
        order = np.random.permutation(kind_indices)
        for offset in range(0, len(order), BATCH_SIZE):
            selected = [docs[int(i)] for i in order[offset:offset + BATCH_SIZE]]
            x, padding, target = _batch(selected)
            x = x.to(DEVICE); padding = padding.to(DEVICE)
            target = {name: value.to(DEVICE) for name, value in target.items()}
            optimizer.zero_grad()
            graph_hat, speed_hat, event_logits, aux_logits, document_aux_logits, macro_logits = model(
                x, padding, causal=kind == "symbol_year"
            )
            valid = ~padding
            task_valid = target["task_mask"]
            graph_mask = target["graph_mask"] * valid.unsqueeze(-1) * task_valid[:, 0, None, None]
            graph_error = nn.functional.smooth_l1_loss(graph_hat, target["graph"], reduction="none")
            graph_loss = (graph_error * graph_mask).sum() / graph_mask.sum().clamp_min(1.0)
            speed_valid = valid * task_valid[:, 1, None].bool()
            speed_error = nn.functional.smooth_l1_loss(speed_hat, target["speed"], reduction="none")
            speed_loss = (speed_error * speed_valid.unsqueeze(-1)).sum() / speed_valid.sum().clamp_min(1.0)
            event_valid = valid * task_valid[:, 2, None].bool()
            event_loss = gnn.event_loss_from_logits(event_logits[event_valid], target["events"][event_valid]) if event_valid.any() else graph_hat.new_zeros(())
            macro_valid = valid * task_valid[:, 4, None].bool()
            macro_loss = torch.tensor(0.0, device=DEVICE)
            if macro_logits is not None and MACRO_EVENT_COLS and macro_valid.any():
                if kind == "cross_sectional":
                    macro_loss = gnn.event_loss_from_logits(
                        macro_logits, target["macro_document_events"]
                    )
                else:
                    macro_loss = gnn.event_loss_from_logits(macro_logits[macro_valid], target["macro_events"][macro_valid])
            aux_loss = torch.tensor(0.0, device=DEVICE)
            for index, name in enumerate(AUX_COLS):
                # Annual causal documents use the final valid token as the
                # document state.  The target is constant across the daily
                # tokens, so reading the first valid target avoids duplicating
                # the document-level supervision across the sequence.
                if kind in ("symbol_year", "cross_sectional"):
                    aux_target = target["aux"][:, 0, index]
                    mask = task_valid[:, 3].bool() & aux_target.ge(0)
                    if mask.any():
                        aux_loss = aux_loss + nn.functional.cross_entropy(
                            document_aux_logits[name][mask], aux_target[mask]
                        )
            task_losses = {"graph": graph_loss, "speed": speed_loss, "event": event_loss, "aux": aux_loss}
        # Keep the optimizer step active when macro prediction is disabled;
        # the macro branch is optional, while graph/auxiliary training is not.
        if model.macro_head is None or MACRO_EVENT_COLS:
            task_losses["macro"] = macro_loss
            if GRADNORM_ENABLED and gradnorm_optimizer is not None:
                if model.gradnorm_initial_losses is None:
                    model.gradnorm_initial_losses = torch.stack([
                        task_losses[name].detach().clamp_min(1e-8)
                        for name in model.gradnorm_task_names
                    ])
                weights = model.gradnorm_weights()
                weighted_losses = [weights[index] * task_losses[name] for index, name in enumerate(model.gradnorm_task_names)]
                grad_norms: list[torch.Tensor] = []
                shared_parameters = model.shared_parameters()
                for weighted_task_loss in weighted_losses:
                    gradients = torch.autograd.grad(
                        weighted_task_loss, shared_parameters, retain_graph=True,
                        create_graph=True, allow_unused=True,
                    )
                    grad_norms.append(torch.sqrt(sum(
                        gradient.pow(2).sum() for gradient in gradients if gradient is not None
                    ).clamp_min(1e-12)))
                gradient_stack = torch.stack(grad_norms)
                with torch.no_grad():
                    relative_loss = torch.stack([
                        task_losses[name].detach().clamp_min(1e-8)
                        for name in model.gradnorm_task_names
                    ]) / model.gradnorm_initial_losses
                    inverse_rate = relative_loss / relative_loss.mean().clamp_min(1e-8)
                    target_norm = gradient_stack.detach().mean() * inverse_rate.pow(GRADNORM_ALPHA)
                gradnorm_loss = torch.abs(gradient_stack - target_norm).sum()
                gradnorm_optimizer.zero_grad()
                gradnorm_loss.backward(retain_graph=True)
                gradnorm_optimizer.step()
                weights = model.gradnorm_weights().detach()
            else:
                default_weights = {"graph": 1.0, "speed": float(os.getenv("TRANSFORMER_SPEED_LOSS_WEIGHT", "0.10")), "event": 1.0, "aux": float(os.getenv("TRANSFORMER_AUX_LOSS_WEIGHT", "0.10")), "macro": float(os.getenv("TRANSFORMER_MACRO_LOSS_WEIGHT", "1.0"))}
                weights = torch.tensor(
                    [default_weights[name] for name in model.gradnorm_task_names], device=DEVICE,
                )
            loss = sum(
                weights[index] * task_losses[name]
                for index, name in enumerate(model.gradnorm_task_names)
            ) + model.routing_regularization()
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
        cross_enabled = CROSS_SECTIONAL_ENABLED and (bool(MACRO_EVENT_COLS) or "year_target" in AUX_COLS)
        cross_train_docs = _make_cross_sectional_docs(normalized, test_year, True) if cross_enabled else []
        cross_test_docs = _make_cross_sectional_docs(normalized, test_year, False) if cross_enabled else []
        if not train_docs or not test_docs:
            continue
        training_docs = train_docs + cross_train_docs
        aux_dims = {name: int(normalized[name].max()) + 1 for name in AUX_COLS}
        model = TransformerMTL(len(feature_cols), aux_dims).to(DEVICE)
        main_parameters = [
            parameter for name, parameter in model.named_parameters()
            if name != "gradnorm_log_weights"
        ]
        optimizer = torch.optim.AdamW(main_parameters, lr=LR, weight_decay=1e-4)
        gradnorm_optimizer = torch.optim.Adam(
            [model.gradnorm_log_weights], lr=GRADNORM_LR
        ) if GRADNORM_ENABLED else None
        print({"tier": tier, "year": test_year, "documents": len(train_docs), "cross_sectional_train_documents": len(cross_train_docs), "test_documents": len(test_docs), "cross_sectional_test_documents": len(cross_test_docs), "tokens": sum(len(d["x"]) for d in train_docs), "macro_tasks": len(MACRO_EVENT_COLS), "trunks": TRUNKS, "routing": ROUTING_MODE, "gradnorm": GRADNORM_ENABLED, "architecture": "mixed_causal_symbol_year_and_bidirectional_cross_sectional_transformer", "device": str(DEVICE)}, flush=True)
        for epoch in range(EPOCHS):
            loss = _train_epoch(model, training_docs, optimizer, gradnorm_optimizer)
            if epoch == 0 or epoch == EPOCHS - 1:
                print({"tier": tier, "year": test_year, "epoch": epoch + 1, "transformer_loss": round(loss, 5)}, flush=True)
        print({"tier": tier, "year": test_year, "task_trunk_weights": model.routing_weights(), "gradnorm_weights": model.gradnorm_weight_values()}, flush=True)
        model.eval()
        predictions: list[pd.DataFrame] = []
        with torch.no_grad():
            for start in range(0, len(test_docs), BATCH_SIZE):
                batch_docs = test_docs[start:start + BATCH_SIZE]
                x, padding, _ = _batch(batch_docs)
                x = x.to(DEVICE); padding = padding.to(DEVICE)
                graph_hat, _, _, _, _, _ = model(x, padding)
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
            summary["family"] = f"causal_symbol_year_transformer_{TRUNKS}trunk_{ROUTING_MODE}routing_speedhits"
            summary["label_source"] = ("transformer_macro_mtl" if MACRO_ENABLED else "transformer_baseline_mtl") + f"_{TRUNKS}trunk_{ROUTING_MODE}routing_speedhits" + ("_gradnorm" if GRADNORM_ENABLED else "")
            summaries.append(summary)
        run_label = ("transformer_macro" if MACRO_ENABLED else "transformer") + f"_{TRUNKS}trunk_{ROUTING_MODE}routing_speedhits" + ("_gradnorm" if GRADNORM_ENABLED else "")
        pd.concat(summaries, ignore_index=True).to_parquet(OUT / f"{tier.lower()}_{run_label}_long_only_through_{test_year}.parquet", index=False)
    result = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    run_label = ("transformer_macro" if MACRO_ENABLED else "transformer") + f"_{TRUNKS}trunk_{ROUTING_MODE}routing_speedhits" + ("_gradnorm" if GRADNORM_ENABLED else "")
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
