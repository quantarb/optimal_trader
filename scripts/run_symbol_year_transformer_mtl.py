"""Causal transformer MTL baseline with one document per symbol-year."""
from __future__ import annotations

import importlib.util
import gc
import os
from contextlib import nullcontext
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
    deduplicate_binary_label_columns,
)

FIRST_TEST_YEAR = int(os.getenv("TRANSFORMER_FIRST_TEST_YEAR", "2021"))
LAST_TEST_YEAR = int(os.getenv("TRANSFORMER_LAST_TEST_YEAR", "2025"))
EPOCHS = int(os.getenv("TRANSFORMER_EPOCHS", "12"))
DUAL_EPOCHS = int(os.getenv("TRANSFORMER_DUAL_EPOCHS", "24"))
MIX_MATCH_START_EPOCH = int(os.getenv("TRANSFORMER_MIX_MATCH_START_EPOCH", "18"))
HIDDEN = int(os.getenv("TRANSFORMER_HIDDEN", "48"))
LAYERS = int(os.getenv("TRANSFORMER_LAYERS", "2"))
HEADS = int(os.getenv("TRANSFORMER_HEADS", "4"))
# Official baseline: two shared trunks with learned task and feature-family routing.
TRUNKS = max(1, int(os.getenv("TRANSFORMER_TRUNKS", "2")))
ISSUER_TRUNKS = max(1, int(os.getenv("TRANSFORMER_ISSUER_TRUNKS", "1")))
INSTRUMENT_TRUNKS = max(1, int(os.getenv("TRANSFORMER_INSTRUMENT_TRUNKS", "1")))
ROUTING_MODE = os.getenv("TRANSFORMER_ROUTING", "soft").strip().lower()
ROUTING_TEMPERATURE = max(0.1, float(os.getenv("TRANSFORMER_ROUTING_TEMPERATURE", "1.0")))
ROUTING_BALANCE_WEIGHT = float(os.getenv("TRANSFORMER_ROUTING_BALANCE_WEIGHT", "0.01"))
GRADNORM_ENABLED = os.getenv("TRANSFORMER_GRADNORM", "0") == "1"
GRADNORM_ALPHA = float(os.getenv("TRANSFORMER_GRADNORM_ALPHA", "0.5"))
GRADNORM_LR = float(os.getenv("TRANSFORMER_GRADNORM_LR", "0.025"))
ASSET_CLASS_LOSS_WEIGHT = float(os.getenv("TRANSFORMER_ASSET_CLASS_LOSS_WEIGHT", "0.10"))
ASSET_CLASS_TASK_ENABLED = os.getenv("TRANSFORMER_ASSET_CLASS_TASK", "0") == "1"
BATCH_SIZE = int(os.getenv("TRANSFORMER_BATCH_SIZE", "32"))
LR = float(os.getenv("TRANSFORMER_LR", "0.002"))
MACRO_ENABLED = os.getenv("TRANSFORMER_MACRO", "0") == "1"
MACRO_COMPACT = os.getenv("TRANSFORMER_MACRO_COMPACT", "0") == "1"
MACRO_DEDUP_IDENTICAL = os.getenv("TRANSFORMER_MACRO_DEDUP_IDENTICAL", "0") == "1"
MACRO_COMPACT_GROUPS = tuple(
    value.strip().lower()
    for value in os.getenv("TRANSFORMER_MACRO_COMPACT_GROUPS", "treasury,mortgage,consumer").split(",")
    if value.strip()
)
CROSS_SECTIONAL_ENABLED = os.getenv("TRANSFORMER_CROSS_SECTIONAL", "1") == "1"
CROSS_SECTIONAL_TOKEN_TASKS = os.getenv("TRANSFORMER_CROSS_TOKEN_TASKS", "1") == "1"
CROSS_SECTIONAL_TRADING = os.getenv("TRANSFORMER_CROSS_TRADING", "0") == "1"
CROSS_SECTIONAL_COMPARE_HEADS = os.getenv("TRANSFORMER_COMPARE_HEADS", "1") == "1"
SPEED_STRATEGY_ENABLED = os.getenv("TRANSFORMER_SPEED_STRATEGY", "1") == "1"
BACKTEST_VARIANTS = tuple(
    value.strip().lower()
    for value in os.getenv("TRANSFORMER_VARIANTS", os.getenv("TRANSFORMER_VARIANT", "long_only,short_only")).split(",")
    if value.strip()
)
if not BACKTEST_VARIANTS or any(value not in {"long_only", "short_only"} for value in BACKTEST_VARIANTS):
    raise ValueError("TRANSFORMER_VARIANTS must contain only 'long_only' and/or 'short_only'")
BACKTEST_VARIANT_TAG = "-".join(BACKTEST_VARIANTS)
DUAL_TOWER_ENABLED = os.getenv("TRANSFORMER_DUAL_TOWER", "0") == "1"
PREFERRED_ENABLED = os.getenv("TRANSFORMER_PREFERRED", "0") == "1"
PREFERRED_PANEL_PATH = os.getenv("TRANSFORMER_PREFERRED_PANEL", "").strip()
RELATED_ASSETS_ENABLED = os.getenv("TRANSFORMER_RELATED_ASSETS", "0") == "1"
RELATED_ASSETS_PANEL_PATH = os.getenv("TRANSFORMER_RELATED_ASSETS_PANEL", "").strip()
RELATED_ASSETS_AS_ROWS = os.getenv("TRANSFORMER_RELATED_AS_ROWS", "0") == "1"
ISSUER_EQUITY_ENABLED = os.getenv("TRANSFORMER_ISSUER_EQUITY", "1") == "1"
HYBRID_INSTRUMENT_ADAPTER = os.getenv("TRANSFORMER_HYBRID_ADAPTER", "1") == "1"
FAMILY_INTERACTIONS = os.getenv("TRANSFORMER_FAMILY_INTERACTIONS", "0") == "1"
SHARED_FEATURE_MIXER = os.getenv("TRANSFORMER_SHARED_FEATURE_MIXER", "1") == "1"
SHARED_MIXER_ENHANCEMENTS = os.getenv("TRANSFORMER_SHARED_MIXER_ENHANCEMENTS", "0") == "1"
COVERAGE_AWARE_FAMILY_ADAPTERS = os.getenv("TRANSFORMER_COVERAGE_AWARE_FAMILIES", "1") == "1"
SELF_SUPERVISED_ENABLED = os.getenv("TRANSFORMER_SELF_SUPERVISED", "0") == "1"
SELF_SUPERVISED_WEIGHT = float(os.getenv("TRANSFORMER_SELF_SUPERVISED_WEIGHT", "0.05"))
SELF_SUPERVISED_PRETRAIN_EPOCHS = max(0, int(os.getenv("TRANSFORMER_SELF_SUPERVISED_PRETRAIN_EPOCHS", "0")))
MASKED_TOKEN_ENABLED = os.getenv("TRANSFORMER_MASKED_TOKEN", "1") == "1"
NEXT_TOKEN_ENABLED = os.getenv("TRANSFORMER_NEXT_TOKEN", "1") == "1"
MASKED_TOKEN_WEIGHT = float(os.getenv("TRANSFORMER_MASKED_TOKEN_WEIGHT", "0.05"))
NEXT_TOKEN_WEIGHT = float(os.getenv("TRANSFORMER_NEXT_TOKEN_WEIGHT", "0.05"))
MASKED_TOKEN_RATE = min(0.5, max(0.01, float(os.getenv("TRANSFORMER_MASKED_TOKEN_RATE", "0.15"))))
CROSS_SECTIONAL_SET_CONTEXT = os.getenv("TRANSFORMER_CROSS_SET_CONTEXT", "1") == "1"
ETF_CORPUS_ENABLED = os.getenv("TRANSFORMER_ETF_CORPUS", "0") == "1"
ETF_CORPUS_PANEL_PATH = os.getenv("TRANSFORMER_ETF_CORPUS_PANEL", "").strip()
ASSET_CLASSES = ("preferred", "warrant", "unit", "note_bond", "adr", "ordinary", "etf")
ASSET_MODALITY_IDS = {name: index + 1 for index, name in enumerate(ASSET_CLASSES)}
# The transformer is deliberately given raw observations in this experiment.
# It should learn rolling price behavior through causal attention instead of
# receiving precomputed indicators such as returns, SMAs, RSI, ATR, or TA
# candle/cycle features.
DISABLED_TECHNICAL_FAMILIES = {
    "price_technicals",
    "technical_candles",
    "technical_cycles",
    "technical_math",
    "technical_momentum",
    "technical_overlap",
    "technical_performance",
}
EXTRA_DISABLED_FAMILIES = {
    value.strip()
    for value in os.getenv("TRANSFORMER_EXCLUDE_FAMILIES", "").split(",")
    if value.strip()
}
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


def _dual_autocast():
    """Use CUDA bfloat16 for dual-tower compute when a CUDA device is active."""
    if DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()
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
CROSS_RANK_TARGET_COLS = [f"cross_{column}" for column in TARGET_COLS]
CROSS_SPEED_RANK_TARGET_COLS = [f"cross_{column}" for column in SPEED_TARGET_COLS]
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


class LearnableFeatureNorm(nn.Module):
    """Fold-fitted input scaling whose affine correction is learned by the model.

    The fold statistics only establish a numerically stable starting point.  The
    scale and offset are parameters, so the network can keep, undo, or refine
    the normalization during training instead of receiving pre-normalized
    features from the data pipeline.
    """

    def __init__(self, mean: np.ndarray | torch.Tensor, std: np.ndarray | torch.Tensor,
                 robust_mask: np.ndarray | torch.Tensor | None = None):
        super().__init__()
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
        std_tensor = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6)
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("std", std_tensor)
        if robust_mask is None:
            robust_mask = torch.zeros_like(mean_tensor, dtype=torch.bool)
        self.register_buffer("robust_mask", torch.as_tensor(robust_mask, dtype=torch.bool))
        self.log_scale = nn.Parameter(torch.zeros_like(mean_tensor))
        self.offset = nn.Parameter(torch.zeros_like(mean_tensor))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For raw price/volume channels use a shape-agnostic, monotonic
        # compression.  It is linear near zero and logarithmic for large
        # magnitudes, which avoids assuming a feature is specifically a price,
        # volume, fundamental, or ratio.
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        robust = torch.asinh(x / self.std)
        zscore = (x - self.mean) / self.std
        scaled = torch.where(self.robust_mask, robust, zscore).clamp(-100.0, 100.0)
        return scaled * torch.exp(self.log_scale).clamp(0.01, 100.0) + self.offset


class CausalAssetAdapter(nn.Module):
    """Learn local auto-TA filters from an asset's raw token sequence."""

    def __init__(self, input_dim: int, hidden: int, mean: np.ndarray | torch.Tensor | None = None,
                 std: np.ndarray | torch.Tensor | None = None,
                 robust_mask: np.ndarray | torch.Tensor | None = None, kernel_size: int = 9):
        super().__init__()
        if mean is None:
            mean = np.zeros(input_dim, dtype=np.float32)
        if std is None:
            std = np.ones(input_dim, dtype=np.float32)
        self.feature_norm = LearnableFeatureNorm(mean, std, robust_mask)
        self.input = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.depthwise = nn.Conv1d(
            hidden, hidden, kernel_size=kernel_size, groups=hidden,
            padding=kernel_size - 1, bias=False,
        )
        self.pointwise = nn.Linear(hidden, hidden * 2)
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.GELU())
        self.gate = nn.Linear(hidden, hidden)
        self.residual = nn.Linear(hidden, hidden)
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.input(self.feature_norm(x))
        filtered = self.depthwise(base.transpose(1, 2)).transpose(1, 2)
        filtered = filtered[:, : x.shape[1], :]
        mixed = self.pointwise(filtered)
        value, gate = mixed.chunk(2, dim=-1)
        value = torch.nn.functional.gelu(value) * torch.sigmoid(gate)
        return self.output(self.residual(base) + value * torch.sigmoid(self.gate(base)))


class CrossSectionalSetBlock(nn.Module):
    """Linear-cost bidirectional same-date context block."""

    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.token = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden)
        )
        self.context = nn.Linear(hidden, hidden)
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid = (~padding_mask).unsqueeze(-1).to(x.dtype)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        context = self.context(pooled).unsqueeze(1)
        return x + torch.sigmoid(self.gate) * self.token(self.norm(x) + context)


class TransformerMTL(nn.Module):
    def __init__(self, feature_dim: int, aux_dims: dict[str, int], extra_task_names: tuple[str, ...] = (), asset_feature_indices: dict[str, tuple[int, ...]] | None = None,
                 family_feature_indices: dict[str, tuple[int, ...]] | None = None,
                 feature_mean: np.ndarray | torch.Tensor | None = None, feature_std: np.ndarray | torch.Tensor | None = None,
                 robust_feature_mask: np.ndarray | torch.Tensor | None = None,
                 max_position: int = 512):
        super().__init__()
        if feature_mean is None:
            feature_mean = np.zeros(feature_dim, dtype=np.float32)
        if feature_std is None:
            feature_std = np.ones(feature_dim, dtype=np.float32)
        if robust_feature_mask is None:
            robust_feature_mask = np.zeros(feature_dim, dtype=bool)
        self.feature_norm = LearnableFeatureNorm(feature_mean, feature_std, robust_feature_mask)
        self.input = nn.Sequential(nn.Linear(feature_dim, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU())
        self.asset_feature_indices = asset_feature_indices or {}
        self.family_feature_indices = family_feature_indices or {}
        self.shared_family_names = tuple(self.family_feature_indices)
        self._coverage_family_states = None
        self._coverage_family_presence = None
        self.coverage_family_names = tuple(
            family for family, indices in self.family_feature_indices.items() if indices
        )
        self.coverage_family_adapters = nn.ModuleDict()
        self.coverage_family_gate = None
        self.coverage_family_bias = None
        if COVERAGE_AWARE_FAMILY_ADAPTERS and self.coverage_family_names:
            # Each family gets a small bottleneck adapter.  The adapter cost is
            # proportional to that family's columns, not to document length
            # squared.  A single shared gate produces one state-dependent
            # weight per family and also receives an explicit coverage mask.
            adapter_rank = max(8, min(16, HIDDEN // 3))
            for family in self.coverage_family_names:
                key = family.replace("-", "__")
                self.coverage_family_adapters[key] = nn.Sequential(
                    nn.Linear(len(self.family_feature_indices[family]), adapter_rank),
                    nn.LayerNorm(adapter_rank),
                    nn.GELU(),
                    nn.Linear(adapter_rank, HIDDEN),
                )
            self.coverage_family_gate = nn.Linear(HIDDEN, len(self.coverage_family_names))
            self.coverage_family_bias = nn.Parameter(torch.full((len(self.coverage_family_names),), -2.0))
        self.asset_inputs = nn.ModuleDict({
            name: CausalAssetAdapter(
                len(indices), HIDDEN,
                mean=np.asarray(feature_mean)[list(indices)], std=np.asarray(feature_std)[list(indices)],
                robust_mask=np.asarray(robust_feature_mask)[list(indices)],
            )
            for name, indices in self.asset_feature_indices.items() if indices
        })
        self.asset_auto_ta_gates = nn.ParameterDict({
            name.replace("-", "__"): nn.Parameter(torch.tensor(-1.0))
            for name in self.asset_inputs if name == "equity"
        })
        # Family adapters preserve column/family identity before the shared
        # trunk.  The interaction path is intentionally low-rank: it lets
        # the model learn relationships such as revenue / employees without
        # materializing pairwise features or paying for attention over every
        # raw column.
        self.family_inputs = nn.ModuleDict({})
        self.family_embeddings = nn.ParameterDict({})
        if FAMILY_INTERACTIONS:
            for family, indices in self.family_feature_indices.items():
                if not indices:
                    continue
                key = family.replace("-", "__")
                self.family_inputs[key] = nn.Sequential(
                    LearnableFeatureNorm(
                        np.asarray(feature_mean)[list(indices)],
                        np.asarray(feature_std)[list(indices)],
                        np.asarray(robust_feature_mask)[list(indices)],
                    ),
                    nn.Linear(len(indices), HIDDEN),
                    nn.LayerNorm(HIDDEN),
                    nn.GELU(),
                )
                self.family_embeddings[key] = nn.Parameter(torch.randn(HIDDEN) * 0.01)
            self.family_interaction = nn.Sequential(
                nn.Linear(HIDDEN * 2, HIDDEN),
                nn.GELU(),
                nn.Linear(HIDDEN, HIDDEN),
            )
            self.family_gates = nn.ParameterDict({
                family.replace("-", "__"): nn.Parameter(torch.tensor(-4.0))
                for family in self.family_feature_indices
                if self.family_feature_indices[family]
            })
        self.shared_feature_mixer = None
        self.shared_mixer_gate = None
        self.shared_cross_down = None
        self.shared_cross_up = None
        self.shared_cross_gate = None
        self.family_presence_embeddings = None
        self.auto_ta_conv = None
        self.auto_ta_gate = None
        if SHARED_FEATURE_MIXER:
            mixer_rank = max(8, min(32, HIDDEN // 2))
            mixer_input_dim = feature_dim + len(self.shared_family_names)
            self.shared_feature_mixer = nn.Sequential(
                nn.Linear(mixer_input_dim, mixer_rank),
                nn.GELU(),
                nn.Linear(mixer_rank, HIDDEN),
            )
            self.shared_mixer_gate = nn.Parameter(torch.tensor(-4.0))
            if SHARED_MIXER_ENHANCEMENTS:
                # Multiplicative low-rank cross path: unlike a plain MLP,
                # this explicitly gives the model a cheap mechanism for
                # learning relationships such as revenue per employee.
                self.shared_cross_down = nn.Linear(mixer_input_dim, mixer_rank, bias=False)
                self.shared_cross_up = nn.Linear(mixer_rank, HIDDEN, bias=False)
                self.shared_cross_gate = nn.Linear(mixer_input_dim, mixer_rank)
                nn.init.constant_(self.shared_cross_gate.bias, -2.0)
                self.family_presence_embeddings = nn.Parameter(
                    torch.randn(len(self.shared_family_names), HIDDEN) * 0.01
                )
                self.auto_ta_conv = nn.Conv1d(
                    HIDDEN, HIDDEN, kernel_size=5, groups=HIDDEN,
                    padding=4, bias=False,
                )
                self.auto_ta_gate = nn.Parameter(torch.tensor(-4.0))
        # Related assets augment the shared equity state through a dynamic,
        # per-channel residual gate.  Zero-initialized weights and a strongly
        # negative bias make the adapter nearly silent at initialization while
        # allowing training to learn when an instrument family is useful.
        self.instrument_fusion_gates = nn.ModuleDict({})
        if HYBRID_INSTRUMENT_ADAPTER:
            for name in self.asset_inputs:
                if name == "equity":
                    continue
                gate = nn.Linear(HIDDEN * 2, HIDDEN)
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, -6.0)
                self.instrument_fusion_gates[name.replace("-", "__")] = gate
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
        self.cross_set_encoders = nn.ModuleList(
            [nn.ModuleList([
                CrossSectionalSetBlock(HIDDEN)
                for _ in range(LAYERS)
            ]) for _ in range(TRUNKS)]
        ) if CROSS_SECTIONAL_SET_CONTEXT else None
        self.position = nn.Parameter(torch.randn(max(512, int(max_position)), HIDDEN) * 0.01)
        self.graph_head = nn.Linear(HIDDEN, len(TARGET_COLS))
        self.masked_feature_head = nn.Linear(HIDDEN, feature_dim) if SELF_SUPERVISED_ENABLED else None
        self.masked_token_head = nn.Linear(HIDDEN, feature_dim) if MASKED_TOKEN_ENABLED else None
        self.next_token_head = nn.Linear(HIDDEN, feature_dim) if NEXT_TOKEN_ENABLED else None
        self.last_token_state = None
        self.speed_head = nn.Linear(HIDDEN, len(SPEED_TARGET_COLS))
        self.cross_graph_head = nn.Linear(HIDDEN, len(TARGET_COLS))
        self.cross_speed_head = nn.Linear(HIDDEN, len(SPEED_TARGET_COLS))
        self.event_head = gnn.EventPrototypeHead(HIDDEN, len(EVENT_COLS))
        # These labels describe the symbol/year document, not each daily
        # token.  The document representation is taken from the final valid
        # causal token below, so the prediction cannot see future tokens.
        self.document_aux_heads = nn.ModuleDict({
            name: PrototypeHead(HIDDEN, max(1, int(size)))
            for name, size in aux_dims.items()
        })
        self.cross_year_head = (
            PrototypeHead(HIDDEN, max(1, int(aux_dims["year_target"])))
            if "year_target" in aux_dims else None
        )
        # Asset class is an auxiliary target for related-asset documents.
        # The class ID is not embedded into the token representation; modality
        # is used only to route a row through its asset-specific adapter and to
        # provide the supervised target.
        self.asset_class_head = nn.Linear(HIDDEN, len(ASSET_CLASSES) + 1) if ASSET_CLASS_TASK_ENABLED else None
        self.macro_head = gnn.EventPrototypeHead(HIDDEN, len(MACRO_EVENT_COLS)) if MACRO_EVENT_COLS else None
        # Each MTL head gets a learned task embedding. A shared router maps
        # task identity to trunk weights, so task grouping is learned rather
        # than assigned by hand. The router starts at equal weights.
        task_names = ["graph", "speed", "event"] + [f"aux_doc:{name}" for name in self.document_aux_heads]
        task_names.extend(("cross_graph", "cross_speed"))
        if self.cross_year_head is not None:
            task_names.append("cross_year")
        if self.asset_class_head is not None:
            task_names.append("asset_class")
        if self.macro_head is not None:
            task_names.append("macro")
        task_names.extend(extra_task_names)
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
        self.trunk_family_router = None
        self.trunk_family_gates = None
        if COVERAGE_AWARE_FAMILY_ADAPTERS and self.coverage_family_names:
            # Each shared trunk learns which feature families it should see
            # before attention. The existing task router then learns which
            # trunk each task should use, creating an effective task-family
            # affinity without running one transformer per task.
            self.trunk_family_router = nn.Parameter(
                torch.zeros(TRUNKS, len(self.coverage_family_names))
            )
            self.trunk_family_gates = nn.Parameter(torch.full((TRUNKS,), -2.0))
        self.gradnorm_task_names = ["graph", "speed", "event", "aux", "cross_graph", "cross_speed"]
        if self.cross_year_head is not None:
            self.gradnorm_task_names.append("cross_year")
        if self.asset_class_head is not None:
            self.gradnorm_task_names.append("asset_class")
        if self.macro_head is not None:
            self.gradnorm_task_names.append("macro")
        self.gradnorm_log_weights = nn.Parameter(torch.zeros(len(self.gradnorm_task_names)))
        self.gradnorm_initial_losses: torch.Tensor | None = None

    def routing_weights(self) -> dict[str, list[float]]:
        hard = ROUTING_MODE == "top1" and TRUNKS > 1
        return {name: self._route(name, hard=hard)[0].detach().cpu().tolist() for name in self.task_names}

    def task_family_weights(self) -> dict[str, list[float]]:
        if self.trunk_family_router is None:
            return {}
        trunk_family = torch.softmax(self.trunk_family_router, dim=-1)
        values: dict[str, list[float]] = {}
        for name in self.task_names:
            task_trunks, _ = self._route(name, hard=False)
            values[name] = task_trunks @ trunk_family
            values[name] = values[name].detach().cpu().tolist()
        return values

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
            for module in [self.input, *self.asset_inputs.values(), *self.encoders]
            for parameter in module.parameters()
            if parameter.requires_grad
        ]

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor, causal: bool = True, modality: torch.Tensor | None = None,
                family_presence: torch.Tensor | None = None):
        length = x.shape[1]
        if length > self.position.shape[0]:
            raise ValueError(f"document length {length} exceeds positional capacity {self.position.shape[0]}")
        normalized_x = self.feature_norm(x)
        h = self.input(normalized_x)
        if COVERAGE_AWARE_FAMILY_ADAPTERS and self.coverage_family_adapters:
            if family_presence is None:
                # Compatibility fallback for auxiliary corpora that predate
                # explicit masks.  Primary fused panels provide the real mask.
                family_presence = torch.stack([
                    x[..., list(self.family_feature_indices[family])].abs().sum(dim=-1).gt(1e-8)
                    for family in self.coverage_family_names
                ], dim=-1).to(normalized_x.dtype)
            else:
                family_presence = family_presence.to(normalized_x.dtype)
            gate_logits = self.coverage_family_gate(h) + self.coverage_family_bias
            family_residuals = []
            family_states = []
            for position, family in enumerate(self.coverage_family_names):
                key = family.replace("-", "__")
                family_state = self.coverage_family_adapters[key](normalized_x[..., list(self.family_feature_indices[family])])
                gate = torch.sigmoid(gate_logits[..., position:position + 1])
                family_state = family_state * gate
                family_states.append(family_state)
                family_residuals.append(family_state * family_presence[..., position:position + 1])
            self._coverage_family_states = torch.stack(family_states, dim=2)
            self._coverage_family_presence = family_presence
            h = h + torch.stack(family_residuals, dim=2).sum(dim=2)
        if SHARED_FEATURE_MIXER and self.shared_feature_mixer is not None:
            # Add cheap family-presence tokens so the shared mixer can
            # distinguish an observed zero from a family with no coverage.
            presence = []
            for indices in self.family_feature_indices.values():
                presence.append(x[..., list(indices)].abs().sum(dim=-1).gt(1e-8).to(normalized_x.dtype))
            presence = torch.stack(presence, dim=-1) if presence else normalized_x.new_zeros((*normalized_x.shape[:-1], 0))
            mixer_input = torch.cat([normalized_x, presence], dim=-1)
            mixed = self.shared_feature_mixer(mixer_input)
            h = h + torch.sigmoid(self.shared_mixer_gate) * mixed
            if SHARED_MIXER_ENHANCEMENTS and self.shared_cross_down is not None:
                cross_basis = self.shared_cross_down(mixer_input)
                cross_gate = torch.sigmoid(self.shared_cross_gate(mixer_input))
                h = h + torch.sigmoid(self.shared_mixer_gate) * self.shared_cross_up(cross_basis * cross_gate)
                if self.family_presence_embeddings is not None and presence.shape[-1]:
                    h = h + presence @ self.family_presence_embeddings
        if FAMILY_INTERACTIONS and self.family_inputs:
            family_states = []
            family_masks = []
            for family, indices in self.family_feature_indices.items():
                key = family.replace("-", "__")
                if key not in self.family_inputs:
                    continue
                family_x = x[..., list(indices)]
                family_states.append(self.family_inputs[key](family_x) + self.family_embeddings[key])
                family_masks.append(family_x.abs().sum(dim=-1).gt(1e-8))
            if family_states:
                family_h = torch.stack(family_states, dim=2)
                family_mask = torch.stack(family_masks, dim=2).unsqueeze(-1).to(family_h.dtype)
                family_context = (family_h * family_mask).sum(dim=2) / family_mask.sum(dim=2).clamp_min(1.0)
                interactions = []
                for position, family in enumerate(self.family_feature_indices):
                    key = family.replace("-", "__")
                    if key not in self.family_inputs:
                        continue
                    pair_input = torch.cat([family_h[:, :, position], family_context], dim=-1)
                    pair_state = self.family_interaction(pair_input)
                    pair_state = pair_state * torch.sigmoid(self.family_gates[key])
                    interactions.append(pair_state * family_mask[:, :, position])
                if interactions:
                    h = h + torch.stack(interactions, dim=2).sum(dim=2)
        if modality is not None:
            for name, adapter in self.asset_inputs.items():
                indices = self.asset_feature_indices[name]
                asset_h = adapter(x[..., list(indices)])
                asset_mask = modality[:, None, None].eq(ASSET_MODALITY_IDS[name])
                if name == "equity":
                    gate = torch.sigmoid(self.asset_auto_ta_gates[name])
                    h = h + torch.where(asset_mask, asset_h * gate, torch.zeros_like(asset_h))
                elif HYBRID_INSTRUMENT_ADAPTER:
                    gate_input = torch.cat([h, asset_h], dim=-1)
                    gate = torch.sigmoid(self.instrument_fusion_gates[name.replace("-", "__")](gate_input))
                    h = h + torch.where(asset_mask, asset_h * gate, torch.zeros_like(asset_h))
                else:
                    h = torch.where(asset_mask, asset_h, h)
        if causal and SHARED_MIXER_ENHANCEMENTS and self.auto_ta_conv is not None:
            # Causal depthwise temporal mixing is applied only to temporal
            # documents; cross-sectional documents retain same-date symmetry.
            conv = self.auto_ta_conv(h.transpose(1, 2)).transpose(1, 2)
            conv = conv[:, :length]
            h = h + torch.sigmoid(self.auto_ta_gate) * conv
        h = h + self.position[:length].unsqueeze(0)
        mask = causal_mask(length, x.device) if causal else None
        trunk_inputs = [h for _ in range(TRUNKS)]
        if self.trunk_family_router is not None and self._coverage_family_states is not None:
            family_weights = torch.softmax(self.trunk_family_router, dim=-1).to(h.dtype)
            presence = self._coverage_family_presence.to(h.dtype)
            for trunk_index in range(TRUNKS):
                weighted_presence = presence * family_weights[trunk_index].view(1, 1, -1)
                family_mix = (
                    self._coverage_family_states * weighted_presence.unsqueeze(-1)
                ).sum(dim=2) / weighted_presence.sum(dim=2, keepdim=True).clamp_min(1.0)
                trunk_inputs[trunk_index] = h + torch.sigmoid(self.trunk_family_gates[trunk_index]) * family_mix
        if not causal and self.cross_set_encoders is not None:
            trunk_states = []
            for trunk_index, encoder in enumerate(self.cross_set_encoders):
                state = trunk_inputs[trunk_index]
                for block in encoder:
                    state = block(state, padding_mask)
                trunk_states.append(state)
        else:
            trunk_states = [
                encoder(trunk_inputs[trunk_index], mask=mask, src_key_padding_mask=padding_mask)
                for trunk_index, encoder in enumerate(self.encoders)
            ]
        graph_h = self._task_state("graph", trunk_states)
        self.last_token_state = graph_h
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
        asset_class_logits = None
        if self.asset_class_head is not None:
            asset_class_h = self._task_state("asset_class", trunk_states)
            if causal:
                asset_class_h = asset_class_h[batch_index, lengths]
            else:
                asset_class_h = (asset_class_h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            asset_class_logits = self.asset_class_head(asset_class_h)
        macro_logits = self.macro_head(macro_h) if macro_h is not None else None
        cross_year_logits = None
        if self.cross_year_head is not None:
            cross_year_h = self._task_state("cross_year", trunk_states)
            if not causal:
                valid = (~padding_mask).unsqueeze(-1)
                cross_year_h = (cross_year_h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            else:
                cross_year_h = cross_year_h[batch_index, lengths]
            cross_year_logits = self.cross_year_head(cross_year_h)
        return (
            self.graph_head(graph_h), self.speed_head(speed_h), self.event_head(event_h),
            {}, {name: head(document_aux_h[name]) for name, head in self.document_aux_heads.items()},
            macro_logits, asset_class_logits,
            self.cross_graph_head(self._task_state("cross_graph", trunk_states)),
            self.cross_speed_head(self._task_state("cross_speed", trunk_states)), cross_year_logits,
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


class DualTowerTransformerMTL(nn.Module):
    """Fused issuer/instrument transformer with one shared head set.

    Both modality encoders process causal daily sequences independently.  The
    fusion state is the only representation exposed to prediction heads, so
    graph, speed, event, document, and asset-class tasks all receive both
    issuer and instrument information.  The model accepts one instrument per
    sampled pair; callers can run the same issuer state against many candidate
    instruments without changing the architecture.
    """

    def __init__(
        self,
        issuer_dim: int,
        instrument_dim: int,
        aux_dims: dict[str, int],
        instrument_types: int = len(ASSET_CLASSES) + 1,
        issuer_mean: np.ndarray | torch.Tensor | None = None,
        issuer_std: np.ndarray | torch.Tensor | None = None,
        issuer_robust_mask: np.ndarray | torch.Tensor | None = None,
        instrument_mean: np.ndarray | torch.Tensor | None = None,
        instrument_std: np.ndarray | torch.Tensor | None = None,
        instrument_robust_mask: np.ndarray | torch.Tensor | None = None,
    ):
        super().__init__()
        self.issuer_norm = LearnableFeatureNorm(
            issuer_mean if issuer_mean is not None else np.zeros(issuer_dim, dtype=np.float32),
            issuer_std if issuer_std is not None else np.ones(issuer_dim, dtype=np.float32),
            issuer_robust_mask,
        )
        self.instrument_norm = LearnableFeatureNorm(
            instrument_mean if instrument_mean is not None else np.zeros(instrument_dim, dtype=np.float32),
            instrument_std if instrument_std is not None else np.ones(instrument_dim, dtype=np.float32),
            instrument_robust_mask,
        )
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
        self.issuer_encoders = nn.ModuleList(
            [nn.TransformerEncoder(issuer_layer, num_layers=LAYERS) for _ in range(ISSUER_TRUNKS)]
        )
        self.instrument_encoders = nn.ModuleList(
            [nn.TransformerEncoder(instrument_layer, num_layers=LAYERS) for _ in range(INSTRUMENT_TRUNKS)]
        )
        self.instrument_type = nn.Embedding(max(1, instrument_types), HIDDEN)
        self.position = nn.Parameter(torch.randn(512, HIDDEN) * 0.01)
        self.issuer_router = nn.Parameter(torch.zeros(ISSUER_TRUNKS))
        self.instrument_router = nn.Parameter(torch.zeros(INSTRUMENT_TRUNKS))
        self.fusion = nn.Sequential(
            nn.Linear(HIDDEN * 4, HIDDEN * 2), nn.LayerNorm(HIDDEN * 2), nn.GELU(),
            nn.Linear(HIDDEN * 2, HIDDEN), nn.LayerNorm(HIDDEN),
        )
        self.graph_head = nn.Linear(HIDDEN, len(TARGET_COLS))
        self.speed_head = nn.Linear(HIDDEN, len(SPEED_TARGET_COLS))
        self.event_head = gnn.EventPrototypeHead(HIDDEN, len(EVENT_COLS))
        self.document_aux_heads = nn.ModuleDict({
            name: PrototypeHead(HIDDEN, max(1, int(size)))
            for name, size in aux_dims.items()
        })
        self.asset_class_head = nn.Linear(HIDDEN, len(ASSET_CLASSES) + 1)

    def forward(
        self,
        issuer_x: torch.Tensor,
        instrument_x: torch.Tensor,
        padding_mask: torch.Tensor,
        instrument_type_ids: torch.Tensor,
    ) -> tuple:
        if issuer_x.ndim != 3 or instrument_x.ndim not in (3, 4):
            raise ValueError("issuer_x must be [batch,time,features] and instrument_x [batch,time,features] or [batch,instruments,time,features]")
        grouped = instrument_x.ndim == 4
        if grouped:
            if issuer_x.shape[0] != instrument_x.shape[0] or issuer_x.shape[1] != instrument_x.shape[2]:
                raise ValueError("grouped issuer and instrument sequences must share batch and time dimensions")
        elif issuer_x.shape[:2] != instrument_x.shape[:2]:
            raise ValueError("issuer and instrument sequences must share batch and time dimensions")
        batch, length, _ = issuer_x.shape
        if length > self.position.shape[0]:
            raise ValueError(f"document length {length} exceeds positional capacity {self.position.shape[0]}")
        if padding_mask.shape != (batch, length):
            raise ValueError("padding_mask must have shape [batch, time]")
        if instrument_type_ids.shape not in ((batch,), instrument_x.shape[:2] if grouped else (batch,)):
            raise ValueError("instrument_type_ids must have shape [batch] or [batch,instruments]")
        mask = causal_mask(length, issuer_x.device)
        issuer_input = self.issuer_input(self.issuer_norm(issuer_x)) + self.position[:length].unsqueeze(0)
        if grouped:
            instruments = instrument_x.shape[1]
            flat_instrument = instrument_x.reshape(batch * instruments, length, instrument_x.shape[-1])
            instrument_input = self.instrument_input(self.instrument_norm(flat_instrument)).reshape(batch, instruments, length, HIDDEN)
            type_ids = instrument_type_ids if instrument_type_ids.ndim == 2 else instrument_type_ids[:, None].expand(batch, instruments)
            instrument_input = instrument_input + self.position[:length].view(1, 1, length, HIDDEN) + self.instrument_type(type_ids).unsqueeze(2)
            instrument_input = instrument_input.reshape(batch * instruments, length, HIDDEN)
        else:
            instrument_input = self.instrument_input(self.instrument_norm(instrument_x)) + self.position[:length].unsqueeze(0) + self.instrument_type(instrument_type_ids).unsqueeze(1)
        issuer_states = torch.stack([
            encoder(issuer_input, mask=mask, src_key_padding_mask=padding_mask)
            for encoder in self.issuer_encoders
        ], dim=0)
        instrument_padding = padding_mask.repeat_interleave(instruments, dim=0) if grouped else padding_mask
        instrument_states = torch.stack([
            encoder(instrument_input, mask=mask, src_key_padding_mask=instrument_padding)
            for encoder in self.instrument_encoders
        ], dim=0)
        if grouped:
            instrument_states = instrument_states.reshape(INSTRUMENT_TRUNKS, batch, instruments, length, HIDDEN)
        issuer_h = (issuer_states * torch.softmax(self.issuer_router, dim=0)[:, None, None, None]).sum(dim=0)
        instrument_weights = torch.softmax(self.instrument_router, dim=0)
        instrument_h = (instrument_states * instrument_weights.view(INSTRUMENT_TRUNKS, *([1] * (instrument_states.ndim - 1)))).sum(dim=0)
        if grouped:
            issuer_h = issuer_h[:, None].expand(-1, instruments, -1, -1)
        fused = self.fusion(torch.cat([
            issuer_h, instrument_h, issuer_h * instrument_h, torch.abs(issuer_h - instrument_h),
        ], dim=-1))
        graph_hat = self.graph_head(fused)
        speed_hat = self.speed_head(fused)
        event_logits = self.event_head(fused)
        valid = (~padding_mask).unsqueeze(-1)
        if grouped:
            valid = valid[:, None].expand(-1, instruments, -1, -1)
        pool_dim = 2 if grouped else 1
        pooled = (fused * valid).sum(dim=pool_dim) / valid.sum(dim=pool_dim).clamp_min(1)
        document_aux_logits = {
            name: head(pooled) for name, head in self.document_aux_heads.items()
        }
        asset_class_logits = self.asset_class_head(pooled)
        return graph_hat, speed_hat, event_logits, {}, document_aux_logits, None, asset_class_logits

    def trunk_weights(self) -> dict[str, list[float]]:
        return {
            "issuer": torch.softmax(self.issuer_router, dim=0).detach().cpu().tolist(),
            "instrument": torch.softmax(self.instrument_router, dim=0).detach().cpu().tolist(),
        }


def _read_fused_panel(index: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    parts: list[pd.DataFrame] = []
    feature_cols: list[str] = []
    for _, meta in index.iterrows():
        panel = pd.read_parquet(meta.panel_path)
        metadata = pd.read_parquet(meta.metadata_path)
        family = str(meta.family)
        if family in DISABLED_TECHNICAL_FAMILIES or family in EXTRA_DISABLED_FAMILIES:
            continue
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
    # Feature families have independent reporting histories and coverage.
    # Preserve the union of (symbol, date) observations; downstream fold-safe
    # imputation and family-presence signals distinguish missing coverage from
    # observed values.  An inner join would silently shrink the universe to
    # the intersection of every family's coverage.
    fused = pd.concat(parts, axis=1, join="outer").reset_index()
    return fused.sort_values(["symbol", "date"]).reset_index(drop=True), feature_cols


def _make_docs(
    base: pd.DataFrame,
    test_year: int,
    train: bool,
    feature_mask: np.ndarray | None = None,
) -> list[dict[str, np.ndarray]]:
    year = base.date.dt.year
    selected = base.loc[year < test_year if train else year == test_year].copy()
    docs: list[dict[str, np.ndarray]] = []
    selected["_year"] = selected.date.dt.year
    for (symbol, _), frame in selected.groupby(["symbol", "_year"], sort=True):
        frame = frame.sort_values("date")
        values = np.stack(frame["__x__"].to_numpy()).astype(np.float32, copy=False)
        if feature_mask is not None:
            values = values * feature_mask
        docs.append({
            "symbol": np.array([symbol] * len(frame)),
            "date": frame.date.to_numpy(),
            "x": values,
            "family_presence": np.stack(frame["__family_presence__"].to_numpy()).astype(np.float32),
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
        # Cross-sectional targets are ranks among symbols on this date.  They
        # reuse the per-symbol temporal HITS labels; no cross-sectional graph
        # or pairwise edge construction is performed.
        cross_graph = frame[TARGET_COLS].apply(
            pd.to_numeric, errors="coerce"
        ).rank(method="average", pct=True).fillna(0.5).to_numpy(np.float32)
        cross_speed = frame[SPEED_TARGET_COLS].apply(
            pd.to_numeric, errors="coerce"
        ).rank(method="average", pct=True).fillna(0.5).to_numpy(np.float32)
        cross_aux = np.full((len(frame), len(AUX_COLS)), -1, dtype=np.int64)
        cross_year = int(frame["year_target"].iloc[0]) if "year_target" in frame else -1
        docs.append({
            "symbol": frame.symbol.to_numpy(),
            "date": frame.date.to_numpy(),
            "x": np.stack(frame["__x__"].to_numpy()),
            "family_presence": np.stack(frame["__family_presence__"].to_numpy()).astype(np.float32),
            # Reuse labels generated by the temporal per-symbol graph when
            # enabled.  No graph is ever constructed over the universe on a
            # cross-sectional date.
            "graph": frame[TARGET_COLS].to_numpy(np.float32),
            "speed": frame[SPEED_TARGET_COLS].to_numpy(np.float32),
            "cross_graph": cross_graph,
            "cross_speed": cross_speed,
            "events": np.zeros((len(frame), len(EVENT_COLS)), dtype=np.float32),
            "macro_events": frame[MACRO_EVENT_COLS].iloc[0].to_numpy(np.float32),
            "cross_year": cross_year,
            "aux": cross_aux,
            "graph_mask": np.stack(frame["__graph_mask__"].to_numpy()) if CROSS_SECTIONAL_TOKEN_TASKS else np.zeros((len(frame), len(TARGET_COLS)), dtype=np.float32),
            "kind": "cross_sectional",
            "task_mask": np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32),
            "cross_task_mask": np.array([float(CROSS_SECTIONAL_TOKEN_TASKS), float(CROSS_SECTIONAL_TOKEN_TASKS)], dtype=np.float32),
        })
    return docs


def _batch(docs: list[dict[str, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    length = max(len(item["x"]) for item in docs)
    dim = docs[0]["x"].shape[1]
    x = np.zeros((len(docs), length, dim), dtype=np.float32)
    padding = np.ones((len(docs), length), dtype=bool)
    graph = np.zeros((len(docs), length, len(TARGET_COLS)), dtype=np.float32)
    speed = np.zeros((len(docs), length, len(SPEED_TARGET_COLS)), dtype=np.float32)
    cross_graph = np.zeros((len(docs), length, len(TARGET_COLS)), dtype=np.float32)
    cross_speed = np.zeros((len(docs), length, len(SPEED_TARGET_COLS)), dtype=np.float32)
    events = np.zeros((len(docs), length, len(EVENT_COLS)), dtype=np.float32)
    macro_events = np.zeros((len(docs), length, len(MACRO_EVENT_COLS)), dtype=np.float32)
    macro_document_events = np.zeros((len(docs), len(MACRO_EVENT_COLS)), dtype=np.float32)
    aux = np.full((len(docs), length, len(AUX_COLS)), -1, dtype=np.int64)
    graph_mask = np.zeros((len(docs), length, len(TARGET_COLS)), dtype=np.float32)
    task_mask = np.zeros((len(docs), 5), dtype=np.float32)
    cross_task_mask = np.zeros((len(docs), 2), dtype=np.float32)
    cross_year = np.full(len(docs), -1, dtype=np.int64)
    modality = np.zeros(len(docs), dtype=np.int64)
    family_count = len(docs[0].get("family_presence", np.zeros(0, dtype=np.float32)[None, :])[0])
    family_presence = np.zeros((len(docs), length, family_count), dtype=np.float32)
    for i, item in enumerate(docs):
        n = len(item["x"])
        x[i, :n] = item["x"]
        if family_count:
            if "family_presence" in item:
                family_presence[i, :n] = item["family_presence"]
            else:
                # Auxiliary corpora may not carry the fused-panel mask.  A
                # zero mask keeps those families inactive rather than turning
                # imputed values into false observations.
                family_presence[i, :n] = 0.0
        padding[i, :n] = False
        graph[i, :n] = item["graph"]
        speed[i, :n] = item["speed"]
        if "cross_graph" in item:
            cross_graph[i, :n] = item["cross_graph"]
        if "cross_speed" in item:
            cross_speed[i, :n] = item["cross_speed"]
        events[i, :n] = item["events"]
        if MACRO_EVENT_COLS:
            if item["kind"] == "cross_sectional":
                macro_document_events[i] = item["macro_events"]
            else:
                macro_events[i, :n] = item["macro_events"]
        aux[i, :n] = item["aux"]
        graph_mask[i, :n] = item["graph_mask"]
        task_mask[i] = item["task_mask"]
        cross_task_mask[i] = item.get("cross_task_mask", np.zeros(2, dtype=np.float32))
        cross_year[i] = int(item.get("cross_year", -1))
        modality[i] = int(item.get("modality", 0))
    return (
        torch.from_numpy(x), torch.from_numpy(padding),
        {"graph": torch.from_numpy(graph), "speed": torch.from_numpy(speed), "cross_graph": torch.from_numpy(cross_graph), "cross_speed": torch.from_numpy(cross_speed), "events": torch.from_numpy(events),
         "macro_events": torch.from_numpy(macro_events), "macro_document_events": torch.from_numpy(macro_document_events), "aux": torch.from_numpy(aux),
         "family_presence": torch.from_numpy(family_presence),
         "graph_mask": torch.from_numpy(graph_mask), "task_mask": torch.from_numpy(task_mask), "cross_task_mask": torch.from_numpy(cross_task_mask), "cross_year": torch.from_numpy(cross_year), "modality": torch.from_numpy(modality)},
    )


def _batch_same_issuer_pairs(pairs: list[dict[str, object]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Batch aligned issuer/instrument pairs for DualTowerTransformerMTL."""
    length = max(len(pair["issuer"]["x"]) for pair in pairs)
    issuer_dim = pairs[0]["issuer"]["x"].shape[1]
    instrument_dim = pairs[0]["instrument"]["x"].shape[1]
    issuer_x = np.zeros((len(pairs), length, issuer_dim), dtype=np.float32)
    instrument_x = np.zeros((len(pairs), length, instrument_dim), dtype=np.float32)
    padding = np.ones((len(pairs), length), dtype=bool)
    graph = np.zeros((len(pairs), length, len(TARGET_COLS)), dtype=np.float32)
    speed = np.zeros((len(pairs), length, len(SPEED_TARGET_COLS)), dtype=np.float32)
    events = np.zeros((len(pairs), length, len(EVENT_COLS)), dtype=np.float32)
    aux = np.full((len(pairs), len(AUX_COLS)), -1, dtype=np.int64)
    modality = np.zeros(len(pairs), dtype=np.int64)
    for row, pair in enumerate(pairs):
        issuer = pair["issuer"]; instrument = pair["instrument"]
        count = min(len(issuer["x"]), len(instrument["x"]))
        issuer_x[row, :count] = issuer["x"][:count]
        instrument_x[row, :count] = instrument["x"][:count]
        padding[row, :count] = False
        graph[row, :count] = instrument["graph"][:count]
        speed[row, :count] = instrument["speed"][:count]
        events[row, :count] = instrument["events"][:count]
        aux[row] = instrument["aux"][0]
        modality[row] = int(pair["modality"])
    return (
        torch.from_numpy(issuer_x), torch.from_numpy(instrument_x), torch.from_numpy(padding),
        {"graph": torch.from_numpy(graph), "speed": torch.from_numpy(speed), "events": torch.from_numpy(events),
         "aux": torch.from_numpy(aux), "modality": torch.from_numpy(modality)},
    )


def _group_same_issuer_pairs(pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    """Group one issuer document with all same-issuer instrument candidates."""
    groups: dict[tuple[str, int], dict[str, object]] = {}
    for pair in pairs:
        issuer = pair["issuer"]
        key = (str(issuer["symbol"][0]).upper(), int(pd.Timestamp(issuer["date"][0]).year))
        group = groups.setdefault(key, {"issuer": issuer, "instruments": [], "modalities": []})
        if not group["instruments"]:
            group["issuer"] = issuer
        group["instruments"].append(pair["instrument"])
        group["modalities"].append(int(pair["modality"]))
    return list(groups.values())


def _batch_same_issuer_groups(groups: list[dict[str, object]]):
    length = max(len(g["issuer"]["x"]) for g in groups)
    candidates = max(len(g["instruments"]) for g in groups)
    fd = groups[0]["issuer"]["x"].shape[1]
    issuer_x = np.zeros((len(groups), length, fd), np.float32)
    instrument_x = np.zeros((len(groups), candidates, length, fd), np.float32)
    padding = np.ones((len(groups), length), bool)
    modality = np.zeros((len(groups), candidates), np.int64)
    graph = np.zeros((len(groups), candidates, length, len(TARGET_COLS)), np.float32)
    speed = np.zeros((len(groups), candidates, length, len(SPEED_TARGET_COLS)), np.float32)
    events = np.zeros((len(groups), candidates, length, len(EVENT_COLS)), np.float32)
    aux = np.full((len(groups), candidates, len(AUX_COLS)), -1, np.int64)
    candidate_mask = np.zeros((len(groups), candidates), bool)
    token_mask = np.zeros((len(groups), candidates, length), bool)
    date_ord = np.full((len(groups), length), -1, np.int64)
    for row, group in enumerate(groups):
        issuer = group["issuer"]; q = len(issuer["x"]); issuer_x[row, :q] = issuer["x"]; padding[row, :q] = False
        date_ord[row, :q] = pd.to_datetime(issuer["date"], errors="coerce").to_numpy(dtype="datetime64[D]").astype(np.int64)
        for col, instrument in enumerate(group["instruments"]):
            candidate_mask[row, col] = True
            n = min(q, len(instrument["x"])); token_mask[row, col, :n] = True; instrument_x[row, col, :n] = instrument["x"][:n]; graph[row, col, :n] = instrument["graph"][:n]; speed[row, col, :n] = instrument["speed"][:n]; events[row, col, :n] = instrument["events"][:n]; aux[row, col] = instrument["aux"][0]; modality[row, col] = group["modalities"][col]
    return (torch.from_numpy(issuer_x), torch.from_numpy(instrument_x), torch.from_numpy(padding),
            {"graph": torch.from_numpy(graph), "speed": torch.from_numpy(speed), "events": torch.from_numpy(events), "aux": torch.from_numpy(aux), "modality": torch.from_numpy(modality), "candidate_mask": torch.from_numpy(candidate_mask), "token_mask": torch.from_numpy(token_mask), "date_ord": torch.from_numpy(date_ord)})


def _mix_batch_in_memory(
    issuer_x: torch.Tensor,
    instrument_x: torch.Tensor,
    padding: torch.Tensor,
    target: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]] | None]:
    """Add random same-date cross-issuer candidates to one batch in memory.

    Every row gets one random donor attempt. A candidate contributes only
    where the two documents share an exact calendar date, so the data itself
    determines how many mix-and-match examples are usable.
    """
    if issuer_x.shape[0] < 2:
        return issuer_x, instrument_x, padding, target, None
    batch, _, length, features = instrument_x.shape
    date_ord = target["date_ord"]
    shift = int(torch.randint(1, batch, (1,)))
    permutation = torch.roll(torch.arange(batch), shifts=shift)
    valid_mixes: list[tuple[int, int, torch.Tensor]] = []
    for row in range(batch):
        donor = int(permutation[row])
        valid_candidates = torch.where(target["candidate_mask"][donor])[0]
        if not len(valid_candidates):
            continue
        donor_candidate = int(valid_candidates[torch.randint(len(valid_candidates), (1,))])
        # Match by exact calendar date, not by position in each issuer's
        # history.  Coverage differences can otherwise pair different dates.
        token_valid = (
            target["token_mask"][row, 0]
            & target["token_mask"][donor, donor_candidate]
            & date_ord[row].eq(date_ord[donor])
        )
        if token_valid.any():
            valid_mixes.append((row, donor_candidate, token_valid))
    # Do not append an all-empty candidate dimension to the batch.
    if not valid_mixes:
        return issuer_x, instrument_x, padding, target, None
    rows = [row for row, _, _ in valid_mixes]
    mixed_issuer_x = issuer_x[rows]
    mixed_padding = padding[rows]
    mixed_x = torch.zeros((len(rows), 1, length, features), dtype=instrument_x.dtype)
    mixed_graph = torch.zeros((len(rows), 1, length, target["graph"].shape[-1]), dtype=target["graph"].dtype)
    mixed_speed = torch.zeros((len(rows), 1, length, target["speed"].shape[-1]), dtype=target["speed"].dtype)
    mixed_events = torch.zeros((len(rows), 1, length, target["events"].shape[-1]), dtype=target["events"].dtype)
    mixed_aux = torch.full((len(rows), 1, target["aux"].shape[-1]), -1, dtype=target["aux"].dtype)
    mixed_modality = torch.zeros((len(rows), 1), dtype=target["modality"].dtype)
    mixed_candidate = torch.ones((len(rows), 1), dtype=torch.bool)
    mixed_token = torch.zeros((len(rows), 1, length), dtype=torch.bool)
    for mixed_row, (row, donor_candidate, token_valid) in enumerate(valid_mixes):
        donor = int(permutation[row])
        mixed_x[mixed_row, 0, token_valid] = instrument_x[donor, donor_candidate, token_valid]
        mixed_graph[mixed_row, 0, token_valid] = target["graph"][donor, donor_candidate, token_valid]
        mixed_speed[mixed_row, 0, token_valid] = target["speed"][donor, donor_candidate, token_valid]
        mixed_events[mixed_row, 0, token_valid] = target["events"][donor, donor_candidate, token_valid]
        mixed_aux[mixed_row, 0] = target["aux"][donor, donor_candidate]
        mixed_modality[mixed_row, 0] = target["modality"][donor, donor_candidate]
        mixed_token[mixed_row, 0] = token_valid
    mixed_target = {
        "graph": mixed_graph, "speed": mixed_speed, "events": mixed_events,
        "aux": mixed_aux, "modality": mixed_modality,
        "candidate_mask": mixed_candidate, "token_mask": mixed_token,
        "date_ord": date_ord[rows],
    }
    return issuer_x, instrument_x, padding, target, (mixed_issuer_x, mixed_x, mixed_padding, mixed_target)


def _load_macro_event_panel(dates: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """Fetch FMP releases once and build labels through quant-warehouse."""
    group_key = "_".join(MACRO_COMPACT_GROUPS) if MACRO_COMPACT else "all"
    cache_version = (
        f"compact_directional_v9_{group_key}"
        if MACRO_COMPACT else "directional_no_unchanged_v9"
    )
    if MACRO_DEDUP_IDENTICAL:
        cache_version += "_dedup"
    cache = OUT / "cache" / f"transformer_macro_event_labels_{cache_version}_{gnn.DATA_END:%Y%m%d}.parquet"
    if cache.exists():
        panel = pd.read_parquet(cache)
    else:
        events = fetch_economy_calendar_range(
            start_date="2020-01-01", end_date=str(gnn.DATA_END.date())
        )
        if MACRO_COUNTRIES and not events.empty:
            events = events.loc[events.country.astype(str).str.upper().isin(MACRO_COUNTRIES)].copy()
        token_dates = pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce").dt.normalize().unique()})
        panel = build_macro_event_label_panel(
            token_dates,
            events,
            directional_only=True,
            compact_directional=MACRO_COMPACT,
            compact_groups=MACRO_COMPACT_GROUPS,
            deduplicate_identical=MACRO_DEDUP_IDENTICAL,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(cache, index=False)
    return panel, [column for column in panel.columns if str(column).startswith("is_")]


def _load_preferred_feature_panel(base_symbols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Load the precomputed raw preferred-security feature family."""
    if not PREFERRED_ENABLED:
        return pd.DataFrame(), []
    path = Path(PREFERRED_PANEL_PATH) if PREFERRED_PANEL_PATH else OUT / "cache" / "preferred_stock_features_100b.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Preferred feature panel is missing: {path}. Run build_preferred_stock_feature_panel.py first."
        )
    panel = pd.read_parquet(path)
    required = {"symbol", "date"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Preferred feature panel missing columns: {sorted(missing)}")
    panel["symbol"] = panel["symbol"].astype(str).str.upper()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel = panel.loc[panel.symbol.isin(base_symbols)].copy()
    feature_cols = [
        column for column in panel.columns
        if str(column).startswith("preferred__") and "__px__" not in str(column)
    ]
    identity_cols = ["symbol", "date"]
    if "asset_class" in panel.columns:
        identity_cols.append("asset_class")
    if "instrument_symbol" in panel.columns:
        identity_cols.append("instrument_symbol")
    if "maturity_date" in panel.columns:
        identity_cols.append("maturity_date")
    return panel[[*identity_cols, *feature_cols]], feature_cols


def _load_related_asset_feature_panel(base_symbols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    if not RELATED_ASSETS_ENABLED:
        return pd.DataFrame(), []
    path = Path(RELATED_ASSETS_PANEL_PATH) if RELATED_ASSETS_PANEL_PATH else OUT / "cache" / "related_asset_features_100b_adjusted.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Related-asset feature panel is missing: {path}")
    panel = pd.read_parquet(path)
    panel["symbol"] = panel["symbol"].astype(str).str.upper()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel = panel.loc[panel.symbol.isin(base_symbols)].copy()
    feature_cols = [
        column for column in panel.columns
        if "__" in str(column)
        and str(column).split("__", 1)[0] in {"preferred", "warrant", "unit", "note_bond", "adr", "ordinary", "etf"}
        and "__px__" not in str(column)
    ]
    identity_cols = ["symbol", "date"]
    if "asset_class" in panel.columns:
        identity_cols.append("asset_class")
    if "instrument_symbol" in panel.columns:
        identity_cols.append("instrument_symbol")
    if "maturity_date" in panel.columns:
        identity_cols.append("maturity_date")
    return panel[[*identity_cols, *feature_cols]], feature_cols


def _load_etf_corpus_panel() -> tuple[pd.DataFrame, list[str]]:
    if not ETF_CORPUS_ENABLED:
        return pd.DataFrame(), []
    path = Path(ETF_CORPUS_PANEL_PATH) if ETF_CORPUS_PANEL_PATH else OUT / "cache" / "etf_document_corpus_100b_adjusted.parquet"
    if not path.exists():
        raise FileNotFoundError(f"ETF corpus panel is missing: {path}. Run build_etf_feature_panel.py first.")
    panel = pd.read_parquet(path)
    panel["symbol"] = panel["symbol"].astype(str).str.upper()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    feature_cols = [
        column for column in panel.columns
        if str(column).startswith("etf__") and "__px__" not in str(column)
    ]
    return panel, feature_cols


def _add_related_instrument_graph_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """Build HITS targets from each related instrument's own price history."""
    if panel.empty or "instrument_symbol" not in panel.columns:
        return panel
    frames: dict[str, pd.DataFrame] = {}
    for security_class in ASSET_CLASSES:
        open_col = f"{security_class}__open"
        high_col = f"{security_class}__high"
        low_col = f"{security_class}__low"
        close_col = f"{security_class}__close"
        volume_col = f"{security_class}__volume"
        if any(column not in panel.columns for column in (open_col, high_col, low_col, close_col)):
            continue
        available_columns = ["symbol", "instrument_symbol", "date", open_col, high_col, low_col, close_col]
        if volume_col in panel.columns:
            available_columns.append(volume_col)
        available = panel[available_columns].dropna(subset=[high_col, low_col])
        if volume_col not in available:
            available[volume_col] = np.nan
        for (issuer, instrument), frame in available.groupby(["symbol", "instrument_symbol"], sort=False):
            frames[f"{issuer}:{security_class}:{instrument}".upper()] = frame.rename(
                columns={open_col: "open", high_col: "high", low_col: "low", close_col: "close", volume_col: "volume"}
            )[["date", "open", "high", "low", "close", "volume"]]
    if not frames:
        return panel
    spec = gnn.HitsLabelSpec(
        max_hold=gnn.MAX_HOLD,
        iterations=gnn.HITS_ITERATIONS,
        tail_quantile=gnn.HITS_TAIL_QUANTILE,
        start_date=str(gnn.DATA_START.date()),
        end_date=str(gnn.DATA_END.date()),
    )
    labels = gnn.build_return_and_speed_hits_labels(frames, spec=spec)
    labels = gnn.add_speed_pagerank_labels(frames, labels)
    labels = gnn.add_pagerank_labels(frames, labels)
    labels = labels.rename(columns={"symbol": "_asset_key"})
    panel = panel.copy()
    panel["_asset_key"] = (
        panel["symbol"].astype(str).str.upper()
        + ":" + panel["asset_class"].astype(str).str.upper()
        + ":" + panel["instrument_symbol"].astype(str).str.upper()
    )
    label_cols = ["_asset_key", "date", *TARGET_COLS, *SPEED_TARGET_COLS]
    return panel.merge(labels[label_cols], on=["_asset_key", "date"], how="left", validate="many_to_one")


def _prepare_data(tier: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame]:
    global MACRO_EVENT_COLS, MACRO_FAMILY_COLS, MACRO_DIRECTION_COLS, MACRO_SURPRISE_COLS
    index = pd.read_csv(gnn.feature_dir(tier) / "index.csv")
    fused, feature_cols = _read_fused_panel(index)
    symbols = sorted(fused.symbol.unique())
    prices, labels = gnn.build_price_and_labels(symbols, tier)
    base = fused.merge(labels, on=["symbol", "date"], how="inner")
    # Keep only the adjusted raw price observations after disabling all
    # precomputed technical families.  Causal attention can derive returns,
    # trends, volatility, and reversal behavior from these historical tokens.
    raw_price_cols = [column for column in ("open", "high", "low", "close", "volume") if column in prices.columns]
    raw_price_names = [f"raw__{column}" for column in raw_price_cols]
    if raw_price_cols:
        raw_prices = prices[["symbol", "date", *raw_price_cols]].rename(
            columns=dict(zip(raw_price_cols, raw_price_names))
        )
        base = base.merge(raw_prices, on=["symbol", "date"], how="left", validate="one_to_one")
        base[raw_price_names] = base[raw_price_names].apply(pd.to_numeric, errors="coerce")
        feature_cols.extend(raw_price_names)
    preferred_panel, preferred_cols = _load_preferred_feature_panel(symbols)
    if not preferred_panel.empty:
        base = base.merge(preferred_panel, on=["symbol", "date"], how="left")
        base[preferred_cols] = base[preferred_cols].apply(pd.to_numeric, errors="coerce")
    elif PREFERRED_ENABLED:
        raise RuntimeError("Preferred features were enabled but no preferred rows overlap the requested tier")
    related_panel, related_cols = _load_related_asset_feature_panel(symbols)
    if not related_panel.empty and RELATED_ASSETS_AS_ROWS:
        related_panel = _add_related_instrument_graph_labels(related_panel)
    if not related_panel.empty:
        if RELATED_ASSETS_AS_ROWS:
            # Reserve the related-asset feature coordinates in the model input,
            # but keep equity documents free of those values.  The related
            # values are emitted as a separate document corpus below.
            base = pd.concat([base, pd.DataFrame(np.nan, index=base.index, columns=related_cols)], axis=1)
        else:
            base = base.merge(related_panel, on=["symbol", "date"], how="left")
            base[related_cols] = base[related_cols].apply(pd.to_numeric, errors="coerce")
    elif RELATED_ASSETS_ENABLED:
        raise RuntimeError("Related-asset features were enabled but no rows overlap the requested tier")
    etf_panel, etf_cols = _load_etf_corpus_panel()
    if ETF_CORPUS_ENABLED:
        base = pd.concat([base, pd.DataFrame(np.nan, index=base.index, columns=etf_cols)], axis=1)
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
    return base, prices, feature_cols + preferred_cols + related_cols + etf_cols, related_panel, etf_panel


def _make_related_docs(
    base: pd.DataFrame,
    related_panel: pd.DataFrame,
    feature_cols: list[str],
    test_year: int,
    train: bool,
) -> list[dict[str, np.ndarray]]:
    """Build a separate causal corpus with one document per issuer/year/class.

    Each row is one related security-class observation for an issuer/date.  The
    class-specific raw features remain in their own coordinates; other class
    coordinates are zero.  Labels are joined by issuer/date only so this corpus
    can shape the shared representation without putting related data into the
    equity documents.
    """
    if not RELATED_ASSETS_AS_ROWS or related_panel.empty:
        return []
    class_names = ("preferred", "warrant", "unit", "note_bond", "adr", "ordinary", "etf")
    class_cols = {
        name: [column for column in feature_cols if column in related_panel.columns and str(column).startswith(f"{name}__")]
        for name in class_names
    }
    issuer_label_frame = base[["symbol", "date", *EVENT_COLS, *AUX_COLS]].copy()
    panel = related_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["symbol"] = panel["symbol"].astype(str).str.upper()
    panel = _add_maturity_delisted_labels(panel)
    panel = panel.merge(issuer_label_frame, on=["symbol", "date"], how="inner", validate="many_to_one")
    if "is_delisted_maturity" in panel.columns:
        panel["is_delisted"] = panel[["is_delisted", "is_delisted_maturity"]].fillna(0.0).max(axis=1)
    year = panel.date.dt.year
    panel = panel.loc[year < test_year if train else year == test_year].copy()
    docs: list[dict[str, np.ndarray]] = []
    for security_class, columns in class_cols.items():
        if not columns:
            continue
        available = panel[columns].notna().any(axis=1)
        class_panel = panel.loc[available].copy()
        if class_panel.empty:
            continue
        class_panel["_asset_class"] = security_class
        group_columns = ["symbol"]
        if "instrument_symbol" in class_panel.columns:
            group_columns.append("instrument_symbol")
        group_columns.extend([class_panel.date.dt.year.rename("_doc_year"), "_asset_class"])
        for group_key, frame in class_panel.groupby(group_columns, sort=True):
            frame = frame.sort_values("date")
            symbol = str(frame["symbol"].iloc[0])
            values = np.zeros((len(frame), len(feature_cols)), dtype=np.float32)
            positions = {column: index for index, column in enumerate(feature_cols)}
            for column in columns:
                values[:, positions[column]] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(np.float32)
            docs.append({
                "symbol": np.array([symbol] * len(frame)), "date": frame.date.to_numpy(),
                "x": values,
                "graph": frame[TARGET_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32),
                "speed": frame[SPEED_TARGET_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32),
                "events": frame[EVENT_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32),
                "macro_events": np.zeros((len(frame), len(MACRO_EVENT_COLS)), dtype=np.float32),
                "aux": frame[AUX_COLS].to_numpy(np.int64),
                "graph_mask": _related_graph_mask(frame),
                "kind": "symbol_year", "task_mask": np.ones(5, dtype=np.float32),
                "modality": ASSET_MODALITY_IDS[security_class],
            })
    return docs


def _make_same_issuer_pairs(
    issuer_docs: list[dict[str, np.ndarray]],
    instrument_docs: list[dict[str, np.ndarray]],
    equity_instrument_docs: list[dict[str, np.ndarray]] | None = None,
) -> list[dict[str, object]]:
    """Align issuer and instrument documents from the same issuer by date.

    The issuer state is reused for every same-issuer instrument candidate, but
    instrument targets remain sourced from that instrument's own labels.
    """
    issuer_index = {
        (str(doc["symbol"][0]).upper(), int(pd.Timestamp(doc["date"][0]).year)): doc
        for doc in issuer_docs if len(doc["date"])
    }
    pairs: list[dict[str, object]] = []
    equity_docs_by_key = {
        (str(doc["symbol"][0]).upper(), int(pd.Timestamp(doc["date"][0]).year)): doc
        for doc in (equity_instrument_docs or issuer_docs) if len(doc["date"])
    }
    for issuer in issuer_docs:
        key = (str(issuer["symbol"][0]).upper(), int(pd.Timestamp(issuer["date"][0]).year))
        equity_instrument = equity_docs_by_key.get(key, issuer)
        pairs.append({"issuer": issuer, "instrument": equity_instrument, "modality": 0})
    for instrument in instrument_docs:
        if not len(instrument["date"]):
            continue
        key = (str(instrument["symbol"][0]).upper(), int(pd.Timestamp(instrument["date"][0]).year))
        issuer = issuer_index.get(key)
        if issuer is None:
            continue
        issuer_dates = {pd.Timestamp(value): index for index, value in enumerate(issuer["date"])}
        instrument_indices = [index for index, value in enumerate(instrument["date"]) if pd.Timestamp(value) in issuer_dates]
        if not instrument_indices:
            continue
        issuer_indices = [issuer_dates[pd.Timestamp(instrument["date"][index])] for index in instrument_indices]

        def select(doc: dict[str, np.ndarray], indices: list[int]) -> dict[str, np.ndarray]:
            selected = dict(doc)
            for name in ("x", "graph", "speed", "events", "date", "symbol"):
                if name in doc and isinstance(doc[name], np.ndarray):
                    selected[name] = doc[name][indices]
            if "aux" in doc:
                selected["aux"] = doc["aux"][indices]
            return selected

        pairs.append({
            "issuer": select(issuer, issuer_indices),
            "instrument": select(instrument, instrument_indices),
            "modality": int(instrument.get("modality", 1)),
        })
    return pairs


def _add_maturity_delisted_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """Mark a related instrument delisted at maturity when FMP lacks history."""
    if panel.empty or "maturity_date" not in panel.columns:
        return panel
    out = panel.copy()
    out["maturity_date"] = pd.to_datetime(out["maturity_date"], errors="coerce").dt.normalize()
    out["is_delisted_maturity"] = 0.0
    group_cols = [column for column in ("symbol", "asset_class", "instrument_symbol") if column in out.columns]
    for _, indices in out.groupby(group_cols, sort=False).groups.items():
        frame = out.loc[indices].sort_values("date")
        maturity = frame["maturity_date"].dropna()
        if maturity.empty:
            continue
        maturity_date = maturity.iloc[0]
        eligible = frame.index[frame["date"].ge(maturity_date)]
        if len(eligible):
            # Use the first available observation on/after maturity so sparse
            # historical coverage still receives the event label once.
            out.loc[eligible[0], "is_delisted_maturity"] = 1.0
    return out


def _related_graph_mask(frame: pd.DataFrame) -> np.ndarray:
    """Mask graph losses using tails of the instrument's own PageRank labels."""
    mask = np.zeros((len(frame), len(TARGET_COLS)), dtype=np.float32)
    for column in ("long_pagerank", "short_pagerank"):
        if column not in frame:
            continue
        ranks = pd.to_numeric(frame[column], errors="coerce").rank(method="first", pct=True)
        values = ranks.le(gnn.HITS_TAIL_QUANTILE) | ranks.ge(1.0 - gnn.HITS_TAIL_QUANTILE)
        mask[:, TARGET_COLS.index(column)] = values.fillna(False).to_numpy(np.float32)
    return mask


def _is_price_volume_feature(column: str) -> bool:
    name = str(column).lower()
    field = name.rsplit("__", 1)[-1]
    return name.startswith("raw__") or field in {"open", "high", "low", "close", "volume"}


def _normalize_related_panel(panel: pd.DataFrame, feature_cols: list[str], test_year: int) -> pd.DataFrame:
    """Keep raw price/volume channels; normalize other channels fold-safely."""
    if panel.empty:
        return panel
    out = panel.copy()
    train = out.date.dt.year < test_year
    present_cols = [column for column in feature_cols if column in out.columns]
    if not present_cols:
        return out
    raw = out[present_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = raw.loc[train].median().fillna(0.0)
    filled = raw.fillna(med)
    raw_cols = [column for column in present_cols if _is_price_volume_feature(column)]
    normalized = filled.astype("float32")
    other_cols = [column for column in present_cols if column not in raw_cols]
    if other_cols:
        mean = filled.loc[train, other_cols].mean().fillna(0.0)
        std = filled.loc[train, other_cols].std().replace(0, 1).fillna(1.0)
        normalized[other_cols] = ((filled[other_cols] - mean) / std).clip(-8, 8).astype("float32")
    out[present_cols] = normalized
    return out


def _make_etf_docs(panel: pd.DataFrame, feature_cols: list[str], test_year: int, train: bool) -> list[dict[str, np.ndarray]]:
    if not ETF_CORPUS_ENABLED or panel.empty:
        return []
    panel = panel.copy()
    selected = panel.loc[panel.date.dt.year < test_year if train else panel.date.dt.year == test_year].copy()
    etf_cols = [column for column in feature_cols if str(column).startswith("etf__") and column in selected.columns]
    docs: list[dict[str, np.ndarray]] = []
    positions = {column: index for index, column in enumerate(feature_cols)}
    for symbol, frame in selected.groupby("symbol", sort=True):
        frame = frame.sort_values("date")
        # Long-lived ETFs can have more than 512 daily rows. Segment only the
        # ETF corpus so the causal encoder stays within positional capacity.
        for start in range(0, len(frame), 512):
            chunk = frame.iloc[start:start + 512]
            values = np.zeros((len(chunk), len(feature_cols)), dtype=np.float32)
            for column in etf_cols:
                values[:, positions[column]] = pd.to_numeric(chunk[column], errors="coerce").fillna(0.0).to_numpy(np.float32)
            graph_mask = np.ones((len(chunk), len(TARGET_COLS)), dtype=np.float32)
            docs.append({
                "symbol": np.array([symbol] * len(chunk)), "date": chunk.date.to_numpy(), "x": values,
                "graph": chunk[TARGET_COLS].to_numpy(np.float32), "speed": chunk[SPEED_TARGET_COLS].to_numpy(np.float32),
                "events": np.zeros((len(chunk), len(EVENT_COLS)), dtype=np.float32),
                "macro_events": np.zeros((len(chunk), len(MACRO_EVENT_COLS)), dtype=np.float32),
                "aux": np.full((len(chunk), len(AUX_COLS)), -1, dtype=np.int64), "graph_mask": graph_mask,
                "kind": "symbol_year", "task_mask": np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "modality": ASSET_MODALITY_IDS["etf"],
            })
    return docs


def _normalize(base: pd.DataFrame, feature_cols: list[str], test_year: int) -> pd.DataFrame:
    """Prepare raw model inputs and labels.

    The historical name is retained because it is part of the training loop,
    but this function no longer computes technical indicators or z-scores.
    It only performs fold-safe missing-value imputation and label cleanup.
    """
    out = base.copy()
    train = out.date.dt.year < test_year
    raw = out[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Preserve vendor coverage before imputation.  The model receives this
    # mask alongside the imputed values, so a missing historical endpoint is
    # not confused with a genuine zero observation.
    families: dict[str, list[str]] = {}
    for column in feature_cols:
        families.setdefault(str(column).split("__", 1)[0], []).append(column)
    family_presence = np.stack([
        raw[columns].notna().any(axis=1).to_numpy(np.float32)
        for columns in families.values()
    ], axis=-1) if families else np.zeros((len(out), 0), dtype=np.float32)
    med = raw.loc[train].median().fillna(0.0)
    filled = raw.fillna(med)
    raw_cols = [column for column in feature_cols if _is_price_volume_feature(column)]
    normalized = filled.astype("float32")
    other_cols = [column for column in feature_cols if column not in raw_cols]
    if other_cols:
        mean = filled.loc[train, other_cols].mean().fillna(0.0)
        std = filled.loc[train, other_cols].std().replace(0, 1).fillna(1.0)
        normalized[other_cols] = ((filled[other_cols] - mean) / std).clip(-8, 8).astype("float32")
    out["__x__"] = list(normalized.to_numpy())
    out["__family_presence__"] = list(family_presence)
    for column in TARGET_COLS + SPEED_TARGET_COLS + EVENT_COLS:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).astype("float32")
    graph_mask = pd.DataFrame(False, index=out.index, columns=TARGET_COLS)
    for column in ("long_pagerank", "short_pagerank"):
        ranks = out.groupby(out.date.dt.year)[column].rank(method="first", pct=True)
        graph_mask[column] = ranks.le(gnn.HITS_TAIL_QUANTILE) | ranks.ge(1.0 - gnn.HITS_TAIL_QUANTILE)
    out["__graph_mask__"] = list(graph_mask.astype("float32").to_numpy())
    return out


def _fit_feature_stats(docs: list[dict[str, np.ndarray]], feature_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit initialization statistics from training documents only.

    Zero-filled channels belonging to another asset modality are excluded when
    possible, so an asset adapter is initialized from the scale of its own raw
    observations.  The returned values are buffers for the learnable model
    normalizer, not frozen preprocessing outputs.
    """
    if not docs:
        return np.zeros(feature_dim, dtype=np.float32), np.ones(feature_dim, dtype=np.float32)
    values = np.concatenate([np.asarray(doc["x"], dtype=np.float32) for doc in docs], axis=0)
    finite = np.isfinite(values)
    observed = finite & (np.abs(values) > 1e-12)
    mean = np.zeros(feature_dim, dtype=np.float32)
    std = np.ones(feature_dim, dtype=np.float32)
    for index in range(feature_dim):
        column = values[observed[:, index], index]
        if column.size == 0:
            continue
        mean[index] = float(np.mean(column))
        deviation = float(np.std(column))
        std[index] = max(deviation, 1e-6)
    return mean, std


def _mean_holding_days(trade_log: pd.DataFrame) -> float:
    """Mean completed position duration from the shared-book trade log."""
    if trade_log is None or trade_log.empty:
        return float("nan")
    opened: dict[tuple[str, str], pd.Timestamp] = {}
    durations: list[float] = []
    for row in trade_log.sort_values("date").itertuples(index=False):
        symbol = str(row.symbol).upper()
        action = str(row.action).lower()
        date = pd.Timestamp(row.date)
        if action in {"enter_long", "enter_short"}:
            side = "long" if action.endswith("long") else "short"
            opened[(symbol, side)] = date
        elif action in {"exit_long", "exit_short"}:
            side = "long" if action.endswith("long") else "short"
            start = opened.pop((symbol, side), None)
            if start is not None:
                durations.append(max(0.0, float((date - start).days)))
    return float(np.mean(durations)) if durations else float("nan")


def _attach_holding_days(summary: pd.DataFrame, trade_log: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    summary = summary.copy()
    holding = {}
    if trade_log is not None and not trade_log.empty:
        for (variant, top_k), group in trade_log.groupby(["variant", "top_k"], dropna=False):
            holding[(str(variant), int(top_k))] = _mean_holding_days(group)
    summary["mean_holding_days"] = [
        holding.get((str(row.variant), int(row.top_k)), float("nan"))
        for row in summary.itertuples(index=False)
    ]
    return summary


def _train_epoch(
    model: TransformerMTL,
    docs: list[dict[str, np.ndarray]],
    optimizer: torch.optim.Optimizer,
    gradnorm_optimizer: torch.optim.Optimizer | None = None,
    *,
    pretrain_only: bool = False,
    include_self_supervised: bool | None = None,
) -> float:
    model.train()
    use_self_supervised = SELF_SUPERVISED_ENABLED if include_self_supervised is None else include_self_supervised
    losses: list[float] = []
    for kind in ("symbol_year", "cross_sectional"):
        kind_indices = [index for index, item in enumerate(docs) if item["kind"] == kind]
        order = np.random.permutation(kind_indices)
        for offset in range(0, len(order), BATCH_SIZE):
            selected = [docs[int(i)] for i in order[offset:offset + BATCH_SIZE]]
            x, padding, target = _batch(selected)
            x = x.to(DEVICE); padding = padding.to(DEVICE)
            clean_x = x
            masked_features = None
            masked_token_positions = None
            next_token_positions = None
            if use_self_supervised and model.masked_feature_head is not None:
                # Mask whole families rather than arbitrary scalar columns.
                # Temporal documents remain causal; cross-sectional documents
                # can use the other same-date tokens to reconstruct the mask.
                masked_features = torch.zeros_like(clean_x, dtype=torch.bool)
                for indices in model.family_feature_indices.values():
                    family_selected = torch.rand(clean_x.shape[:2], device=DEVICE).lt(0.08)
                    masked_features[..., list(indices)] |= family_selected.unsqueeze(-1)
                masked_features &= (~padding).unsqueeze(-1)
                masked_features &= clean_x.abs().gt(1e-8)
                x = clean_x.masked_fill(masked_features, 0.0)
            if MASKED_TOKEN_ENABLED and model.masked_token_head is not None:
                # Hide complete token embeddings.  On temporal documents the
                # causal mask means token t can only reconstruct itself from
                # earlier tokens; on cross-sectional documents it can use the
                # other same-date tokens.
                token_valid = (~padding) & clean_x.abs().sum(dim=-1).gt(1e-8)
                masked_token_positions = token_valid & torch.rand_like(token_valid, dtype=torch.float32).lt(MASKED_TOKEN_RATE)
                x = x.masked_fill(masked_token_positions.unsqueeze(-1), 0.0)
            if NEXT_TOKEN_ENABLED and kind == "symbol_year" and model.next_token_head is not None:
                # Predict the next observed token from the current token state.
                # Cross-sectional token order is not temporal, so it is excluded.
                token_valid = (~padding) & clean_x.abs().sum(dim=-1).gt(1e-8)
                next_token_positions = torch.zeros_like(token_valid)
                next_token_positions[:, :-1] = token_valid[:, :-1] & token_valid[:, 1:]
            target = {name: value.to(DEVICE) for name, value in target.items()}
            optimizer.zero_grad()
            graph_hat, speed_hat, event_logits, aux_logits, document_aux_logits, macro_logits, asset_class_logits, cross_graph_hat, cross_speed_hat, cross_year_logits = model(
                x, padding, causal=kind == "symbol_year", modality=target["modality"],
                family_presence=target["family_presence"],
            )
            valid = ~padding
            task_valid = target["task_mask"]
            graph_mask = target["graph_mask"] * valid.unsqueeze(-1) * task_valid[:, 0, None, None]
            graph_error = nn.functional.smooth_l1_loss(graph_hat, target["graph"], reduction="none")
            graph_loss = (graph_error * graph_mask).sum() / graph_mask.sum().clamp_min(1.0)
            speed_valid = valid * task_valid[:, 1, None].bool()
            speed_error = nn.functional.smooth_l1_loss(speed_hat, target["speed"], reduction="none")
            speed_loss = (speed_error * speed_valid.unsqueeze(-1)).sum() / speed_valid.sum().clamp_min(1.0)
            cross_task_valid = target["cross_task_mask"]
            cross_graph_valid = valid * cross_task_valid[:, 0, None].bool()
            cross_graph_error = nn.functional.smooth_l1_loss(cross_graph_hat, target["cross_graph"], reduction="none")
            cross_graph_loss = (cross_graph_error * cross_graph_valid.unsqueeze(-1)).sum() / cross_graph_valid.sum().clamp_min(1.0)
            cross_speed_valid = valid * cross_task_valid[:, 1, None].bool()
            cross_speed_error = nn.functional.smooth_l1_loss(cross_speed_hat, target["cross_speed"], reduction="none")
            cross_speed_loss = (cross_speed_error * cross_speed_valid.unsqueeze(-1)).sum() / cross_speed_valid.sum().clamp_min(1.0)
            cross_year_loss = graph_hat.new_zeros(())
            if cross_year_logits is not None:
                cross_year_valid = (
                    target["cross_year"].ge(0)
                    if kind == "cross_sectional"
                    else torch.zeros_like(target["cross_year"], dtype=torch.bool)
                )
                if cross_year_valid.any():
                    cross_year_loss = nn.functional.cross_entropy(
                        cross_year_logits[cross_year_valid], target["cross_year"][cross_year_valid]
                    )
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
            task_losses = {
                "graph": graph_loss, "speed": speed_loss, "event": event_loss, "aux": aux_loss,
                "cross_graph": cross_graph_loss, "cross_speed": cross_speed_loss,
                "cross_year": cross_year_loss,
            }
            masked_feature_loss = graph_hat.new_zeros(())
            if (
                use_self_supervised
                and model.masked_feature_head is not None
                and masked_features is not None
                and masked_features.any()
            ):
                reconstruction = model.masked_feature_head(model.last_token_state)
                reconstruction_target = model.feature_norm(clean_x).detach()
                reconstruction_error = nn.functional.smooth_l1_loss(
                    reconstruction, reconstruction_target, reduction="none"
                )
                masked_feature_loss = reconstruction_error[masked_features].mean()
            masked_token_loss = graph_hat.new_zeros(())
            if (
                MASKED_TOKEN_ENABLED
                and model.masked_token_head is not None
                and masked_token_positions is not None
                and masked_token_positions.any()
            ):
                token_reconstruction = model.masked_token_head(model.last_token_state)
                token_target = model.feature_norm(clean_x).detach()
                token_error = nn.functional.smooth_l1_loss(
                    token_reconstruction, token_target, reduction="none"
                )
                masked_token_loss = token_error[masked_token_positions].mean()
            next_token_loss = graph_hat.new_zeros(())
            if (
                NEXT_TOKEN_ENABLED
                and model.next_token_head is not None
                and next_token_positions is not None
                and next_token_positions.any()
            ):
                next_prediction = model.next_token_head(model.last_token_state[:, :-1])
                next_target = model.feature_norm(clean_x[:, 1:]).detach()
                next_error = nn.functional.smooth_l1_loss(
                    next_prediction, next_target, reduction="none"
                )
                next_token_loss = next_error[next_token_positions[:, :-1]].mean()
            if asset_class_logits is not None:
                asset_class_valid = target["modality"].gt(0)
                task_losses["asset_class"] = (
                    nn.functional.cross_entropy(asset_class_logits[asset_class_valid], target["modality"][asset_class_valid])
                    if asset_class_valid.any() else graph_hat.new_zeros(())
                )
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
                    default_weights = {"graph": 1.0, "speed": float(os.getenv("TRANSFORMER_SPEED_LOSS_WEIGHT", "0.10")), "event": 1.0, "aux": float(os.getenv("TRANSFORMER_AUX_LOSS_WEIGHT", "0.10")), "asset_class": ASSET_CLASS_LOSS_WEIGHT, "macro": float(os.getenv("TRANSFORMER_MACRO_LOSS_WEIGHT", "1.0")), "cross_graph": float(os.getenv("TRANSFORMER_CROSS_RANK_LOSS_WEIGHT", "1.0")), "cross_speed": float(os.getenv("TRANSFORMER_CROSS_RANK_LOSS_WEIGHT", "1.0")), "cross_year": float(os.getenv("TRANSFORMER_CROSS_YEAR_LOSS_WEIGHT", "1.0"))}
                    weights = torch.tensor(
                        [default_weights[name] for name in model.gradnorm_task_names], device=DEVICE,
                    )
                if pretrain_only:
                    loss = (
                        masked_feature_loss
                        + MASKED_TOKEN_WEIGHT * masked_token_loss
                        + NEXT_TOKEN_WEIGHT * next_token_loss
                    )
                else:
                    loss = sum(
                        weights[index] * task_losses[name]
                        for index, name in enumerate(model.gradnorm_task_names)
                    ) + model.routing_regularization() + (
                        SELF_SUPERVISED_WEIGHT * masked_feature_loss if use_self_supervised else 0.0
                    ) + MASKED_TOKEN_WEIGHT * masked_token_loss + NEXT_TOKEN_WEIGHT * next_token_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else 0.0


def _dual_batch_loss(
    model: DualTowerTransformerMTL,
    issuer_x: torch.Tensor,
    instrument_x: torch.Tensor,
    padding: torch.Tensor,
    target: dict[str, torch.Tensor],
) -> torch.Tensor:
    with _dual_autocast():
        graph_hat, speed_hat, event_logits, _, document_aux_logits, _, asset_logits = model(
            issuer_x, instrument_x, padding, target["modality"],
        )
        valid = target["token_mask"].unsqueeze(-1).to(graph_hat.dtype)
        loss = (nn.functional.smooth_l1_loss(graph_hat, target["graph"], reduction="none") * valid).mean()
        loss = loss + float(os.getenv("TRANSFORMER_SPEED_LOSS_WEIGHT", "0.10")) * (nn.functional.smooth_l1_loss(speed_hat, target["speed"], reduction="none") * valid).mean()
        token_valid = target["token_mask"].expand_as(target["events"][..., 0])
        if token_valid.any():
            loss = loss + gnn.event_loss_from_logits(event_logits[token_valid], target["events"][token_valid])
        for index, name in enumerate(AUX_COLS):
            labels = target["aux"][..., index]; mask = labels.ge(0) & target["candidate_mask"]
            if mask.any():
                loss = loss + float(os.getenv("TRANSFORMER_AUX_LOSS_WEIGHT", "0.10")) * nn.functional.cross_entropy(document_aux_logits[name][mask], labels[mask])
        mask = target["modality"].gt(0) & target["candidate_mask"]
        if mask.any():
            loss = loss + ASSET_CLASS_LOSS_WEIGHT * nn.functional.cross_entropy(asset_logits[mask], target["modality"][mask])
    return loss.float()


def _train_dual_epoch(
    model: DualTowerTransformerMTL,
    pairs: list[dict[str, object]],
    optimizer: torch.optim.Optimizer,
    cached_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]] | None = None,
    mix_match: bool = True,
) -> float:
    model.train(); losses: list[float] = []
    if cached_batches is None:
        groups = _group_same_issuer_pairs(pairs)
        cached_batches = [
            _batch_same_issuer_groups(groups[start:start + BATCH_SIZE])
            for start in range(0, len(groups), BATCH_SIZE)
        ]
    order = np.random.permutation(len(cached_batches))
    for batch_index in order:
        issuer_x, instrument_x, padding, target = cached_batches[int(batch_index)]
        if mix_match:
            issuer_x, instrument_x, padding, target, mixed_batch = _mix_batch_in_memory(issuer_x, instrument_x, padding, target)
        else:
            mixed_batch = None
        issuer_x = issuer_x.to(DEVICE); instrument_x = instrument_x.to(DEVICE); padding = padding.to(DEVICE)
        target = {name: value.to(DEVICE) for name, value in target.items()}
        optimizer.zero_grad()
        loss = _dual_batch_loss(model, issuer_x, instrument_x, padding, target)
        if mixed_batch is not None:
            mixed_issuer_x, mixed_instrument_x, mixed_padding, mixed_target = mixed_batch
            mixed_issuer_x = mixed_issuer_x.to(DEVICE); mixed_instrument_x = mixed_instrument_x.to(DEVICE); mixed_padding = mixed_padding.to(DEVICE)
            mixed_target = {name: value.to(DEVICE) for name, value in mixed_target.items()}
            # The mixed examples use the same existing MTL objectives; this is
            # not a separately weighted compatibility or retrieval task.
            loss = (loss + _dual_batch_loss(model, mixed_issuer_x, mixed_instrument_x, mixed_padding, mixed_target)) / 2
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite dual-tower loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else 0.0


def _run_dual_tier(tier: str) -> pd.DataFrame:
    started = perf_counter(); base, prices, feature_cols, related_panel, _ = _prepare_data(tier)
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index().ffill(); next_returns = close.pct_change().shift(-1)
    summaries: list[pd.DataFrame] = []; effective_top_k = min(20, len(close.columns))
    related_prefixes = tuple(f"{name}__" for name in ASSET_CLASSES)
    issuer_feature_mask = np.ones(len(feature_cols), dtype=np.float32)
    if not ISSUER_EQUITY_ENABLED:
        issuer_feature_mask = np.asarray(
            [float(str(column).startswith(related_prefixes)) for column in feature_cols],
            dtype=np.float32,
        )
    for test_year in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1):
        normalized = _normalize(base, feature_cols, test_year)
        related = _normalize_related_panel(related_panel, feature_cols, test_year)
        issuer_train_docs = _make_docs(normalized, test_year, True, issuer_feature_mask)
        equity_train_docs = _make_docs(normalized, test_year, True)
        related_train_docs = _make_related_docs(normalized, related, feature_cols, test_year, True)
        train_pairs = _make_same_issuer_pairs(issuer_train_docs, related_train_docs, equity_train_docs)
        test_issuers = _make_docs(normalized, test_year, False, issuer_feature_mask)
        equity_test_docs = _make_docs(normalized, test_year, False)
        test_pairs = _make_same_issuer_pairs(test_issuers, [], equity_test_docs)
        if not train_pairs or not test_pairs: continue
        mean, std = _fit_feature_stats([pair["issuer"] for pair in train_pairs], len(feature_cols))
        dims = {name: int(normalized[name].max()) + 1 for name in AUX_COLS}
        robust = np.asarray([_is_price_volume_feature(column) for column in feature_cols], dtype=bool)
        model = DualTowerTransformerMTL(len(feature_cols), len(feature_cols), dims, issuer_mean=mean, issuer_std=std, issuer_robust_mask=robust, instrument_mean=mean, instrument_std=std, instrument_robust_mask=robust).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        print({"tier": tier, "year": test_year, "dual_tower": True, "issuer_equity": ISSUER_EQUITY_ENABLED, "pairs": len(train_pairs), "mix_match": "in_memory_random_same_date", "issuer_trunks": ISSUER_TRUNKS, "instrument_trunks": INSTRUMENT_TRUNKS}, flush=True)
        train_groups = _group_same_issuer_pairs(train_pairs)
        train_batches = [
            _batch_same_issuer_groups(train_groups[start:start + BATCH_SIZE])
            for start in range(0, len(train_groups), BATCH_SIZE)
        ]
        for epoch in range(DUAL_EPOCHS):
            mix_match = epoch >= MIX_MATCH_START_EPOCH
            loss = _train_dual_epoch(model, train_pairs, optimizer, train_batches, mix_match=mix_match)
            if epoch in (0, MIX_MATCH_START_EPOCH - 1, MIX_MATCH_START_EPOCH, DUAL_EPOCHS - 1):
                print({"tier": tier, "year": test_year, "epoch": epoch + 1, "dual_loss": round(loss, 5), "mix_match": mix_match}, flush=True)
        model.eval(); rows: list[pd.DataFrame] = []; test_groups = _group_same_issuer_pairs(test_pairs)
        with torch.no_grad():
            for start in range(0, len(test_groups), BATCH_SIZE):
                b = test_groups[start:start + BATCH_SIZE]; ix, ax, pad, target = _batch_same_issuer_groups(b)
                with _dual_autocast():
                    gh = model(ix.to(DEVICE), ax.to(DEVICE), pad.to(DEVICE), target["modality"].to(DEVICE))[0]
                gh = gh.float().cpu().numpy()
                for row, group in enumerate(b):
                    count = len(group["issuer"]["x"]); frame = pd.DataFrame({"symbol": group["issuer"]["symbol"][:count], "date": group["issuer"]["date"][:count]}); frame[TARGET_COLS] = gh[row, 0, :count]; rows.append(frame)
        pred = pd.concat(rows, ignore_index=True)
        for column in TARGET_COLS: pred[column] = pred.groupby("date")[column].rank(pct=True, method="average")
        pred["long_score"], pred["long_exit_score"] = pred.long_hub, pred.long_authority; pred["short_score"], pred["short_exit_score"] = pred.short_hub, pred.short_authority; pred["long_agree_count"] = (pred.long_score >= pred.short_score).astype(int); pred["short_agree_count"] = (pred.short_score > pred.long_score).astype(int); pred["model_count"] = 1
        dates = pd.DatetimeIndex(next_returns.index[(next_returns.index >= f"{test_year}-01-01") & (next_returns.index <= f"{test_year}-12-31")])
        summary, _, _ = gnn.run_shared_book_framework_comparison(scores=pred[["symbol", "date", "long_score", "short_score", "long_exit_score", "short_exit_score", "long_agree_count", "short_agree_count", "model_count"]], next_returns=next_returns, symbols=tuple(close.columns), dates=dates, variants=BACKTEST_VARIANTS, top_k_values=(effective_top_k,), entry_threshold=.5, exit_threshold=.5, cost_models={"family_common": gnn.SharedBookCostModel(.5, 5.)})
        summary["tier"] = tier; summary["year"] = test_year; summary["family"] = f"dual_tower_{ISSUER_TRUNKS}issuer_{ISSUER_EQUITY_ENABLED}equity_{INSTRUMENT_TRUNKS}instrument"; summaries.append(summary)
    result = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    result_path = OUT / f"{tier.lower()}_dual_tower_{ISSUER_TRUNKS}issuer_{ISSUER_EQUITY_ENABLED}equity_{INSTRUMENT_TRUNKS}instrument_wfo_results.csv"
    result.to_csv(result_path, index=False)
    print(result.to_string(index=False)); print({"tier": tier, "dual_seconds": round(perf_counter() - started, 1), "result_path": str(result_path)}, flush=True); return result


def _run_single_fit_tier(tier: str) -> pd.DataFrame:
    """Smoke mode: fit once through the pre-2021 cutoff, then score all OOS years."""
    started = perf_counter()
    base, prices, feature_cols, related_panel, etf_panel = _prepare_data(tier)
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    next_returns = close.pct_change().shift(-1)
    effective_top_k = min(20, len(close.columns))
    representation_label = "_single_fit"
    if COVERAGE_AWARE_FAMILY_ADAPTERS:
        representation_label += "_coverage_family_adapters"
    if MASKED_TOKEN_ENABLED:
        representation_label += "_masked_token"
    if NEXT_TOKEN_ENABLED:
        representation_label += "_next_token"
    representation_label += os.getenv("TRANSFORMER_RUN_SUFFIX", "")
    normalized = _normalize(base, feature_cols, FIRST_TEST_YEAR)
    train_docs = _make_docs(normalized, FIRST_TEST_YEAR, True)
    normalized_related = _normalize_related_panel(related_panel, feature_cols, FIRST_TEST_YEAR)
    related_train_docs = _make_related_docs(normalized, normalized_related, feature_cols, FIRST_TEST_YEAR, True)
    normalized_etf = _normalize_related_panel(etf_panel, feature_cols, FIRST_TEST_YEAR)
    etf_train_docs = _make_etf_docs(normalized_etf, feature_cols, FIRST_TEST_YEAR, True)
    cross_enabled = CROSS_SECTIONAL_ENABLED and (bool(MACRO_EVENT_COLS) or "year_target" in AUX_COLS)
    cross_train_docs = _make_cross_sectional_docs(normalized, FIRST_TEST_YEAR, True) if cross_enabled else []
    test_docs_by_year = {
        year: _make_docs(normalized, year, False)
        for year in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1)
    }
    cross_test_by_year = {
        year: _make_cross_sectional_docs(normalized, year, False) if cross_enabled else []
        for year in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1)
    }
    training_docs = train_docs + cross_train_docs + related_train_docs + etf_train_docs
    if not training_docs or not any(test_docs_by_year.values()):
        return pd.DataFrame()
    feature_mean, feature_std = _fit_feature_stats(training_docs, len(feature_cols))
    aux_dims = {name: int(normalized[name].max()) + 1 for name in AUX_COLS}
    asset_feature_indices = {
        name: tuple(index for index, column in enumerate(feature_cols) if str(column).startswith(f"{name}__"))
        for name in ASSET_CLASSES
    }
    family_feature_indices = {}
    for index, column in enumerate(feature_cols):
        family = str(column).split("__", 1)[0]
        family_feature_indices.setdefault(family, []).append(index)
    family_feature_indices = {
        family: tuple(indices) for family, indices in family_feature_indices.items() if indices
    }
    robust_feature_mask = np.asarray([_is_price_volume_feature(column) for column in feature_cols], dtype=bool)
    all_docs = training_docs + [doc for docs in test_docs_by_year.values() for doc in docs]
    all_docs += [doc for docs in cross_test_by_year.values() for doc in docs]
    max_position = max((len(doc["x"]) for doc in all_docs), default=512)
    model = TransformerMTL(
        len(feature_cols), aux_dims, asset_feature_indices=asset_feature_indices,
        family_feature_indices=family_feature_indices,
        feature_mean=feature_mean, feature_std=feature_std,
        robust_feature_mask=robust_feature_mask, max_position=max_position,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    gradnorm_optimizer = torch.optim.Adam([model.gradnorm_log_weights], lr=GRADNORM_LR) if GRADNORM_ENABLED else None
    print({"tier": tier, "single_fit": True, "train_through": str(pd.Timestamp(f"{FIRST_TEST_YEAR - 1}-12-31").date()), "train_documents": len(train_docs), "cross_train_documents": len(cross_train_docs), "trunks": TRUNKS, "masked_token": MASKED_TOKEN_ENABLED, "next_token": NEXT_TOKEN_ENABLED, "device": str(DEVICE)}, flush=True)
    for epoch in range(SELF_SUPERVISED_PRETRAIN_EPOCHS):
        loss = _train_epoch(
            model, training_docs, optimizer, gradnorm_optimizer,
            pretrain_only=True, include_self_supervised=True,
        )
        print({"tier": tier, "single_fit": True, "pretrain_epoch": epoch + 1, "masked_reconstruction_loss": round(loss, 5)}, flush=True)
    for epoch in range(EPOCHS):
        loss = _train_epoch(
            model, training_docs, optimizer, gradnorm_optimizer,
            include_self_supervised=False,
        )
        if epoch == 0 or epoch == EPOCHS - 1:
            print({"tier": tier, "single_fit": True, "epoch": epoch + 1, "transformer_loss": round(loss, 5)}, flush=True)
    model.eval()
    summaries: list[pd.DataFrame] = []
    for test_year in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1):
        test_docs = test_docs_by_year[test_year]
        cross_test_docs = cross_test_by_year[test_year]
        temporal_predictions: list[pd.DataFrame] = []
        temporal_speed_predictions: list[pd.DataFrame] = []
        with torch.no_grad():
            for start in range(0, len(test_docs), BATCH_SIZE):
                batch_docs = test_docs[start:start + BATCH_SIZE]
                x, padding, batch_target = _batch(batch_docs)
                outputs = model(
                    x.to(DEVICE), padding.to(DEVICE), modality=batch_target["modality"].to(DEVICE),
                    family_presence=batch_target["family_presence"].to(DEVICE),
                )
                values = outputs[0].cpu().numpy()
                speed_values = outputs[1].cpu().numpy()
                for row, doc in enumerate(batch_docs):
                    n = len(doc["x"])
                    frame = pd.DataFrame({"symbol": doc["symbol"], "date": doc["date"]})
                    frame[TARGET_COLS] = values[row, :n]
                    temporal_predictions.append(frame)
                    if SPEED_STRATEGY_ENABLED:
                        speed_frame = pd.DataFrame({"symbol": doc["symbol"], "date": doc["date"]})
                        speed_frame[TARGET_COLS] = speed_values[row, :n, :len(TARGET_COLS)]
                        temporal_speed_predictions.append(speed_frame)
        head_predictions = {"temporal": pd.concat(temporal_predictions, ignore_index=True)}
        if SPEED_STRATEGY_ENABLED:
            head_predictions["temporal_speed"] = pd.concat(temporal_speed_predictions, ignore_index=True)
        if CROSS_SECTIONAL_COMPARE_HEADS or CROSS_SECTIONAL_TRADING:
            cross_predictions: list[pd.DataFrame] = []
            with torch.no_grad():
                for start in range(0, len(cross_test_docs), BATCH_SIZE):
                    batch_docs = cross_test_docs[start:start + BATCH_SIZE]
                    x, padding, batch_target = _batch(batch_docs)
                    outputs = model(
                        x.to(DEVICE), padding.to(DEVICE), causal=False,
                        modality=batch_target["modality"].to(DEVICE),
                        family_presence=batch_target["family_presence"].to(DEVICE),
                    )
                    values = outputs[7].cpu().numpy()
                    for row, doc in enumerate(batch_docs):
                        n = len(doc["x"])
                        frame = pd.DataFrame({"symbol": doc["symbol"], "date": doc["date"]})
                        frame[TARGET_COLS] = values[row, :n]
                        cross_predictions.append(frame)
            cross_pred = pd.concat(cross_predictions, ignore_index=True)
            for column in TARGET_COLS:
                cross_pred[column] = cross_pred.groupby("date")[column].rank(pct=True, method="average")
            head_predictions["cross_sectional"] = cross_pred
            if SPEED_STRATEGY_ENABLED:
                cross_speed_pred = cross_pred.copy()
                # Cross-sectional speed scores are produced by the speed head
                # under the same bidirectional document attention.
                cross_speed_predictions: list[pd.DataFrame] = []
                with torch.no_grad():
                    for start in range(0, len(cross_test_docs), BATCH_SIZE):
                        batch_docs = cross_test_docs[start:start + BATCH_SIZE]
                        x, padding, batch_target = _batch(batch_docs)
                        outputs = model(
                            x.to(DEVICE), padding.to(DEVICE), causal=False,
                            modality=batch_target["modality"].to(DEVICE),
                            family_presence=batch_target["family_presence"].to(DEVICE),
                        )
                        speed_values = outputs[8].cpu().numpy()
                        for row, doc in enumerate(batch_docs):
                            n = len(doc["x"])
                            frame = pd.DataFrame({"symbol": doc["symbol"], "date": doc["date"]})
                            frame[TARGET_COLS] = speed_values[row, :n, :len(TARGET_COLS)]
                            cross_speed_predictions.append(frame)
                cross_speed_pred = pd.concat(cross_speed_predictions, ignore_index=True)
                for column in TARGET_COLS:
                    cross_speed_pred[column] = cross_speed_pred.groupby("date")[column].rank(pct=True, method="average")
                head_predictions["cross_sectional_speed"] = cross_speed_pred
        if CROSS_SECTIONAL_COMPARE_HEADS:
            selected_heads = head_predictions.items()
        elif CROSS_SECTIONAL_TRADING:
            selected_heads = (("cross_sectional", head_predictions["cross_sectional"]),)
        else:
            selected_heads = (("temporal", head_predictions["temporal"]),)
            if SPEED_STRATEGY_ENABLED:
                selected_heads = (*selected_heads, ("temporal_speed", head_predictions["temporal_speed"]))
        dates = pd.DatetimeIndex(next_returns.index[(next_returns.index >= f"{test_year}-01-01") & (next_returns.index <= f"{test_year}-12-31")])
        for head_name, pred in selected_heads:
            for column in TARGET_COLS:
                pred[column] = pred.groupby("date")[column].rank(pct=True, method="average")
            pred["long_score"], pred["long_exit_score"] = pred.long_hub, pred.long_authority
            pred["short_score"], pred["short_exit_score"] = pred.short_hub, pred.short_authority
            pred["long_agree_count"] = (pred.long_score >= pred.short_score).astype(int)
            pred["short_agree_count"] = (pred.short_score > pred.long_score).astype(int)
            pred["model_count"] = 1
            summary, trade_log, _ = gnn.run_shared_book_framework_comparison(
                scores=pred[["symbol", "date", "long_score", "short_score", "long_exit_score", "short_exit_score", "long_agree_count", "short_agree_count", "model_count"]],
                next_returns=next_returns, symbols=tuple(close.columns), dates=dates,
                variants=BACKTEST_VARIANTS, top_k_values=(effective_top_k,), entry_threshold=0.5,
                exit_threshold=0.5, cost_models={"family_common": gnn.SharedBookCostModel(0.5, 5.0)},
            )
            summary = _attach_holding_days(summary, trade_log)
            if not summary.empty:
                summary["tier"] = tier; summary["year"] = test_year
                summary["family"] = f"causal_symbol_year_transformer_{TRUNKS}trunk_single_fit_{head_name}head"
                summary["label_source"] = "transformer_hits_single_fit_smoke"
                summaries.append(summary)
    result = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    run_suffix = os.getenv("TRANSFORMER_RUN_SUFFIX", "")
    path = OUT / f"{tier.lower()}_transformer_single_fit_hits_smoke{run_suffix}_wfo_results.csv"
    result.to_csv(path, index=False)
    print(result.to_string(index=False) if not result.empty else result)
    print({"tier": tier, "single_fit": True, "seconds": round(perf_counter() - started, 1), "result_rows": len(result), "result_path": str(path)}, flush=True)
    return result


def run_tier(tier: str) -> pd.DataFrame:
    if DUAL_TOWER_ENABLED:
        return _run_dual_tier(tier)
    if os.getenv("TRANSFORMER_SINGLE_FIT", "0") == "1":
        return _run_single_fit_tier(tier)
    started = perf_counter()
    if CROSS_SECTIONAL_TRADING and not CROSS_SECTIONAL_ENABLED:
        raise ValueError("TRANSFORMER_CROSS_TRADING=1 requires TRANSFORMER_CROSS_SECTIONAL=1")
    base, prices, feature_cols, related_panel, etf_panel = _prepare_data(tier)
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    next_returns = close.pct_change().shift(-1)
    effective_top_k = min(20, len(close.columns))
    summaries: list[pd.DataFrame] = []
    representation_label = "_related_rows" if RELATED_ASSETS_ENABLED and RELATED_ASSETS_AS_ROWS else ""
    if ETF_CORPUS_ENABLED:
        representation_label += "_etf_corpus"
    if COVERAGE_AWARE_FAMILY_ADAPTERS:
        representation_label += "_coverage_family_adapters"
    if MASKED_TOKEN_ENABLED:
        representation_label += "_masked_token"
    if NEXT_TOKEN_ENABLED:
        representation_label += "_next_token"
    representation_label += os.getenv("TRANSFORMER_RUN_SUFFIX", "")
    for test_year in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1):
        normalized = _normalize(base, feature_cols, test_year)
        train_docs = _make_docs(normalized, test_year, True)
        test_docs = _make_docs(normalized, test_year, False)
        normalized_related = _normalize_related_panel(related_panel, feature_cols, test_year)
        related_train_docs = _make_related_docs(normalized, normalized_related, feature_cols, test_year, True)
        normalized_etf = _normalize_related_panel(etf_panel, feature_cols, test_year)
        etf_train_docs = _make_etf_docs(normalized_etf, feature_cols, test_year, True)
        cross_enabled = CROSS_SECTIONAL_ENABLED and (bool(MACRO_EVENT_COLS) or "year_target" in AUX_COLS)
        cross_train_docs = _make_cross_sectional_docs(normalized, test_year, True) if cross_enabled else []
        cross_test_docs = _make_cross_sectional_docs(normalized, test_year, False) if cross_enabled else []
        if not train_docs or not test_docs:
            continue
        training_docs = train_docs + cross_train_docs + related_train_docs + etf_train_docs
        feature_mean, feature_std = _fit_feature_stats(training_docs, len(feature_cols))
        aux_dims = {name: int(normalized[name].max()) + 1 for name in AUX_COLS}
        asset_feature_indices = {
            name: tuple(index for index, column in enumerate(feature_cols) if str(column).startswith(f"{name}__"))
            for name in ASSET_CLASSES
        }
        family_feature_indices = {}
        for index, column in enumerate(feature_cols):
            family = str(column).split("__", 1)[0]
            family_feature_indices.setdefault(family, []).append(index)
        family_feature_indices = {
            family: tuple(indices) for family, indices in family_feature_indices.items() if indices
        }
        robust_feature_mask = np.asarray([_is_price_volume_feature(column) for column in feature_cols], dtype=bool)
        max_position = max(
            (len(document["x"]) for document in (
                train_docs + test_docs + cross_train_docs + cross_test_docs
                + related_train_docs + etf_train_docs
            )),
            default=512,
        )
        model = TransformerMTL(
            len(feature_cols), aux_dims, asset_feature_indices=asset_feature_indices,
            family_feature_indices=family_feature_indices,
            feature_mean=feature_mean, feature_std=feature_std,
            robust_feature_mask=robust_feature_mask, max_position=max_position,
        ).to(DEVICE)
        main_parameters = [
            parameter for name, parameter in model.named_parameters()
            if name != "gradnorm_log_weights"
        ]
        optimizer = torch.optim.AdamW(main_parameters, lr=LR, weight_decay=1e-4)
        gradnorm_optimizer = torch.optim.Adam(
            [model.gradnorm_log_weights], lr=GRADNORM_LR
        ) if GRADNORM_ENABLED else None
        architecture = "shared_mixer_plus_linear_cross_set_context" if CROSS_SECTIONAL_SET_CONTEXT else "shared_mixer_cross_presence_causal_conv" if SHARED_MIXER_ENHANCEMENTS else "shared_low_rank_feature_mixer_plus_shared_trunks" if SHARED_FEATURE_MIXER else "family_cross_interactions_plus_shared_trunks" if FAMILY_INTERACTIONS else "shared_equity_plus_gated_instrument_adapters" if HYBRID_INSTRUMENT_ADAPTER else "separate_related_and_etf_document_corpora"
        print({"tier": tier, "year": test_year, "documents": len(train_docs), "related_documents": len(related_train_docs), "etf_documents": len(etf_train_docs), "cross_sectional_train_documents": len(cross_train_docs), "test_documents": len(test_docs), "cross_sectional_test_documents": len(cross_test_docs), "tokens": sum(len(d["x"]) for d in train_docs), "related_tokens": sum(len(d["x"]) for d in related_train_docs), "etf_tokens": sum(len(d["x"]) for d in etf_train_docs), "macro_tasks": len(MACRO_EVENT_COLS), "trunks": TRUNKS, "masked_token": MASKED_TOKEN_ENABLED, "next_token": NEXT_TOKEN_ENABLED, "routing": ROUTING_MODE, "gradnorm": GRADNORM_ENABLED, "architecture": architecture, "device": str(DEVICE)}, flush=True)
        for epoch in range(SELF_SUPERVISED_PRETRAIN_EPOCHS):
            loss = _train_epoch(
                model, training_docs, optimizer, gradnorm_optimizer,
                pretrain_only=True, include_self_supervised=True,
            )
            print({"tier": tier, "year": test_year, "pretrain_epoch": epoch + 1, "masked_reconstruction_loss": round(loss, 5)}, flush=True)
        for epoch in range(EPOCHS):
            loss = _train_epoch(
                model, training_docs, optimizer, gradnorm_optimizer,
                include_self_supervised=False,
            )
            if epoch == 0 or epoch == EPOCHS - 1:
                print({"tier": tier, "year": test_year, "epoch": epoch + 1, "transformer_loss": round(loss, 5)}, flush=True)
        print({"tier": tier, "year": test_year, "task_trunk_weights": model.routing_weights(), "task_family_weights": model.task_family_weights(), "gradnorm_weights": model.gradnorm_weight_values()}, flush=True)
        checkpoint_dir = OUT / "transformer_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "tier": tier, "test_year": test_year, "model_state_dict": model.state_dict(),
            "feature_cols": feature_cols, "feature_mean": feature_mean, "feature_std": feature_std,
            "aux_dims": aux_dims, "trunks": TRUNKS, "cross_sectional": bool(cross_enabled),
            "masked_token": MASKED_TOKEN_ENABLED, "next_token": NEXT_TOKEN_ENABLED,
        }, checkpoint_dir / f"{tier.lower()}_{test_year}_2trunk_cross_year.pt")
        model.eval()
        predictions: list[pd.DataFrame] = []
        with torch.no_grad():
            for start in range(0, len(test_docs), BATCH_SIZE):
                batch_docs = test_docs[start:start + BATCH_SIZE]
                x, padding, batch_target = _batch(batch_docs)
                x = x.to(DEVICE); padding = padding.to(DEVICE)
                graph_hat, speed_hat, _, _, _, _, _, _, _, _ = model(
                    x, padding, modality=batch_target["modality"].to(DEVICE),
                    family_presence=batch_target["family_presence"].to(DEVICE),
                )
                values = graph_hat.cpu().numpy()
                speed_values = speed_hat.cpu().numpy()
                for row, doc in enumerate(batch_docs):
                    n = len(doc["x"])
                    pred = pd.DataFrame({"symbol": doc["symbol"], "date": doc["date"]})
                    pred[TARGET_COLS] = values[row, :n]
                    pred[SPEED_TARGET_COLS] = speed_values[row, :n]
                    predictions.append(pred)
        temporal_pred = pd.concat(predictions, ignore_index=True)
        head_predictions = {"temporal": temporal_pred}
        if SPEED_STRATEGY_ENABLED:
            head_predictions["temporal_speed"] = temporal_pred.copy()
        if CROSS_SECTIONAL_TRADING or CROSS_SECTIONAL_COMPARE_HEADS:
            # Evaluate the cross-sectional head directly.  Each date is one
            # bidirectional document, so the scores are produced jointly for
            # all equities on that date rather than copied from temporal
            # symbol/year documents.
            cross_predictions: list[pd.DataFrame] = []
            with torch.no_grad():
                for start in range(0, len(cross_test_docs), BATCH_SIZE):
                    batch_docs = cross_test_docs[start:start + BATCH_SIZE]
                    x, padding, batch_target = _batch(batch_docs)
                    x = x.to(DEVICE); padding = padding.to(DEVICE)
                    outputs = model(
                        x, padding, causal=False,
                        modality=batch_target["modality"].to(DEVICE),
                        family_presence=batch_target["family_presence"].to(DEVICE),
                    )
                    cross_values = outputs[7].cpu().numpy()
                    cross_speed_values = outputs[8].cpu().numpy()
                    for row, doc in enumerate(batch_docs):
                        n = len(doc["x"])
                        cross_pred = pd.DataFrame({"symbol": doc["symbol"], "date": doc["date"]})
                        cross_pred[TARGET_COLS] = cross_values[row, :n]
                        cross_pred[SPEED_TARGET_COLS] = cross_speed_values[row, :n]
                        cross_predictions.append(cross_pred)
            cross_pred = pd.concat(cross_predictions, ignore_index=True)
            for column in TARGET_COLS:
                cross_pred[column] = cross_pred.groupby("date")[column].rank(pct=True, method="average")
            head_predictions["cross_sectional"] = cross_pred
            if SPEED_STRATEGY_ENABLED:
                head_predictions["cross_sectional_speed"] = cross_pred.copy()
        if CROSS_SECTIONAL_COMPARE_HEADS:
            selected_heads = head_predictions.items()
        elif CROSS_SECTIONAL_TRADING:
            selected_heads = [("cross_sectional", head_predictions["cross_sectional"])]
            if SPEED_STRATEGY_ENABLED:
                selected_heads.append(("cross_sectional_speed", head_predictions["cross_sectional_speed"]))
        else:
            selected_heads = [("temporal", temporal_pred)]
            if SPEED_STRATEGY_ENABLED:
                selected_heads.append(("temporal_speed", head_predictions["temporal_speed"]))
        hybrid_label = "_hybrid_adapter" if HYBRID_INSTRUMENT_ADAPTER else ""
        dates = pd.DatetimeIndex(next_returns.index[(next_returns.index >= f"{test_year}-01-01") & (next_returns.index <= f"{test_year}-12-31")])
        for head_name, pred in selected_heads:
            is_speed_strategy = head_name.endswith("_speed")
            strategy_targets = SPEED_TARGET_COLS if is_speed_strategy else TARGET_COLS
            for column in strategy_targets:
                pred[column] = pred.groupby("date")[column].rank(pct=True, method="average")
            long_hub = "speed_long_hub" if is_speed_strategy else "long_hub"
            long_authority = "speed_long_authority" if is_speed_strategy else "long_authority"
            short_hub = "speed_short_hub" if is_speed_strategy else "short_hub"
            short_authority = "speed_short_authority" if is_speed_strategy else "short_authority"
            pred["long_score"], pred["long_exit_score"] = pred[long_hub], pred[long_authority]
            pred["short_score"], pred["short_exit_score"] = pred[short_hub], pred[short_authority]
            pred["long_agree_count"] = (pred.long_score >= pred.short_score).astype(int)
            pred["short_agree_count"] = (pred.short_score > pred.long_score).astype(int)
            pred["model_count"] = 1
            strategy_label = f"_{head_name}head" if CROSS_SECTIONAL_COMPARE_HEADS else ("_crosshead" if CROSS_SECTIONAL_TRADING else "")
            strategy_label += "_speed" if is_speed_strategy and not strategy_label.endswith("_speed") else ""
            strategy_kind = "speed_hits" if is_speed_strategy else "return_hits"
            prediction_label = ("transformer_macro" if MACRO_ENABLED else "transformer") + f"_{TRUNKS}trunk_{ROUTING_MODE}routing_speedhits{representation_label}{hybrid_label}{strategy_label}"
            pred.to_parquet(OUT / f"{tier.lower()}_{prediction_label}_predictions_through_{test_year}.parquet", index=False)
            summary, trade_log, _ = gnn.run_shared_book_framework_comparison(
                scores=pred[["symbol", "date", "long_score", "short_score", "long_exit_score", "short_exit_score", "long_agree_count", "short_agree_count", "model_count"]],
                next_returns=next_returns, symbols=tuple(close.columns), dates=dates,
                variants=BACKTEST_VARIANTS, top_k_values=(effective_top_k,), entry_threshold=0.5,
                exit_threshold=0.5, cost_models={"family_common": gnn.SharedBookCostModel(0.5, 5.0)},
            )
            summary = _attach_holding_days(summary, trade_log)
            if not summary.empty:
                summary["tier"] = tier
                summary["year"] = test_year
                summary["family"] = f"causal_symbol_year_transformer_{TRUNKS}trunk_{ROUTING_MODE}routing_{strategy_kind}{representation_label}{strategy_label}"
                summary["label_source"] = ("transformer_macro_mtl" if MACRO_ENABLED else "transformer_baseline_mtl") + f"_{TRUNKS}trunk_{ROUTING_MODE}routing_{strategy_kind}{representation_label}_{head_name}head" + ("_gradnorm" if GRADNORM_ENABLED else "")
                summaries.append(summary)
            run_label = ("transformer_macro" if MACRO_ENABLED else "transformer") + f"_{TRUNKS}trunk_{ROUTING_MODE}routing_speedhits{representation_label}{strategy_label}" + ("_gradnorm" if GRADNORM_ENABLED else "")
            pd.concat(summaries, ignore_index=True).to_parquet(OUT / f"{tier.lower()}_{run_label}_{BACKTEST_VARIANT_TAG}_through_{test_year}.parquet", index=False)
        # Large universes retain millions of token arrays per fold.  Release
        # the completed fold before constructing the next year's documents.
        del model, optimizer, training_docs, train_docs, test_docs
        del cross_train_docs, cross_test_docs, related_train_docs, etf_train_docs
        del normalized, normalized_related, normalized_etf, predictions, head_predictions
        del selected_heads, pred, temporal_pred, cross_pred, cross_predictions
        del batch_docs, x, padding, batch_target, values, cross_values
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    result = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    run_label = ("transformer_macro" if MACRO_ENABLED else "transformer") + f"_{TRUNKS}trunk_{ROUTING_MODE}routing_speedhits{representation_label}{('_compare_heads' if CROSS_SECTIONAL_COMPARE_HEADS else ('_crosshead' if CROSS_SECTIONAL_TRADING else ''))}" + ("_gradnorm" if GRADNORM_ENABLED else "")
    result.to_csv(OUT / f"{tier.lower()}_{run_label}_{BACKTEST_VARIANT_TAG}_wfo_results.csv", index=False)
    print(result.to_string(index=False) if not result.empty else result)
    print({"tier": tier, "transformer_seconds": round(perf_counter() - started, 1), "result_rows": len(result)}, flush=True)
    return result


def main() -> None:
    requested = tuple(x.strip().upper() for x in os.getenv("TRANSFORMER_TIERS", "1T").split(",") if x.strip())
    for tier in requested:
        run_tier(tier)


if __name__ == "__main__":
    main()
