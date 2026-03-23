from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import math
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ml.metrics import (
    build_action_classification_report_df,
    build_action_f1_comparison_df,
    build_flair_action_report,
    build_regression_task_report_df,
    format_metric_report,
    safe_accuracy,
    safe_macro_f1,
    safe_mae,
    safe_mean,
    safe_rmse,
    safe_spearman,
)


@dataclass(frozen=True)
class ContextFamilyMTLDataSpec:
    state_market_cols: Sequence[str]
    state_fundamental_cols: Sequence[str]
    state_macro_cols: Sequence[str]
    entry_market_cols: Sequence[str]
    entry_fundamental_cols: Sequence[str]
    entry_macro_cols: Sequence[str]
    exit_market_cols: Sequence[str]
    exit_fundamental_cols: Sequence[str]
    exit_macro_cols: Sequence[str]
    state_market_stats: dict[str, dict[str, float]]
    state_fundamental_stats: dict[str, dict[str, float]]
    state_macro_stats: dict[str, dict[str, float]]
    exit_market_stats: dict[str, dict[str, float]]
    exit_fundamental_stats: dict[str, dict[str, float]]
    exit_macro_stats: dict[str, dict[str, float]]
    entry_action_to_id: dict[str, int]
    exit_action_to_id: dict[str, int]

    @property
    def id_to_entry_action(self) -> dict[int, str]:
        return {value: key for key, value in self.entry_action_to_id.items()}

    @property
    def id_to_exit_action(self) -> dict[int, str]:
        return {value: key for key, value in self.exit_action_to_id.items()}


def load_local_first_transformer(model_name: str):
    try:
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
        print(f"Loaded transformer backbone from local cache: {model_name}")
        return model
    except Exception as exc:
        print(f"Local transformer cache unavailable for {model_name}: {exc}")
        print("Falling back to the default Hugging Face loader.")
        return AutoModel.from_pretrained(model_name)


def load_local_first_tokenizer(model_name: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        print(f"Loaded tokenizer from local cache: {model_name}")
        return tokenizer
    except Exception as exc:
        print(f"Local tokenizer cache unavailable for {model_name}: {exc}")
        print("Falling back to the default Hugging Face loader.")
        return AutoTokenizer.from_pretrained(model_name)


def resolve_torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def finite_or_zero(value: Any) -> float:
    return 0.0 if value is None or math.isnan(float(value)) else float(value)


def smooth_l1_compat(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(prediction.contiguous(), target.contiguous())


def frame_batches(frame: pd.DataFrame, batch_size: int, shuffle: bool = True, seed: int = 42):
    indices = np.arange(len(frame))
    if shuffle and len(indices) > 0:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield frame.iloc[batch_idx].reset_index(drop=True)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return masked.sum(dim=1) / denom


def batch_numeric_tensor(frame: pd.DataFrame, cols: Sequence[str], device: torch.device) -> torch.Tensor:
    if not cols:
        return torch.zeros((len(frame), 0), dtype=torch.float32, device=device)
    return torch.tensor(frame[list(cols)].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)


def batch_standardized_numeric_tensor(
    frame: pd.DataFrame,
    cols: Sequence[str],
    stats: dict[str, dict[str, float]],
    device: torch.device,
) -> torch.Tensor:
    if not cols:
        return torch.zeros((len(frame), 0), dtype=torch.float32, device=device)
    raw = frame[list(cols)].to_numpy(dtype=np.float32)
    means = np.asarray([float(stats["mean"].get(col, 0.0)) for col in cols], dtype=np.float32)
    stds = np.asarray([max(float(stats["std"].get(col, 1.0)), 1e-6) for col in cols], dtype=np.float32)
    scaled = (raw - means) / stds
    return torch.tensor(scaled, dtype=torch.float32, device=device)


class FTNumericEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        token_dim: int = 64,
        output_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.token_dim = int(token_dim)
        self.output_dim = int(output_dim)
        if self.input_dim == 0:
            self.register_parameter("value_weight", None)
            self.register_parameter("value_bias", None)
            self.register_parameter("feature_embedding", None)
            self.register_parameter("cls_token", None)
            self.transformer = None
            self.output_projection = None
            return
        self.value_weight = nn.Parameter(torch.empty(self.input_dim, self.token_dim))
        self.value_bias = nn.Parameter(torch.zeros(self.input_dim, self.token_dim))
        self.feature_embedding = nn.Parameter(torch.empty(self.input_dim, self.token_dim))
        self.cls_token = nn.Parameter(torch.empty(1, 1, self.token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=num_heads,
            dim_feedforward=max(self.token_dim * 4, self.output_dim * 2),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.output_dim),
            nn.GELU(),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.input_dim == 0:
            return
        nn.init.xavier_uniform_(self.value_weight)
        nn.init.xavier_uniform_(self.feature_embedding)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch_size = values.shape[0]
        if self.input_dim == 0:
            return torch.zeros((batch_size, self.output_dim), dtype=values.dtype, device=values.device)
        value_tokens = values.unsqueeze(-1) * self.value_weight.unsqueeze(0)
        value_tokens = value_tokens + self.value_bias.unsqueeze(0) + self.feature_embedding.unsqueeze(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        token_sequence = torch.cat([cls_token, value_tokens], dim=1)
        encoded = self.transformer(token_sequence)
        cls_state = encoded[:, 0, :]
        return self.output_projection(cls_state)


class OutcomeHead(nn.Module):
    def __init__(self, input_dim: int, action_classes: int = 2):
        super().__init__()
        self.action_head = nn.Linear(input_dim, action_classes)
        self.return_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, 1),
        )
        self.signed_return_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, 1),
        )
        self.duration_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_logits = self.action_head(embeddings)
        return_pred = torch.sigmoid(self.return_head(embeddings)).squeeze(-1)
        signed_return_pred = torch.sigmoid(self.signed_return_head(embeddings)).squeeze(-1)
        duration_pred = torch.sigmoid(self.duration_head(embeddings)).squeeze(-1)
        return action_logits, return_pred, signed_return_pred, duration_pred


class FamilyDecoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.10):
        super().__init__()
        self.output_dim = int(output_dim)
        if self.output_dim == 0:
            self.network = None
        else:
            hidden_dim = max(self.output_dim, input_dim)
            self.network = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.output_dim),
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.output_dim == 0:
            return torch.zeros((values.shape[0], 0), dtype=values.dtype, device=values.device)
        return self.network(values)


class ContextFamilyStateModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        market_input_dim: int,
        fundamental_input_dim: int,
        macro_input_dim: int,
        tokenizer=None,
        tokenizer_max_length: int | None = None,
        text_projection_dim: int = 192,
        family_embedding_dim: int = 64,
        family_num_heads: int = 4,
        family_num_layers: int = 2,
        fusion_dim: int = 256,
        bottleneck_hidden_1: int = 288,
        bottleneck_hidden_2: int = 240,
        action_classes: int = 2,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = tokenizer if tokenizer is not None else load_local_first_tokenizer(model_name)
        self.tokenizer_max_length = tokenizer_max_length
        self.encoder = load_local_first_transformer(model_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.text_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, text_projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.market_encoder = FTNumericEncoder(
            market_input_dim,
            token_dim=family_embedding_dim,
            output_dim=family_embedding_dim,
            num_heads=family_num_heads,
            num_layers=family_num_layers,
            dropout=dropout,
        )
        self.fundamental_encoder = FTNumericEncoder(
            fundamental_input_dim,
            token_dim=family_embedding_dim,
            output_dim=family_embedding_dim,
            num_heads=family_num_heads,
            num_layers=family_num_layers,
            dropout=dropout,
        )
        self.macro_encoder = FTNumericEncoder(
            macro_input_dim,
            token_dim=family_embedding_dim,
            output_dim=family_embedding_dim,
            num_heads=family_num_heads,
            num_layers=family_num_layers,
            dropout=dropout,
        )
        fused_input_dim = text_projection_dim + (family_embedding_dim * 3)
        self.fusion_projection = nn.Sequential(
            nn.LayerNorm(fused_input_dim),
            nn.Linear(fused_input_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.shared_bottleneck = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, bottleneck_hidden_1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_hidden_1, bottleneck_hidden_2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.entry_outcome_head = OutcomeHead(fusion_dim, action_classes=action_classes)
        self.exit_outcome_head = OutcomeHead(fusion_dim, action_classes=action_classes)
        self.context_decoder = FamilyDecoder(bottleneck_hidden_2, text_projection_dim, dropout=dropout)
        self.market_decoder = FamilyDecoder(bottleneck_hidden_2, market_input_dim, dropout=dropout)
        self.fundamental_decoder = FamilyDecoder(bottleneck_hidden_2, fundamental_input_dim, dropout=dropout)
        self.macro_decoder = FamilyDecoder(bottleneck_hidden_2, macro_input_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def encode_context(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer(
            [str(text) for text in texts],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.tokenizer_max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = self.encoder(**encoded)
        pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        return self.text_projection(pooled)

    def encode_state(
        self,
        texts: Sequence[str],
        market_tensor: torch.Tensor,
        fundamental_tensor: torch.Tensor,
        macro_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_embedding = self.encode_context(texts, market_tensor.device)
        market_embedding = self.market_encoder(market_tensor)
        fundamental_embedding = self.fundamental_encoder(fundamental_tensor)
        macro_embedding = self.macro_encoder(macro_tensor)
        fused = torch.cat([context_embedding, market_embedding, fundamental_embedding, macro_embedding], dim=1)
        state_embedding = F.normalize(self.fusion_projection(fused), p=2, dim=1)
        return state_embedding, context_embedding

    def predict_entry_outcomes(self, entry_embeddings: torch.Tensor):
        return self.entry_outcome_head(self.dropout(entry_embeddings))

    def predict_exit_outcomes(self, exit_embeddings: torch.Tensor):
        return self.exit_outcome_head(self.dropout(exit_embeddings))

    def compress_state(self, state_embeddings: torch.Tensor) -> torch.Tensor:
        return self.shared_bottleneck(self.dropout(state_embeddings))

    def decode_state(self, compressed_state: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "context": self.context_decoder(compressed_state),
            "market": self.market_decoder(compressed_state),
            "fundamental": self.fundamental_decoder(compressed_state),
            "macro": self.macro_decoder(compressed_state),
        }

    def forward_state(
        self,
        texts: Sequence[str],
        market_tensor: torch.Tensor,
        fundamental_tensor: torch.Tensor,
        macro_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_state(texts, market_tensor, fundamental_tensor, macro_tensor)

    def forward_pair(
        self,
        entry_texts: Sequence[str],
        entry_market_tensor: torch.Tensor,
        entry_fundamental_tensor: torch.Tensor,
        entry_macro_tensor: torch.Tensor,
        exit_texts: Sequence[str],
        exit_market_tensor: torch.Tensor,
        exit_fundamental_tensor: torch.Tensor,
        exit_macro_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        entry_embedding, entry_context_embedding = self.encode_state(
            entry_texts,
            entry_market_tensor,
            entry_fundamental_tensor,
            entry_macro_tensor,
        )
        exit_embedding, exit_context_embedding = self.encode_state(
            exit_texts,
            exit_market_tensor,
            exit_fundamental_tensor,
            exit_macro_tensor,
        )
        duration_distance = 0.5 * (1.0 - F.cosine_similarity(entry_embedding, exit_embedding, dim=1))
        predicted_exit = self.decode_state(self.compress_state(entry_embedding))
        return entry_embedding, entry_context_embedding, exit_embedding, exit_context_embedding, duration_distance, predicted_exit


def _state_losses(
    model: ContextFamilyStateModel,
    batch: pd.DataFrame,
    data_spec: ContextFamilyMTLDataSpec,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    market_tensor = batch_numeric_tensor(batch, data_spec.state_market_cols, device)
    fundamental_tensor = batch_numeric_tensor(batch, data_spec.state_fundamental_cols, device)
    macro_tensor = batch_numeric_tensor(batch, data_spec.state_macro_cols, device)
    state_embeddings, context_targets = model.forward_state(batch["text"].tolist(), market_tensor, fundamental_tensor, macro_tensor)
    market_targets = batch_standardized_numeric_tensor(batch, data_spec.state_market_cols, data_spec.state_market_stats, device)
    fundamental_targets = batch_standardized_numeric_tensor(
        batch,
        data_spec.state_fundamental_cols,
        data_spec.state_fundamental_stats,
        device,
    )
    macro_targets = batch_standardized_numeric_tensor(batch, data_spec.state_macro_cols, data_spec.state_macro_stats, device)
    event_roles = batch["event_role"].astype(str).tolist()
    entry_mask = torch.tensor([role == "entry" for role in event_roles], dtype=torch.bool, device=device)
    exit_mask = torch.tensor([role == "exit" for role in event_roles], dtype=torch.bool, device=device)
    action_values = batch["action"].astype(str).str.strip().str.lower().tolist()
    return_targets = torch.tensor(batch["trade_return_pct"].tolist(), dtype=torch.float32, device=device)
    signed_return_targets = torch.tensor(batch["signed_trade_return_pct"].tolist(), dtype=torch.float32, device=device)
    duration_targets = torch.tensor(batch["duration_pct"].tolist(), dtype=torch.float32, device=device)
    zero = torch.tensor(0.0, dtype=torch.float32, device=device)
    losses: dict[str, torch.Tensor] = {
        "entry_action_loss": zero,
        "entry_return_loss": zero,
        "entry_signed_return_loss": zero,
        "entry_duration_loss": zero,
        "entry_context_recon_loss": zero,
        "entry_market_recon_loss": zero,
        "entry_fundamental_recon_loss": zero,
        "entry_macro_recon_loss": zero,
        "exit_action_loss": zero,
        "exit_return_loss": zero,
        "exit_signed_return_loss": zero,
        "exit_duration_loss": zero,
        "exit_context_recon_loss": zero,
        "exit_market_recon_loss": zero,
        "exit_fundamental_recon_loss": zero,
        "exit_macro_recon_loss": zero,
    }

    for role_name, mask, predict_fn in (("entry", entry_mask, model.predict_entry_outcomes), ("exit", exit_mask, model.predict_exit_outcomes)):
        if not bool(mask.any()):
            continue
        action_to_id = data_spec.entry_action_to_id if role_name == "entry" else data_spec.exit_action_to_id
        mask_list = mask.detach().cpu().tolist()
        role_action_targets = torch.tensor(
            [action_to_id[value] for value, keep in zip(action_values, mask_list) if keep],
            dtype=torch.long,
            device=device,
        )
        role_embeddings = state_embeddings[mask]
        role_context_targets = context_targets[mask].detach()
        role_market_targets = market_targets[mask]
        role_fundamental_targets = fundamental_targets[mask]
        role_macro_targets = macro_targets[mask]
        logits, return_pred, signed_return_pred, duration_pred = predict_fn(role_embeddings)
        decoded = model.decode_state(model.compress_state(role_embeddings))
        losses[f"{role_name}_action_loss"] = F.cross_entropy(logits, role_action_targets)
        losses[f"{role_name}_return_loss"] = smooth_l1_compat(return_pred, return_targets[mask])
        losses[f"{role_name}_signed_return_loss"] = smooth_l1_compat(signed_return_pred, signed_return_targets[mask])
        losses[f"{role_name}_duration_loss"] = smooth_l1_compat(duration_pred, duration_targets[mask])
        losses[f"{role_name}_context_recon_loss"] = smooth_l1_compat(decoded["context"], role_context_targets)
        if role_market_targets.shape[1] > 0:
            losses[f"{role_name}_market_recon_loss"] = smooth_l1_compat(decoded["market"], role_market_targets)
        if role_fundamental_targets.shape[1] > 0:
            losses[f"{role_name}_fundamental_recon_loss"] = smooth_l1_compat(decoded["fundamental"], role_fundamental_targets)
        if role_macro_targets.shape[1] > 0:
            losses[f"{role_name}_macro_recon_loss"] = smooth_l1_compat(decoded["macro"], role_macro_targets)
    return losses


def _pair_losses(
    model: ContextFamilyStateModel,
    batch: pd.DataFrame,
    data_spec: ContextFamilyMTLDataSpec,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    entry_market_tensor = batch_numeric_tensor(batch, data_spec.entry_market_cols, device)
    entry_fundamental_tensor = batch_numeric_tensor(batch, data_spec.entry_fundamental_cols, device)
    entry_macro_tensor = batch_numeric_tensor(batch, data_spec.entry_macro_cols, device)
    exit_market_tensor = batch_numeric_tensor(batch, data_spec.exit_market_cols, device)
    exit_fundamental_tensor = batch_numeric_tensor(batch, data_spec.exit_fundamental_cols, device)
    exit_macro_tensor = batch_numeric_tensor(batch, data_spec.exit_macro_cols, device)
    _, _, exit_embedding, exit_context_embedding, duration_distance, predicted_exit = model.forward_pair(
        batch["entry_text"].tolist(),
        entry_market_tensor,
        entry_fundamental_tensor,
        entry_macro_tensor,
        batch["exit_text"].tolist(),
        exit_market_tensor,
        exit_fundamental_tensor,
        exit_macro_tensor,
    )
    duration_target = torch.tensor(batch["duration_pct"].tolist(), dtype=torch.float32, device=device)
    exit_market_target = batch_standardized_numeric_tensor(batch, data_spec.exit_market_cols, data_spec.exit_market_stats, device)
    exit_fundamental_target = batch_standardized_numeric_tensor(
        batch,
        data_spec.exit_fundamental_cols,
        data_spec.exit_fundamental_stats,
        device,
    )
    exit_macro_target = batch_standardized_numeric_tensor(batch, data_spec.exit_macro_cols, data_spec.exit_macro_stats, device)
    duration_contrastive_loss = smooth_l1_compat(duration_distance, duration_target)
    transition_losses = {
        "transition_context_loss": smooth_l1_compat(predicted_exit["context"], exit_context_embedding.detach()),
        "transition_market_loss": smooth_l1_compat(predicted_exit["market"], exit_market_target)
        if exit_market_target.shape[1] > 0
        else torch.tensor(0.0, device=device),
        "transition_fundamental_loss": smooth_l1_compat(predicted_exit["fundamental"], exit_fundamental_target)
        if exit_fundamental_target.shape[1] > 0
        else torch.tensor(0.0, device=device),
        "transition_macro_loss": smooth_l1_compat(predicted_exit["macro"], exit_macro_target)
        if exit_macro_target.shape[1] > 0
        else torch.tensor(0.0, device=device),
        "transition_context_cos": F.cosine_similarity(predicted_exit["context"], exit_context_embedding.detach(), dim=1).mean(),
    }
    return duration_contrastive_loss, transition_losses


@torch.no_grad()
def evaluate_context_family_mtl_model(
    model: ContextFamilyStateModel,
    state_frame: pd.DataFrame,
    pair_frame: pd.DataFrame,
    data_spec: ContextFamilyMTLDataSpec,
    device: torch.device,
    batch_size: int = 8,
    return_buffers: bool = False,
):
    model.eval()
    task_buffers = {
        "entry": defaultdict(list),
        "exit": defaultdict(list),
    }
    for batch in frame_batches(state_frame, batch_size=batch_size, shuffle=False):
        market_tensor = batch_numeric_tensor(batch, data_spec.state_market_cols, device)
        fundamental_tensor = batch_numeric_tensor(batch, data_spec.state_fundamental_cols, device)
        macro_tensor = batch_numeric_tensor(batch, data_spec.state_macro_cols, device)
        state_embeddings, context_targets = model.forward_state(batch["text"].tolist(), market_tensor, fundamental_tensor, macro_tensor)
        market_targets = batch_standardized_numeric_tensor(batch, data_spec.state_market_cols, data_spec.state_market_stats, device)
        fundamental_targets = batch_standardized_numeric_tensor(
            batch,
            data_spec.state_fundamental_cols,
            data_spec.state_fundamental_stats,
            device,
        )
        macro_targets = batch_standardized_numeric_tensor(batch, data_spec.state_macro_cols, data_spec.state_macro_stats, device)
        event_roles = batch["event_role"].astype(str).tolist()
        action_values = batch["action"].astype(str).str.strip().str.lower().tolist()
        return_values = batch["trade_return_pct"].tolist()
        signed_return_values = batch["signed_trade_return_pct"].tolist()
        duration_values = batch["duration_pct"].tolist()

        for role_name, predict_fn, id_to_action in (
            ("entry", model.predict_entry_outcomes, data_spec.id_to_entry_action),
            ("exit", model.predict_exit_outcomes, data_spec.id_to_exit_action),
        ):
            mask = [role == role_name for role in event_roles]
            if not any(mask):
                continue
            role_indices = torch.tensor(mask, dtype=torch.bool, device=device)
            role_embeddings = state_embeddings[role_indices]
            role_context_targets = context_targets[role_indices]
            role_market_targets = market_targets[role_indices]
            role_fundamental_targets = fundamental_targets[role_indices]
            role_macro_targets = macro_targets[role_indices]
            logits, return_pred, signed_return_pred, duration_pred = predict_fn(role_embeddings)
            decoded = model.decode_state(model.compress_state(role_embeddings))
            numeric_errors: list[torch.Tensor] = []
            if role_market_targets.shape[1] > 0:
                numeric_errors.append(torch.abs(decoded["market"] - role_market_targets).mean(dim=1))
            if role_fundamental_targets.shape[1] > 0:
                numeric_errors.append(torch.abs(decoded["fundamental"] - role_fundamental_targets).mean(dim=1))
            if role_macro_targets.shape[1] > 0:
                numeric_errors.append(torch.abs(decoded["macro"] - role_macro_targets).mean(dim=1))
            numeric_recon_mae = (
                torch.stack(numeric_errors, dim=1).mean(dim=1).detach().cpu().tolist() if numeric_errors else [0.0] * len(role_embeddings)
            )
            task_buffers[role_name]["action_true"].extend([value for value, keep in zip(action_values, mask) if keep])
            task_buffers[role_name]["action_pred"].extend([id_to_action[int(idx)] for idx in logits.argmax(dim=1).detach().cpu().tolist()])
            task_buffers[role_name]["return_true"].extend([float(value) for value, keep in zip(return_values, mask) if keep])
            task_buffers[role_name]["return_pred"].extend(return_pred.detach().cpu().tolist())
            task_buffers[role_name]["signed_return_true"].extend([float(value) for value, keep in zip(signed_return_values, mask) if keep])
            task_buffers[role_name]["signed_return_pred"].extend(signed_return_pred.detach().cpu().tolist())
            task_buffers[role_name]["duration_true"].extend([float(value) for value, keep in zip(duration_values, mask) if keep])
            task_buffers[role_name]["duration_pred"].extend(duration_pred.detach().cpu().tolist())
            task_buffers[role_name]["context_cosine"].extend(
                F.cosine_similarity(decoded["context"], role_context_targets, dim=1).detach().cpu().tolist()
            )
            task_buffers[role_name]["numeric_recon_mae"].extend(numeric_recon_mae)

    pair_duration_true: list[float] = []
    pair_duration_pred: list[float] = []
    transition_context_cos: list[float] = []
    transition_numeric_mae: list[float] = []
    for batch in frame_batches(pair_frame, batch_size=batch_size, shuffle=False):
        entry_market_tensor = batch_numeric_tensor(batch, data_spec.entry_market_cols, device)
        entry_fundamental_tensor = batch_numeric_tensor(batch, data_spec.entry_fundamental_cols, device)
        entry_macro_tensor = batch_numeric_tensor(batch, data_spec.entry_macro_cols, device)
        exit_market_tensor = batch_numeric_tensor(batch, data_spec.exit_market_cols, device)
        exit_fundamental_tensor = batch_numeric_tensor(batch, data_spec.exit_fundamental_cols, device)
        exit_macro_tensor = batch_numeric_tensor(batch, data_spec.exit_macro_cols, device)
        _, _, _, exit_context_embedding, duration_distance, predicted_exit = model.forward_pair(
            batch["entry_text"].tolist(),
            entry_market_tensor,
            entry_fundamental_tensor,
            entry_macro_tensor,
            batch["exit_text"].tolist(),
            exit_market_tensor,
            exit_fundamental_tensor,
            exit_macro_tensor,
        )
        exit_market_target = batch_standardized_numeric_tensor(batch, data_spec.exit_market_cols, data_spec.exit_market_stats, device)
        exit_fundamental_target = batch_standardized_numeric_tensor(
            batch,
            data_spec.exit_fundamental_cols,
            data_spec.exit_fundamental_stats,
            device,
        )
        exit_macro_target = batch_standardized_numeric_tensor(batch, data_spec.exit_macro_cols, data_spec.exit_macro_stats, device)
        numeric_errors: list[torch.Tensor] = []
        if exit_market_target.shape[1] > 0:
            numeric_errors.append(torch.abs(predicted_exit["market"] - exit_market_target).mean(dim=1))
        if exit_fundamental_target.shape[1] > 0:
            numeric_errors.append(torch.abs(predicted_exit["fundamental"] - exit_fundamental_target).mean(dim=1))
        if exit_macro_target.shape[1] > 0:
            numeric_errors.append(torch.abs(predicted_exit["macro"] - exit_macro_target).mean(dim=1))
        if numeric_errors:
            transition_numeric_mae.extend(torch.stack(numeric_errors, dim=1).mean(dim=1).detach().cpu().tolist())
        pair_duration_true.extend(batch["duration_pct"].tolist())
        pair_duration_pred.extend(duration_distance.detach().cpu().tolist())
        transition_context_cos.extend(F.cosine_similarity(predicted_exit["context"], exit_context_embedding, dim=1).detach().cpu().tolist())

    metrics: dict[str, Any] = {}
    for role_name in ("entry", "exit"):
        buf = task_buffers[role_name]
        metrics[f"{role_name}_support"] = int(len(buf["action_true"]))
        metrics[f"{role_name}_action_accuracy"] = safe_accuracy(buf["action_true"], buf["action_pred"])
        metrics[f"{role_name}_action_macro_f1"] = safe_macro_f1(buf["action_true"], buf["action_pred"])
        metrics[f"{role_name}_return_mae"] = safe_mae(buf["return_true"], buf["return_pred"])
        metrics[f"{role_name}_return_rmse"] = safe_rmse(buf["return_true"], buf["return_pred"])
        metrics[f"{role_name}_return_spearman"] = safe_spearman(buf["return_true"], buf["return_pred"])
        metrics[f"{role_name}_signed_return_mae"] = safe_mae(buf["signed_return_true"], buf["signed_return_pred"])
        metrics[f"{role_name}_signed_return_rmse"] = safe_rmse(buf["signed_return_true"], buf["signed_return_pred"])
        metrics[f"{role_name}_signed_return_spearman"] = safe_spearman(buf["signed_return_true"], buf["signed_return_pred"])
        metrics[f"{role_name}_duration_mae"] = safe_mae(buf["duration_true"], buf["duration_pred"])
        metrics[f"{role_name}_duration_rmse"] = safe_rmse(buf["duration_true"], buf["duration_pred"])
        metrics[f"{role_name}_duration_spearman"] = safe_spearman(buf["duration_true"], buf["duration_pred"])
        metrics[f"{role_name}_context_recon_cosine_mean"] = safe_mean(buf["context_cosine"])
        metrics[f"{role_name}_numeric_recon_mae"] = safe_mean(buf["numeric_recon_mae"])

    metrics["pair_support"] = int(len(pair_duration_true))
    metrics["transition_support"] = int(len(transition_context_cos))
    metrics["pair_duration_mae"] = safe_mae(pair_duration_true, pair_duration_pred)
    metrics["pair_duration_rmse"] = safe_rmse(pair_duration_true, pair_duration_pred)
    metrics["pair_duration_spearman"] = safe_spearman(pair_duration_true, pair_duration_pred)
    metrics["transition_context_cosine_mean"] = safe_mean(transition_context_cos)
    metrics["transition_numeric_recon_mae"] = safe_mean(transition_numeric_mae)
    score_terms = [
        finite_or_zero(metrics["entry_return_spearman"]),
        finite_or_zero(metrics["exit_return_spearman"]),
        finite_or_zero(metrics["entry_duration_spearman"]),
        finite_or_zero(metrics["exit_duration_spearman"]),
        finite_or_zero(metrics["pair_duration_spearman"]),
        finite_or_zero(metrics["transition_context_cosine_mean"]),
        1.0 / (1.0 + max(finite_or_zero(metrics["transition_numeric_recon_mae"]), 0.0)),
        1.0 / (1.0 + max(finite_or_zero(metrics["entry_numeric_recon_mae"]), 0.0)),
        1.0 / (1.0 + max(finite_or_zero(metrics["exit_numeric_recon_mae"]), 0.0)),
    ]
    metrics["model_score"] = float(np.mean(score_terms))
    if return_buffers:
        task_buffers["pair"] = {"duration_true": list(pair_duration_true), "duration_pred": list(pair_duration_pred)}
        task_buffers["transition"] = {"context_cosine": list(transition_context_cos), "numeric_recon_mae": list(transition_numeric_mae)}
        return metrics, task_buffers
    return metrics


def train_context_family_mtl_model(
    *,
    model: ContextFamilyStateModel,
    state_train_df: pd.DataFrame,
    pair_train_df: pd.DataFrame,
    state_dev_df: pd.DataFrame,
    pair_dev_df: pd.DataFrame,
    state_test_df: pd.DataFrame,
    pair_test_df: pd.DataFrame,
    data_spec: ContextFamilyMTLDataSpec,
    training_cfg: dict[str, Any],
    device: torch.device,
    artifact_dir: str | Path,
    run_training: bool = True,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = artifact_dir / "best_model.pt"
    history_path = artifact_dir / "training_history.csv"
    training_history: list[dict[str, Any]] = []
    best_dev_score = float("-inf")

    if run_training:
        for epoch in range(1, training_cfg["epochs"] + 1):
            model.train()
            epoch_started = time.time()
            running: defaultdict[str, float] = defaultdict(float)
            state_batches = list(frame_batches(state_train_df, batch_size=training_cfg["batch_size"], shuffle=True, seed=42 + epoch))
            pair_batches = list(frame_batches(pair_train_df, batch_size=training_cfg["batch_size"], shuffle=True, seed=142 + epoch))
            step_count = max(len(state_batches), len(pair_batches))
            progress_interval = max(1, step_count // 10)
            print(
                f"Starting epoch {epoch}/{training_cfg['epochs']} with {step_count} steps "
                f"(state_batches={len(state_batches)}, pair_batches={len(pair_batches)}, progress_interval={progress_interval})",
                flush=True,
            )
            for step_idx in range(step_count):
                optimizer.zero_grad()
                total_loss = torch.tensor(0.0, device=device)
                if step_idx < len(state_batches):
                    state_loss_values = _state_losses(model, state_batches[step_idx], data_spec, device)
                    entry_task_loss = (
                        state_loss_values["entry_action_loss"]
                        + state_loss_values["entry_return_loss"]
                        + state_loss_values["entry_signed_return_loss"]
                        + state_loss_values["entry_duration_loss"]
                    )
                    exit_task_loss = (
                        state_loss_values["exit_action_loss"]
                        + state_loss_values["exit_return_loss"]
                        + state_loss_values["exit_signed_return_loss"]
                        + state_loss_values["exit_duration_loss"]
                    )
                    state_recon_loss = (
                        state_loss_values["entry_context_recon_loss"]
                        + state_loss_values["entry_market_recon_loss"]
                        + state_loss_values["entry_fundamental_recon_loss"]
                        + state_loss_values["entry_macro_recon_loss"]
                        + state_loss_values["exit_context_recon_loss"]
                        + state_loss_values["exit_market_recon_loss"]
                        + state_loss_values["exit_fundamental_recon_loss"]
                        + state_loss_values["exit_macro_recon_loss"]
                    )
                    total_loss = total_loss + training_cfg["entry_task_loss_weight"] * entry_task_loss
                    total_loss = total_loss + training_cfg["exit_task_loss_weight"] * exit_task_loss
                    total_loss = total_loss + training_cfg["state_reconstruction_loss_weight"] * state_recon_loss
                    for key, value in state_loss_values.items():
                        running[key] += float(value.detach().cpu())
                    running["state_steps"] += 1.0
                if step_idx < len(pair_batches):
                    duration_contrastive_loss, transition_losses = _pair_losses(model, pair_batches[step_idx], data_spec, device)
                    transition_recon_loss = (
                        transition_losses["transition_context_loss"]
                        + transition_losses["transition_market_loss"]
                        + transition_losses["transition_fundamental_loss"]
                        + transition_losses["transition_macro_loss"]
                    )
                    total_loss = total_loss + training_cfg["duration_contrastive_loss_weight"] * duration_contrastive_loss
                    total_loss = total_loss + training_cfg["transition_reconstruction_loss_weight"] * transition_recon_loss
                    running["duration_contrastive_loss"] += float(duration_contrastive_loss.detach().cpu())
                    for key, value in transition_losses.items():
                        running[key] += float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
                    running["pair_steps"] += 1.0
                total_loss.backward()
                optimizer.step()
                running["total_loss"] += float(total_loss.detach().cpu())
                if ((step_idx + 1) % progress_interval == 0) or (step_idx + 1 == step_count):
                    elapsed = time.time() - epoch_started
                    print(
                        f"epoch={epoch} step={step_idx + 1}/{step_count} "
                        f"avg_loss={running['total_loss'] / (step_idx + 1):.4f} elapsed_sec={elapsed:.1f}",
                        flush=True,
                    )
            train_metrics = evaluate_context_family_mtl_model(
                model,
                state_train_df,
                pair_train_df,
                data_spec,
                device,
                batch_size=training_cfg["batch_size"],
            )
            dev_metrics = evaluate_context_family_mtl_model(
                model,
                state_dev_df,
                pair_dev_df,
                data_spec,
                device,
                batch_size=training_cfg["batch_size"],
            )
            epoch_row = {
                "epoch": epoch,
                "train_total_loss": running["total_loss"] / max(step_count, 1),
                "train_entry_action_loss": running["entry_action_loss"] / max(running["state_steps"], 1.0),
                "train_entry_return_loss": running["entry_return_loss"] / max(running["state_steps"], 1.0),
                "train_entry_signed_return_loss": running["entry_signed_return_loss"] / max(running["state_steps"], 1.0),
                "train_entry_duration_loss": running["entry_duration_loss"] / max(running["state_steps"], 1.0),
                "train_entry_context_recon_loss": running["entry_context_recon_loss"] / max(running["state_steps"], 1.0),
                "train_entry_market_recon_loss": running["entry_market_recon_loss"] / max(running["state_steps"], 1.0),
                "train_entry_fundamental_recon_loss": running["entry_fundamental_recon_loss"] / max(running["state_steps"], 1.0),
                "train_entry_macro_recon_loss": running["entry_macro_recon_loss"] / max(running["state_steps"], 1.0),
                "train_exit_action_loss": running["exit_action_loss"] / max(running["state_steps"], 1.0),
                "train_exit_return_loss": running["exit_return_loss"] / max(running["state_steps"], 1.0),
                "train_exit_signed_return_loss": running["exit_signed_return_loss"] / max(running["state_steps"], 1.0),
                "train_exit_duration_loss": running["exit_duration_loss"] / max(running["state_steps"], 1.0),
                "train_exit_context_recon_loss": running["exit_context_recon_loss"] / max(running["state_steps"], 1.0),
                "train_exit_market_recon_loss": running["exit_market_recon_loss"] / max(running["state_steps"], 1.0),
                "train_exit_fundamental_recon_loss": running["exit_fundamental_recon_loss"] / max(running["state_steps"], 1.0),
                "train_exit_macro_recon_loss": running["exit_macro_recon_loss"] / max(running["state_steps"], 1.0),
                "train_duration_contrastive_loss": running["duration_contrastive_loss"] / max(running["pair_steps"], 1.0),
                "train_transition_context_loss": running["transition_context_loss"] / max(running["pair_steps"], 1.0),
                "train_transition_market_loss": running["transition_market_loss"] / max(running["pair_steps"], 1.0),
                "train_transition_fundamental_loss": running["transition_fundamental_loss"] / max(running["pair_steps"], 1.0),
                "train_transition_macro_loss": running["transition_macro_loss"] / max(running["pair_steps"], 1.0),
                "dev_model_score": dev_metrics["model_score"],
                "seconds": time.time() - epoch_started,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"dev_{key}": value for key, value in dev_metrics.items()},
            }
            training_history.append(epoch_row)
            print(f"epoch={epoch} loss={epoch_row['train_total_loss']:.4f} dev_score={dev_metrics['model_score']:.4f}", flush=True)
            print(format_metric_report(f"epoch={epoch} train", train_metrics), flush=True)
            print(format_metric_report(f"epoch={epoch} dev", dev_metrics), flush=True)
            if dev_metrics["model_score"] > best_dev_score:
                best_dev_score = dev_metrics["model_score"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "training_cfg": training_cfg,
                        "model_name": model.model_name,
                    },
                    best_checkpoint_path,
                )
        history_df = pd.DataFrame(training_history)
    elif history_path.exists():
        history_df = pd.read_csv(history_path)
    else:
        history_df = pd.DataFrame()

    if not best_checkpoint_path.exists():
        raise FileNotFoundError(f"Expected checkpoint at {best_checkpoint_path}")
    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dev_metrics, dev_task_buffers = evaluate_context_family_mtl_model(
        model,
        state_dev_df,
        pair_dev_df,
        data_spec,
        device,
        batch_size=training_cfg["batch_size"],
        return_buffers=True,
    )
    test_metrics, test_task_buffers = evaluate_context_family_mtl_model(
        model,
        state_test_df,
        pair_test_df,
        data_spec,
        device,
        batch_size=training_cfg["batch_size"],
        return_buffers=True,
    )
    metrics_df = pd.DataFrame([{"split": "dev", **dev_metrics}, {"split": "test", **test_metrics}])
    metrics_long_df = metrics_df.set_index("split").T.reset_index().rename(columns={"index": "metric"})
    action_report_df = pd.concat(
        [
            build_action_classification_report_df(
                "dev",
                "entry",
                dev_task_buffers["entry"]["action_true"],
                dev_task_buffers["entry"]["action_pred"],
                list(data_spec.entry_action_to_id.keys()),
            ),
            build_action_classification_report_df(
                "dev",
                "exit",
                dev_task_buffers["exit"]["action_true"],
                dev_task_buffers["exit"]["action_pred"],
                list(data_spec.exit_action_to_id.keys()),
            ),
            build_action_classification_report_df(
                "test",
                "entry",
                test_task_buffers["entry"]["action_true"],
                test_task_buffers["entry"]["action_pred"],
                list(data_spec.entry_action_to_id.keys()),
            ),
            build_action_classification_report_df(
                "test",
                "exit",
                test_task_buffers["exit"]["action_true"],
                test_task_buffers["exit"]["action_pred"],
                list(data_spec.exit_action_to_id.keys()),
            ),
        ],
        ignore_index=True,
    )
    action_detailed_results_df = pd.DataFrame(
        [
            {
                "split": "dev",
                **build_flair_action_report(
                    "entry",
                    dev_task_buffers["entry"]["action_true"],
                    dev_task_buffers["entry"]["action_pred"],
                    list(data_spec.entry_action_to_id.keys()),
                ),
            },
            {
                "split": "dev",
                **build_flair_action_report(
                    "exit",
                    dev_task_buffers["exit"]["action_true"],
                    dev_task_buffers["exit"]["action_pred"],
                    list(data_spec.exit_action_to_id.keys()),
                ),
            },
            {
                "split": "test",
                **build_flair_action_report(
                    "entry",
                    test_task_buffers["entry"]["action_true"],
                    test_task_buffers["entry"]["action_pred"],
                    list(data_spec.entry_action_to_id.keys()),
                ),
            },
            {
                "split": "test",
                **build_flair_action_report(
                    "exit",
                    test_task_buffers["exit"]["action_true"],
                    test_task_buffers["exit"]["action_pred"],
                    list(data_spec.exit_action_to_id.keys()),
                ),
            },
        ]
    )[["split", "task", "accuracy", "detailed_result"]]
    action_f1_comparison_df = pd.concat(
        [
            build_action_f1_comparison_df(
                "dev",
                dev_metrics,
                dev_task_buffers,
                {"entry": list(data_spec.entry_action_to_id.keys()), "exit": list(data_spec.exit_action_to_id.keys())},
            ),
            build_action_f1_comparison_df(
                "test",
                test_metrics,
                test_task_buffers,
                {"entry": list(data_spec.entry_action_to_id.keys()), "exit": list(data_spec.exit_action_to_id.keys())},
            ),
        ],
        ignore_index=True,
    )
    task_report_df = pd.concat(
        [
            build_regression_task_report_df(
                "dev",
                dev_metrics,
                dev_task_buffers,
                regression_specs=[
                    ("entry_return", dev_task_buffers["entry"]["return_true"], dev_task_buffers["entry"]["return_pred"]),
                    (
                        "entry_signed_return",
                        dev_task_buffers["entry"]["signed_return_true"],
                        dev_task_buffers["entry"]["signed_return_pred"],
                    ),
                    ("entry_duration", dev_task_buffers["entry"]["duration_true"], dev_task_buffers["entry"]["duration_pred"]),
                    ("exit_return", dev_task_buffers["exit"]["return_true"], dev_task_buffers["exit"]["return_pred"]),
                    (
                        "exit_signed_return",
                        dev_task_buffers["exit"]["signed_return_true"],
                        dev_task_buffers["exit"]["signed_return_pred"],
                    ),
                    ("exit_duration", dev_task_buffers["exit"]["duration_true"], dev_task_buffers["exit"]["duration_pred"]),
                    ("pair_duration", dev_task_buffers["pair"]["duration_true"], dev_task_buffers["pair"]["duration_pred"]),
                ],
                context_numeric_specs=[
                    (
                        "entry_reconstruction",
                        int(dev_metrics.get("entry_support", 0)),
                        dev_metrics.get("entry_context_recon_cosine_mean", float("nan")),
                        dev_metrics.get("entry_numeric_recon_mae", float("nan")),
                    ),
                    (
                        "exit_reconstruction",
                        int(dev_metrics.get("exit_support", 0)),
                        dev_metrics.get("exit_context_recon_cosine_mean", float("nan")),
                        dev_metrics.get("exit_numeric_recon_mae", float("nan")),
                    ),
                    (
                        "transition",
                        int(dev_metrics.get("transition_support", 0)),
                        dev_metrics.get("transition_context_cosine_mean", float("nan")),
                        dev_metrics.get("transition_numeric_recon_mae", float("nan")),
                    ),
                ],
            ),
            build_regression_task_report_df(
                "test",
                test_metrics,
                test_task_buffers,
                regression_specs=[
                    ("entry_return", test_task_buffers["entry"]["return_true"], test_task_buffers["entry"]["return_pred"]),
                    (
                        "entry_signed_return",
                        test_task_buffers["entry"]["signed_return_true"],
                        test_task_buffers["entry"]["signed_return_pred"],
                    ),
                    ("entry_duration", test_task_buffers["entry"]["duration_true"], test_task_buffers["entry"]["duration_pred"]),
                    ("exit_return", test_task_buffers["exit"]["return_true"], test_task_buffers["exit"]["return_pred"]),
                    (
                        "exit_signed_return",
                        test_task_buffers["exit"]["signed_return_true"],
                        test_task_buffers["exit"]["signed_return_pred"],
                    ),
                    ("exit_duration", test_task_buffers["exit"]["duration_true"], test_task_buffers["exit"]["duration_pred"]),
                    ("pair_duration", test_task_buffers["pair"]["duration_true"], test_task_buffers["pair"]["duration_pred"]),
                ],
                context_numeric_specs=[
                    (
                        "entry_reconstruction",
                        int(test_metrics.get("entry_support", 0)),
                        test_metrics.get("entry_context_recon_cosine_mean", float("nan")),
                        test_metrics.get("entry_numeric_recon_mae", float("nan")),
                    ),
                    (
                        "exit_reconstruction",
                        int(test_metrics.get("exit_support", 0)),
                        test_metrics.get("exit_context_recon_cosine_mean", float("nan")),
                        test_metrics.get("exit_numeric_recon_mae", float("nan")),
                    ),
                    (
                        "transition",
                        int(test_metrics.get("transition_support", 0)),
                        test_metrics.get("transition_context_cosine_mean", float("nan")),
                        test_metrics.get("transition_numeric_recon_mae", float("nan")),
                    ),
                ],
            ),
        ],
        ignore_index=True,
    )
    history_df.to_csv(history_path, index=False)
    metrics_df.to_csv(artifact_dir / "metrics.csv", index=False)
    return {
        "history_df": history_df,
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "metrics_df": metrics_df,
        "metrics_long_df": metrics_long_df,
        "action_report_df": action_report_df,
        "action_detailed_results_df": action_detailed_results_df,
        "action_f1_comparison_df": action_f1_comparison_df,
        "task_report_df": task_report_df,
        "dev_task_buffers": dev_task_buffers,
        "test_task_buffers": test_task_buffers,
        "artifact_dir": artifact_dir,
        "best_checkpoint_path": best_checkpoint_path,
    }


@torch.no_grad()
def predict_state_frame(
    model: ContextFamilyStateModel,
    frame: pd.DataFrame,
    data_spec: ContextFamilyMTLDataSpec,
    device: torch.device,
    batch_size: int = 8,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model.eval()
    for batch in frame_batches(frame, batch_size=batch_size, shuffle=False):
        market_tensor = batch_numeric_tensor(batch, data_spec.state_market_cols, device)
        fundamental_tensor = batch_numeric_tensor(batch, data_spec.state_fundamental_cols, device)
        macro_tensor = batch_numeric_tensor(batch, data_spec.state_macro_cols, device)
        state_embeddings, context_targets = model.forward_state(batch["text"].tolist(), market_tensor, fundamental_tensor, macro_tensor)
        market_targets = batch_standardized_numeric_tensor(batch, data_spec.state_market_cols, data_spec.state_market_stats, device)
        fundamental_targets = batch_standardized_numeric_tensor(
            batch,
            data_spec.state_fundamental_cols,
            data_spec.state_fundamental_stats,
            device,
        )
        macro_targets = batch_standardized_numeric_tensor(batch, data_spec.state_macro_cols, data_spec.state_macro_stats, device)
        event_roles = batch["event_role"].astype(str).tolist()
        role_masks = {"entry": [role == "entry" for role in event_roles], "exit": [role == "exit" for role in event_roles]}
        role_predictions: dict[str, dict[str, list[Any]]] = {}
        for role_name, predict_fn in (("entry", model.predict_entry_outcomes), ("exit", model.predict_exit_outcomes)):
            mask = role_masks[role_name]
            if any(mask):
                role_indices = torch.tensor(mask, dtype=torch.bool, device=device)
                role_embeddings = state_embeddings[role_indices]
                role_context_targets = context_targets[role_indices]
                role_market_targets = market_targets[role_indices]
                role_fund_targets = fundamental_targets[role_indices]
                role_macro_targets = macro_targets[role_indices]
                logits, return_pred, signed_return_pred, duration_pred = predict_fn(role_embeddings)
                decoded = model.decode_state(model.compress_state(role_embeddings))
                numeric_errors: list[torch.Tensor] = []
                if role_market_targets.shape[1] > 0:
                    numeric_errors.append(torch.abs(decoded["market"] - role_market_targets).mean(dim=1))
                if role_fund_targets.shape[1] > 0:
                    numeric_errors.append(torch.abs(decoded["fundamental"] - role_fund_targets).mean(dim=1))
                if role_macro_targets.shape[1] > 0:
                    numeric_errors.append(torch.abs(decoded["macro"] - role_macro_targets).mean(dim=1))
                numeric_recon_mae = (
                    torch.stack(numeric_errors, dim=1).mean(dim=1).detach().cpu().tolist() if numeric_errors else [0.0] * len(role_embeddings)
                )
                id_to_action = data_spec.id_to_entry_action if role_name == "entry" else data_spec.id_to_exit_action
                role_predictions[role_name] = {
                    "action": [id_to_action[int(idx)] for idx in logits.argmax(dim=1).detach().cpu().tolist()],
                    "return": return_pred.detach().cpu().tolist(),
                    "signed_return": signed_return_pred.detach().cpu().tolist(),
                    "duration": duration_pred.detach().cpu().tolist(),
                    "context_cosine": F.cosine_similarity(decoded["context"], role_context_targets, dim=1).detach().cpu().tolist(),
                    "numeric_recon_mae": numeric_recon_mae,
                }
            else:
                role_predictions[role_name] = {
                    "action": [],
                    "return": [],
                    "signed_return": [],
                    "duration": [],
                    "context_cosine": [],
                    "numeric_recon_mae": [],
                }
        role_offsets = {"entry": 0, "exit": 0}
        for row, embedding in zip(batch.to_dict(orient="records"), state_embeddings.detach().cpu().tolist()):
            role_name = str(row["event_role"])
            offset = role_offsets[role_name]
            role_offsets[role_name] += 1
            rows.append(
                {
                    **row,
                    "pred_action": role_predictions[role_name]["action"][offset],
                    "pred_trade_return_pct": float(role_predictions[role_name]["return"][offset]),
                    "pred_signed_trade_return_pct": float(role_predictions[role_name]["signed_return"][offset]),
                    "pred_duration_pct": float(role_predictions[role_name]["duration"][offset]),
                    "state_context_recon_cosine": float(role_predictions[role_name]["context_cosine"][offset]),
                    "state_numeric_recon_mae": float(role_predictions[role_name]["numeric_recon_mae"][offset]),
                    "embedding_norm": float(np.linalg.norm(np.asarray(embedding))),
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def predict_pair_frame(
    model: ContextFamilyStateModel,
    frame: pd.DataFrame,
    data_spec: ContextFamilyMTLDataSpec,
    device: torch.device,
    batch_size: int = 8,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model.eval()
    for batch in frame_batches(frame, batch_size=batch_size, shuffle=False):
        entry_market_tensor = batch_numeric_tensor(batch, data_spec.entry_market_cols, device)
        entry_fundamental_tensor = batch_numeric_tensor(batch, data_spec.entry_fundamental_cols, device)
        entry_macro_tensor = batch_numeric_tensor(batch, data_spec.entry_macro_cols, device)
        exit_market_tensor = batch_numeric_tensor(batch, data_spec.exit_market_cols, device)
        exit_fundamental_tensor = batch_numeric_tensor(batch, data_spec.exit_fundamental_cols, device)
        exit_macro_tensor = batch_numeric_tensor(batch, data_spec.exit_macro_cols, device)
        _, _, _, exit_context_embedding, duration_distance, predicted_exit = model.forward_pair(
            batch["entry_text"].tolist(),
            entry_market_tensor,
            entry_fundamental_tensor,
            entry_macro_tensor,
            batch["exit_text"].tolist(),
            exit_market_tensor,
            exit_fundamental_tensor,
            exit_macro_tensor,
        )
        exit_market_target = batch_standardized_numeric_tensor(batch, data_spec.exit_market_cols, data_spec.exit_market_stats, device)
        exit_fundamental_target = batch_standardized_numeric_tensor(
            batch,
            data_spec.exit_fundamental_cols,
            data_spec.exit_fundamental_stats,
            device,
        )
        exit_macro_target = batch_standardized_numeric_tensor(batch, data_spec.exit_macro_cols, data_spec.exit_macro_stats, device)
        numeric_errors: list[torch.Tensor] = []
        if exit_market_target.shape[1] > 0:
            numeric_errors.append(torch.abs(predicted_exit["market"] - exit_market_target).mean(dim=1))
        if exit_fundamental_target.shape[1] > 0:
            numeric_errors.append(torch.abs(predicted_exit["fundamental"] - exit_fundamental_target).mean(dim=1))
        if exit_macro_target.shape[1] > 0:
            numeric_errors.append(torch.abs(predicted_exit["macro"] - exit_macro_target).mean(dim=1))
        transition_numeric_mae = (
            torch.stack(numeric_errors, dim=1).mean(dim=1).detach().cpu().tolist() if numeric_errors else [0.0] * len(batch)
        )
        transition_context_cosine = F.cosine_similarity(predicted_exit["context"], exit_context_embedding, dim=1).detach().cpu().tolist()
        for row, pred_duration, context_cos, numeric_mae in zip(
            batch.to_dict(orient="records"),
            duration_distance.detach().cpu().tolist(),
            transition_context_cosine,
            transition_numeric_mae,
        ):
            rows.append(
                {
                    **row,
                    "pred_duration_pct_from_distance": float(pred_duration),
                    "transition_context_cosine": float(context_cos),
                    "transition_numeric_recon_mae": float(numeric_mae),
                    "abs_duration_error": abs(float(pred_duration) - float(row["duration_pct"])),
                }
            )
    return pd.DataFrame(rows)
