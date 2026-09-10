"""Annual/quarterly issuer memory and shared multi-asset daily decoder."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .config import MultiRateModelConfig
from .heads import build_task_heads, expand_hits_task_inventory


class FeatureTokenProjection(nn.Module):
    """Project values while retaining feature identity and missingness."""

    def __init__(self, input_dim: int, d_model: int, feature_token_mode: bool = False):
        super().__init__()
        self.feature_token_mode = feature_token_mode
        self.value = nn.Linear(1, d_model)
        self.feature_embedding = nn.Embedding(max(1, input_dim), d_model)
        self.missing_embedding = nn.Embedding(2, d_model)
        self.row_projection = nn.Sequential(nn.Linear(input_dim * 2, d_model), nn.LayerNorm(d_model), nn.GELU())

    def forward(self, values: torch.Tensor, missing: torch.Tensor | None = None) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("feature values must have shape [batch, sequence, features]")
        if missing is None:
            missing = ~torch.isfinite(values)
        missing = missing.bool()
        clean = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.feature_token_mode:
            return self.row_projection(torch.cat([clean, missing.to(clean.dtype)], dim=-1))
        batch, sequence, features = clean.shape
        ids = torch.arange(features, device=clean.device)
        token = self.value(clean.unsqueeze(-1))
        token = token + self.feature_embedding(ids).view(1, 1, features, -1)
        token = token + self.missing_embedding(missing.long())
        return token


class InputAdapter(nn.Module):
    """Asset-specific adapter with a common output dimension."""

    def __init__(self, input_dim: int, d_model: int, asset_class: str, feature_token_mode: bool = False):
        super().__init__()
        self.asset_class = asset_class
        self.projection = FeatureTokenProjection(input_dim, d_model, feature_token_mode)
        self.asset_embedding = nn.Parameter(torch.randn(d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, values: torch.Tensor, missing: torch.Tensor | None = None) -> torch.Tensor:
        output = self.projection(values, missing)
        output = self.norm(output + self.asset_embedding)
        if self.projection.feature_token_mode:
            return output.mean(dim=2)
        return output


class EquityInputAdapter(InputAdapter):
    def __init__(self, input_dim: int, d_model: int, feature_token_mode: bool = False):
        super().__init__(input_dim, d_model, "equity", feature_token_mode)


class CorporateBondInputAdapter(InputAdapter):
    def __init__(self, input_dim: int, d_model: int, feature_token_mode: bool = False):
        super().__init__(input_dim, d_model, "corporate_bond", feature_token_mode)


class PreferredShareInputAdapter(InputAdapter):
    def __init__(self, input_dim: int, d_model: int, feature_token_mode: bool = False):
        super().__init__(input_dim, d_model, "preferred", feature_token_mode)


class WarrantInputAdapter(InputAdapter):
    def __init__(self, input_dim: int, d_model: int, feature_token_mode: bool = False):
        super().__init__(input_dim, d_model, "warrant", feature_token_mode)


class OptionInputAdapter(InputAdapter):
    def __init__(self, input_dim: int, d_model: int, feature_token_mode: bool = False):
        super().__init__(input_dim, d_model, "option", feature_token_mode)


class GenericAssetInputAdapter(InputAdapter):
    pass


ADAPTERS: dict[str, type[InputAdapter]] = {
    "equity": EquityInputAdapter,
    "common_equity": EquityInputAdapter,
    "corporate_bond": CorporateBondInputAdapter,
    "note_bond": CorporateBondInputAdapter,
    "preferred": PreferredShareInputAdapter,
    "preferred_share": PreferredShareInputAdapter,
    "warrant": WarrantInputAdapter,
    "option": OptionInputAdapter,
    "options": OptionInputAdapter,
}


def _encoder(d_model: int, heads: int, layers: int, dropout: float) -> nn.TransformerEncoder:
    block = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=heads,
        dim_feedforward=d_model * 4,
        dropout=dropout,
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(block, num_layers=layers)


class MultiRateMultiAssetTransformer(nn.Module):
    """Shared decoder conditioned on annual and quarterly issuer memories.

    Inputs use ``[B, S, F]`` for issuer histories and daily instrument values.
    The forward method returns one ``[B, S, D]`` representation plus dynamic
    task-head outputs when heads have been registered.
    """

    def __init__(
        self,
        feature_dims: Mapping[str, int],
        config: MultiRateModelConfig | None = None,
        task_heads: Mapping[str, nn.Module] | None = None,
        task_specs: list[Mapping[str, object]] | None = None,
    ):
        super().__init__()
        self.config = config or MultiRateModelConfig()
        self.feature_dims = dict(feature_dims)
        self.annual_encoder = _encoder(self.config.d_model, self.config.num_heads, self.config.annual_layers, self.config.dropout)
        self.quarterly_encoder = _encoder(self.config.d_model, self.config.num_heads, self.config.quarterly_layers, self.config.dropout)
        self.annual_projection = FeatureTokenProjection(self.feature_dims.get("annual", 1), self.config.d_model, self.config.feature_token_mode)
        self.quarterly_projection = FeatureTokenProjection(self.feature_dims.get("quarterly", 1), self.config.d_model, self.config.feature_token_mode)
        self.quarterly_annual_attention = nn.MultiheadAttention(self.config.d_model, self.config.num_heads, dropout=self.config.dropout, batch_first=True)
        self.annual_norm = nn.LayerNorm(self.config.d_model)
        self.quarterly_norm = nn.LayerNorm(self.config.d_model)
        self.adapters = nn.ModuleDict({
            asset: ADAPTERS.get(asset, GenericAssetInputAdapter)(
                self.feature_dims.get(asset, self.feature_dims.get("daily", 1)),
                self.config.d_model,
                self.config.feature_token_mode,
            )
            for asset in self.config.asset_classes
            if asset not in self.config.excluded_asset_classes
        })
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.d_model * 4,
            dropout=self.config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.daily_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.config.decoder_layers)
        self.quarterly_cross_attention = nn.MultiheadAttention(self.config.d_model, self.config.num_heads, dropout=self.config.dropout, batch_first=True)
        self.annual_cross_attention = nn.MultiheadAttention(self.config.d_model, self.config.num_heads, dropout=self.config.dropout, batch_first=True)
        self.output_norm = nn.LayerNorm(self.config.d_model)
        if task_heads is not None and task_specs is not None:
            raise ValueError("provide task_heads or task_specs, not both")
        if task_heads is None:
            if task_specs is None:
                from .task_registry import resolve_current_tasks

                task_specs = expand_hits_task_inventory(resolve_current_tasks())
            task_heads = build_task_heads(task_specs, self.config.d_model)
        self.task_heads = nn.ModuleDict(task_heads)

    @property
    def task_names(self) -> tuple[str, ...]:
        return tuple(self.task_heads.keys())

    @staticmethod
    def causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)

    def _issuer_memory(
        self,
        values: torch.Tensor,
        projection: FeatureTokenProjection,
        encoder: nn.TransformerEncoder,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        states = projection(values)
        if projection.feature_token_mode:
            states = states.mean(dim=2)
        return encoder(states, src_key_padding_mask=padding_mask)

    def forward(
        self,
        daily_values: torch.Tensor,
        annual_values: torch.Tensor,
        quarterly_values: torch.Tensor,
        *,
        asset_class: str = "equity",
        daily_padding_mask: torch.Tensor | None = None,
        annual_padding_mask: torch.Tensor | None = None,
        quarterly_padding_mask: torch.Tensor | None = None,
        daily_missing: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if asset_class not in self.adapters:
            raise ValueError(f"unsupported asset class: {asset_class}")
        annual_memory = self._issuer_memory(annual_values, self.annual_projection, self.annual_encoder, annual_padding_mask)
        quarterly_states = self._issuer_memory(quarterly_values, self.quarterly_projection, self.quarterly_encoder, quarterly_padding_mask)
        quarterly_memory, _ = self.quarterly_annual_attention(
            quarterly_states, annual_memory, annual_memory, key_padding_mask=annual_padding_mask
        )
        quarterly_memory = self.quarterly_norm(quarterly_states + quarterly_memory)
        daily = self.adapters[asset_class](daily_values, daily_missing)
        daily = self.daily_decoder(
            daily,
            quarterly_memory,
            tgt_mask=self.causal_mask(daily.shape[1], daily.device),
            tgt_key_padding_mask=daily_padding_mask,
            memory_key_padding_mask=quarterly_padding_mask,
        )
        daily_quarterly, _ = self.quarterly_cross_attention(
            daily, quarterly_memory, quarterly_memory, key_padding_mask=quarterly_padding_mask
        )
        daily_annual, _ = self.annual_cross_attention(
            daily, annual_memory, annual_memory, key_padding_mask=annual_padding_mask
        )
        representation = self.output_norm(daily + daily_quarterly + daily_annual)
        return {
            "representation": representation,
            "annual_memory": self.annual_norm(annual_memory),
            "quarterly_memory": quarterly_memory,
            "outputs": {name: head(representation) for name, head in self.task_heads.items()},
        }


def build_asset_adapters(feature_dims: Mapping[str, int], config: MultiRateModelConfig | None = None) -> nn.ModuleDict:
    cfg = config or MultiRateModelConfig()
    return nn.ModuleDict({
        asset: ADAPTERS.get(asset, GenericAssetInputAdapter)(feature_dims.get(asset, feature_dims.get("daily", 1)), cfg.d_model, cfg.feature_token_mode)
        for asset in cfg.asset_classes
        if asset not in cfg.excluded_asset_classes
    })
