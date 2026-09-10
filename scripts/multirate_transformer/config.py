"""Typed configuration for the multi-rate transformer workflow."""

from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_ASSET_CLASSES = (
    "equity",
    "corporate_bond",
    "preferred",
    "warrant",
    "unit",
    "note_bond",
    "adr",
    "ordinary",
    "etf",
    "option",
)


@dataclass(frozen=True)
class MultiRateModelConfig:
    d_model: int = 256
    num_heads: int = 8
    annual_layers: int = 4
    quarterly_layers: int = 4
    decoder_layers: int = 6
    dropout: float = 0.1
    annual_sequence_length: int = 5
    quarterly_sequence_length: int = 12
    daily_sequence_length: int = 256
    quarterly_condition_on_annual: bool = True
    separate_cross_attention_by_rate: bool = True
    shared_instrument_decoder: bool = True
    asset_specific_adapters: bool = True
    feature_token_mode: bool = False
    asset_classes: tuple[str, ...] = field(default_factory=lambda: DEFAULT_ASSET_CLASSES)
    # Asset class changes the input adapter and feature schema, not the
    # supervised task inventory.  A task with an unavailable label is masked
    # at the sample/task level by the trainer.
    excluded_asset_classes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if any(value <= 0 for value in (self.annual_layers, self.quarterly_layers, self.decoder_layers)):
            raise ValueError("encoder and decoder layer counts must be positive")
        if any(value <= 0 for value in (self.annual_sequence_length, self.quarterly_sequence_length, self.daily_sequence_length)):
            raise ValueError("sequence lengths must be positive")
        excluded = {name.lower() for name in self.excluded_asset_classes}
        if any(name.lower() in excluded for name in self.asset_classes):
            raise ValueError(f"excluded asset class was requested: {sorted(excluded)}")


@dataclass(frozen=True)
class MultiRateTaskConfig:
    """Task selection and loss priorities resolved from the repository registry."""

    task_names: tuple[str, ...] = ()
    raw_hits_weight: float = 1.0
    asset_rank_weight: float = 0.75
    issuer_rank_weight: float = 0.5
    return_risk_weight: float = 0.5
    event_weight: float = 0.25
    context_weight: float = 0.1
    use_pairwise_ranking: bool = True
    use_listwise_ranking: bool = False
    uncertainty_weighting: bool = False
    gradnorm: bool = False
