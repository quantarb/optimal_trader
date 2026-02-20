from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AutoEncoderConfig(BaseModel):
    """Config for TorchAutoEncoder.

    Notes:
      - If embed_dim <= 0, the model is numeric-only (no categorical path).
      - If embed_dim > 0, the model is hybrid and requires categorical columns.
    """
    model_config = ConfigDict(extra="forbid")

    hidden_dim: int = 128
    bottleneck_dim: int = 32
    n_layers: int = Field(default=2, ge=1)
    min_layer_dim: int = Field(default=2, ge=1)
    embed_dim: int = 16

    epochs: int = 30
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-4

    device: str = "cpu"

    # Stability / preprocessing
    z_clip: float = Field(default=10.0, ge=0.0)
    robust_clip_lo_pct: float = Field(default=0.1, ge=0.0, le=50.0)
    robust_clip_hi_pct: float = Field(default=99.9, ge=50.0, le=100.0)
    denoise_std: float = Field(default=0.02, ge=0.0)
    loss_topk_frac: float = Field(default=1.0, ge=0.0, le=1.0)
    score_topk_frac: float = Field(default=0.2, ge=0.0, le=1.0)
    feature_weight_clip_min: float = Field(default=0.25, gt=0.0)
    feature_weight_clip_max: float = Field(default=4.0, gt=0.0)

    # Latent-manifold reference
    latent_ref_max_points: int = Field(default=50000, ge=1000)
    latent_ref_random_state: int = 1337
