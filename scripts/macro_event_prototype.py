"""Shared company-response prototype head for macro event tokens."""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn


MACRO_RESPONSE_CLASSES = (
    "strong_negative",
    "negative",
    "neutral",
    "positive",
    "strong_positive",
)


class MacroEventResponsePrototypeHead(nn.Module):
    """Predict company response classes with one shared event head.

    The head is shared across all macro event types.  Event type, country,
    surprise, impact, and other release metadata are represented by the
    ``event_features`` input; the company embedding conditions the response.
    A single set of response prototypes is used for every event type.
    """

    def __init__(
        self,
        company_dim: int,
        event_dim: int,
        n_classes: int = len(MACRO_RESPONSE_CLASSES),
        metric_dim: int | None = None,
    ):
        super().__init__()
        metric_dim = metric_dim or int(os.getenv("GNN_MACRO_METRIC_DIM", "24"))
        self.company_projection = nn.Sequential(
            nn.Linear(company_dim, metric_dim), nn.LayerNorm(metric_dim), nn.GELU()
        )
        self.event_projection = nn.Sequential(
            nn.Linear(event_dim, metric_dim), nn.LayerNorm(metric_dim), nn.GELU()
        )
        self.interaction = nn.Sequential(
            nn.Linear(metric_dim * 3, metric_dim), nn.LayerNorm(metric_dim), nn.GELU()
        )
        self.prototypes = nn.Parameter(torch.randn(n_classes, metric_dim) * 0.02)
        self.log_temperature = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))

    def forward(self, company_embedding: torch.Tensor, event_features: torch.Tensor) -> torch.Tensor:
        company = nn.functional.normalize(self.company_projection(company_embedding), dim=-1)
        event = nn.functional.normalize(self.event_projection(event_features), dim=-1)
        pair = self.interaction(torch.cat([company, event, company * event], dim=-1))
        pair = nn.functional.normalize(pair, dim=-1)
        prototypes = nn.functional.normalize(self.prototypes, dim=-1)
        temperature = self.log_temperature.exp().clamp(1.0, 50.0)
        return temperature * pair @ prototypes.T


def macro_response_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Prototype classification loss shared across all event types."""
    return nn.functional.cross_entropy(logits, targets)
