"""Masked pointwise, pairwise, and multi-task objectives."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


def percentile_huber_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    error = nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    mask = mask.to(error.dtype)
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(-1)
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def pairwise_logistic_ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    group_ids: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Compare only rows in the same group and ignore tied targets."""
    losses = []
    valid = valid_mask.bool()
    for group in torch.unique(group_ids[valid]):
        indices = torch.nonzero(valid & group_ids.eq(group), as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        left, right = torch.triu_indices(indices.numel(), indices.numel(), offset=1, device=indices.device)
        left_index, right_index = indices[left], indices[right]
        direction = torch.sign(target[left_index] - target[right_index])
        keep = direction.ne(0)
        if keep.any():
            losses.append(torch.nn.functional.softplus(-(prediction[left_index][keep] - prediction[right_index][keep]) * direction[keep]))
    if not losses:
        return prediction.new_zeros(())
    return torch.cat(losses).mean()


class MultiTaskLoss(nn.Module):
    """Masked task-family weighting with optional uncertainty parameters."""

    def __init__(self, task_weights: Mapping[str, float] | None = None, uncertainty_weighting: bool = False):
        super().__init__()
        self.task_weights = dict(task_weights or {})
        self.uncertainty_weighting = uncertainty_weighting
        self.log_variance = nn.ParameterDict({
            name.replace(":", "__"): nn.Parameter(torch.zeros(()))
            for name in self.task_weights
        }) if uncertainty_weighting else nn.ParameterDict()

    def forward(self, losses: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        weighted: dict[str, torch.Tensor] = {}
        total = next(iter(losses.values())).new_zeros(()) if losses else torch.tensor(0.0)
        for name, loss in losses.items():
            weight = float(self.task_weights.get(name, 1.0))
            if self.uncertainty_weighting and name.replace(":", "__") in self.log_variance:
                variance = self.log_variance[name.replace(":", "__")]
                contribution = weight * (torch.exp(-variance) * loss + variance)
            else:
                contribution = weight * loss
            weighted[name] = contribution
            total = total + contribution
        return total, weighted

