"""Reusable multi-rate, multi-asset transformer research components."""

from .alignment import (
    align_available_rows,
    assert_point_in_time,
    build_asof_memory,
)
from .config import MultiRateModelConfig, MultiRateTaskConfig
from .losses import MultiTaskLoss, pairwise_logistic_ranking_loss, percentile_huber_loss
from .heads import build_task_heads, expand_hits_task_inventory
from .ranking import add_hits_rank_targets, percentile_rank_within_group
from .walk_forward import WalkForwardFold, anchored_folds
from .dataset import MultiRateWindow, build_multirate_window
from .warehouse_data import build_rate_views, load_existing_feature_family_panel
from .trading_policy import build_legacy_compatible_scores, task_output_frame

__all__ = [
    "MultiRateModelConfig",
    "MultiRateTaskConfig",
    "MultiTaskLoss",
    "add_hits_rank_targets",
    "align_available_rows",
    "assert_point_in_time",
    "build_asof_memory",
    "build_task_heads",
    "expand_hits_task_inventory",
    "pairwise_logistic_ranking_loss",
    "percentile_huber_loss",
    "WalkForwardFold",
    "anchored_folds",
    "MultiRateWindow",
    "build_multirate_window",
    "build_rate_views",
    "load_existing_feature_family_panel",
    "build_legacy_compatible_scores",
    "task_output_frame",
]
