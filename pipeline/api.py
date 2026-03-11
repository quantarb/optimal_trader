"""Canonical public API.

This module is now a small public façade over focused service modules.
It remains the stable application entrypoint for raw-stack workflows, but the
real implementations live next to their owning concerns.
"""

from __future__ import annotations

from pipeline.api_common import summarize_technical_features
from pipeline.api_datasets import (
    build_dataset_artifacts,
    prepare_entry2exit_dataset,
    prepare_ml_dataset,
)
from pipeline.api_features import (
    build_fundamental_dataframe,
    build_macro_dataframe,
    build_technical_dataframe,
    build_time_dataframe,
)
from pipeline.api_labels import build_label_dataframe
from pipeline.api_models import backtest, predict_panel, train_model

__all__ = [
    "backtest",
    "build_dataset_artifacts",
    "build_fundamental_dataframe",
    "build_label_dataframe",
    "build_macro_dataframe",
    "build_technical_dataframe",
    "build_time_dataframe",
    "predict_panel",
    "prepare_entry2exit_dataset",
    "prepare_ml_dataset",
    "summarize_technical_features",
    "train_model",
]
