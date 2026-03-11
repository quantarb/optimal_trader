from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceSpec:
    """Optional sequence/text settings for architecture-agnostic training specs."""

    input_col: str = "input_text"
    target_col: str = "target_text"
    sequence_id_col: str | None = None
    time_col: str | None = None
    padding: str = "max_length"
    max_source_length: int = 512
    max_target_length: int = 512


@dataclass(frozen=True)
class FitSpec:
    """Training spec shared across model adapters."""

    feature_cols: Sequence[str]
    target_col: str | None = None
    weight_col: str | None = None
    split_ratio: float = 0.8
    signal: str | None = None
    model_tag: str | None = None
    task_type: str = "tabular"
    sequence: SequenceSpec | None = None

    def is_sequence_task(self) -> bool:
        return self.sequence is not None or self.task_type in {"seq2seq", "sequence"}


@dataclass(frozen=True)
class ArtifactSelectionSpec:
    """Artifact IDs selected for a train or score workflow."""

    feature_artifact_id: int
    label_artifact_id: int | None = None
    prediction_artifact_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ModelTrainingSpec:
    """Typed training workflow configuration."""

    model_name: str
    algorithm: str
    task_type: str
    target_col: str
    framework: str = "sklearn"
    split_ratio: float = 0.8
    params: dict[str, Any] = field(default_factory=dict)
    start_date: str | None = None
    end_date: str | None = None
    feature_family: str | None = None
    feature_families: tuple[str, ...] = ()
    label_k: int | None = None
    label_ks: tuple[int, ...] = ()
    min_abs_trade_return: float | None = None
    max_hold_days: int | None = None
    sample_weight_mode: str = "uniform"
    oracle_cluster_keys: tuple[str, ...] = ()
    prediction_artifact_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ModelScoringSpec:
    """Typed scoring workflow configuration."""

    saved_model_id: int
    label_artifact_id: int | None = None
    start_date: str | None = None
    end_date: str | None = None
    prediction_artifact_ids: tuple[int, ...] = ()


def metrics_with_feature_importance(
    metrics: dict[str, Any],
    feature_importance: dict[str, float],
    *,
    top_n: int = 30,
) -> dict[str, Any]:
    out = dict(metrics)
    if feature_importance:
        top = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)[: int(top_n)]
        out["feature_importance_top"] = top
    return out


def copy_feature_importance(feature_importance: dict[str, float]) -> dict[str, float]:
    return dict(feature_importance)


class ModelProtocol:
    def fit(self, df_train: pd.DataFrame, spec: FitSpec) -> "ModelProtocol":
        raise NotImplementedError

    def predict(self, df: pd.DataFrame, *, feature_cols: Sequence[str]) -> np.ndarray:
        raise NotImplementedError
