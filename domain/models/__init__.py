"""Modeling domain contracts and typed specs."""

from domain.models.datasets import dedupe_label_frame, feature_columns_from_frame, filter_frame_by_date
from domain.models.feature_families import infer_feature_family_columns
from domain.models.interfaces import ArtifactRepository, BacktestRunner, ModelScorer, ModelTrainer
from domain.models.specs import (
    ArtifactSelectionSpec,
    ArtifactTrainingDatasetSpec,
    FitSpec,
    ModelScoringSpec,
    ModelTrainingSpec,
    SequenceSpec,
    metrics_with_feature_importance,
    copy_feature_importance,
)

__all__ = [
    "ArtifactRepository",
    "ArtifactSelectionSpec",
    "ArtifactTrainingDatasetSpec",
    "BacktestRunner",
    "FitSpec",
    "ModelScorer",
    "ModelTrainingSpec",
    "ModelScoringSpec",
    "ModelTrainer",
    "SequenceSpec",
    "copy_feature_importance",
    "dedupe_label_frame",
    "feature_columns_from_frame",
    "filter_frame_by_date",
    "infer_feature_family_columns",
    "metrics_with_feature_importance",
]
