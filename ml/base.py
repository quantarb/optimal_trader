from typing import Protocol

from domain.models.interfaces import ArtifactRepository, BacktestRunner, ModelScorer, ModelTrainer
from domain.models.specs import FitSpec, SequenceSpec, copy_feature_importance, metrics_with_feature_importance


class Model(ModelScorer, ModelTrainer, Protocol):
    """Compatibility alias for the domain model protocols."""


def print_model_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  DIAGNOSTIC: {title.upper()}")
    print("=" * 60)


__all__ = [
    "ArtifactRepository",
    "BacktestRunner",
    "FitSpec",
    "Model",
    "ModelScorer",
    "ModelTrainer",
    "SequenceSpec",
    "copy_feature_importance",
    "metrics_with_feature_importance",
    "print_model_section",
]
