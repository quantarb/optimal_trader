from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd

from domain.models.specs import FitSpec


class ModelTrainer(Protocol):
    """Framework-agnostic model trainer contract."""

    def fit(self, df_train: pd.DataFrame, spec: FitSpec, verbose: bool = True, validation_df: pd.DataFrame | None = None) -> Any:
        ...


class ModelScorer(Protocol):
    """Framework-agnostic scoring contract."""

    def predict(self, df: pd.DataFrame, *, feature_cols: Sequence[str]) -> np.ndarray:
        ...


class ArtifactRepository(Protocol):
    """Repository interface for pipeline/model artifacts."""

    def get_pipeline_artifact(self, artifact_id: int, *, artifact_type: str | None = None) -> Any | None:
        ...

    def list_pipeline_artifacts(self, artifact_ids: Sequence[int], *, artifact_types: Sequence[str] | None = None) -> list[Any]:
        ...

    def get_saved_model(self, saved_model_id: int) -> Any | None:
        ...


class BacktestRunner(Protocol):
    """Backtest workflow interface for strategy evaluation."""

    def run(self, frame: pd.DataFrame, *, config: dict[str, Any]) -> pd.DataFrame:
        ...
