from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Sequence

import pandas as pd

PricesFrame = pd.DataFrame
FeaturesFrame = pd.DataFrame
LabelsFrame = pd.DataFrame
PanelFrame = pd.DataFrame


@dataclass
class DatasetArtifacts:
    daily_by_symbol: Dict[str, pd.DataFrame]
    training_df: pd.DataFrame
    inference_panel: PanelFrame
    feature_cols: Sequence[str]
    meta: Dict[str, Any]


@dataclass(frozen=True)
class ModelArtifact:
    model: Any
    meta: Dict[str, Any]
    feature_cols: Sequence[str]
    target_col: str
    weight_col: Optional[str] = None


@dataclass(frozen=True)
class PredictionsArtifact:
    panel: PanelFrame
    pred_cols: Sequence[str]
    meta: Dict[str, Any]


class Trainer(Protocol):
    def fit(
        self,
        *,
        train_df: pd.DataFrame,
        feature_cols: Sequence[str],
        target_col: str,
        weight_col: Optional[str] = None,
    ) -> ModelArtifact:
        ...


class Predictor(Protocol):
    def predict(self, *, model_artifact: ModelArtifact, panel: PanelFrame) -> PredictionsArtifact:
        ...
