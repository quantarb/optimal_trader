from __future__ import annotations

from typing import Any, Optional, Sequence

import pandas as pd

from backtest import run_backtest
from pipeline.contracts_runtime import DatasetArtifacts, ModelArtifact, PredictionsArtifact
from utils.panel import ensure_panel_index


def train_model(
    *,
    trainer,
    dataset: DatasetArtifacts,
    feature_cols: Sequence[str],
    target_col: str,
    weight_col: Optional[str] = None,
) -> ModelArtifact:
    return trainer.fit(
        train_df=dataset.training_df,
        feature_cols=list(feature_cols),
        target_col=str(target_col),
        weight_col=str(weight_col) if weight_col else None,
    )


def predict_panel(
    *,
    predictor,
    model_artifact: ModelArtifact,
    panel: pd.DataFrame,
) -> PredictionsArtifact:
    panel = ensure_panel_index(panel)
    return predictor.predict(model_artifact=model_artifact, panel=panel)


def backtest(
    *,
    panel: pd.DataFrame,
    strategy,
    title: Optional[str] = None,
    **engine_kwargs: Any,
):
    panel = ensure_panel_index(panel)
    return run_backtest(panel=panel, strategy=strategy, title=title, **engine_kwargs)


__all__ = ["backtest", "predict_panel", "train_model"]
