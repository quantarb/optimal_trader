from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from modules.models.base import FitSpec, Model, copy_feature_importance, metrics_with_feature_importance
from modules.models.metrics import regression_metrics


class AutoGluonRegressor(Model):
    """AutoGluon Tabular regressor. predict() returns yhat."""

    def __init__(
        self,
        *,
        presets: str = "medium_quality",
        time_limit: int | None = None,
        test_size: float = 0.2,
        random_state: int = 1337,
    ) -> None:
        self.presets = str(presets)
        self.time_limit = time_limit
        self.test_size = float(test_size)
        self.random_state = int(random_state)
        self._pred = None
        self._metrics: Dict[str, Any] = {}
        self._feature_importance: Dict[str, float] = {}

    def fit(self, df_train: pd.DataFrame, spec: FitSpec) -> "AutoGluonRegressor":
        if not spec.target_col:
            raise ValueError("AutoGluonRegressor requires spec.target_col")

        from autogluon.tabular import TabularPredictor  # type: ignore

        df = df_train.dropna(subset=[spec.target_col]).copy()
        X_cols = list(spec.feature_cols)
        y_col = spec.target_col

        d = df[X_cols + [y_col]].copy()
        d[y_col] = pd.to_numeric(d[y_col], errors="coerce")

        train_df, val_df = train_test_split(
            d,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
        )

        self._pred = TabularPredictor(label=y_col, problem_type="regression").fit(
            train_data=train_df,
            presets=self.presets,
            time_limit=self.time_limit,
            tuning_data=val_df,
            verbosity=0,
        )

        y_pred = np.asarray(self._pred.predict(val_df[X_cols]), dtype=float).reshape(-1)
        y_true = val_df[y_col].to_numpy(dtype=float)
        self._metrics = regression_metrics(y_true, y_pred)

        # feature importance (AutoGluon permutation importance)
        try:
            fi_df = self._pred.feature_importance(val_df)
            if "importance" in fi_df.columns:
                self._feature_importance = {str(k): float(v) for k, v in fi_df["importance"].to_dict().items()}
        except Exception:
            self._feature_importance = {}

        return self

    def predict(self, df: pd.DataFrame, *, feature_cols) -> np.ndarray:
        if self._pred is None:
            raise RuntimeError("AutoGluonRegressor not fit()")

        X = df[list(feature_cols)].copy()
        yhat = self._pred.predict(X)
        return np.asarray(yhat, dtype=float).reshape(-1)

    def metrics_report(self) -> Dict[str, Any]:
        return metrics_with_feature_importance(self._metrics, self._feature_importance, top_n=30)

    def feature_importance(self) -> Dict[str, float]:
        """Full feature importance mapping (may be empty)."""
        return copy_feature_importance(self._feature_importance)
