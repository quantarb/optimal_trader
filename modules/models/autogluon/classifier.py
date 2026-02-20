from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from modules.models.base import FitSpec, Model, copy_feature_importance, metrics_with_feature_importance
from modules.models.metrics import binary_classifier_metrics_from_scores


class AutoGluonClassifier(Model):
    """AutoGluon Tabular binary classifier. predict() returns P(y=1)."""

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

    def fit(self, df_train: pd.DataFrame, spec: FitSpec) -> "AutoGluonClassifier":
        if not spec.target_col:
            raise ValueError("AutoGluonClassifier requires spec.target_col")

        from autogluon.tabular import TabularPredictor  # type: ignore

        df = df_train.dropna(subset=[spec.target_col]).copy()
        X_cols = list(spec.feature_cols)
        y_col = spec.target_col

        d = df[X_cols + [y_col]].copy()
        d[y_col] = pd.to_numeric(d[y_col], errors="coerce").fillna(0).astype(int)

        train_df, val_df = train_test_split(
            d,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
            stratify=d[y_col],
        )

        self._pred = TabularPredictor(label=y_col, problem_type="binary").fit(
            train_data=train_df,
            presets=self.presets,
            time_limit=self.time_limit,
            tuning_data=val_df,
            verbosity=0,
        )

        # metrics on val
        proba_df = self._pred.predict_proba(val_df[X_cols])
        proba = np.asarray(proba_df)[:, -1]
        y_true = val_df[y_col].to_numpy()
        self._metrics = binary_classifier_metrics_from_scores(y_true, proba, threshold=0.5)

        # feature importance (AutoGluon permutation importance)
        try:
            fi_df = self._pred.feature_importance(val_df)
            # expected columns: "importance" (and maybe "stddev") with index=feature name
            if "importance" in fi_df.columns:
                self._feature_importance = {str(k): float(v) for k, v in fi_df["importance"].to_dict().items()}
        except Exception:
            self._feature_importance = {}

        return self

    def predict(self, df: pd.DataFrame, *, feature_cols) -> np.ndarray:
        if self._pred is None:
            raise RuntimeError("AutoGluonClassifier not fit()")

        X = df[list(feature_cols)].copy()
        proba_df = self._pred.predict_proba(X)
        arr = np.asarray(proba_df)
        return arr[:, -1].astype(float)

    def metrics_report(self) -> Dict[str, Any]:
        return metrics_with_feature_importance(self._metrics, self._feature_importance, top_n=30)

    def feature_importance(self) -> Dict[str, float]:
        """Full feature importance mapping (may be empty)."""
        return copy_feature_importance(self._feature_importance)
