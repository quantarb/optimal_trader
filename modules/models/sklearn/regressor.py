from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from modules.models.base import (
    FitSpec,
    Model,
    copy_feature_importance,
    metrics_with_feature_importance,
    print_model_section,
)
from modules.models.metrics import regression_metrics


class SklearnRFRegressor(Model):
    """RandomForest regressor handling continuous targets with detailed diagnostics."""

    def __init__(
            self,
            *,
            test_size: float = 0.2,
            random_state: int = 1337,
            **rf_kwargs: Any,
    ) -> None:
        self.test_size = float(test_size)
        self.random_state = int(random_state)
        self.model = RandomForestRegressor(**rf_kwargs)
        self._is_fit = False
        self._used_features: list[str] = []
        self._metrics: Dict[str, Any] = {}
        self._feature_importance: Dict[str, float] = {}
        self._train_stats: Dict[str, Any] = {}

    def fit(self, df_train: pd.DataFrame, spec: FitSpec, verbose: bool = True) -> "SklearnRFRegressor":
        """Fit model with feature filtering and regression diagnostics."""
        if not spec.target_col:
            raise ValueError("SklearnRFRegressor requires spec.target_col")

        df = df_train.dropna(subset=[spec.target_col]).copy()

        # 1. Feature Filtering (Numeric Only)
        full_X = df[list(spec.feature_cols)]
        self._used_features = full_X.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric = [c for c in spec.feature_cols if c not in self._used_features]
        X = df[self._used_features]

        if X.empty:
            print(f"! ERROR: No numeric features found for regression.")
            return self

        # 2. Target Processing
        y = pd.to_numeric(df[spec.target_col], errors="coerce").fillna(0.0).astype(float)

        # 3. Weights
        w = None
        if spec.weight_col and spec.weight_col in df.columns:
            w = pd.to_numeric(df[spec.weight_col], errors="coerce").fillna(1.0).to_numpy()

        # 4. Train/Test Split (optional in no-holdout mode)
        use_holdout = (self.test_size > 0.0) and (self.test_size < 1.0)
        if use_holdout:
            X_tr, X_va, y_tr, y_va, w_tr, w_va = train_test_split(
                X, y, w,
                test_size=self.test_size,
                random_state=self.random_state,
                shuffle=True,
            )
        else:
            # Full-fit mode: no internal holdout split. Diagnostics are in-sample.
            X_tr, y_tr, w_tr = X, y, w
            X_va, y_va, w_va = X, y, w

        # 5. Training
        self.model.fit(X_tr, y_tr, sample_weight=w_tr)
        self._is_fit = True

        # 6. Metrics and Stats
        y_pred = self.model.predict(X_va)
        self._metrics = regression_metrics(y_va.to_numpy(), y_pred)

        self._train_stats = {
            "n_obs": len(df),
            "n_train": len(X_tr),
            "n_test": len(X_va),
            "eval_mode": "holdout" if use_holdout else "in_sample",
            "target_mean": y.mean(),
            "target_std": y.std(),
            "pred_mean": y_pred.mean(),
            "pred_std": y_pred.std(),
            "dropped_count": len(non_numeric)
        }

        # 7. Feature Importance
        fi = getattr(self.model, "feature_importances_", None)
        if fi is not None:
            self._feature_importance = {str(col): float(val) for col, val in zip(self._used_features, fi)}

        if verbose:
            self.summarize()

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("SklearnRFRegressor not fit()")
        X = df[self._used_features]
        return self.model.predict(X).astype(float)

    def metrics_report(self) -> Dict[str, Any]:
        return metrics_with_feature_importance(self._metrics, self._feature_importance, top_n=30)

    def feature_importance(self) -> Dict[str, float]:
        return copy_feature_importance(self._feature_importance)

    def summarize(self) -> None:
        """Prints regression-specific diagnostics."""
        if not self._is_fit: return
        print_model_section("Sklearn Random Forest Regressor (Diagnostics)")
        s, m = self._train_stats, self._metrics

        print(f"DATASET: {s['n_obs']:,} obs | {len(self._used_features)} features")
        if s.get("eval_mode") == "in_sample":
            print("  - Split Mode: In-sample eval (no internal holdout split).")
        print(f"  - Dropped {s['dropped_count']} non-numeric features.")

        print(f"\nTARGET VS PREDICTION DIST:")
        print(f"  - Actual Mean: {s['target_mean']:.6f} | Std: {s['target_std']:.6f}")
        print(f"  - Pred   Mean: {s['pred_mean']:.6f} | Std: {s['pred_std']:.6f}")

        print(f"\nPERFORMANCE:")
        print(f"  - R-Squared:  {m.get('r2', 0):.4f}")
        print(f"  - MAE:        {m.get('mae', 0):.6f}")
        print(f"  - MSE:        {m.get('mse', 0):.6f}")

        if self._feature_importance:
            print(f"\nTOP 10 FEATURES:")
            top = sorted(self._feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for col, val in top:
                print(f"  - {col}: {val:.4f}")
        print("=" * 60)
