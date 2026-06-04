from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from ml.base import FitSpec, Model, copy_feature_importance, metrics_with_feature_importance, print_model_section
from ml.metrics import regression_metrics


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
        self._metrics: dict[str, Any] = {}
        self._feature_importance: dict[str, float] = {}
        self._train_stats: dict[str, Any] = {}

    def fit(
        self,
        df_train: pd.DataFrame,
        spec: FitSpec,
        verbose: bool = True,
        validation_df: pd.DataFrame | None = None,
    ) -> "SklearnRFRegressor":
        if not spec.target_col:
            raise ValueError("SklearnRFRegressor requires spec.target_col")

        df = df_train.dropna(subset=[spec.target_col]).copy()
        val_df = None
        if validation_df is not None:
            val_df = validation_df.dropna(subset=[spec.target_col]).copy()
            if val_df.empty:
                raise ValueError("validation_df has no usable rows for SklearnRFRegressor.")

        full_x = df[list(spec.feature_cols)]
        self._used_features = full_x.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric = [col for col in spec.feature_cols if col not in self._used_features]
        x = df[self._used_features]
        if x.empty:
            raise ValueError("No numeric features found for regression.")

        y = pd.to_numeric(df[spec.target_col], errors="coerce").fillna(0.0).astype(float)

        weights = None
        if spec.weight_col and spec.weight_col in df.columns:
            weights = pd.to_numeric(df[spec.weight_col], errors="coerce").fillna(1.0).to_numpy()

        if val_df is not None:
            x_tr, y_tr, w_tr = x, y, weights
            x_va = val_df[self._used_features]
            y_va = pd.to_numeric(val_df[spec.target_col], errors="coerce").fillna(0.0).astype(float)
            w_va = None
            if spec.weight_col and spec.weight_col in val_df.columns:
                w_va = pd.to_numeric(val_df[spec.weight_col], errors="coerce").fillna(1.0).to_numpy()
            eval_mode = "external_validation"
        else:
            use_holdout = (self.test_size > 0.0) and (self.test_size < 1.0)
            if use_holdout:
                x_tr, x_va, y_tr, y_va, w_tr, w_va = train_test_split(
                    x,
                    y,
                    weights,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    shuffle=True,
                )
            else:
                x_tr, y_tr, w_tr = x, y, weights
                x_va, y_va, w_va = x, y, weights
            eval_mode = "holdout" if use_holdout else "in_sample"

        self.model.fit(x_tr, y_tr, sample_weight=w_tr)
        self._is_fit = True

        y_pred = self.model.predict(x_va)
        self._metrics = regression_metrics(y_va.to_numpy(), y_pred)
        self._train_stats = {
            "n_obs": len(df),
            "n_train": len(x_tr),
            "n_test": len(x_va),
            "eval_mode": eval_mode,
            "target_col": str(spec.target_col),
            "model_tag": str(spec.model_tag or ""),
            "signal": str(spec.signal or ""),
            "target_mean": y.mean(),
            "target_std": y.std(),
            "pred_mean": y_pred.mean(),
            "pred_std": y_pred.std(),
            "dropped_count": len(non_numeric),
        }

        feature_importances = getattr(self.model, "feature_importances_", None)
        if feature_importances is not None:
            self._feature_importance = {str(col): float(val) for col, val in zip(self._used_features, feature_importances)}

        if verbose:
            self.summarize()
        return self

    def predict(self, df: pd.DataFrame, *, feature_cols=None) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("SklearnRFRegressor not fit()")
        used_features = list(feature_cols) if feature_cols is not None else list(self._used_features)
        x = df[used_features]
        return self.model.predict(x).astype(float)

    def metrics_report(self) -> dict[str, Any]:
        return metrics_with_feature_importance(self._metrics, self._feature_importance, top_n=30)

    def feature_importance(self) -> dict[str, float]:
        return copy_feature_importance(self._feature_importance)

    def summarize(self) -> None:
        if not self._is_fit:
            return
        print_model_section("Sklearn Random Forest Regressor (Diagnostics)")
        stats = self._train_stats
        metrics = self._metrics

        print(f"DATASET: {stats['n_obs']:,} obs | {len(self._used_features)} features")
        print(f"  - Target:     {stats.get('target_col') or ''}")
        if stats.get("model_tag"):
            print(f"  - Model Role: {stats['model_tag']}")
        if stats.get("signal"):
            print(f"  - Signal:     {stats['signal']}")
        if stats.get("eval_mode") == "in_sample":
            print("  - Split Mode: In-sample eval (no internal holdout split).")
        elif stats.get("eval_mode") == "external_validation":
            print("  - Split Mode: External validation frame.")
        print(f"  - Dropped {stats['dropped_count']} non-numeric features.")

        print("\nTARGET VS PREDICTION DIST:")
        print(f"  - Actual Mean: {stats['target_mean']:.6f} | Std: {stats['target_std']:.6f}")
        print(f"  - Pred   Mean: {stats['pred_mean']:.6f} | Std: {stats['pred_std']:.6f}")

        print("\nPERFORMANCE:")
        print(f"  - R-Squared:  {metrics.get('r2', 0):.4f}")
        print(f"  - MAE:        {metrics.get('mae', 0):.6f}")
        print(f"  - MSE:        {metrics.get('mse', 0):.6f}")

        if self._feature_importance:
            print("\nTOP 10 FEATURES:")
            top = sorted(self._feature_importance.items(), key=lambda item: item[1], reverse=True)[:10]
            for col, val in top:
                print(f"  - {col}: {val:.4f}")
        print("=" * 60)
