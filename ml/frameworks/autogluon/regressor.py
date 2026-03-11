from __future__ import annotations

import inspect
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.base import (
    FitSpec,
    Model,
    copy_feature_importance,
    metrics_with_feature_importance,
    print_model_section,
)
from ml.metrics import regression_metrics


class AutoGluonRegressor(Model):
    """AutoGluon regressor aligned with the SklearnRFRegressor adapter behavior."""

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
        self._is_fit = False
        self._used_features: list[str] = []
        self._metrics: Dict[str, Any] = {}
        self._feature_importance: Dict[str, float] = {}
        self._train_stats: Dict[str, Any] = {}

    def fit(self, df_train: pd.DataFrame, spec: FitSpec, verbose: bool = True) -> "AutoGluonRegressor":
        if not spec.target_col:
            raise ValueError("AutoGluonRegressor requires spec.target_col")

        from autogluon.tabular import TabularPredictor  # type: ignore

        df = df_train.dropna(subset=[spec.target_col]).copy()
        full_X = df[list(spec.feature_cols)]
        self._used_features = full_X.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric = [c for c in spec.feature_cols if c not in self._used_features]
        if not self._used_features:
            print("! ERROR: No numeric features found for regression.")
            return self

        y_col = spec.target_col

        d = df[self._used_features + [y_col]].copy()
        d[y_col] = pd.to_numeric(d[y_col], errors="coerce")

        split_ratio = float(getattr(spec, "split_ratio", 0.8))
        test_size = 1.0 - split_ratio

        weight_col = None
        if spec.weight_col and spec.weight_col in df.columns:
            weight_col = "__sample_weight__"
            # Align by row position (not index) to support non-unique MultiIndex.
            d[weight_col] = pd.to_numeric(df[spec.weight_col], errors="coerce").fillna(1.0).astype(float).to_numpy()

        d = d.dropna(subset=[y_col])
        # AutoGluon requires unique indices in train/tuning datasets.
        d = d.reset_index(drop=True)

        train_df, val_df = train_test_split(
            d,
            test_size=test_size,
            random_state=self.random_state,
            shuffle=True,
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        predictor_kwargs: Dict[str, Any] = {"label": y_col, "problem_type": "regression"}
        if weight_col is not None:
            try:
                init_sig = inspect.signature(TabularPredictor.__init__).parameters
                if "sample_weight" in init_sig:
                    predictor_kwargs["sample_weight"] = weight_col
            except Exception:
                pass

        self._pred = TabularPredictor(**predictor_kwargs).fit(
            train_data=train_df,
            presets=self.presets,
            time_limit=self.time_limit,
            tuning_data=val_df,
            verbosity=0,
        )
        self._is_fit = True

        y_pred = np.asarray(self._pred.predict(val_df[self._used_features]), dtype=float).reshape(-1)
        y_true = val_df[y_col].to_numpy(dtype=float)
        self._metrics = regression_metrics(y_true, y_pred)

        # feature importance (AutoGluon permutation importance)
        try:
            fi_df = self._pred.feature_importance(val_df)
            if "importance" in fi_df.columns:
                self._feature_importance = {str(k): float(v) for k, v in fi_df["importance"].to_dict().items()}
        except Exception:
            self._feature_importance = {}

        self._train_stats = {
            "n_obs": len(d),
            "n_train": len(train_df),
            "n_test": len(val_df),
            "split_ratio": split_ratio,
            "n_features_numeric": len(self._used_features),
            "dropped_count": len(non_numeric),
            "target_mean": float(np.mean(y_true)),
            "target_std": float(np.std(y_true)),
            "pred_mean": float(np.mean(y_pred)),
            "pred_std": float(np.std(y_pred)),
        }

        if verbose:
            self.summarize()

        return self

    def predict(self, df: pd.DataFrame, *, feature_cols=None) -> np.ndarray:
        if self._pred is None or not self._is_fit:
            raise RuntimeError("AutoGluonRegressor not fit()")

        cols = list(feature_cols) if feature_cols is not None else list(self._used_features)
        X = df[cols].copy()
        yhat = self._pred.predict(X)
        return np.asarray(yhat, dtype=float).reshape(-1)

    def metrics_report(self) -> Dict[str, Any]:
        return metrics_with_feature_importance(self._metrics, self._feature_importance, top_n=30)

    def feature_importance(self) -> Dict[str, float]:
        """Full feature importance mapping (may be empty)."""
        return copy_feature_importance(self._feature_importance)

    def summarize(self) -> None:
        if not self._is_fit:
            return

        print_model_section("AutoGluon Regressor (Diagnostics)")
        s, m = self._train_stats, self._metrics

        print(f"DATASET: {s['n_obs']:,} obs | {s['n_features_numeric']} features")
        print(f"  - Random Split:       {s['split_ratio']:.1%} Train / {1 - s['split_ratio']:.1%} Test")
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
