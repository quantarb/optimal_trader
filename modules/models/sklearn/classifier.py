from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from modules.models.base import FitSpec, Model, print_model_section
from modules.models.metrics import binary_classifier_metrics_from_scores, classification_metrics


class SklearnRFClassifier(Model):
    """RandomForest classifier handling binary and multi-class targets with detailed diagnostics."""

    def __init__(
            self,
            *,
            random_state: int = 1337,
            **rf_kwargs: Any,
    ) -> None:
        self.random_state = int(random_state)
        self.model = RandomForestClassifier(**rf_kwargs)
        self._is_fit = False
        self._used_features: list[str] = []
        self._metrics: Dict[str, Any] = {}
        self._feature_importance: Dict[str, float] = {}
        self._train_stats: Dict[str, Any] = {}
        self._class_mapping: Dict[int, str] = {}

    def fit(self, df_train: pd.DataFrame, spec: FitSpec, verbose: bool = True) -> "SklearnRFClassifier":
        """Fit model with internal categorical filtering, random splits, and distribution checks."""
        if not spec.target_col:
            raise ValueError("SklearnRFClassifier requires spec.target_col")

        df = df_train.dropna(subset=[spec.target_col]).copy()

        # 1. Feature Filtering
        full_X = df[list(spec.feature_cols)]
        self._used_features = full_X.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric = [c for c in spec.feature_cols if c not in self._used_features]
        X = df[self._used_features]

        if X.empty:
            print(f"! ERROR: No numeric features found. Check your feature_cols.")
            return self

        # 2. Target Processing (Fixed: Uses factorize to prevent string-drop)
        target_series = df[spec.target_col]
        if target_series.dtype == 'object' or target_series.dtype.name == 'category':
            codes, uniques = pd.factorize(target_series)
            y = pd.Series(codes, index=df.index)
            self._class_mapping = {i: str(val) for i, val in enumerate(uniques)}
        else:
            y = pd.to_numeric(target_series, errors="coerce").fillna(0).astype(int)
            unique_vals = np.sort(y.unique())
            self._class_mapping = {int(v): str(v) for v in unique_vals}

        unique_classes = np.sort(y.unique())
        n_classes = len(unique_classes)

        if n_classes < 2:
            print(f"\n! ERROR: Target '{spec.target_col}' has only one class: {unique_classes}.")
            return self

        # 3. Random Shuffle Split (optional in no-holdout mode)
        split_ratio = getattr(spec, "split_ratio", 0.8)
        test_size = 1.0 - split_ratio

        w = None
        if spec.weight_col and spec.weight_col in df.columns:
            w = pd.to_numeric(df[spec.weight_col], errors="coerce").fillna(1.0).to_numpy()

        use_holdout = (test_size > 0.0) and (test_size < 1.0)
        if use_holdout:
            X_tr, X_va, y_tr, y_va, w_tr, w_va = train_test_split(
                X, y, w,
                test_size=test_size,
                random_state=self.random_state,
                shuffle=True,
                stratify=y,
            )
        else:
            # Full-fit mode: no internal holdout split. Diagnostics are in-sample.
            X_tr, y_tr, w_tr = X, y, w
            X_va, y_va, w_va = X, y, w

        # 4. Capture stats
        self._train_stats = {
            "n_obs": len(df),
            "n_train": len(X_tr),
            "n_test": len(X_va),
            "split_ratio": split_ratio,
            "eval_mode": "holdout" if use_holdout else "in_sample",
            "n_features_numeric": len(self._used_features),
            "dropped_count": len(non_numeric),
            "classes": unique_classes.tolist(),
            "is_binary": (n_classes == 2),
            "class_dist": {
                "train": y_tr.value_counts().to_dict(),
                "test": y_va.value_counts().to_dict()
            }
        }

        # 5. Model Training
        self.model.fit(X_tr, y_tr, sample_weight=w_tr)
        self._is_fit = True

        # 6. Metric Calculation (Fixed: Explicit Confusion Matrix)
        y_pred = self.model.predict(X_va)

        if self._train_stats["is_binary"]:
            proba_matrix = self.model.predict_proba(X_va)
            proba = proba_matrix[:, 1] if proba_matrix.shape[1] > 1 else proba_matrix[:, 0]
            self._metrics = binary_classifier_metrics_from_scores(y_va.to_numpy(), proba, threshold=0.5)
        else:
            self._metrics = classification_metrics(y_va.to_numpy(), y_pred, labels=unique_classes.tolist())

        # Force add confusion matrix for visualization
        self._metrics["confusion_matrix"] = confusion_matrix(y_va, y_pred, labels=unique_classes)

        # 7. Feature Importance
        fi = getattr(self.model, "feature_importances_", None)
        if fi is not None:
            self._feature_importance = {str(col): float(val) for col, val in zip(self._used_features, fi)}

        if verbose:
            self.summarize()

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("SklearnRFClassifier not fit()")
        X = df[self._used_features]
        return self.model.predict(X).astype(float)

    def summarize(self) -> None:
        """Detailed diagnostic including multi-set class distribution and confusion matrix."""
        if not self._is_fit: return

        print_model_section("Sklearn Random Forest Classifier (Test Results)")
        s = self._train_stats
        m = self._metrics

        print(f"DATASET & SPLIT:")
        print(f"  - Total Observations: {s['n_obs']:,}")
        if s.get("eval_mode") == "holdout":
            print(f"  - Random Split:       {s['split_ratio']:.1%} Train / {1 - s['split_ratio']:.1%} Test")
        else:
            print("  - Split Mode:         In-sample eval (no internal holdout split)")
        print(f"  - Features:           {s['n_features_numeric']} numeric (filtered {s['dropped_count']} strings)")

        print(f"\nCLASS DISTRIBUTION (Mapping: {self._class_mapping}):")
        print(f"               Train Set              Test Set")
        dist = s['class_dist']
        for c in s['classes']:
            tr_cnt = dist['train'].get(c, 0)
            va_cnt = dist['test'].get(c, 0)
            tr_pct = tr_cnt / s['n_train']
            va_pct = va_cnt / s['n_test']
            name = self._class_mapping.get(c, str(c))
            print(f"  - {name:>8}: {tr_cnt:>7,} ({tr_pct:>5.1%})   {va_cnt:>7,} ({va_pct:>5.1%})")

        print(f"\nTEST PERFORMANCE:")
        print(f"  - Accuracy:           {m.get('accuracy', 0):.2%}")
        if s['is_binary']:
            print(f"  - ROC AUC:            {m.get('roc_auc', 0):.4f}")

        # --- DYNAMIC CONFUSION MATRIX (RESTORED) ---
        cm = m.get("confusion_matrix")
        if cm is not None:
            print(f"\nCONFUSION MATRIX (Test Set):")
            classes = s['classes']
            header = "            " + "".join([f"Pred {self._class_mapping.get(c, c):>7} " for c in classes])
            print(header)
            for i, row_label in enumerate(classes):
                name = self._class_mapping.get(row_label, row_label)
                row_str = f"True {name:>7}: "
                row_str += "".join([f"{val:>12} " for val in cm[i]])
                print(row_str)

        if self._feature_importance:
            print(f"\nTOP 10 FEATURES:")
            top = sorted(self._feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for col, val in top:
                print(f"  - {col}: {val:.4f}")
        print("=" * 60)

    def metrics_report(self) -> Dict[str, Any]:
        return dict(self._metrics)
