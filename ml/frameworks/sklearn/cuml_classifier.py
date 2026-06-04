from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from ml.base import FitSpec, Model, copy_feature_importance, metrics_with_feature_importance, print_model_section
from ml.metrics import binary_classifier_metrics_from_scores, classification_metrics


class CumlRFClassifier(Model):
    """cuML RandomForest classifier with the same public shape as SklearnRFClassifier."""

    _UNSUPPORTED_KWARGS = {"class_weight", "n_jobs"}

    def __init__(
        self,
        *,
        random_state: int = 1337,
        **rf_kwargs: Any,
    ) -> None:
        try:
            from cuml.ensemble import RandomForestClassifier
        except Exception as exc:  # pragma: no cover - depends on local CUDA env
            raise ImportError("CumlRFClassifier requires RAPIDS cuML in the active Python environment.") from exc

        self.random_state = int(random_state)
        cleaned_kwargs = {k: v for k, v in rf_kwargs.items() if k not in self._UNSUPPORTED_KWARGS}
        cleaned_kwargs.setdefault("output_type", "numpy")
        cleaned_kwargs.setdefault("random_state", self.random_state)
        self.model = RandomForestClassifier(**cleaned_kwargs)
        self._is_fit = False
        self._used_features: list[str] = []
        self._metrics: dict[str, Any] = {}
        self._feature_importance: dict[str, float] = {}
        self._train_stats: dict[str, Any] = {}
        self._class_mapping: dict[int, str] = {}
        self._classes: list[int] = []

    def positive_class_index(self) -> int:
        if not self._classes:
            return -1
        if 1 in self._classes:
            return self._classes.index(1)
        if "1" in self._classes:
            return self._classes.index("1")
        return max(len(self._classes) - 1, 0)

    def fit(
        self,
        df_train: pd.DataFrame,
        spec: FitSpec,
        verbose: bool = True,
        validation_df: pd.DataFrame | None = None,
    ) -> "CumlRFClassifier":
        if not spec.target_col:
            raise ValueError("CumlRFClassifier requires spec.target_col")

        df = df_train.dropna(subset=[spec.target_col]).copy()
        val_df = None
        if validation_df is not None:
            val_df = validation_df.dropna(subset=[spec.target_col]).copy()
            if val_df.empty:
                raise ValueError("validation_df has no usable rows for CumlRFClassifier.")

        full_x = df[list(spec.feature_cols)]
        self._used_features = full_x.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric = [col for col in spec.feature_cols if col not in self._used_features]
        x = df[self._used_features]

        if x.empty:
            raise ValueError("No numeric features found. Check your feature_cols.")

        target_series = df[spec.target_col]
        if target_series.dtype == "object" or target_series.dtype.name == "category":
            codes, uniques = pd.factorize(target_series)
            y = pd.Series(codes, index=df.index)
            self._class_mapping = {i: str(value) for i, value in enumerate(uniques)}
        else:
            y = pd.to_numeric(target_series, errors="coerce").fillna(0).astype(int)
            unique_vals = np.sort(y.unique())
            self._class_mapping = {int(value): str(value) for value in unique_vals}

        unique_classes = np.sort(y.unique())
        n_classes = len(unique_classes)
        if n_classes < 2:
            raise ValueError(f"Target '{spec.target_col}' has only one class: {unique_classes.tolist()}.")
        self._classes = [int(value) for value in unique_classes.tolist()]

        split_ratio = getattr(spec, "split_ratio", 0.8)
        test_size = 1.0 - split_ratio

        if val_df is not None:
            x_tr, y_tr = x, y
            x_va = val_df[self._used_features]
            val_target_series = val_df[spec.target_col]
            if target_series.dtype == "object" or target_series.dtype.name == "category":
                y_va = pd.Series(pd.Categorical(val_target_series.astype(str), categories=list(self._class_mapping.values())).codes, index=val_df.index)
            else:
                y_va = pd.to_numeric(val_target_series, errors="coerce").fillna(0).astype(int)
            if (y_va < 0).any():
                raise ValueError(f"validation_df contains unseen labels for target '{spec.target_col}'.")
            use_holdout = False
            eval_mode = "external_validation"
        else:
            use_holdout = (test_size > 0.0) and (test_size < 1.0)
            if use_holdout:
                x_tr, x_va, y_tr, y_va = train_test_split(
                    x,
                    y,
                    test_size=test_size,
                    random_state=self.random_state,
                    shuffle=True,
                    stratify=y,
                )
            else:
                x_tr, y_tr = x, y
                x_va, y_va = x, y
            eval_mode = "holdout" if use_holdout else "in_sample"

        self._train_stats = {
            "n_obs": len(df),
            "n_train": len(x_tr),
            "n_test": len(x_va),
            "split_ratio": split_ratio,
            "eval_mode": eval_mode,
            "target_col": str(spec.target_col),
            "model_tag": str(spec.model_tag or ""),
            "signal": str(spec.signal or ""),
            "n_features_numeric": len(self._used_features),
            "dropped_count": len(non_numeric),
            "classes": unique_classes.tolist(),
            "is_binary": n_classes == 2,
            "class_dist": {
                "train": y_tr.value_counts().to_dict(),
                "test": y_va.value_counts().to_dict(),
            },
        }

        self.model.fit(
            x_tr.astype("float32", copy=False),
            y_tr.astype("int32", copy=False),
        )
        self._is_fit = True

        y_pred = np.asarray(self.model.predict(x_va.astype("float32", copy=False)), dtype=int)
        if self._train_stats["is_binary"]:
            proba_matrix = np.asarray(self.model.predict_proba(x_va.astype("float32", copy=False)), dtype=float)
            positive_idx = self.positive_class_index()
            if positive_idx < 0 or positive_idx >= proba_matrix.shape[1]:
                positive_idx = min(1, proba_matrix.shape[1] - 1)
            proba = proba_matrix[:, positive_idx] if proba_matrix.shape[1] > positive_idx else proba_matrix[:, -1]
            self._metrics = binary_classifier_metrics_from_scores(y_va.to_numpy(), proba, threshold=0.5)
        else:
            self._metrics = classification_metrics(y_va.to_numpy(), y_pred, labels=unique_classes.tolist())

        self._metrics["confusion_matrix"] = confusion_matrix(y_va, y_pred, labels=unique_classes)

        feature_importances = getattr(self.model, "feature_importances_", None)
        if feature_importances is not None:
            self._feature_importance = {str(col): float(val) for col, val in zip(self._used_features, np.asarray(feature_importances))}

        if verbose:
            self.summarize()
        return self

    def predict(self, df: pd.DataFrame, *, feature_cols=None) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("CumlRFClassifier not fit()")
        used_features = list(feature_cols) if feature_cols is not None else list(self._used_features)
        x = df[used_features].astype("float32", copy=False)
        return np.asarray(self.model.predict(x), dtype=float)

    def predict_proba(self, df: pd.DataFrame, *, feature_cols=None) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("CumlRFClassifier not fit()")
        used_features = list(feature_cols) if feature_cols is not None else list(self._used_features)
        x = df[used_features].astype("float32", copy=False)
        return np.asarray(self.model.predict_proba(x), dtype=float)

    def positive_class_index(self) -> int:
        if not self._classes:
            return -1
        positive_class = self._classes[-1]
        return self._classes.index(positive_class)

    def summarize(self) -> None:
        if not self._is_fit:
            return

        print_model_section("cuML Random Forest Classifier (Test Results)")
        stats = self._train_stats
        metrics = self._metrics

        print("DATASET & SPLIT:")
        print(f"  - Total Observations: {stats['n_obs']:,}")
        print(f"  - Target:             {stats.get('target_col') or ''}")
        if stats.get("model_tag"):
            print(f"  - Model Role:         {stats['model_tag']}")
        if stats.get("signal"):
            print(f"  - Signal:             {stats['signal']}")
        if stats.get("eval_mode") == "holdout":
            print(f"  - Random Split:       {stats['split_ratio']:.1%} Train / {1 - stats['split_ratio']:.1%} Test")
        elif stats.get("eval_mode") == "external_validation":
            print("  - Split Mode:         External validation frame")
        else:
            print("  - Split Mode:         In-sample eval (no internal holdout split)")
        print(f"  - Features:           {stats['n_features_numeric']} numeric (filtered {stats['dropped_count']} strings)")
        if stats["is_binary"]:
            positive_idx = self.positive_class_index()
            positive_class = stats["classes"][positive_idx] if 0 <= positive_idx < len(stats["classes"]) else stats["classes"][-1]
            positive_name = self._class_mapping.get(positive_class, str(positive_class))
            print(f"  - Positive Class:     {positive_class} => {positive_name}")

        print(f"\nCLASS DISTRIBUTION (Mapping: {self._class_mapping}):")
        print("               Train Set              Test Set")
        dist = stats["class_dist"]
        for cls in stats["classes"]:
            tr_cnt = dist["train"].get(cls, 0)
            va_cnt = dist["test"].get(cls, 0)
            tr_pct = tr_cnt / stats["n_train"]
            va_pct = va_cnt / stats["n_test"]
            name = self._class_mapping.get(cls, str(cls))
            print(f"  - {name:>8}: {tr_cnt:>7,} ({tr_pct:>5.1%})   {va_cnt:>7,} ({va_pct:>5.1%})")

        print("\nTEST PERFORMANCE:")
        print(f"  - Accuracy:           {metrics.get('accuracy', 0):.2%}")
        if stats["is_binary"]:
            print(f"  - ROC AUC:            {metrics.get('roc_auc', 0):.4f}")

        confusion = metrics.get("confusion_matrix")
        if confusion is not None:
            print("\nCONFUSION MATRIX (Test Set):")
            classes = stats["classes"]
            header = "            " + "".join([f"Pred {self._class_mapping.get(cls, cls):>7} " for cls in classes])
            print(header)
            for idx, row_label in enumerate(classes):
                name = self._class_mapping.get(row_label, row_label)
                row_str = f"True {name:>7}: "
                row_str += "".join([f"{val:>12} " for val in confusion[idx]])
                print(row_str)

        if self._feature_importance:
            print("\nTOP 10 FEATURES:")
            top = sorted(self._feature_importance.items(), key=lambda item: item[1], reverse=True)[:10]
            for col, val in top:
                print(f"  - {col}: {val:.4f}")
        print("=" * 60)

    def metrics_report(self) -> dict[str, Any]:
        return metrics_with_feature_importance(self._metrics, self._feature_importance, top_n=30)

    def feature_importance(self) -> dict[str, float]:
        return copy_feature_importance(self._feature_importance)
