from __future__ import annotations

import inspect
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from ml.base import (
    FitSpec,
    Model,
    copy_feature_importance,
    metrics_with_feature_importance,
    print_model_section,
)
from ml.metrics import binary_classifier_metrics_from_scores, classification_metrics


class AutoGluonClassifier(Model):
    """AutoGluon classifier aligned with the SklearnRFClassifier adapter behavior."""

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
        self._class_mapping: Dict[int, str] = {}

    def fit(self, df_train: pd.DataFrame, spec: FitSpec, verbose: bool = True) -> "AutoGluonClassifier":
        if not spec.target_col:
            raise ValueError("AutoGluonClassifier requires spec.target_col")

        from autogluon.tabular import TabularPredictor  # type: ignore

        df = df_train.dropna(subset=[spec.target_col]).copy()
        full_X = df[list(spec.feature_cols)]
        self._used_features = full_X.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric = [c for c in spec.feature_cols if c not in self._used_features]
        if not self._used_features:
            print("! ERROR: No numeric features found. Check your feature_cols.")
            return self

        y_col = spec.target_col

        target_series = df[y_col]
        if target_series.dtype == "object" or target_series.dtype.name == "category":
            codes, uniques = pd.factorize(target_series)
            y = pd.Series(codes, index=df.index)
            self._class_mapping = {i: str(v) for i, v in enumerate(uniques)}
        else:
            y = pd.to_numeric(target_series, errors="coerce").fillna(0).astype(int)
            unique_vals = np.sort(y.unique())
            self._class_mapping = {int(v): str(v) for v in unique_vals}

        unique_classes = np.sort(y.unique())
        n_classes = len(unique_classes)
        if n_classes < 2:
            print(f"! ERROR: Target '{y_col}' has only one class: {unique_classes}.")
            return self

        split_ratio = float(getattr(spec, "split_ratio", 0.8))
        test_size = 1.0 - split_ratio

        d = df[self._used_features].copy()
        d[y_col] = y
        weight_col = None
        if spec.weight_col and spec.weight_col in df.columns:
            weight_col = "__sample_weight__"
            d[weight_col] = pd.to_numeric(df[spec.weight_col], errors="coerce").fillna(1.0).astype(float)
        # AutoGluon requires unique indices in train/tuning datasets.
        d = d.reset_index(drop=True)

        train_df, val_df = train_test_split(
            d,
            test_size=test_size,
            random_state=self.random_state,
            shuffle=True,
            stratify=y,
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        ag_problem_type = "binary" if n_classes == 2 else "multiclass"
        predictor_kwargs: Dict[str, Any] = {"label": y_col, "problem_type": ag_problem_type}
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

        proba_df = self._pred.predict_proba(val_df[self._used_features])
        proba_cols = list(getattr(proba_df, "columns", []))
        if 1 in proba_cols:
            proba = np.asarray(proba_df[1], dtype=float)
        else:
            proba = np.asarray(proba_df)[:, -1]
        y_true = val_df[y_col].to_numpy()
        y_pred = np.asarray(self._pred.predict(val_df[self._used_features]), dtype=int).reshape(-1)

        if n_classes == 2:
            self._metrics = binary_classifier_metrics_from_scores(y_true, proba, threshold=0.5)
        else:
            self._metrics = classification_metrics(y_true, y_pred, labels=unique_classes.tolist())
        self._metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=unique_classes)

        try:
            fi_df = self._pred.feature_importance(val_df)
            if "importance" in fi_df.columns:
                self._feature_importance = {str(k): float(v) for k, v in fi_df["importance"].to_dict().items()}
        except Exception:
            self._feature_importance = {}

        self._train_stats = {
            "n_obs": len(df),
            "n_train": len(train_df),
            "n_test": len(val_df),
            "split_ratio": split_ratio,
            "n_features_numeric": len(self._used_features),
            "dropped_count": len(non_numeric),
            "classes": unique_classes.tolist(),
            "is_binary": (n_classes == 2),
            "class_dist": {
                "train": train_df[y_col].value_counts().to_dict(),
                "test": val_df[y_col].value_counts().to_dict(),
            },
        }

        if verbose:
            self.summarize()

        return self

    def predict(self, df: pd.DataFrame, *, feature_cols=None) -> np.ndarray:
        if self._pred is None or not self._is_fit:
            raise RuntimeError("AutoGluonClassifier not fit()")

        cols = list(feature_cols) if feature_cols is not None else list(self._used_features)
        X = df[cols].copy()
        proba_df = self._pred.predict_proba(X)
        arr = np.asarray(proba_df)
        return arr[:, -1].astype(float)

    def metrics_report(self) -> Dict[str, Any]:
        return metrics_with_feature_importance(self._metrics, self._feature_importance, top_n=30)

    def feature_importance(self) -> Dict[str, float]:
        """Full feature importance mapping (may be empty)."""
        return copy_feature_importance(self._feature_importance)

    def summarize(self) -> None:
        if not self._is_fit:
            return

        print_model_section("AutoGluon Classifier (Test Results)")
        s = self._train_stats
        m = self._metrics

        print(f"DATASET & SPLIT:")
        print(f"  - Total Observations: {s['n_obs']:,}")
        print(f"  - Random Split:       {s['split_ratio']:.1%} Train / {1 - s['split_ratio']:.1%} Test")
        print(f"  - Features:           {s['n_features_numeric']} numeric (filtered {s['dropped_count']} strings)")

        print(f"\nCLASS DISTRIBUTION (Mapping: {self._class_mapping}):")
        print(f"               Train Set              Test Set")
        dist = s["class_dist"]
        for c in s["classes"]:
            tr_cnt = dist["train"].get(c, 0)
            va_cnt = dist["test"].get(c, 0)
            tr_pct = tr_cnt / max(s["n_train"], 1)
            va_pct = va_cnt / max(s["n_test"], 1)
            name = self._class_mapping.get(c, str(c))
            print(f"  - {name:>8}: {tr_cnt:>7,} ({tr_pct:>5.1%})   {va_cnt:>7,} ({va_pct:>5.1%})")

        print(f"\nTEST PERFORMANCE:")
        print(f"  - Accuracy:           {m.get('accuracy', 0):.2%}")
        if s["is_binary"]:
            print(f"  - ROC AUC:            {m.get('roc_auc', 0):.4f}")

        cm = m.get("confusion_matrix")
        if cm is not None:
            print(f"\nCONFUSION MATRIX (Test Set):")
            classes = s["classes"]
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
