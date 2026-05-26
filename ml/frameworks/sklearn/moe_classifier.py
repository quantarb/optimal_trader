from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from ml.base import FitSpec, Model, copy_feature_importance, metrics_with_feature_importance, print_model_section
from ml.metrics import binary_classifier_metrics_from_scores, classification_metrics


@dataclass
class _FamilyForest:
    family: str
    features: list[str]
    model: RandomForestClassifier


class SklearnMoERFClassifier(Model):
    """One sklearn-style MoE object that trains optimized family forests one at a time."""

    def __init__(
        self,
        feature_families: Mapping[str, list[str] | tuple[str, ...]],
        *,
        family_weights: Mapping[str, float] | None = None,
        random_state: int = 1337,
        n_estimators: int = 200,
        bootstrap: bool = True,
        max_samples: int | float | None = None,
        clip_abs: float | None = 1e12,
        n_jobs: int | None = None,
        **rf_kwargs: Any,
    ) -> None:
        self.feature_families = {
            str(name): [str(col) for col in cols]
            for name, cols in dict(feature_families or {}).items()
            if str(name).strip() and cols
        }
        if not self.feature_families:
            raise ValueError("SklearnMoERFClassifier requires at least one feature family.")
        self.family_weights = {str(name): float(weight) for name, weight in dict(family_weights or {}).items()}
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)
        self.bootstrap = bool(bootstrap)
        self.max_samples = max_samples
        self.clip_abs = None if clip_abs is None else float(clip_abs)
        self.n_jobs = n_jobs
        self.rf_kwargs = dict(rf_kwargs)

        self.model: list[_FamilyForest] = []
        self._is_fit = False
        self._used_features: list[str] = []
        self._family_features: dict[str, list[str]] = {}
        self._family_medians: dict[str, pd.Series] = {}
        self._forests_by_family: dict[str, _FamilyForest] = {}
        self._trees_by_family: dict[str, list[Any]] = {}
        self._classes: np.ndarray = np.array([])
        self._metrics: dict[str, Any] = {}
        self._feature_importance: dict[str, float] = {}
        self._train_stats: dict[str, Any] = {}
        self._class_mapping: dict[int, str] = {}

    def _sanitize_numeric_frame(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in cols:
            out[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
        out = out.replace([np.inf, -np.inf], np.nan)
        if self.clip_abs is not None and self.clip_abs > 0:
            out = out.clip(lower=-self.clip_abs, upper=self.clip_abs)
        return out.astype(np.float32)

    def _family_available_mask(self, df: pd.DataFrame, family: str) -> pd.Series:
        cols = self._family_features.get(family, [])
        if not cols:
            return pd.Series(False, index=df.index)
        return self._sanitize_numeric_frame(df, cols).notna().any(axis=1)

    def _family_matrix(self, df: pd.DataFrame, family: str) -> pd.DataFrame:
        cols = self._family_features[family]
        x = self._sanitize_numeric_frame(df, cols)
        medians = self._family_medians.get(family, pd.Series(0.0, index=cols))
        return x.fillna(medians).fillna(0.0).astype(np.float32)

    def _positive_class_index(self) -> int:
        classes = list(self._classes)
        if 1 in classes:
            return classes.index(1)
        if "1" in classes:
            return classes.index("1")
        return max(len(classes) - 1, 0)

    def _allocate_tree_counts(self) -> dict[str, int]:
        families = list(self._family_features)
        weights = np.asarray([max(float(self.family_weights.get(name, 1.0)), 0.0) for name in families], dtype=float)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
            weights = np.ones(len(families), dtype=float)
        total = max(int(self.n_estimators), len(families))
        raw = weights / weights.sum() * total
        counts = np.floor(raw).astype(int)
        counts[counts == 0] = 1
        while int(counts.sum()) > total and int(counts.max()) > 1:
            counts[int(np.argmax(counts))] -= 1
        while int(counts.sum()) < total:
            counts[int(np.argmax(raw - np.floor(raw)))] += 1
        return {family: int(count) for family, count in zip(families, counts)}

    def fit(self, df_train: pd.DataFrame, spec: FitSpec, verbose: bool = True) -> "SklearnMoERFClassifier":
        if not spec.target_col:
            raise ValueError("SklearnMoERFClassifier requires spec.target_col")
        if self.n_estimators <= 0:
            raise ValueError("SklearnMoERFClassifier requires n_estimators > 0")

        df = df_train.dropna(subset=[spec.target_col]).copy()
        requested_features = [str(col) for col in spec.feature_cols if str(col) in df.columns]
        numeric_cols = df[requested_features].select_dtypes(include=[np.number]).columns.tolist()
        numeric_set = set(numeric_cols)
        self._family_features = {
            name: [col for col in cols if col in numeric_set]
            for name, cols in self.feature_families.items()
        }
        self._family_features = {name: cols for name, cols in self._family_features.items() if cols}
        if not self._family_features:
            raise ValueError("No numeric feature-family columns were found for SklearnMoERFClassifier.")
        self._used_features = sorted({col for cols in self._family_features.values() for col in cols})

        available_any = pd.Series(False, index=df.index)
        for family in self._family_features:
            available_any |= self._family_available_mask(df, family)
        df = df.loc[available_any].copy()
        if df.empty:
            raise ValueError("No rows with finite feature-family coverage were available for training.")

        target_series = df[spec.target_col]
        if target_series.dtype == "object" or target_series.dtype.name == "category":
            codes, uniques = pd.factorize(target_series)
            y = pd.Series(codes, index=df.index)
            self._class_mapping = {i: str(value) for i, value in enumerate(uniques)}
        else:
            y = pd.to_numeric(target_series, errors="coerce").fillna(0).astype(int)
            self._class_mapping = {int(value): str(value) for value in np.sort(y.unique())}
        self._classes = np.sort(y.unique())
        if len(self._classes) < 2:
            raise ValueError(f"Target '{spec.target_col}' has only one class: {self._classes.tolist()}.")

        weights = None
        if spec.weight_col and spec.weight_col in df.columns:
            weights = pd.to_numeric(df[spec.weight_col], errors="coerce").fillna(1.0)

        split_ratio = float(getattr(spec, "split_ratio", 0.8))
        test_size = 1.0 - split_ratio
        use_holdout = (test_size > 0.0) and (test_size < 1.0)
        if use_holdout:
            train_index, valid_index = train_test_split(
                df.index,
                test_size=test_size,
                random_state=self.random_state,
                shuffle=True,
                stratify=y,
            )
        else:
            train_index = df.index
            valid_index = df.index

        train_df = df.loc[train_index]
        y_train = y.loc[train_index]
        valid_df = df.loc[valid_index]
        y_valid = y.loc[valid_index]
        w_train = weights.loc[train_index] if weights is not None else None

        tree_counts = self._allocate_tree_counts()
        self.model = []
        self._forests_by_family = {}
        self._trees_by_family = {family: [] for family in self._family_features}

        for family, tree_count in tree_counts.items():
            available = self._family_available_mask(train_df, family)
            if not bool(available.any()):
                continue
            family_train_df = train_df.loc[available]
            family_x = self._sanitize_numeric_frame(family_train_df, self._family_features[family])
            self._family_medians[family] = family_x.median(axis=0).fillna(0.0)
            family_x = family_x.fillna(self._family_medians[family]).fillna(0.0).astype(np.float32)
            family_y = y_train.loc[family_x.index]
            family_w = w_train.loc[family_x.index].to_numpy(dtype=float) if w_train is not None else None

            rf = RandomForestClassifier(
                n_estimators=int(tree_count),
                random_state=self.random_state + len(self.model) + 1,
                bootstrap=self.bootstrap,
                max_samples=self.max_samples,
                n_jobs=self.n_jobs,
                **self.rf_kwargs,
            )
            rf.fit(family_x, family_y, sample_weight=family_w)
            wrapped = _FamilyForest(family=family, features=list(self._family_features[family]), model=rf)
            self.model.append(wrapped)
            self._forests_by_family[family] = wrapped
            self._trees_by_family[family] = list(getattr(rf, "estimators_", []))
            del family_train_df, family_x, family_y, family_w
            gc.collect()

        if not self.model:
            raise ValueError("No family forests were trained.")
        self._is_fit = True
        self._build_feature_importance()

        valid_proba = self.predict_proba(valid_df)
        valid_pred = self._classes[np.nanargmax(valid_proba, axis=1)]
        is_binary = len(self._classes) == 2
        if is_binary:
            self._metrics = binary_classifier_metrics_from_scores(
                y_valid.to_numpy(),
                valid_proba[:, self._positive_class_index()],
                threshold=0.5,
            )
        else:
            self._metrics = classification_metrics(y_valid.to_numpy(), valid_pred, labels=self._classes.tolist())
        self._metrics["confusion_matrix"] = confusion_matrix(y_valid, valid_pred, labels=self._classes)

        self._train_stats = {
            "n_obs": len(df),
            "n_train": len(train_df),
            "n_test": len(valid_df),
            "split_ratio": split_ratio,
            "eval_mode": "holdout" if use_holdout else "in_sample",
            "target_col": str(spec.target_col),
            "model_tag": str(spec.model_tag or "optimized family-forest MoE"),
            "n_features_numeric": len(self._used_features),
            "family_count": len(self._family_features),
            "families": list(self._family_features),
            "tree_counts": {family: len(trees) for family, trees in self._trees_by_family.items()},
            "classes": self._classes.tolist(),
            "is_binary": is_binary,
            "class_dist": {"train": y_train.value_counts().to_dict(), "test": y_valid.value_counts().to_dict()},
        }
        if verbose:
            self.summarize()
        return self

    def _aligned_forest_proba(self, forest: RandomForestClassifier, x: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(forest.predict_proba(x), dtype=np.float32)
        aligned = np.zeros((len(x), len(self._classes)), dtype=np.float32)
        for raw_idx, cls in enumerate(forest.classes_):
            positions = np.where(self._classes == cls)[0]
            if len(positions):
                aligned[:, int(positions[0])] = raw[:, raw_idx]
        return aligned

    def _predict_family_matrix(self, df: pd.DataFrame, family: str) -> tuple[np.ndarray, pd.Series]:
        wrapped = self._forests_by_family.get(family)
        available = self._family_available_mask(df, family)
        proba = np.full((len(df), len(self._classes)), np.nan, dtype=np.float32)
        if wrapped is None or not bool(available.any()):
            return proba, available
        x = self._family_matrix(df, family)
        family_proba = self._aligned_forest_proba(wrapped.model, x[wrapped.features])
        proba[available.to_numpy(dtype=bool), :] = family_proba[available.to_numpy(dtype=bool), :]
        return proba, available

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("SklearnMoERFClassifier not fit()")
        weighted_sum = np.zeros((len(df), len(self._classes)), dtype=np.float32)
        weight_sum = np.zeros(len(df), dtype=np.float32)
        for family in self._family_features:
            family_proba, available = self._predict_family_matrix(df, family)
            valid = available.to_numpy(dtype=bool) & np.isfinite(family_proba).any(axis=1)
            if not bool(valid.any()):
                continue
            weight = float(self.family_weights.get(family, 1.0))
            weighted_sum[valid, :] += weight * family_proba[valid, :]
            weight_sum[valid] += weight
        out = np.zeros_like(weighted_sum)
        valid_rows = weight_sum > 0
        out[valid_rows, :] = weighted_sum[valid_rows, :] / weight_sum[valid_rows, None]
        if (~valid_rows).any():
            out[~valid_rows, :] = 1.0 / float(len(self._classes))
        return out

    def predict(self, df: pd.DataFrame, *, feature_cols=None) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("SklearnMoERFClassifier not fit()")
        return self._classes[np.argmax(self.predict_proba(df), axis=1)].astype(float)

    def predict_positive_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(df)[:, self._positive_class_index()]

    def predict_family_proba_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fit:
            raise RuntimeError("SklearnMoERFClassifier not fit()")
        out = pd.DataFrame(index=df.index)
        positive_idx = self._positive_class_index()
        for family in self._family_features:
            family_proba, available = self._predict_family_matrix(df, family)
            out[f"{family}__prob_buy"] = pd.Series(family_proba[:, positive_idx], index=df.index).where(available, np.nan)
        return out

    def predict_moe_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.predict_family_proba_frame(df)
        weighted_sum = pd.Series(0.0, index=out.index, dtype=float)
        weight_sum = pd.Series(0.0, index=out.index, dtype=float)
        expert_count = pd.Series(0, index=out.index, dtype=int)
        for family in self._family_features:
            col = f"{family}__prob_buy"
            prob = pd.to_numeric(out[col], errors="coerce")
            valid = prob.notna()
            weight = float(self.family_weights.get(family, 1.0))
            weighted_sum.loc[valid] += weight * prob.loc[valid]
            weight_sum.loc[valid] += weight
            expert_count.loc[valid] += 1
        out["clf__prob_1"] = (weighted_sum / weight_sum.replace(0.0, np.nan)).clip(0.0, 1.0)
        out["clf"] = out["clf__prob_1"]
        out["ranking"] = out["clf__prob_1"]
        out["combined_score"] = out["clf__prob_1"]
        out["ae_familiarity"] = 1.0
        out["moe_available_experts"] = expert_count
        out["moe_active_experts"] = len(self._family_features)
        return out

    def _build_feature_importance(self) -> None:
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for wrapped in self.model:
            importances = getattr(wrapped.model, "feature_importances_", None)
            if importances is None:
                continue
            for col, value in zip(wrapped.features, importances):
                totals[col] = totals.get(col, 0.0) + float(value)
                counts[col] = counts.get(col, 0) + 1
        self._feature_importance = {col: totals[col] / max(counts.get(col, 1), 1) for col in totals}

    def summarize(self) -> None:
        if not self._is_fit:
            return
        print_model_section("Sklearn Optimized Family-Forest MoE Classifier")
        stats = self._train_stats
        metrics = self._metrics
        print("DATASET & SPLIT:")
        print(f"  - Total Observations: {stats['n_obs']:,}")
        print(f"  - Target:             {stats.get('target_col') or ''}")
        print(f"  - Model Role:         {stats.get('model_tag') or ''}")
        print(f"  - Families:           {stats['family_count']} ({', '.join(stats['families'])})")
        print(f"  - Trees:              {sum(stats['tree_counts'].values()):,} total | {stats['tree_counts']}")
        print(f"  - Features:           {stats['n_features_numeric']} numeric across families")
        if stats.get("eval_mode") == "holdout":
            print(f"  - Random Split:       {stats['split_ratio']:.1%} Train / {1 - stats['split_ratio']:.1%} Test")
        else:
            print("  - Split Mode:         In-sample eval (no internal holdout split)")
        print("\nTEST PERFORMANCE:")
        print(f"  - Accuracy:           {metrics.get('accuracy', 0):.2%}")
        if stats["is_binary"]:
            print(f"  - ROC AUC:            {metrics.get('roc_auc', 0):.4f}")
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
