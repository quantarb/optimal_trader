from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from ml.base import FitSpec, Model, copy_feature_importance, metrics_with_feature_importance, print_model_section
from ml.metrics import binary_classifier_metrics_from_scores, classification_metrics


@dataclass
class _RoutedExpert:
    route_value: str
    model: RandomForestClassifier
    training_rows: int


class SklearnRoutedMoERFClassifier(Model):
    """Route each row to an FMP classification expert with a global fallback."""

    def __init__(
        self,
        route_col: str,
        *,
        min_expert_rows: int = 100,
        random_state: int = 1337,
        n_estimators: int = 200,
        clip_abs: float | None = 1e12,
        n_jobs: int | None = None,
        **rf_kwargs: Any,
    ) -> None:
        self.route_col = str(route_col or "").strip()
        if not self.route_col:
            raise ValueError("SklearnRoutedMoERFClassifier requires route_col.")
        self.min_expert_rows = max(2, int(min_expert_rows))
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)
        self.clip_abs = None if clip_abs is None else float(clip_abs)
        self.n_jobs = n_jobs
        self.rf_kwargs = dict(rf_kwargs)

        self.model: dict[str, RandomForestClassifier] = {}
        self._global_model: RandomForestClassifier | None = None
        self._experts: dict[str, _RoutedExpert] = {}
        self._used_features: list[str] = []
        self._medians = pd.Series(dtype=float)
        self._classes = np.array([])
        self._is_fit = False
        self._metrics: dict[str, Any] = {}
        self._feature_importance: dict[str, float] = {}
        self._train_stats: dict[str, Any] = {}

    @staticmethod
    def _normalize_routes(values: pd.Series) -> pd.Series:
        routes = values.fillna("").astype(str).str.strip()
        return routes.mask(routes.eq(""), "Unknown")

    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self._used_features:
            out[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
        out = out.replace([np.inf, -np.inf], np.nan)
        if self.clip_abs is not None and self.clip_abs > 0:
            out = out.clip(lower=-self.clip_abs, upper=self.clip_abs)
        return out.fillna(self._medians).fillna(0.0).astype(np.float32)

    def _new_forest(self, seed_offset: int) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state + int(seed_offset),
            n_jobs=self.n_jobs,
            **self.rf_kwargs,
        )

    def fit(
        self,
        df_train: pd.DataFrame,
        spec: FitSpec,
        verbose: bool = True,
        validation_df: pd.DataFrame | None = None,
    ) -> "SklearnRoutedMoERFClassifier":
        if not spec.target_col:
            raise ValueError("SklearnRoutedMoERFClassifier requires spec.target_col.")
        if self.route_col not in df_train.columns:
            raise ValueError(f"Training frame is missing routing column {self.route_col!r}.")
        if self.n_estimators <= 0:
            raise ValueError("SklearnRoutedMoERFClassifier requires n_estimators > 0.")

        df = df_train.dropna(subset=[spec.target_col]).copy()
        requested = [str(col) for col in spec.feature_cols if str(col) in df.columns]
        self._used_features = df[requested].select_dtypes(include=[np.number]).columns.tolist()
        if not self._used_features:
            raise ValueError("No numeric features were found for routed MoE training.")
        finite_any = df[self._used_features].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
        df = df.loc[finite_any].copy()
        if df.empty:
            raise ValueError("No rows with finite feature coverage were available for routed MoE training.")

        target = df[spec.target_col]
        if target.dtype == "object" or target.dtype.name == "category":
            codes, _uniques = pd.factorize(target)
            y = pd.Series(codes, index=df.index)
        else:
            y = pd.to_numeric(target, errors="coerce").fillna(0).astype(int)
        self._classes = np.sort(y.unique())
        if len(self._classes) < 2:
            raise ValueError(f"Target {spec.target_col!r} has only one class: {self._classes.tolist()}.")

        weights = None
        if spec.weight_col and spec.weight_col in df.columns:
            weights = pd.to_numeric(df[spec.weight_col], errors="coerce").fillna(1.0)

        if validation_df is not None:
            train_df = df
            y_train = y
            valid_df = validation_df.dropna(subset=[spec.target_col]).copy()
            y_valid = pd.to_numeric(valid_df[spec.target_col], errors="coerce").fillna(0).astype(int)
            w_train = weights
            eval_mode = "external_validation"
        else:
            test_size = 1.0 - float(spec.split_ratio)
            if 0.0 < test_size < 1.0:
                train_idx, valid_idx = train_test_split(
                    df.index,
                    test_size=test_size,
                    random_state=self.random_state,
                    shuffle=True,
                    stratify=y,
                )
            else:
                train_idx = valid_idx = df.index
            train_df = df.loc[train_idx]
            y_train = y.loc[train_idx]
            valid_df = df.loc[valid_idx]
            y_valid = y.loc[valid_idx]
            w_train = weights.loc[train_idx] if weights is not None else None
            eval_mode = "holdout" if 0.0 < test_size < 1.0 else "in_sample"

        raw_train_x = train_df[self._used_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if self.clip_abs is not None and self.clip_abs > 0:
            raw_train_x = raw_train_x.clip(lower=-self.clip_abs, upper=self.clip_abs)
        self._medians = raw_train_x.median(axis=0).fillna(0.0)
        train_x = raw_train_x.fillna(self._medians).fillna(0.0).astype(np.float32)

        self._global_model = self._new_forest(0)
        self._global_model.fit(
            train_x,
            y_train,
            sample_weight=w_train.to_numpy(dtype=float) if w_train is not None else None,
        )

        routes = self._normalize_routes(train_df[self.route_col])
        self._experts = {}
        for offset, (route_value, route_index) in enumerate(routes.groupby(routes).groups.items(), start=1):
            route_index = pd.Index(route_index)
            route_y = y_train.loc[route_index]
            if len(route_index) < self.min_expert_rows or route_y.nunique() < 2:
                continue
            expert = self._new_forest(offset)
            expert.fit(
                train_x.loc[route_index],
                route_y,
                sample_weight=w_train.loc[route_index].to_numpy(dtype=float) if w_train is not None else None,
            )
            self._experts[str(route_value)] = _RoutedExpert(str(route_value), expert, len(route_index))
        self.model = {"__global__": self._global_model, **{key: value.model for key, value in self._experts.items()}}
        self._is_fit = True
        self._build_feature_importance()

        valid_proba = self.predict_proba(valid_df)
        valid_pred = self._classes[np.argmax(valid_proba, axis=1)]
        if len(self._classes) == 2:
            positive_idx = list(self._classes).index(1) if 1 in self._classes else len(self._classes) - 1
            self._metrics = binary_classifier_metrics_from_scores(y_valid.to_numpy(), valid_proba[:, positive_idx])
        else:
            self._metrics = classification_metrics(y_valid.to_numpy(), valid_pred, labels=self._classes.tolist())
        self._metrics["confusion_matrix"] = confusion_matrix(y_valid, valid_pred, labels=self._classes)
        self._train_stats = {
            "n_obs": len(df),
            "n_train": len(train_df),
            "n_test": len(valid_df),
            "eval_mode": eval_mode,
            "route_col": self.route_col,
            "expert_count": len(self._experts),
            "experts": {key: expert.training_rows for key, expert in self._experts.items()},
            "fallback": "global",
            "n_features_numeric": len(self._used_features),
            "classes": self._classes.tolist(),
        }
        if verbose:
            self.summarize()
        return self

    def _aligned_proba(self, model: RandomForestClassifier, x: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(model.predict_proba(x), dtype=np.float32)
        out = np.zeros((len(x), len(self._classes)), dtype=np.float32)
        for idx, cls in enumerate(model.classes_):
            matches = np.where(self._classes == cls)[0]
            if len(matches):
                out[:, int(matches[0])] = raw[:, idx]
        return out

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self._is_fit or self._global_model is None:
            raise RuntimeError("SklearnRoutedMoERFClassifier not fit().")
        x = self._matrix(df)
        out = self._aligned_proba(self._global_model, x)
        routes = self._normalize_routes(df.get(self.route_col, pd.Series("Unknown", index=df.index)))
        for route_value, expert in self._experts.items():
            mask = routes.eq(route_value).to_numpy(dtype=bool)
            if mask.any():
                out[mask, :] = self._aligned_proba(expert.model, x.loc[mask])
        return out

    def predict(self, df: pd.DataFrame, *, feature_cols=None) -> np.ndarray:
        return self._classes[np.argmax(self.predict_proba(df), axis=1)].astype(float)

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        proba = self.predict_proba(df)
        positive_idx = list(self._classes).index(1) if 1 in self._classes else len(self._classes) - 1
        routes = self._normalize_routes(df.get(self.route_col, pd.Series("Unknown", index=df.index)))
        expert_used = routes.where(routes.isin(self._experts), "__global__")
        out = pd.DataFrame(index=df.index)
        out["prediction"] = self._classes[np.argmax(proba, axis=1)]
        out["clf__prob_1"] = proba[:, positive_idx]
        out["clf"] = out["clf__prob_1"]
        out["prob_buy"] = out["clf__prob_1"]
        out["prediction_score"] = out["clf__prob_1"]
        out["ranking"] = out["clf__prob_1"]
        out["combined_score"] = out["clf__prob_1"]
        out["ae_familiarity"] = 1.0
        out["moe_route"] = routes
        out["moe_expert"] = expert_used
        out["moe_used_fallback"] = expert_used.eq("__global__")
        out["moe_active_experts"] = len(self._experts) + 1
        return out

    def _build_feature_importance(self) -> None:
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for model in self.model.values():
            for col, value in zip(self._used_features, getattr(model, "feature_importances_", [])):
                totals[col] = totals.get(col, 0.0) + float(value)
                counts[col] = counts.get(col, 0) + 1
        self._feature_importance = {col: totals[col] / counts[col] for col in totals}

    def metrics_report(self) -> dict[str, Any]:
        out = metrics_with_feature_importance(self._metrics, self._feature_importance, top_n=30)
        out["route_col"] = self.route_col
        out["expert_count"] = len(self._experts)
        out["experts"] = dict(self._train_stats.get("experts") or {})
        return out

    def feature_importance(self) -> dict[str, float]:
        return copy_feature_importance(self._feature_importance)

    def summarize(self) -> None:
        if not self._is_fit:
            return
        print_model_section(f"Sklearn {self.route_col.title()} Routed MoE Classifier")
        print(f"  - Routing column: {self.route_col}")
        print(f"  - Specialist experts: {len(self._experts)}")
        print(f"  - Global fallback: yes")
        print(f"  - Expert rows: {self._train_stats.get('experts', {})}")
