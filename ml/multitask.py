from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml.base import FitSpec
from ml.frameworks.sklearn import SklearnRFClassifier, SklearnRFRegressor


def derive_oracle_cluster_labels(df: pd.DataFrame) -> pd.Series:
    work = df.copy()
    side = work["side"].astype(str).str.strip().str.lower() if "side" in work.columns else pd.Series("unknown", index=work.index)
    freq = work["freq"].astype(str).str.strip() if "freq" in work.columns else pd.Series("unknown", index=work.index)
    k = pd.to_numeric(work["k"], errors="coerce").fillna(0).astype(int) if "k" in work.columns else pd.Series(0, index=work.index)
    hold_days = pd.to_numeric(work["hold_days"], errors="coerce").fillna(0.0)
    trade_return = pd.to_numeric(work["trade_return"], errors="coerce").fillna(0.0)

    def hold_bucket(value: float) -> str:
        if value <= 10:
            return "hold_1_10"
        if value <= 30:
            return "hold_11_30"
        if value <= 90:
            return "hold_31_90"
        return "hold_91_plus"

    hold_labels = hold_days.apply(lambda value: hold_bucket(float(value)))
    bucket_count = min(4, int(trade_return.nunique()))
    if bucket_count >= 2:
        return_bucket = pd.qcut(
            trade_return.rank(method="first"),
            q=bucket_count,
            duplicates="drop",
        ).astype(str)
    else:
        return_bucket = pd.Series("single_bucket", index=trade_return.index)
    return side + "|" + freq + "|k=" + k.astype(str) + "|" + hold_labels + "|" + return_bucket


@dataclass
class MultiTaskForestBundle:
    used_features: list[str]
    label_model: Any | None = None
    trade_return_model: Any | None = None
    hold_days_model: Any | None = None
    cluster_model: Any | None = None
    cluster_mapping: dict[int, str] | None = None
    head_metrics: dict[str, Any] | None = None

    def metrics_report(self) -> dict[str, Any]:
        return dict(self.head_metrics or {})

    def summarize(self) -> None:
        active_heads = [name for name, model in {
            "label": self.label_model,
            "trade_return": self.trade_return_model,
            "hold_days": self.hold_days_model,
            "cluster": self.cluster_model,
        }.items() if model is not None]
        print(f"MultiTaskForestBundle heads: {', '.join(active_heads)}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        frame = self.predict_frame(df)
        if "prediction" in frame.columns:
            return pd.to_numeric(frame["prediction"], errors="coerce").fillna(0.0).to_numpy()
        return np.zeros(len(df), dtype=float)

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.used_features:
            return pd.DataFrame(index=df.index)
        X_df = df[self.used_features].copy()
        X_df[self.used_features] = X_df[self.used_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        out = pd.DataFrame(index=df.index)

        if self.label_model is not None:
            out["mtl_label"] = self.label_model.predict(X_df)
            predict_proba = getattr(getattr(self.label_model, "model", None), "predict_proba", None)
            if callable(predict_proba):
                try:
                    proba = predict_proba(X_df[self.used_features])
                    if getattr(proba, "shape", None) is not None and len(proba.shape) == 2 and proba.shape[1] >= 2:
                        out["mtl_prob_buy"] = proba[:, 1]
                except Exception:
                    pass

        if self.trade_return_model is not None:
            out["mtl_trade_return"] = self.trade_return_model.predict(X_df)
            out["prediction"] = out["mtl_trade_return"]

        if self.hold_days_model is not None:
            out["mtl_hold_days"] = self.hold_days_model.predict(X_df)

        if self.cluster_model is not None:
            cluster_codes = pd.Series(self.cluster_model.predict(X_df), index=df.index).fillna(0).astype(int)
            mapping = dict(self.cluster_mapping or {})
            out["mtl_cluster_code"] = cluster_codes
            out["mtl_cluster_key"] = cluster_codes.map(mapping).fillna(cluster_codes.astype(str))
            predict_proba = getattr(getattr(self.cluster_model, "model", None), "predict_proba", None)
            if callable(predict_proba):
                try:
                    cluster_proba = predict_proba(X_df[self.used_features])
                    if getattr(cluster_proba, "shape", None) is not None and len(cluster_proba.shape) == 2:
                        out["mtl_cluster_confidence"] = np.max(cluster_proba, axis=1)
                except Exception:
                    pass

        if "mtl_prob_buy" in out.columns:
            out["prediction_score"] = out["mtl_prob_buy"]
        elif "prediction" in out.columns:
            out["prediction_score"] = pd.to_numeric(out["prediction"], errors="coerce")
        else:
            out["prediction_score"] = pd.Series(0.0, index=df.index, dtype=float)
            out["prediction"] = pd.Series(0.0, index=df.index, dtype=float)

        return out


def train_multi_task_forest_bundle(
    *,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    split_ratio: float,
    model_params: dict[str, Any] | None = None,
    include_cluster_head: bool = True,
) -> MultiTaskForestBundle:
    params = dict(model_params or {})
    working = train_df.copy()
    if "sample_weight" not in working.columns:
        working["sample_weight"] = 1.0

    label_model = None
    trade_return_model = None
    hold_days_model = None
    cluster_model = None
    cluster_mapping: dict[int, str] = {}
    head_metrics: dict[str, Any] = {}

    if "label" in working.columns and working["label"].notna().any():
        unique_label_classes = sorted(pd.Series(working["label"]).dropna().astype(int).unique().tolist())
        if len(unique_label_classes) >= 2:
            label_model = SklearnRFClassifier(random_state=1337, **params)
            label_model.fit(
                working,
                FitSpec(feature_cols=list(feature_cols), target_col="label", weight_col="sample_weight", split_ratio=float(split_ratio)),
                verbose=False,
            )
            head_metrics["label"] = label_model.metrics_report()
        else:
            head_metrics["label"] = {
                "status": "skipped",
                "reason": "single_class_target",
                "class_values": unique_label_classes,
            }

    if "trade_return" in working.columns and working["trade_return"].notna().any():
        trade_return_model = SklearnRFRegressor(
            test_size=max(0.0, 1.0 - float(split_ratio)),
            random_state=1337,
            **params,
        )
        trade_return_model.fit(
            working,
            FitSpec(feature_cols=list(feature_cols), target_col="trade_return", weight_col="sample_weight", split_ratio=float(split_ratio)),
            verbose=False,
        )
        head_metrics["trade_return"] = trade_return_model.metrics_report()

    if "hold_days" in working.columns and pd.to_numeric(working["hold_days"], errors="coerce").notna().any():
        working["hold_days"] = pd.to_numeric(working["hold_days"], errors="coerce")
        hold_days_model = SklearnRFRegressor(
            test_size=max(0.0, 1.0 - float(split_ratio)),
            random_state=1337,
            **params,
        )
        hold_days_model.fit(
            working,
            FitSpec(feature_cols=list(feature_cols), target_col="hold_days", weight_col="sample_weight", split_ratio=float(split_ratio)),
            verbose=False,
        )
        head_metrics["hold_days"] = hold_days_model.metrics_report()

    if include_cluster_head and {"trade_return", "hold_days"}.issubset(set(working.columns)):
        working["cluster_target"] = derive_oracle_cluster_labels(working)
        cluster_counts = working["cluster_target"].value_counts(dropna=True)
        min_cluster_examples = int(cluster_counts.min()) if not cluster_counts.empty else 0
        use_holdout = 0.0 < float(split_ratio) < 1.0
        if working["cluster_target"].nunique() >= 2 and (not use_holdout or min_cluster_examples >= 2):
            cluster_model = SklearnRFClassifier(random_state=1337, **params)
            cluster_model.fit(
                working,
                FitSpec(feature_cols=list(feature_cols), target_col="cluster_target", weight_col="sample_weight", split_ratio=float(split_ratio)),
                verbose=False,
            )
            cluster_mapping = dict(getattr(cluster_model, "_class_mapping", {}) or {})
            head_metrics["cluster"] = cluster_model.metrics_report()
        elif working["cluster_target"].nunique() >= 2:
            head_metrics["cluster"] = {
                "status": "skipped",
                "reason": "insufficient_examples_per_cluster",
                "min_cluster_examples": int(min_cluster_examples),
            }

    bundle = MultiTaskForestBundle(
        used_features=list(feature_cols),
        label_model=label_model,
        trade_return_model=trade_return_model,
        hold_days_model=hold_days_model,
        cluster_model=cluster_model,
        cluster_mapping=cluster_mapping,
        head_metrics=head_metrics,
    )
    return bundle
