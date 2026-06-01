from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import uuid
from typing import Any, Mapping, Sequence

import pandas as pd

from domain.models.datasets import dedupe_label_frame
from domain.models.feature_families import infer_feature_family_columns
from ml.base import FitSpec
from ml.frameworks.sklearn import SklearnMoERFClassifier, SklearnRFClassifier, SklearnRFRegressor
from pipeline.contracts import normalize_prediction_output_frame
from settings import BASE_DIR

from .multitask import train_multi_task_forest_bundle

PREDICTION_ARTIFACT_DIR = Path(BASE_DIR) / "data" / "pipeline_artifacts"


def _attach_target(train_df: pd.DataFrame, *, target_col: str, task_type: str) -> pd.DataFrame:
    df = train_df.copy()
    if target_col in df.columns and df[target_col].notna().any():
        return df
    close = pd.to_numeric(df["close"], errors="coerce")
    if "symbol" in df.columns:
        next_return = close.groupby(df["symbol"]).pct_change()
        next_return = next_return.groupby(df["symbol"]).shift(-1)
    else:
        next_return = close.pct_change().shift(-1)
    if task_type == "classification":
        df[target_col] = (next_return > 0).astype(int)
    else:
        df[target_col] = next_return.astype(float)
    return df


def _train_classifier(
    *,
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    model_params: dict[str, Any],
    target_col: str,
    split_ratio: float,
) -> Any:
    df = _attach_target(train_df, target_col=target_col, task_type="classification")
    spec = FitSpec(
        feature_cols=list(feature_cols),
        target_col=target_col,
        weight_col="sample_weight",
        split_ratio=float(split_ratio),
    )
    model = SklearnRFClassifier(random_state=1337, **model_params)
    model.fit(df, spec, verbose=False)
    return model


def _train_regressor(
    *,
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    model_params: dict[str, Any],
    target_col: str,
    split_ratio: float,
) -> Any:
    df = _attach_target(train_df, target_col=target_col, task_type="regression")
    model = SklearnRFRegressor(
        test_size=max(0.0, 1.0 - float(split_ratio)),
        random_state=1337,
        **model_params,
    )
    spec = FitSpec(
        feature_cols=list(feature_cols),
        target_col=target_col,
        weight_col="sample_weight",
        split_ratio=float(split_ratio),
    )
    model.fit(df, spec, verbose=False)
    return model


def _train_autoencoder(train_df: pd.DataFrame, feature_cols: Sequence[str]) -> Any:
    from ml.raw_stack import train_ae as _train_ae
    ae_model, _numeric_cols = _train_ae(train_df, feature_cols, verbose=False)
    setattr(ae_model, "_used_features", list(feature_cols))
    return ae_model


def _train_multi_task_bundle(
    *,
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    model_params: dict[str, Any],
    split_ratio: float,
) -> Any:
    bundle = train_multi_task_forest_bundle(
        train_df=train_df,
        feature_cols=list(feature_cols),
        split_ratio=float(split_ratio),
        model_params=model_params,
        include_cluster_head=True,
    )
    setattr(bundle, "_used_features", list(feature_cols))
    return bundle


def _train_moe_classifier(
    *,
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    model_params: dict[str, Any],
    target_col: str,
    split_ratio: float,
    feature_families: Mapping[str, Sequence[str]] | None = None,
) -> Any:
    """Train a Mixture-of-Experts classifier with per-family expert forests."""
    df = _attach_target(train_df, target_col=target_col, task_type="classification")

    # Build feature family mapping: prefer explicit, fall back to prefix inference
    if feature_families:
        families: dict[str, list[str]] = {
            str(name): [str(c) for c in cols if str(c) in df.columns]
            for name, cols in dict(feature_families).items()
            if str(name).strip() and cols
        }
    else:
        families = infer_feature_family_columns(list(feature_cols))

    # Remove empty families
    families = {name: cols for name, cols in families.items() if cols}

    # Drop excluded families (analyst estimates, insider trading, ratings, etc.)
    exclude = {str(e).strip().lower() for e in model_params.get("exclude_families", [])}
    if exclude:
        families = {name: cols for name, cols in families.items() if name.lower() not in exclude}

    # Merge economic_indicators + treasury_rates into a single "macro" family
    macro_cols: list[str] = []
    for key in ("economic_indicators", "treasury_rates"):
        if key in families:
            macro_cols.extend(families.pop(key))
    if macro_cols:
        families["macro"] = sorted(set(macro_cols))

    # Drop families with fewer than 2 features (can't train a useful tree)
    families = {name: cols for name, cols in families.items() if len(cols) >= 2}

    if not families:
        raise ValueError("MoE classifier requires at least one family with ≥2 features.")

    spec = FitSpec(
        feature_cols=list(feature_cols),
        target_col=target_col,
        weight_col="sample_weight",
        split_ratio=float(split_ratio),
        model_tag="moe-family-forest",
    )

    model_params_clean = dict(model_params)
    model_params_clean.pop("feature_families", None)
    model_params_clean.pop("family_weights", None)
    model_params_clean.pop("exclude_families", None)

    family_weights_raw = dict(model_params).get("family_weights")
    family_weights: dict[str, float] | None = None
    if isinstance(family_weights_raw, dict):
        family_weights = {str(k): float(v) for k, v in family_weights_raw.items()}

    model = SklearnMoERFClassifier(
        feature_families=families,
        family_weights=family_weights,
        **model_params_clean,
    )
    model.fit(df, spec, verbose=False)
    return model


def fit_model_for_algorithm(
    *,
    algorithm: str,
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    model_params: dict[str, Any],
    target_col: str,
    split_ratio: float,
) -> Any:
    """Fit the selected model adapter on an artifact-backed training frame."""

    algorithm_value = str(algorithm or "").strip().lower()
    if algorithm_value == "random_forest_classifier":
        return _train_classifier(
            train_df=train_df,
            feature_cols=feature_cols,
            model_params=model_params,
            target_col=target_col,
            split_ratio=float(split_ratio),
        )
    if algorithm_value == "random_forest_regressor":
        return _train_regressor(
            train_df=train_df,
            feature_cols=feature_cols,
            model_params=model_params,
            target_col=target_col,
            split_ratio=float(split_ratio),
        )
    if algorithm_value == "autoencoder":
        return _train_autoencoder(train_df=train_df, feature_cols=feature_cols)
    if algorithm_value == "multi_task_forest":
        return _train_multi_task_bundle(
            train_df=train_df,
            feature_cols=feature_cols,
            model_params=model_params,
            split_ratio=float(split_ratio),
        )
    if algorithm_value == "moe_random_forest_classifier":
        return _train_moe_classifier(
            train_df=train_df,
            feature_cols=feature_cols,
            model_params=model_params,
            target_col=target_col,
            split_ratio=float(split_ratio),
            feature_families=model_params.get("feature_families"),
        )
    raise ValueError(f"Unsupported pipeline training algorithm: {algorithm!r}")


def metrics_for(model_obj: Any) -> dict[str, Any]:
    """Extract best-effort metrics from a fitted model adapter."""

    metrics_fn = getattr(model_obj, "metrics_report", None)
    if callable(metrics_fn):
        try:
            metrics = metrics_fn()
            return dict(metrics or {})
        except Exception:
            return {}
    return {}


def model_summary(model_obj: Any) -> str:
    """Capture the model adapter summary without leaking stdout."""

    summarize_fn = getattr(model_obj, "summarize", None)
    if not callable(summarize_fn):
        return ""
    buf = StringIO()
    try:
        with redirect_stdout(buf):
            summarize_fn()
    except Exception:
        return ""
    return buf.getvalue().strip()


def score_artifact_rows(
    *,
    model_obj: Any,
    feature_df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_df: pd.DataFrame | None = None,
    missing_feature_policy: str = "any_coverage",
) -> pd.DataFrame:
    """Score artifact-backed rows with a fitted model adapter."""

    prediction_df = feature_df.copy()
    usable_cols = [col for col in list(feature_cols) if col in prediction_df.columns]
    used_features = list(getattr(model_obj, "_used_features", []) or usable_cols)
    used_features = [col for col in used_features if col in prediction_df.columns]
    if not used_features:
        return pd.DataFrame()
    prediction_df = prediction_df.copy()
    for col in used_features:
        prediction_df[col] = pd.to_numeric(prediction_df[col], errors="coerce")
    policy_value = str(missing_feature_policy or "any_coverage").strip().lower() or "any_coverage"
    if policy_value in {"complete_case", "drop_missing"}:
        prediction_df = prediction_df[prediction_df[used_features].notna().all(axis=1)].copy()
    if prediction_df.empty or prediction_df[used_features].notna().any(axis=1).sum() == 0:
        return pd.DataFrame()
    prediction_df[used_features] = prediction_df[used_features].fillna(0.0)

    recon_error = getattr(model_obj, "recon_error", None)
    familiarity = getattr(model_obj, "familiarity", None)
    predict_frame = getattr(model_obj, "predict_frame", None)
    prediction_cols: dict[str, Any] = {}
    if callable(recon_error) and callable(familiarity):
        try:
            prediction_cols["prediction"] = recon_error(
                prediction_df,
                numeric_cols=used_features,
                categorical_cols=(),
            )
            prediction_cols["prediction_score"] = familiarity(
                prediction_df,
                numeric_cols=used_features,
                categorical_cols=(),
            )
        except Exception:
            return pd.DataFrame()
    elif callable(predict_frame):
        try:
            bundle_frame = predict_frame(prediction_df)
        except Exception:
            return pd.DataFrame()
        for col in list(bundle_frame.columns):
            prediction_cols[col] = bundle_frame[col]
    else:
        preds = model_obj.predict(prediction_df)
        prediction_cols["prediction"] = list(preds)
        predict_proba = getattr(getattr(model_obj, "model", None), "predict_proba", None)
        if callable(predict_proba):
            try:
                proba = predict_proba(prediction_df[used_features])
                if getattr(proba, "shape", None) is not None and len(proba.shape) == 2 and proba.shape[1] >= 2:
                    prediction_cols["prediction_score"] = proba[:, 1]
            except Exception:
                pass
    if prediction_cols:
        # Add all prediction outputs in one concat to avoid repeated column inserts.
        prediction_frame = pd.DataFrame(prediction_cols, index=prediction_df.index)
        prediction_df = pd.concat([prediction_df, prediction_frame], axis=1)

    if label_df is not None and not label_df.empty:
        label_df = dedupe_label_frame(label_df)
        merge_cols = [
            col
            for col in ["date", "symbol", "label", "market_position", "trade_return", "hold_days", "side", "freq", "k"]
            if col in label_df.columns
        ]
        if "date" in merge_cols and "symbol" in merge_cols:
            prediction_df = prediction_df.merge(
                label_df[merge_cols].copy(),
                on=["date", "symbol"],
                how="left",
                suffixes=("", "_label"),
            )
    artifact_type = "PREDICTIONS"
    if callable(recon_error) and callable(familiarity):
        artifact_type = "AUTOENCODER_SCORES"
    elif "mtl_prob_buy" in prediction_df.columns or "mtl_trade_return" in prediction_df.columns:
        artifact_type = "MTL_PREDICTIONS"
    elif "prediction_score" in prediction_df.columns:
        artifact_type = "CLASSIFIER_PREDICTIONS"
    elif "prediction" in prediction_df.columns:
        artifact_type = "REGRESSOR_PREDICTIONS"
    prediction_df = normalize_prediction_output_frame(prediction_df, artifact_type=artifact_type)
    return prediction_df.sort_values(["symbol", "date"]).reset_index(drop=True)


def write_prediction_rows_csv(name: str, prediction_df: pd.DataFrame) -> str:
    """Persist scored rows for downstream diagnostics."""

    PREDICTION_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTION_ARTIFACT_DIR / f"ml_predictions_{uuid.uuid4().hex}.csv"
    prediction_df.to_csv(path, index=False)
    return str(path)
