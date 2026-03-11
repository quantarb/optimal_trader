from __future__ import annotations

from typing import Any
import uuid

from domain.models import ModelScoringSpec, ModelTrainingSpec
from infra.repositories import DjangoArtifactRepository
from ml.execution import score_model_from_artifact_inputs, train_model_from_artifact_inputs
from pipeline.contracts import STATE_PANEL_ARTIFACT_TYPES


def build_model_training_spec(config: dict[str, Any]) -> ModelTrainingSpec:
    """Convert raw job config into a typed training spec."""

    target_col = str(config.get("target_col") or ("trade_return" if str(config.get("task_type") or "").strip().lower() == "regression" else "label")).strip()
    algorithm = str(config.get("algorithm") or "random_forest_classifier").strip().lower()
    framework = str(config.get("framework") or ("torch" if algorithm == "autoencoder" else "sklearn")).strip()
    task_type = str(config.get("task_type") or "classification").strip().lower()
    feature_families = tuple(
        str(value).strip() for value in list(config.get("feature_families") or []) if str(value).strip()
    )
    label_ks: list[int] = []
    for value in list(config.get("label_ks") or []):
        try:
            parsed = int(value)
        except Exception:
            continue
        if parsed > 0 and parsed not in label_ks:
            label_ks.append(parsed)
    try:
        label_k = int(config["label_k"]) if config.get("label_k") not in (None, "") else None
    except Exception:
        label_k = None
    try:
        min_abs_trade_return = (
            max(0.0, float(config.get("min_abs_trade_return_pct")) / 100.0)
            if config.get("min_abs_trade_return_pct") not in (None, "")
            else None
        )
    except Exception:
        min_abs_trade_return = None
    try:
        max_hold_days = max(1, int(config.get("max_hold_days"))) if config.get("max_hold_days") not in (None, "") else None
    except Exception:
        max_hold_days = None
    oracle_cluster_keys: list[str] = []
    for value in list(config.get("oracle_cluster_keys") or []):
        key = str(value).strip()
        if key and key not in oracle_cluster_keys:
            oracle_cluster_keys.append(key)
    prediction_artifact_ids: list[int] = []
    for value in list(config.get("prediction_artifact_ids") or []):
        try:
            parsed = int(value)
        except Exception:
            continue
        if parsed > 0 and parsed not in prediction_artifact_ids:
            prediction_artifact_ids.append(parsed)
    return ModelTrainingSpec(
        model_name=str(config.get("model_name") or config.get("name") or f"pipeline_{algorithm}_{uuid.uuid4().hex[:8]}").strip(),
        algorithm=algorithm,
        task_type=task_type,
        target_col=target_col,
        framework=framework,
        split_ratio=float(config.get("split_ratio") or 0.8),
        params=dict(config.get("params") or {}),
        start_date=str(config.get("train_start_date") or "").strip() or None,
        end_date=str(config.get("train_end_date") or "").strip() or None,
        feature_family=str(config.get("feature_family") or "").strip() or None,
        feature_families=feature_families,
        label_k=label_k,
        label_ks=tuple(label_ks),
        min_abs_trade_return=min_abs_trade_return,
        max_hold_days=max_hold_days,
        sample_weight_mode=str(config.get("sample_weight_mode") or "uniform").strip().lower() or "uniform",
        oracle_cluster_keys=tuple(oracle_cluster_keys),
        prediction_artifact_ids=tuple(prediction_artifact_ids),
    )


def build_model_scoring_spec(config: dict[str, Any], *, saved_model_id: int) -> ModelScoringSpec:
    """Convert raw job config into a typed scoring spec."""

    prediction_artifact_ids: list[int] = []
    for value in list(config.get("prediction_artifact_ids") or []):
        try:
            parsed = int(value)
        except Exception:
            continue
        if parsed > 0 and parsed not in prediction_artifact_ids:
            prediction_artifact_ids.append(parsed)
    label_artifact_id = int(config.get("label_artifact_id") or 0)
    return ModelScoringSpec(
        saved_model_id=int(saved_model_id),
        label_artifact_id=label_artifact_id if label_artifact_id > 0 else None,
        start_date=str(config.get("score_start_date") or config.get("start_date") or "").strip() or None,
        end_date=str(config.get("score_end_date") or config.get("end_date") or "").strip() or None,
        prediction_artifact_ids=tuple(prediction_artifact_ids),
    )


def train_model_workflow(
    *,
    spec: ModelTrainingSpec,
    feature_artifact,
    label_artifact,
    artifact_repo: DjangoArtifactRepository | None = None,
    progress_callback=None,
):
    """Train a saved model from feature/label artifacts."""

    repo = artifact_repo or DjangoArtifactRepository()
    extra_panels = repo.list_pipeline_artifacts(spec.prediction_artifact_ids, artifact_types=STATE_PANEL_ARTIFACT_TYPES)
    return train_model_from_artifact_inputs(
        name=spec.model_name,
        algorithm=spec.algorithm,
        task_type=spec.task_type,
        target_col=spec.target_col,
        feature_artifact=feature_artifact,
        label_artifact=label_artifact,
        extra_panel_artifacts=extra_panels,
        framework=spec.framework,
        split_ratio=spec.split_ratio,
        params=spec.params,
        start_date=spec.start_date,
        end_date=spec.end_date,
        feature_family=spec.feature_family,
        feature_families=spec.feature_families,
        label_k=spec.label_k,
        label_ks=spec.label_ks,
        min_abs_trade_return=spec.min_abs_trade_return,
        max_hold_days=spec.max_hold_days,
        sample_weight_mode=spec.sample_weight_mode,
        oracle_cluster_keys=spec.oracle_cluster_keys,
        progress_callback=progress_callback,
    )


def score_model_workflow(
    *,
    spec: ModelScoringSpec,
    feature_artifact,
    artifact_repo: DjangoArtifactRepository | None = None,
    progress_callback=None,
):
    """Score a saved model against a feature artifact."""

    repo = artifact_repo or DjangoArtifactRepository()
    saved_model = repo.get_saved_model(spec.saved_model_id)
    if saved_model is None:
        raise ValueError(f"Saved model artifact #{spec.saved_model_id} was not found.")
    extra_panels = repo.list_pipeline_artifacts(spec.prediction_artifact_ids, artifact_types=STATE_PANEL_ARTIFACT_TYPES)
    label_artifact = None
    if spec.label_artifact_id is not None:
        label_artifact = repo.get_pipeline_artifact(spec.label_artifact_id, artifact_type="LABELS")
        if label_artifact is None:
            raise ValueError(f"Label artifact #{spec.label_artifact_id} was not found.")
    return score_model_from_artifact_inputs(
        model_record=saved_model,
        feature_artifact=feature_artifact,
        extra_panel_artifacts=extra_panels,
        label_artifact=label_artifact,
        start_date=spec.start_date,
        end_date=spec.end_date,
        return_metadata=True,
        progress_callback=progress_callback,
    )
