from __future__ import annotations

import uuid
from typing import Any

from ml.execution import (
    build_feature_frame_from_artifacts,
    load_artifact_csv_frame,
    score_model_from_artifact_inputs,
    train_model_from_artifact_inputs,
)
from ml.models import ModelArtifact as SavedModelArtifact

from .contracts import (
    PREDICTION_REQUIRED_COLUMNS,
    STATE_PANEL_ARTIFACT_TYPES,
    build_schema_metadata,
    normalize_prediction_output_frame,
    validate_frame_columns,
)
from .progress import ProgressReporter
from .service_runtime import (
    BuiltOutput,
    PipelineExecutionError,
    read_json_artifact,
    write_frame_artifact,
    write_payload_artifact,
)
from .models import Artifact


def execute_train(
    config: dict[str, Any],
    labels_artifact,
    features_artifact,
    *,
    pipeline_run=None,
    job_run=None,
    performance_tracer=None,
) -> BuiltOutput:
    return execute_fit_model(
        config,
        labels_artifact,
        features_artifact,
        algorithm=str(config.get("algorithm") or "random_forest_classifier").strip().lower(),
        task_type=str(config.get("task_type") or "classification").strip().lower(),
        artifact_type="MODEL",
        pipeline_run=pipeline_run,
        job_run=job_run,
        performance_tracer=performance_tracer,
    )


def execute_fit_model(
    config: dict[str, Any],
    labels_artifact,
    features_artifact,
    *,
    algorithm: str,
    task_type: str,
    artifact_type: str,
    pipeline_run=None,
    job_run=None,
    performance_tracer=None,
) -> BuiltOutput:
    progress = ProgressReporter(pipeline_run=pipeline_run, job_run=job_run)
    extra_panel_ids = [int(v) for v in list(config.get("prediction_artifact_ids") or []) if int(v or 0) > 0]
    extra_panels = list(
        Artifact.objects.filter(id__in=extra_panel_ids, artifact_type__in=STATE_PANEL_ARTIFACT_TYPES).order_by("id")
    )
    target_col = str(config.get("target_col") or ("trade_return" if task_type == "regression" else "label")).strip()
    framework = str(config.get("framework") or ("torch" if algorithm == "autoencoder" else "sklearn")).strip()
    split_ratio = float(config.get("split_ratio") or 0.8)
    model_params = dict(config.get("params") or {})
    model_name = str(
        config.get("model_name") or config.get("name") or f"pipeline_{algorithm}_{uuid.uuid4().hex[:8]}"
    ).strip()
    train_start_date = str(config.get("train_start_date") or "").strip() or None
    train_end_date = str(config.get("train_end_date") or "").strip() or None
    feature_family = str(config.get("feature_family") or "").strip() or None
    feature_families = [str(value).strip() for value in list(config.get("feature_families") or []) if str(value).strip()]
    label_k_raw = config.get("label_k")
    label_k = None
    try:
        if label_k_raw not in (None, ""):
            label_k = int(label_k_raw)
    except Exception:
        label_k = None
    label_ks: list[int] = []
    for value in list(config.get("label_ks") or []):
        try:
            parsed = int(value)
        except Exception:
            continue
        if parsed > 0 and parsed not in label_ks:
            label_ks.append(parsed)
    min_abs_trade_return_pct_raw = config.get("min_abs_trade_return_pct")
    try:
        min_abs_trade_return = (
            max(0.0, float(min_abs_trade_return_pct_raw) / 100.0)
            if min_abs_trade_return_pct_raw not in (None, "")
            else None
        )
    except Exception:
        min_abs_trade_return = None
    max_hold_days_raw = config.get("max_hold_days")
    try:
        max_hold_days = max(1, int(max_hold_days_raw)) if max_hold_days_raw not in (None, "") else None
    except Exception:
        max_hold_days = None
    sample_weight_mode = str(config.get("sample_weight_mode") or "uniform").strip().lower() or "uniform"
    oracle_cluster_keys: list[str] = []
    for value in list(config.get("oracle_cluster_keys") or []):
        key = str(value).strip()
        if key and key not in oracle_cluster_keys:
            oracle_cluster_keys.append(key)
    try:
        progress.update(
            phase="prepare_training_dataset",
            phase_label="Prepare training dataset",
            phase_index=1,
            phase_total=3,
            force=True,
        )
        stage_ctx = (
            performance_tracer.stage(
                "model.fit",
                category="model_training",
                workload_type="batched",
                metadata={"algorithm": algorithm, "task_type": task_type},
            )
            if performance_tracer is not None
            else None
        )
        if stage_ctx is None:
            saved_model = train_model_from_artifact_inputs(
                name=model_name,
                algorithm=algorithm,
                task_type=task_type,
                target_col=target_col,
                feature_artifact=features_artifact,
                label_artifact=labels_artifact,
                extra_panel_artifacts=extra_panels,
                framework=framework,
                split_ratio=split_ratio,
                params=model_params,
                start_date=train_start_date,
                end_date=train_end_date,
                feature_family=feature_family,
                feature_families=feature_families,
                label_k=label_k,
                label_ks=label_ks,
                min_abs_trade_return=min_abs_trade_return,
                max_hold_days=max_hold_days,
                sample_weight_mode=sample_weight_mode,
                oracle_cluster_keys=oracle_cluster_keys,
                progress_callback=lambda **kwargs: progress.update(**kwargs),
            )
        else:
            with stage_ctx:
                saved_model = train_model_from_artifact_inputs(
                    name=model_name,
                    algorithm=algorithm,
                    task_type=task_type,
                    target_col=target_col,
                    feature_artifact=features_artifact,
                    label_artifact=labels_artifact,
                    extra_panel_artifacts=extra_panels,
                    framework=framework,
                    split_ratio=split_ratio,
                    params=model_params,
                    start_date=train_start_date,
                    end_date=train_end_date,
                    feature_family=feature_family,
                    feature_families=feature_families,
                    label_k=label_k,
                    label_ks=label_ks,
                    min_abs_trade_return=min_abs_trade_return,
                    max_hold_days=max_hold_days,
                    sample_weight_mode=sample_weight_mode,
                    oracle_cluster_keys=oracle_cluster_keys,
                    progress_callback=lambda **kwargs: progress.update(**kwargs),
                )
        progress.complete(message="Model fit completed")
    except Exception as exc:
        raise PipelineExecutionError(str(exc)) from exc

    saved_metadata = dict(saved_model.metadata or {})
    payload = {
        "model_artifact_id": int(saved_model.id),
        "name": str(saved_model.name),
        "version": int(saved_model.version),
        "framework": str(saved_model.framework),
        "algorithm": algorithm,
        "task_type": str(saved_model.task_type),
        "target_col": str(saved_model.target_col),
        "feature_cols": list(saved_model.feature_cols or []),
        "metrics": dict(saved_model.metrics or {}),
        "metadata": saved_metadata,
    }
    key = f"model_{uuid.uuid4().hex}"
    stored = write_payload_artifact(key, payload)
    return BuiltOutput(
        artifact_type=artifact_type,
        content={
            "model_artifact_id": int(saved_model.id),
            "trained_rows": int((saved_model.metadata or {}).get("joined_rows") or 0),
            "symbols": int((saved_model.metadata or {}).get("symbols_count") or 0),
            "metrics": dict(saved_model.metrics or {}),
            "train_start_date": str(train_start_date or ""),
            "train_end_date": str(train_end_date or ""),
            "dataset_build_seconds": float(saved_metadata.get("dataset_build_seconds") or 0.0),
            "fit_seconds": float(saved_metadata.get("fit_seconds") or 0.0),
            "train_prediction_seconds": float(saved_metadata.get("train_prediction_seconds") or 0.0),
            "feature_family": str(saved_metadata.get("feature_family") or ""),
            "feature_families": list(saved_metadata.get("feature_families") or []),
            "label_k": saved_metadata.get("label_k"),
            "label_ks": list(saved_metadata.get("label_ks") or []),
            "coverage_start_date": str(saved_metadata.get("coverage_start_date") or ""),
            "coverage_end_date": str(saved_metadata.get("coverage_end_date") or ""),
            "min_abs_trade_return_pct": round(float(min_abs_trade_return or 0.0) * 100.0, 6) if min_abs_trade_return is not None else None,
            "max_hold_days": int(max_hold_days) if max_hold_days is not None else None,
            "sample_weight_mode": sample_weight_mode,
            "oracle_cluster_scope": str(saved_metadata.get("oracle_cluster_scope") or "generalist"),
            "oracle_cluster_keys": list(saved_metadata.get("oracle_cluster_keys") or []),
        },
        metadata={
            "saved_model_artifact_id": int(saved_model.id),
            "source_labels_artifact_id": labels_artifact.id,
            "source_features_artifact_id": features_artifact.id,
            "source_prediction_artifact_ids": [int(artifact.id) for artifact in extra_panels],
            "predictions_uri": str(saved_metadata.get("predictions_uri") or ""),
            "train_start_date": str(train_start_date or ""),
            "train_end_date": str(train_end_date or ""),
            "dataset_build_seconds": float(saved_metadata.get("dataset_build_seconds") or 0.0),
            "fit_seconds": float(saved_metadata.get("fit_seconds") or 0.0),
            "train_prediction_seconds": float(saved_metadata.get("train_prediction_seconds") or 0.0),
            "feature_family": str(saved_metadata.get("feature_family") or ""),
            "feature_families": list(saved_metadata.get("feature_families") or []),
            "feature_family_columns": list(saved_metadata.get("feature_family_columns") or []),
            "available_feature_families": list(saved_metadata.get("available_feature_families") or []),
            "label_k": saved_metadata.get("label_k"),
            "label_ks": list(saved_metadata.get("label_ks") or []),
            "coverage_start_date": str(saved_metadata.get("coverage_start_date") or ""),
            "coverage_end_date": str(saved_metadata.get("coverage_end_date") or ""),
            "coverage_rows": int(saved_metadata.get("coverage_rows") or 0),
            "label_rows_after_filters": int(saved_metadata.get("label_rows_after_filters") or 0),
            "label_rows_before_trade_filters": int(saved_metadata.get("label_rows_before_trade_filters") or 0),
            "min_abs_trade_return_pct": round(float(min_abs_trade_return or 0.0) * 100.0, 6) if min_abs_trade_return is not None else None,
            "max_hold_days": int(max_hold_days) if max_hold_days is not None else None,
            "sample_weight_mode": sample_weight_mode,
            "oracle_cluster_scope": str(saved_metadata.get("oracle_cluster_scope") or "generalist"),
            "oracle_cluster_keys": list(saved_metadata.get("oracle_cluster_keys") or []),
            "cluster_rows_before_filter": int(saved_metadata.get("cluster_rows_before_filter") or 0),
            "cluster_rows_after_filter": int(saved_metadata.get("cluster_rows_after_filter") or 0),
            **stored.storage_metadata(),
        },
        uri=stored.uri,
    )


def execute_predict(
    config: dict[str, Any],
    model_artifact,
    features_artifact,
    *,
    pipeline_run=None,
    job_run=None,
    performance_tracer=None,
) -> BuiltOutput:
    return execute_score_model(
        config,
        model_artifact,
        features_artifact,
        expected_pipeline_artifact_type="MODEL",
        output_artifact_type="PREDICTIONS",
        pipeline_run=pipeline_run,
        job_run=job_run,
        performance_tracer=performance_tracer,
    )


def execute_score_model(
    config: dict[str, Any],
    model_artifact,
    features_artifact,
    *,
    expected_pipeline_artifact_type: str,
    output_artifact_type: str,
    pipeline_run=None,
    job_run=None,
    performance_tracer=None,
) -> BuiltOutput:
    progress = ProgressReporter(pipeline_run=pipeline_run, job_run=job_run)
    if str(model_artifact.artifact_type) != expected_pipeline_artifact_type:
        raise PipelineExecutionError(
            f"Expected model artifact type {expected_pipeline_artifact_type!r}, got {model_artifact.artifact_type!r}."
        )
    model_payload = read_json_artifact(model_artifact)
    saved_model_id = int(
        model_payload.get("model_artifact_id") or (model_artifact.metadata or {}).get("saved_model_artifact_id") or 0
    )
    if saved_model_id <= 0:
        raise PipelineExecutionError("Pipeline MODEL artifact is missing saved_model_artifact_id.")
    saved_model = SavedModelArtifact.objects.filter(pk=saved_model_id).first()
    if saved_model is None:
        raise PipelineExecutionError(f"Saved model artifact #{saved_model_id} was not found.")

    extra_panel_ids = [int(v) for v in list(config.get("prediction_artifact_ids") or []) if int(v or 0) > 0]
    extra_panels = list(
        Artifact.objects.filter(id__in=extra_panel_ids, artifact_type__in=STATE_PANEL_ARTIFACT_TYPES).order_by("id")
    )
    label_artifact_id = int(config.get("label_artifact_id") or 0)
    score_start_date = str(config.get("score_start_date") or config.get("start_date") or "").strip() or None
    score_end_date = str(config.get("score_end_date") or config.get("end_date") or "").strip() or None
    label_artifact = Artifact.objects.filter(pk=label_artifact_id, artifact_type="LABELS").first() if label_artifact_id > 0 else None
    try:
        progress.update(
            phase="load_scoring_inputs",
            phase_label="Load scoring inputs",
            phase_index=1,
            phase_total=2,
            force=True,
        )
        stage_ctx = (
            performance_tracer.stage(
                "model.score",
                category="inference",
                workload_type="batched",
                metadata={"artifact_type": output_artifact_type},
            )
            if performance_tracer is not None
            else None
        )
        if stage_ctx is None:
            prediction_df, score_meta = score_model_from_artifact_inputs(
                model_record=saved_model,
                feature_artifact=features_artifact,
                extra_panel_artifacts=extra_panels,
                label_artifact=label_artifact,
                start_date=score_start_date,
                end_date=score_end_date,
                return_metadata=True,
                progress_callback=lambda **kwargs: progress.update(**kwargs),
            )
        else:
            with stage_ctx:
                prediction_df, score_meta = score_model_from_artifact_inputs(
                    model_record=saved_model,
                    feature_artifact=features_artifact,
                    extra_panel_artifacts=extra_panels,
                    label_artifact=label_artifact,
                    start_date=score_start_date,
                    end_date=score_end_date,
                    return_metadata=True,
                    progress_callback=lambda **kwargs: progress.update(**kwargs),
                )
        progress.complete(message="Model scoring completed")
    except Exception as exc:
        raise PipelineExecutionError(str(exc)) from exc
    if prediction_df.empty:
        raise PipelineExecutionError("No feature rows available for prediction.")
    prediction_df = normalize_prediction_output_frame(prediction_df, artifact_type=output_artifact_type)
    validate_frame_columns(prediction_df, PREDICTION_REQUIRED_COLUMNS, artifact_type=output_artifact_type)

    key = f"predictions_{uuid.uuid4().hex}"
    storage_format = str(config.get("artifact_storage_format") or "csv").strip().lower() or "csv"
    if performance_tracer is not None:
        with performance_tracer.stage(
            "model.serialize_predictions",
            category="serialization",
            workload_type="batched",
            metadata={"rows": int(len(prediction_df)), "storage_format": storage_format},
        ):
            stored = write_frame_artifact(
                key,
                frame=prediction_df,
                fieldnames=list(prediction_df.columns),
                storage_format=storage_format,
            )
    else:
        stored = write_frame_artifact(
            key,
            frame=prediction_df,
            fieldnames=list(prediction_df.columns),
            storage_format=storage_format,
        )
    return BuiltOutput(
        artifact_type=output_artifact_type,
        content={
            "rows": int(len(prediction_df)),
            "model_artifact_id": int(saved_model.id),
            "task_type": str(saved_model.task_type),
            "score_start_date": str(score_start_date or ""),
            "score_end_date": str(score_end_date or ""),
            "score_seconds": float(score_meta.get("score_seconds") or 0.0),
        },
        metadata={
            "source_model_artifact_id": model_artifact.id,
            "source_features_artifact_id": features_artifact.id,
            "saved_model_artifact_id": int(saved_model.id),
            "source_prediction_artifact_ids": [int(artifact.id) for artifact in extra_panels],
            "source_label_artifact_id": int(label_artifact.id) if label_artifact is not None else 0,
            "schema": build_schema_metadata(
                artifact_type=output_artifact_type,
                required_columns=PREDICTION_REQUIRED_COLUMNS,
                actual_columns=list(prediction_df.columns),
            ),
            "score_start_date": str(score_start_date or ""),
            "score_end_date": str(score_end_date or ""),
            "score_seconds": float(score_meta.get("score_seconds") or 0.0),
            "rows_scored": int(score_meta.get("rows_scored") or 0),
            "feature_family": str((saved_model.metadata or {}).get("feature_family") or ""),
            "feature_families": list((saved_model.metadata or {}).get("feature_families") or []),
            "label_k": (saved_model.metadata or {}).get("label_k"),
            "label_ks": list((saved_model.metadata or {}).get("label_ks") or []),
            **stored.storage_metadata(),
        },
        uri=stored.uri,
    )


__all__ = [
    "execute_fit_model",
    "execute_predict",
    "execute_score_model",
    "execute_train",
]
