from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import pandas as pd

from domain.models import ArtifactTrainingDatasetSpec
from domain.models.datasets import (
    dedupe_label_frame as _domain_dedupe_label_frame,
    feature_columns_from_frame as _domain_feature_columns_from_frame,
    filter_frame_by_date as _domain_filter_frame_by_date,
)
from domain.models.feature_families import infer_feature_family_columns as _domain_infer_feature_family_columns
from pipeline.models import Artifact

from .artifact_datasets import (
    build_feature_frame_from_artifacts as _build_feature_frame_from_artifacts,
    build_training_frame_from_panel_artifacts as _build_training_frame_from_panel_artifacts,
    load_artifact_csv_frame as _load_artifact_csv_frame,
)
from .model_runtime import (
    fit_model_for_algorithm,
    metrics_for as _metrics_for,
    model_summary as _model_summary,
    score_artifact_rows as _score_artifact_rows,
    write_prediction_rows_csv as _write_prediction_rows_csv,
)
from .models import ModelArtifact
from .store import save_model_artifact

logger = logging.getLogger(__name__)

JOB_CONTEXT_KEY = "__job_context__"


def merge_job_params(
    model_params: dict[str, Any],
    *,
    symbol: str | None = None,
    symbols: Sequence[str] | None = None,
) -> dict[str, Any]:
    payload = dict(model_params)
    context: dict[str, Any] = {}
    if symbols:
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in symbols:
            code = str(raw).strip().upper()
            if not code or code in seen:
                continue
            seen.add(code)
            cleaned.append(code)
        if cleaned:
            context["symbols"] = cleaned
            context["symbol"] = cleaned[0]
    elif symbol:
        context["symbol"] = str(symbol).strip().upper()
    payload[JOB_CONTEXT_KEY] = context
    return payload


def extract_model_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in dict(params or {}).items() if key != JOB_CONTEXT_KEY}


def extract_training_symbol(params: dict[str, Any]) -> str:
    symbols = extract_training_symbols(params)
    if symbols:
        return symbols[0]
    context = dict(params or {}).get(JOB_CONTEXT_KEY)
    if isinstance(context, dict):
        raw = context.get("symbol")
        if raw:
            return str(raw).strip().upper()
    return ""


def extract_training_symbols(params: dict[str, Any]) -> list[str]:
    context = dict(params or {}).get(JOB_CONTEXT_KEY)
    if not isinstance(context, dict):
        return []
    raw_symbols = context.get("symbols")
    out: list[str] = []
    seen: set[str] = set()
    if isinstance(raw_symbols, (list, tuple)):
        for raw in raw_symbols:
            code = str(raw).strip().upper()
            if not code or code in seen:
                continue
            seen.add(code)
            out.append(code)
    if out:
        return out
    raw_symbol = context.get("symbol")
    if raw_symbol:
        code = str(raw_symbol).strip().upper()
        if code:
            return [code]
    return []


def _filter_frame_by_date(
    df: pd.DataFrame,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    return _domain_filter_frame_by_date(df, start_date=start_date, end_date=end_date)


def _dedupe_label_frame(label_df: pd.DataFrame) -> pd.DataFrame:
    return _domain_dedupe_label_frame(label_df)


def load_artifact_csv_frame(artifact: Artifact) -> pd.DataFrame:
    return _load_artifact_csv_frame(artifact)


def _feature_columns_from_artifact_df(feature_df: pd.DataFrame) -> list[str]:
    return _domain_feature_columns_from_frame(feature_df)


def infer_feature_family_columns(feature_cols: Sequence[str]) -> dict[str, list[str]]:
    return _domain_infer_feature_family_columns(feature_cols)


def build_feature_frame_from_artifacts(
    *,
    base_feature_artifact: Artifact,
    extra_panel_artifacts: Sequence[Artifact] = (),
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    return _build_feature_frame_from_artifacts(
        base_feature_artifact=base_feature_artifact,
        extra_panel_artifacts=extra_panel_artifacts,
    )


def build_training_frame_from_panel_artifacts(
    *,
    base_feature_artifact: Artifact,
    label_artifact: Artifact,
    extra_panel_artifacts: Sequence[Artifact] = (),
    start_date: str | None = None,
    end_date: str | None = None,
    feature_family: str | None = None,
    feature_families: Sequence[str] = (),
    label_k: int | None = None,
    label_ks: Sequence[int] = (),
    min_abs_trade_return: float | None = None,
    max_hold_days: int | None = None,
    sample_weight_mode: str = "uniform",
    oracle_cluster_keys: Sequence[str] = (),
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    dataset_spec = ArtifactTrainingDatasetSpec(
        start_date=start_date,
        end_date=end_date,
        feature_family=feature_family,
        feature_families=tuple(feature_families),
        label_k=label_k,
        label_ks=tuple(label_ks),
        min_abs_trade_return=min_abs_trade_return,
        max_hold_days=max_hold_days,
        sample_weight_mode=sample_weight_mode,
        oracle_cluster_keys=tuple(oracle_cluster_keys),
    )
    return _build_training_frame_from_panel_artifacts(
        base_feature_artifact=base_feature_artifact,
        label_artifact=label_artifact,
        spec=dataset_spec,
        extra_panel_artifacts=extra_panel_artifacts,
    )


def train_model_from_artifact_inputs(
    *,
    name: str,
    algorithm: str,
    task_type: str,
    target_col: str,
    feature_artifact: Artifact,
    label_artifact: Artifact,
    extra_panel_artifacts: Sequence[Artifact] = (),
    framework: str = "sklearn",
    split_ratio: float = 0.8,
    params: dict[str, Any] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    feature_family: str | None = None,
    feature_families: Sequence[str] = (),
    label_k: int | None = None,
    label_ks: Sequence[int] = (),
    min_abs_trade_return: float | None = None,
    max_hold_days: int | None = None,
    sample_weight_mode: str = "uniform",
    oracle_cluster_keys: Sequence[str] = (),
    progress_callback=None,
) -> ModelArtifact:
    merged_params = merge_job_params(dict(params or {}))
    context = dict(merged_params.get(JOB_CONTEXT_KEY) or {})
    context["feature_artifact_id"] = int(feature_artifact.id)
    context["label_artifact_id"] = int(label_artifact.id)
    context["extra_panel_artifact_ids"] = [int(artifact.id) for artifact in extra_panel_artifacts]
    merged_params[JOB_CONTEXT_KEY] = context

    if callable(progress_callback):
        progress_callback(
            phase="prepare_training_dataset",
            phase_label="Prepare training dataset",
            phase_index=1,
            phase_total=3,
            force=True,
        )
    dataset_spec = ArtifactTrainingDatasetSpec(
        start_date=start_date,
        end_date=end_date,
        feature_family=feature_family,
        feature_families=tuple(feature_families),
        label_k=label_k,
        label_ks=tuple(label_ks),
        min_abs_trade_return=min_abs_trade_return,
        max_hold_days=max_hold_days,
        sample_weight_mode=sample_weight_mode,
        oracle_cluster_keys=tuple(oracle_cluster_keys),
    )
    dataset_started = time.perf_counter()
    train_df, feature_cols, source_meta = _build_training_frame_from_panel_artifacts(
        base_feature_artifact=feature_artifact,
        label_artifact=label_artifact,
        spec=dataset_spec,
        extra_panel_artifacts=extra_panel_artifacts,
    )
    dataset_build_seconds = time.perf_counter() - dataset_started
    logger.info(
        "ml.train dataset_ready feature_artifact=%s label_artifact=%s rows=%s features=%s elapsed_s=%.3f",
        feature_artifact.id,
        label_artifact.id,
        len(train_df),
        len(feature_cols),
        dataset_build_seconds,
    )

    model_params = extract_model_params(merged_params)
    task_type_value = str(task_type or "").strip().lower()
    if callable(progress_callback):
        progress_callback(
            phase="fit_model",
            phase_label="Fit model",
            phase_index=2,
            phase_total=3,
            message=f"{len(train_df):,} rows | {len(feature_cols):,} features",
            force=True,
        )
    fit_started = time.perf_counter()
    model_obj = fit_model_for_algorithm(
        algorithm=algorithm,
        train_df=train_df,
        feature_cols=feature_cols,
        model_params=model_params,
        target_col=target_col,
        split_ratio=float(split_ratio),
    )
    fit_seconds = time.perf_counter() - fit_started

    metadata = {
        "symbols": list(source_meta.get("symbols") or []),
        "symbols_count": int(source_meta.get("symbols_count") or 0),
        "source_feature_artifact_id": int(feature_artifact.id),
        "source_label_artifact_id": int(label_artifact.id),
        "source_panel_artifact_ids": [int(artifact.id) for artifact in extra_panel_artifacts],
        "extra_panel_sources": list(source_meta.get("extra_panel_sources") or []),
        "joined_rows": int(source_meta.get("joined_rows") or 0),
        "model_summary": _model_summary(model_obj),
        "train_start_date": str(start_date or ""),
        "train_end_date": str(end_date or ""),
        "dataset_build_seconds": round(float(dataset_build_seconds), 6),
        "fit_seconds": round(float(fit_seconds), 6),
        "feature_family": str(source_meta.get("feature_family") or ""),
        "feature_families": list(source_meta.get("feature_families") or []),
        "feature_family_columns": list(source_meta.get("feature_family_columns") or []),
        "available_feature_families": list(source_meta.get("available_feature_families") or []),
        "label_k": source_meta.get("label_k"),
        "label_ks": list(source_meta.get("label_ks") or []),
        "coverage_start_date": str((source_meta.get("coverage_after") or {}).get("coverage_start_date") or ""),
        "coverage_end_date": str((source_meta.get("coverage_after") or {}).get("coverage_end_date") or ""),
        "coverage_rows": int((source_meta.get("coverage_after") or {}).get("coverage_rows") or 0),
        "label_rows_before_trade_filters": int(source_meta.get("label_rows_before_trade_filters") or 0),
        "label_rows_after_filters": int(source_meta.get("label_rows_after_filters") or 0),
        "min_abs_trade_return": source_meta.get("min_abs_trade_return"),
        "max_hold_days": source_meta.get("max_hold_days"),
        "sample_weight_mode": str(source_meta.get("sample_weight_mode") or "uniform"),
        "oracle_cluster_keys": list(source_meta.get("oracle_cluster_keys") or []),
        "oracle_cluster_scope": str(source_meta.get("oracle_cluster_scope") or "generalist"),
        "cluster_rows_before_filter": int(source_meta.get("cluster_rows_before_filter") or 0),
        "cluster_rows_after_filter": int(source_meta.get("cluster_rows_after_filter") or 0),
    }
    artifact = save_model_artifact(
        name=name,
        model_obj=model_obj,
        framework=framework,
        task_type=task_type_value,
        target_col=target_col,
        feature_cols=feature_cols,
        metrics=_metrics_for(model_obj),
        params=model_params,
        metadata=metadata,
    )

    if callable(progress_callback):
        progress_callback(
            phase="generate_train_predictions",
            phase_label="Generate train diagnostics",
            phase_index=3,
            phase_total=3,
            message="Scoring training rows for saved diagnostics",
            force=True,
        )
    prediction_started = time.perf_counter()
    prediction_df = _score_artifact_rows(
        model_obj=model_obj,
        feature_df=source_meta["feature_df"],
        feature_cols=feature_cols,
        label_df=source_meta["label_df"],
    )
    train_prediction_seconds = time.perf_counter() - prediction_started
    predictions_uri = _write_prediction_rows_csv(name, prediction_df)
    artifact.metadata = dict(artifact.metadata or {})
    artifact.metadata["predictions_uri"] = predictions_uri
    artifact.metadata["prediction_rows"] = int(len(prediction_df))
    artifact.metadata["train_prediction_seconds"] = round(float(train_prediction_seconds), 6)
    artifact.save(update_fields=["metadata", "updated_at"])
    if callable(progress_callback):
        progress_callback(
            phase="generate_train_predictions",
            phase_label="Generate train diagnostics",
            phase_index=3,
            phase_total=3,
            total_units=1,
            completed_units=1,
            message="Completed",
            force=True,
        )
    return artifact


def score_model_from_artifact_inputs(
    *,
    model_record: ModelArtifact,
    feature_artifact: Artifact,
    extra_panel_artifacts: Sequence[Artifact] = (),
    label_artifact: Artifact | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    return_metadata: bool = False,
    progress_callback=None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    score_started = time.perf_counter()
    if callable(progress_callback):
        progress_callback(
            phase="load_scoring_inputs",
            phase_label="Load scoring inputs",
            phase_index=1,
            phase_total=2,
            force=True,
        )
    feature_df, feature_cols, _panel_meta = _build_feature_frame_from_artifacts(
        base_feature_artifact=feature_artifact,
        extra_panel_artifacts=extra_panel_artifacts,
    )
    feature_df = _filter_frame_by_date(feature_df, start_date=start_date, end_date=end_date)
    label_df = _load_artifact_csv_frame(label_artifact) if label_artifact is not None else None
    if label_df is not None:
        label_df = _filter_frame_by_date(label_df, start_date=start_date, end_date=end_date)
    model_obj = model_record.get_artifact()
    if callable(progress_callback):
        progress_callback(
            phase="score_rows",
            phase_label="Score rows",
            phase_index=2,
            phase_total=2,
            message=f"{len(feature_df):,} candidate rows",
            force=True,
        )
    prediction_df = _score_artifact_rows(
        model_obj=model_obj,
        feature_df=feature_df,
        feature_cols=list(model_record.feature_cols or feature_cols),
        label_df=label_df,
    )
    if callable(progress_callback):
        progress_callback(
            phase="score_rows",
            phase_label="Score rows",
            phase_index=2,
            phase_total=2,
            total_units=1,
            completed_units=1,
            message=f"{len(prediction_df):,} rows scored",
            force=True,
        )
    metadata = {
        "score_seconds": round(float(time.perf_counter() - score_started), 6),
        "rows_scored": int(len(prediction_df)),
        "score_start_date": str(start_date or ""),
        "score_end_date": str(end_date or ""),
    }
    if return_metadata:
        return prediction_df, metadata
    return prediction_df
