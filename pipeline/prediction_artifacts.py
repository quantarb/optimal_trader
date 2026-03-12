from __future__ import annotations

import uuid
from typing import Any

import pandas as pd
from django.utils import timezone

from .contracts import (
    PREDICTION_REQUIRED_COLUMNS,
    build_schema_metadata,
    normalize_prediction_output_frame,
    validate_frame_columns,
)
from .models import Artifact, PipelineRun
from .service_runtime import artifact_payload_hash, json_safe_value, write_frame_artifact


def save_prediction_frame_artifact(
    prediction_df: pd.DataFrame,
    *,
    artifact_type: str = "REGRESSOR_PREDICTIONS",
    requested_job: str = "custom_prediction_scores",
    run_name: str = "Custom Prediction Scores",
    config: dict[str, Any] | None = None,
    content: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    storage_format: str = "csv",
) -> Artifact:
    if prediction_df.empty:
        raise ValueError("prediction_df must contain at least one row.")
    normalized = normalize_prediction_output_frame(pd.DataFrame(prediction_df).copy(), artifact_type=str(artifact_type))
    validate_frame_columns(normalized, PREDICTION_REQUIRED_COLUMNS, artifact_type=str(artifact_type))

    key = f"predictions_{uuid.uuid4().hex}"
    stored = write_frame_artifact(
        key,
        frame=normalized,
        fieldnames=list(normalized.columns),
        storage_format=str(storage_format or "csv"),
    )
    now = timezone.now()
    pipeline_run = PipelineRun.objects.create(
        name=str(run_name or requested_job),
        requested_job=str(requested_job or "custom_prediction_scores"),
        mode=PipelineRun.Mode.STRICT,
        status=PipelineRun.Status.SUCCEEDED,
        config=json_safe_value(dict(config or {})),
        started_at=now,
        finished_at=now,
    )
    artifact_content = {
        "rows": int(len(normalized)),
        **dict(content or {}),
    }
    artifact_metadata = {
        "schema": build_schema_metadata(
            artifact_type=str(artifact_type),
            required_columns=PREDICTION_REQUIRED_COLUMNS,
            actual_columns=list(normalized.columns),
        ),
        **stored.storage_metadata(),
        **dict(metadata or {}),
    }
    return Artifact.objects.create(
        pipeline_run=pipeline_run,
        artifact_type=str(artifact_type),
        key=key,
        uri=str(stored.uri),
        content=json_safe_value(artifact_content),
        metadata=json_safe_value(artifact_metadata),
        payload_hash=artifact_payload_hash(artifact_content, str(stored.uri)),
    )


__all__ = [
    "save_prediction_frame_artifact",
]
