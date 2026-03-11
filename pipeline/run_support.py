from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from django.utils import timezone

from .progress import progress_from_config
from . import service_runtime as runtime
from .models import PipelineRun


def _serialize_job_run(job) -> dict[str, Any]:
    progress = progress_from_config(job.config or {})
    return {
        "job_run_id": int(job.id),
        "job_type": str(job.job_type),
        "status": str(job.status),
        "progress": progress,
        "progress_total_symbols": int((progress or {}).get("total_units") or (job.config or {}).get("progress_total_symbols") or 0),
        "progress_completed_symbols": int((progress or {}).get("completed_units") or (job.config or {}).get("progress_completed_symbols") or 0),
        "progress_current_symbol": str((progress or {}).get("current_item") or (job.config or {}).get("progress_current_symbol") or ""),
        "input_artifact_ids": [int(v) for v in job.input_artifacts.values_list("id", flat=True)],
        "output_artifact_ids": [int(v) for v in job.produced_artifacts.values_list("id", flat=True)],
        "error": str(job.error or ""),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


def _serialize_artifact(artifact) -> dict[str, Any]:
    return {
        "artifact_id": int(artifact.id),
        "artifact_type": str(artifact.artifact_type),
        "producer_job_id": int(artifact.producer_job_id) if artifact.producer_job_id else None,
        "uri": str(artifact.uri or ""),
        "content": dict(artifact.content or {}),
        "metadata": dict(artifact.metadata or {}),
        "payload_hash": str(artifact.payload_hash or ""),
        "created_at": artifact.created_at.isoformat() if artifact.created_at else None,
    }


def serialize_pipeline_run(pipeline_run: PipelineRun) -> dict[str, Any]:
    job_runs = list(pipeline_run.job_runs.order_by("created_at", "id"))
    artifacts = list(pipeline_run.artifacts.order_by("created_at", "id"))
    pipeline_progress = progress_from_config(pipeline_run.config or {})
    if not pipeline_progress:
        for job in reversed(job_runs):
            pipeline_progress = progress_from_config(job.config or {})
            if pipeline_progress:
                break

    return {
        "pipeline_run_id": int(pipeline_run.id),
        "name": str(pipeline_run.name or ""),
        "requested_job": str(pipeline_run.requested_job),
        "mode": str(pipeline_run.mode),
        "status": str(pipeline_run.status),
        "error": str(pipeline_run.error or ""),
        "started_at": pipeline_run.started_at.isoformat() if pipeline_run.started_at else None,
        "finished_at": pipeline_run.finished_at.isoformat() if pipeline_run.finished_at else None,
        "config": dict(pipeline_run.config or {}),
        "progress": pipeline_progress,
        "job_runs": [_serialize_job_run(job) for job in job_runs],
        "artifacts": [_serialize_artifact(artifact) for artifact in artifacts],
    }


def _start_pipeline_run_thread(
    pipeline_run: PipelineRun,
    *,
    target_job: str,
    mode: str,
    config: dict[str, Any],
    input_artifact_ids: list[int],
) -> None:
    from .tasks import run_pipeline_job_task

    thread = threading.Thread(
        target=lambda: run_pipeline_job_task.run(
            int(pipeline_run.id),
            str(target_job or "").strip().lower(),
            str(mode or PipelineRun.Mode.STRICT).strip().lower(),
            dict(config or {}),
            list(input_artifact_ids or []),
        ),
        daemon=True,
    )
    thread.start()


def _launch_pipeline_run(*, name: str, target_job: str, mode: str, config: dict[str, Any], input_artifact_ids: list[int]) -> PipelineRun:
    pipeline_run = PipelineRun.objects.create(
        name=str(name or "").strip(),
        requested_job=str(target_job or "").strip().lower(),
        mode=str(mode or PipelineRun.Mode.STRICT).strip().lower(),
        status=PipelineRun.Status.PENDING,
        config=dict(config or {}),
    )
    _start_pipeline_run_thread(
        pipeline_run,
        target_job=target_job,
        mode=mode,
        config=config,
        input_artifact_ids=input_artifact_ids,
    )
    return pipeline_run


def _maybe_unstick_pipeline_run(run: PipelineRun, stale_seconds: int = 20) -> None:
    status = str(run.status)
    if status not in {PipelineRun.Status.PENDING, PipelineRun.Status.RUNNING}:
        return
    started_at = run.started_at or run.created_at
    if started_at is None:
        return
    age = (timezone.now() - started_at).total_seconds()
    if age < stale_seconds:
        return
    if run.job_runs.exists():
        return
    run.status = PipelineRun.Status.FAILED
    if not str(run.error or "").strip():
        run.error = "Run watchdog: no job steps were created."
    run.finished_at = timezone.now()
    run.save(update_fields=["status", "error", "finished_at", "updated_at"])


def _safe_delete_artifact_files(run: PipelineRun) -> int:
    base_dir = Path(runtime.ARTIFACT_DIR).resolve()
    deleted = 0
    for artifact in run.artifacts.all().only("uri"):
        uri = str(artifact.uri or "").strip()
        if not uri:
            continue
        path = Path(uri)
        if not path.is_absolute():
            continue
        try:
            resolved = path.resolve()
        except Exception:
            continue
        if base_dir not in resolved.parents:
            continue
        if not resolved.exists() or not resolved.is_file():
            continue
        try:
            resolved.unlink()
            deleted += 1
        except Exception:
            continue
    return deleted


__all__ = [
    "_launch_pipeline_run",
    "_maybe_unstick_pipeline_run",
    "_safe_delete_artifact_files",
    "_start_pipeline_run_thread",
    "serialize_pipeline_run",
]
