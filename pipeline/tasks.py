from __future__ import annotations

from typing import Any

from django.utils import timezone

from .models import PipelineRun
from .services import execute_pipeline_run

try:
    from celery import shared_task
except Exception:  # pragma: no cover
    def shared_task(*_args, **_kwargs):
        def _decorator(func):
            return func

        return _decorator

@shared_task(bind=True, ignore_result=True)
def run_pipeline_job_task(
    self,
    pipeline_run_id: int,
    target_job: str,
    mode: str,
    config: dict[str, Any] | None = None,
    input_artifact_ids: list[int] | None = None,
) -> dict[str, Any]:
    pipeline_run = PipelineRun.objects.filter(pk=int(pipeline_run_id)).first()
    if pipeline_run is None:
        return {"status": "missing", "pipeline_run_id": int(pipeline_run_id)}

    try:
        execute_pipeline_run(
            pipeline_run=pipeline_run,
            target_job=target_job,
            mode=mode,
            config=dict(config or {}),
            input_artifact_ids=list(input_artifact_ids or []),
        )
        pipeline_run.refresh_from_db()
    except Exception as exc:
        pipeline_run.refresh_from_db()
        if str(pipeline_run.status) in {PipelineRun.Status.PENDING, PipelineRun.Status.RUNNING}:
            pipeline_run.status = PipelineRun.Status.FAILED
            pipeline_run.error = str(exc)
            pipeline_run.finished_at = timezone.now()
            pipeline_run.save(update_fields=["status", "error", "finished_at", "updated_at"])
        return {
            "pipeline_run_id": int(pipeline_run.id),
            "status": str(pipeline_run.status),
            "requested_job": str(pipeline_run.requested_job),
            "error": str(pipeline_run.error or str(exc)),
        }

    pipeline_run.refresh_from_db()
    return {
        "pipeline_run_id": int(pipeline_run.id),
        "status": str(pipeline_run.status),
        "requested_job": str(pipeline_run.requested_job),
    }
