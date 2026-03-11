from __future__ import annotations

import json
import threading
from datetime import timedelta
from typing import Any

from django.db import transaction
from django.http import Http404, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from .artifact_support import _build_equity_curve_context
from .models import Artifact, JobRun, PipelineRun
from .progress import progress_from_config
from .research_suite import run_optimal_trade_research_suite
from .run_support import _maybe_unstick_pipeline_run, _safe_delete_artifact_files, _start_pipeline_run_thread, serialize_pipeline_run
from .services import ARTIFACT_DIR, JOB_OUTPUT_ARTIFACT, JOB_TYPES, UI_JOB_TYPES
from .test_support import MAG7_SYMBOLS
from .view_support import (
    _clean_ids,
)
from .views_artifacts import artifact_detail, artifact_preview, artifact_symbol_breakdown, backtest_detail, strategy_detail
from .views_insights import (
    pipeline_market_state_api,
    pipeline_market_situations,
    pipeline_opportunities,
    pipeline_portfolio_analysis,
    pipeline_stock_intelligence,
    pipeline_top_opportunities,
    symbol_research_view,
)
from .views_reports import (
    pipeline_cohorts_view,
    pipeline_diagnostic_reports_view,
    pipeline_feature_attribution_reports_view,
    pipeline_oracle_reports_view,
    pipeline_research_reports_view,
    pipeline_rl_policy_reports_view,
)
from .views_workbench import (
    pipeline_lab_view,
    pipeline_strategies_view,
    pipeline_ui_view,
    strategy_definition_edit_view,
    strategy_definition_list_view,
)


@require_POST
def pipeline_run_start(request):
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    target_job = str(payload.get("target_job") or "").strip().lower()
    if target_job not in JOB_TYPES:
        return JsonResponse({"error": f"Invalid target_job. Supported: {', '.join(JOB_TYPES)}"}, status=400)

    mode = str(payload.get("mode") or PipelineRun.Mode.STRICT).strip().lower()
    if mode not in {PipelineRun.Mode.STRICT, PipelineRun.Mode.AUTO_BUILD_MISSING}:
        return JsonResponse({"error": "Invalid mode. Supported: strict, auto_build_missing"}, status=400)

    config_payload = dict(payload.get("config") or {})
    force_overwrite = str(payload.get("force_overwrite") or "").strip().lower() in {"1", "true", "yes"}
    overwrite_run_id_raw = payload.get("overwrite_pipeline_run_id")
    if overwrite_run_id_raw in (None, "", 0):
        overwrite_run_id_raw = config_payload.get("source_label_run_id")
    try:
        overwrite_run_id = int(overwrite_run_id_raw or 0)
    except Exception:
        overwrite_run_id = 0

    pipeline_run: PipelineRun
    if overwrite_run_id > 0:
        existing = PipelineRun.objects.filter(pk=overwrite_run_id).first()
        if existing is None:
            return JsonResponse({"error": f"Pipeline run #{overwrite_run_id} not found."}, status=404)
        if str(existing.status) in {PipelineRun.Status.PENDING, PipelineRun.Status.RUNNING} and not force_overwrite:
            return JsonResponse(
                {
                    "error": (
                        f"Pipeline run #{overwrite_run_id} is pending/running and cannot be overwritten. "
                        "Retry with force_overwrite=1."
                    )
                },
                status=409,
            )
        with transaction.atomic():
            existing.name = str(payload.get("name") or existing.name or "").strip()
            existing.requested_job = target_job
            existing.mode = mode
            existing.status = PipelineRun.Status.PENDING
            existing.config = config_payload
            existing.error = ""
            existing.started_at = None
            existing.finished_at = None
            existing.save(
                update_fields=[
                    "name",
                    "requested_job",
                    "mode",
                    "status",
                    "config",
                    "error",
                    "started_at",
                    "finished_at",
                    "updated_at",
                ]
            )
            pipeline_run = existing
    else:
        pipeline_run = PipelineRun.objects.create(
            name=str(payload.get("name") or "").strip(),
            requested_job=target_job,
            mode=mode,
            status=PipelineRun.Status.PENDING,
            config=config_payload,
        )

    input_artifact_ids = _clean_ids(payload.get("input_artifact_ids") or [])

    try:
        _start_pipeline_run_thread(
            pipeline_run,
            target_job=target_job,
            mode=mode,
            config=config_payload,
            input_artifact_ids=input_artifact_ids,
        )
    except Exception as exc:
        pipeline_run.status = PipelineRun.Status.FAILED
        pipeline_run.error = str(exc)
        pipeline_run.save(update_fields=["status", "error", "updated_at"])
        return JsonResponse({"error": f"Could not queue pipeline run: {exc}"}, status=503)

    pipeline_run.refresh_from_db()
    return JsonResponse(serialize_pipeline_run(pipeline_run))


@require_GET
def pipeline_run_status(request, pipeline_run_id: int):
    pipeline_run = PipelineRun.objects.filter(pk=int(pipeline_run_id)).first()
    if pipeline_run is None:
        raise Http404("Pipeline run not found.")
    _maybe_unstick_pipeline_run(pipeline_run)
    pipeline_run.refresh_from_db()
    return JsonResponse(serialize_pipeline_run(pipeline_run))


@require_GET
def pipeline_run_latest(request):
    pipeline_run = PipelineRun.objects.order_by("-created_at", "-id").first()
    if pipeline_run is None:
        return JsonResponse({"pipeline_run": None})
    return JsonResponse(serialize_pipeline_run(pipeline_run))


@require_GET
def pipeline_run_list(request):
    raw_job = str(request.GET.get("job_type") or "").strip().lower()
    try:
        limit = int(request.GET.get("limit") or 300)
    except Exception:
        limit = 300
    limit = max(1, min(1000, limit))
    summary_only = str(request.GET.get("summary") or "0").strip().lower() in {"1", "true", "yes"}
    qs = PipelineRun.objects.all().order_by("-created_at", "-id")
    if raw_job:
        qs = qs.filter(requested_job=raw_job)
    rows = []
    for run in qs[:limit]:
        _maybe_unstick_pipeline_run(run)
        run.refresh_from_db()
        duration_seconds: int | None = None
        if run.started_at and run.finished_at:
            try:
                delta: timedelta = run.finished_at - run.started_at
                duration_seconds = max(0, int(delta.total_seconds()))
            except Exception:
                duration_seconds = None
        if summary_only:
            job_runs = []
            artifacts = []
        else:
            job_runs = list(run.job_runs.values("id", "job_type", "status").order_by("created_at", "id"))
            artifacts = list(run.artifacts.values("id", "artifact_type", "producer_job_id", "created_at").order_by("created_at", "id"))
        rows.append(
            {
                "pipeline_run_id": int(run.id),
                "name": str(run.name or ""),
                "requested_job": str(run.requested_job),
                "mode": str(run.mode),
                "status": str(run.status),
                "config": dict(run.config or {}),
                "progress": progress_from_config(run.config or {}),
                "error": str(run.error or ""),
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                "duration_seconds": duration_seconds,
                "job_runs": [
                    {"job_run_id": int(job["id"]), "job_type": str(job["job_type"]), "status": str(job["status"])}
                    for job in job_runs
                ],
                "artifacts": [
                    {
                        "artifact_id": int(artifact["id"]),
                        "artifact_type": str(artifact["artifact_type"]),
                        "producer_job_id": int(artifact["producer_job_id"]) if artifact["producer_job_id"] else None,
                        "created_at": artifact["created_at"].isoformat() if artifact["created_at"] else None,
                    }
                    for artifact in artifacts
                ],
            }
        )
    return JsonResponse({"runs": rows})


@require_GET
def latest_artifact_by_type(request):
    raw_type = str(request.GET.get("type") or "").strip().upper()
    if not raw_type:
        return JsonResponse({"error": "Missing required query parameter: type"}, status=400)
    artifact = Artifact.objects.filter(artifact_type=raw_type).order_by("-created_at", "-id").first()
    if artifact is None:
        return JsonResponse({"artifact": None})
    return JsonResponse(
        {
            "artifact": {
                "artifact_id": int(artifact.id),
                "artifact_type": str(artifact.artifact_type),
                "pipeline_run_id": int(artifact.pipeline_run_id),
                "producer_job_id": int(artifact.producer_job_id) if artifact.producer_job_id else None,
                "uri": str(artifact.uri or ""),
                "content": dict(artifact.content or {}),
                "metadata": dict(artifact.metadata or {}),
                "created_at": artifact.created_at.isoformat() if artifact.created_at else None,
            }
        }
    )


@require_GET
def artifact_list_by_type(request):
    raw_type = str(request.GET.get("type") or "").strip().upper()
    if not raw_type:
        return JsonResponse({"error": "Missing required query parameter: type"}, status=400)
    try:
        limit = int(request.GET.get("limit") or 50)
    except Exception:
        limit = 50
    limit = max(1, min(500, limit))
    rows = Artifact.objects.filter(artifact_type=raw_type).select_related("pipeline_run").order_by("-created_at", "-id")[:limit]
    return JsonResponse(
        {
            "artifacts": [
                {
                    "artifact_id": int(row.id),
                    "artifact_type": str(row.artifact_type),
                    "pipeline_run_id": int(row.pipeline_run_id),
                    "pipeline_run_name": str((row.pipeline_run.name if row.pipeline_run else "") or ""),
                    "uri": str(row.uri or ""),
                    "content": dict(row.content or {}),
                    "metadata": dict(row.metadata or {}),
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]
        }
    )


@require_GET
def pipeline_job_catalog(request):
    return JsonResponse(
        {
            "jobs": [{"job_type": job, "output_artifact_type": JOB_OUTPUT_ARTIFACT.get(job, "")} for job in UI_JOB_TYPES],
            "modes": [PipelineRun.Mode.STRICT, PipelineRun.Mode.AUTO_BUILD_MISSING],
        }
    )


def pipeline_ui(request):
    return pipeline_ui_view(request)


def pipeline_lab(request):
    return pipeline_lab_view(
        request,
        run_optimal_trade_research_suite_fn=run_optimal_trade_research_suite,
        threading_module=threading,
        mag7_symbols=MAG7_SYMBOLS,
    )


def pipeline_strategies(request):
    return pipeline_strategies_view(request)


def pipeline_cohorts(request):
    return pipeline_cohorts_view(request, artifact_dir=ARTIFACT_DIR)


def pipeline_research_reports(request):
    return pipeline_research_reports_view(request, artifact_dir=ARTIFACT_DIR)


def pipeline_diagnostic_reports(request):
    return pipeline_diagnostic_reports_view(request, artifact_dir=ARTIFACT_DIR)


def pipeline_oracle_reports(request):
    return pipeline_oracle_reports_view(request, artifact_dir=ARTIFACT_DIR)


def pipeline_feature_attribution_reports(request):
    return pipeline_feature_attribution_reports_view(request, artifact_dir=ARTIFACT_DIR)


def pipeline_rl_policy_reports(request):
    return pipeline_rl_policy_reports_view(request)


def strategy_definition_list(request):
    return strategy_definition_list_view(request)


def strategy_definition_edit(request, definition_id: int):
    return strategy_definition_edit_view(request, definition_id)


@require_GET
def pipeline_status_board(request):
    rows = []
    for job_type in UI_JOB_TYPES:
        latest = JobRun.objects.filter(job_type=job_type).select_related("pipeline_run").order_by("-created_at", "-id").first()
        if latest is None:
            rows.append(
                {
                    "job_type": job_type,
                    "status": "never_run",
                    "pipeline_run_id": None,
                    "job_run_id": None,
                    "finished_at": None,
                    "error": "",
                }
            )
            continue
        rows.append(
            {
                "job_type": job_type,
                "status": str(latest.status),
                "pipeline_run_id": int(latest.pipeline_run_id),
                "pipeline_run_name": str((latest.pipeline_run.name if latest.pipeline_run else "") or ""),
                "job_run_id": int(latest.id),
                "finished_at": latest.finished_at.isoformat() if latest.finished_at else None,
                "error": str(latest.error or ""),
            }
        )
    return JsonResponse({"jobs": rows})


@require_POST
def pipeline_run_delete(request, pipeline_run_id: int):
    run = PipelineRun.objects.filter(pk=int(pipeline_run_id)).first()
    if run is None:
        raise Http404("Pipeline run not found.")
    force = str(request.GET.get("force") or "").strip().lower() in {"1", "true", "yes"}
    if str(run.status) in {PipelineRun.Status.PENDING, PipelineRun.Status.RUNNING} and not force:
        return JsonResponse({"error": "Cannot delete a pending/running pipeline run without force=1."}, status=409)
    deleted_files = _safe_delete_artifact_files(run)
    run.delete()
    return JsonResponse({"ok": True, "deleted_pipeline_run_id": int(pipeline_run_id), "deleted_files": int(deleted_files)})


@require_POST
def pipeline_run_rename(request, pipeline_run_id: int):
    run = PipelineRun.objects.filter(pk=int(pipeline_run_id)).first()
    if run is None:
        raise Http404("Pipeline run not found.")
    if str(run.status) in {PipelineRun.Status.PENDING, PipelineRun.Status.RUNNING}:
        return JsonResponse({"error": "Cannot rename a pending/running pipeline run."}, status=409)
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}
    new_name = str(payload.get("name") or "").strip()
    if not new_name:
        return JsonResponse({"error": "Name is required."}, status=400)
    if len(new_name) > 255:
        return JsonResponse({"error": "Name must be 255 characters or fewer."}, status=400)
    run.name = new_name
    run.save(update_fields=["name", "updated_at"])
    return JsonResponse({"ok": True, "pipeline_run_id": int(run.id), "name": str(run.name)})


__all__ = [
    "ARTIFACT_DIR",
    "_build_equity_curve_context",
    "artifact_detail",
    "artifact_list_by_type",
    "artifact_preview",
    "artifact_symbol_breakdown",
    "backtest_detail",
    "latest_artifact_by_type",
    "pipeline_cohorts",
    "pipeline_diagnostic_reports",
    "pipeline_feature_attribution_reports",
    "pipeline_job_catalog",
    "pipeline_lab",
    "pipeline_market_state_api",
    "pipeline_market_situations",
    "pipeline_opportunities",
    "pipeline_portfolio_analysis",
    "pipeline_research_reports",
    "pipeline_rl_policy_reports",
    "pipeline_run_delete",
    "pipeline_run_latest",
    "pipeline_run_list",
    "pipeline_run_rename",
    "pipeline_run_start",
    "pipeline_run_status",
    "pipeline_status_board",
    "pipeline_stock_intelligence",
    "pipeline_strategies",
    "pipeline_top_opportunities",
    "pipeline_ui",
    "run_optimal_trade_research_suite",
    "serialize_pipeline_run",
    "strategy_definition_edit",
    "strategy_definition_list",
    "strategy_detail",
    "symbol_research_view",
    "threading",
]
