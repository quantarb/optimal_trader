from __future__ import annotations

from typing import Any

from django.utils import timezone

from .models import Symbol, UniverseDownloadJob
from data import FMPClient

MAX_JOB_ERRORS = 50


try:
    from celery import shared_task
except Exception:  # pragma: no cover - dependency optional at runtime
    def shared_task(*_args, **_kwargs):
        def _decorator(func):
            return func
        return _decorator


def _append_job_error(job: UniverseDownloadJob, message: str) -> list[str]:
    errors = list(job.errors or [])
    if len(errors) < MAX_JOB_ERRORS:
        errors.append(str(message))
    return errors


@shared_task(bind=True, ignore_result=True)
def run_universe_download_job_task(self, job_id: str, api_key: str) -> dict[str, Any]:
    from .views import _refresh_all_symbol_sections

    job = UniverseDownloadJob.objects.filter(pk=str(job_id).strip()).first()
    if job is None:
        return {"status": "missing", "job_id": str(job_id)}

    job.status = UniverseDownloadJob.STATUS_RUNNING
    if job.started_at is None:
        job.started_at = timezone.now()
    if getattr(self, "request", None) is not None and getattr(self.request, "id", None):
        job.celery_task_id = str(self.request.id)
    job.save(update_fields=["status", "started_at", "celery_task_id", "updated_at"])

    symbols = [str(s).strip().upper() for s in list(job.symbols or []) if str(s).strip()]
    total = max(1, int(job.total or len(symbols) or 0))

    completed = int(job.completed or 0)
    success_count = int(job.success_count or 0)
    failed_count = int(job.failed_count or 0)
    errors = list(job.errors or [])
    metrics = dict(job.metrics or {})
    metrics.setdefault("retry", {"strategy": "exponential_backoff", "scope": "per_section_request"})
    metrics.setdefault("sections_total", 0)
    metrics.setdefault("sections_fetched", 0)
    metrics.setdefault("sections_skipped", 0)
    metrics.setdefault("partial_sections", 0)
    metrics.setdefault("retry_attempts", 0)
    metrics.setdefault("symbol_duration_s_total", 0.0)
    metrics.setdefault("symbols_processed", 0)

    try:
        client = FMPClient(api_key=api_key, timeout_s=30.0, max_retries=2)
        for sym in symbols:
            job.current_symbol = sym
            job.save(update_fields=["current_symbol", "updated_at"])

            try:
                symbol_obj = Symbol.objects.filter(symbol__iexact=sym).first()
                if symbol_obj is None:
                    symbol_obj = Symbol.objects.create(symbol=sym)

                section_errors, section_stats = _refresh_all_symbol_sections(symbol_obj, client)
                metrics["sections_total"] = int(metrics.get("sections_total", 0)) + int(section_stats.get("sections_total", 0))
                metrics["sections_fetched"] = int(metrics.get("sections_fetched", 0)) + int(section_stats.get("sections_fetched", 0))
                metrics["sections_skipped"] = int(metrics.get("sections_skipped", 0)) + int(section_stats.get("sections_skipped", 0))
                metrics["partial_sections"] = int(metrics.get("partial_sections", 0)) + int(section_stats.get("partial_sections", 0))
                metrics["retry_attempts"] = int(metrics.get("retry_attempts", 0)) + int(section_stats.get("retry_attempts", 0))
                metrics["symbol_duration_s_total"] = float(metrics.get("symbol_duration_s_total", 0.0)) + float(section_stats.get("duration_s", 0.0))
                metrics["symbols_processed"] = int(metrics.get("symbols_processed", 0)) + 1
                if section_errors:
                    failed_count += 1
                    errors = _append_job_error(
                        job,
                        f"{sym}: " + "; ".join(f"{k}={v}" for k, v in section_errors.items()),
                    )
                else:
                    success_count += 1
            except Exception as exc:
                failed_count += 1
                errors = _append_job_error(job, f"{sym}: {exc}")

            completed += 1
            job.completed = completed
            job.success_count = success_count
            job.failed_count = failed_count
            job.errors = errors
            if int(metrics.get("symbols_processed", 0)) > 0:
                metrics["avg_symbol_duration_s"] = round(
                    float(metrics.get("symbol_duration_s_total", 0.0)) / float(metrics["symbols_processed"]),
                    3,
                )
            job.metrics = metrics
            job.save(update_fields=["completed", "success_count", "failed_count", "errors", "metrics", "updated_at"])

        job.status = (
            UniverseDownloadJob.STATUS_COMPLETED
            if failed_count == 0
            else UniverseDownloadJob.STATUS_COMPLETED_WITH_ERRORS
        )
        job.current_symbol = ""
        job.finished_at = timezone.now()
        job.metrics = metrics
        job.save(update_fields=["status", "current_symbol", "finished_at", "metrics", "updated_at"])
    except Exception as exc:
        errors = _append_job_error(job, f"Worker failed: {exc}")
        job.status = UniverseDownloadJob.STATUS_FAILED
        job.current_symbol = ""
        job.errors = errors
        job.metrics = metrics
        job.finished_at = timezone.now()
        job.save(update_fields=["status", "current_symbol", "errors", "metrics", "finished_at", "updated_at"])

    return {
        "job_id": str(job.pk),
        "status": job.status,
        "completed": int(job.completed or 0),
        "total": total,
        "success_count": int(job.success_count or 0),
        "failed_count": int(job.failed_count or 0),
    }
