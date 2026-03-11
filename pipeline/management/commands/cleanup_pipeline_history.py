from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from ml.models import ModelArtifact
from pipeline import service_runtime as runtime
from pipeline.models import Artifact, PipelineRun
from pipeline.run_support import _safe_delete_artifact_files


def _parse_csv(raw: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for token in str(raw or "").split(","):
        value = str(token or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _parse_date(raw: str) -> date | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except Exception as exc:
        raise CommandError(f"Invalid date {text!r}; expected YYYY-MM-DD.") from exc


def _saved_model_id_from_artifact(artifact: Artifact) -> int:
    candidates = [
        (artifact.content or {}).get("model_artifact_id"),
        (artifact.metadata or {}).get("saved_model_artifact_id"),
    ]
    for value in candidates:
        try:
            parsed = int(value or 0)
        except Exception:
            parsed = 0
        if parsed > 0:
            return parsed
    return 0


def _safe_delete_path(path_value: str, *, base_dir: Path) -> bool:
    path_text = str(path_value or "").strip()
    if not path_text:
        return False
    path = Path(path_text)
    if not path.is_absolute():
        return False
    try:
        resolved = path.resolve()
    except Exception:
        return False
    if base_dir not in resolved.parents:
        return False
    if not resolved.exists() or not resolved.is_file():
        return False
    try:
        resolved.unlink()
        return True
    except Exception:
        return False


class Command(BaseCommand):
    help = "Delete old pipeline/job/artifact history and orphaned saved models with dry-run safeguards."

    def add_arguments(self, parser):
        parser.add_argument("--before-date", default="", help="Only delete runs created before YYYY-MM-DD.")
        parser.add_argument("--job-types", default="", help="Optional comma-separated pipeline job types to target.")
        parser.add_argument("--keep-latest-per-job", type=int, default=10, help="Protect the newest N runs per job type.")
        parser.add_argument("--keep-run-ids", default="", help="Optional comma-separated pipeline run ids to preserve.")
        parser.add_argument("--dry-run", action="store_true", help="Report what would be removed without deleting anything.")

    def handle(self, *args, **options):
        before_date = _parse_date(str(options.get("before_date") or ""))
        job_types = _parse_csv(str(options.get("job_types") or ""))
        keep_run_ids = {int(value) for value in _parse_csv(str(options.get("keep_run_ids") or "")) if str(value).isdigit()}
        keep_latest_per_job = max(0, int(options.get("keep_latest_per_job") or 0))
        dry_run = bool(options.get("dry_run"))

        candidate_qs = PipelineRun.objects.exclude(
            status__in=[PipelineRun.Status.PENDING, PipelineRun.Status.RUNNING]
        )
        if before_date is not None:
            candidate_qs = candidate_qs.filter(created_at__date__lt=before_date)
        if job_types:
            candidate_qs = candidate_qs.filter(requested_job__in=job_types)

        protected_run_ids = set(keep_run_ids)
        if keep_latest_per_job > 0:
            protected_scope = PipelineRun.objects.all()
            if job_types:
                protected_scope = protected_scope.filter(requested_job__in=job_types)
            for job_type in protected_scope.values_list("requested_job", flat=True).distinct():
                protected_run_ids.update(
                    protected_scope.filter(requested_job=job_type)
                    .order_by("-created_at", "-id")
                    .values_list("id", flat=True)[:keep_latest_per_job]
                )

        delete_runs = list(candidate_qs.exclude(id__in=protected_run_ids).order_by("created_at", "id"))
        delete_run_ids = [int(run.id) for run in delete_runs]
        artifact_rows = list(Artifact.objects.filter(pipeline_run_id__in=delete_run_ids))
        delete_saved_model_ids = {
            saved_model_id
            for saved_model_id in (_saved_model_id_from_artifact(artifact) for artifact in artifact_rows)
            if saved_model_id > 0
        }

        remaining_saved_model_ids = {
            saved_model_id
            for saved_model_id in (
                _saved_model_id_from_artifact(artifact)
                for artifact in Artifact.objects.exclude(pipeline_run_id__in=delete_run_ids).only("content", "metadata")
            )
            if saved_model_id > 0
        }
        orphan_model_qs = ModelArtifact.objects.exclude(id__in=remaining_saved_model_ids)
        if before_date is not None:
            orphan_model_qs = orphan_model_qs.filter(created_at__date__lt=before_date)
        delete_model_ids = sorted(
            (delete_saved_model_ids - remaining_saved_model_ids)
            | set(orphan_model_qs.values_list("id", flat=True))
        )

        base_dir = Path(runtime.ARTIFACT_DIR).resolve()
        delete_job_counts = Counter(run.requested_job for run in delete_runs)
        artifact_type_counts = Counter(artifact.artifact_type for artifact in artifact_rows)
        prediction_paths = [
            str((model.metadata or {}).get("predictions_uri") or "").strip()
            for model in ModelArtifact.objects.filter(id__in=delete_model_ids).only("metadata")
            if str((model.metadata or {}).get("predictions_uri") or "").strip()
        ]
        payload = {
            "dry_run": dry_run,
            "filters": {
                "before_date": before_date.isoformat() if before_date is not None else "",
                "job_types": list(job_types),
                "keep_latest_per_job": keep_latest_per_job,
                "keep_run_ids": sorted(protected_run_ids),
            },
            "delete_pipeline_run_count": len(delete_run_ids),
            "delete_pipeline_run_ids_preview": delete_run_ids[:50],
            "delete_pipeline_run_job_types": dict(delete_job_counts),
            "delete_artifact_count": len(artifact_rows),
            "delete_artifact_types": dict(artifact_type_counts),
            "delete_model_artifact_count": len(delete_model_ids),
            "delete_model_artifact_ids_preview": delete_model_ids[:50],
            "delete_prediction_file_count": len(prediction_paths),
            "artifact_base_dir": str(base_dir),
        }

        if not dry_run:
            deleted_artifact_files = 0
            for run in delete_runs:
                deleted_artifact_files += _safe_delete_artifact_files(run)
            deleted_prediction_files = 0
            for prediction_path in prediction_paths:
                if _safe_delete_path(prediction_path, base_dir=base_dir):
                    deleted_prediction_files += 1
            if delete_run_ids:
                PipelineRun.objects.filter(id__in=delete_run_ids).delete()
            if delete_model_ids:
                ModelArtifact.objects.filter(id__in=delete_model_ids).delete()
            payload["deleted_artifact_file_count"] = int(deleted_artifact_files)
            payload["deleted_prediction_file_count"] = int(deleted_prediction_files)

        self.stdout.write(json.dumps(payload, indent=2, sort_keys=True))
