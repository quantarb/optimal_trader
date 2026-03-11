from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand

from pipeline.models import Artifact, PipelineRun
from pipeline.research_suite import RESEARCH_REPORT_SCHEMA_VERSION
from pipeline.services import ARTIFACT_DIR


def _is_summary_like(path: Path) -> bool:
    name = path.name
    return "cohort" in name or "research" in name or name.startswith("mag7_")


def _is_current_report_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    return (
        int(payload.get("schema_version") or 0) == RESEARCH_REPORT_SCHEMA_VERSION
        and
        isinstance(payload.get("leaderboard_rows"), list)
        and isinstance(payload.get("rejected_rows"), list)
        and isinstance(payload.get("report_summary"), dict)
        and isinstance(payload.get("research_profile"), dict)
    )


class Command(BaseCommand):
    help = "Delete stale summary/report JSONs and DB artifact rows that no longer point at real files."

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Report what would be deleted without changing anything.")

    def handle(self, *args, **options):
        dry_run = bool(options["dry_run"])
        base = Path(ARTIFACT_DIR)
        removed_files: list[str] = []

        if base.exists():
            for path in sorted(base.glob("*.json")):
                if not _is_summary_like(path):
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    payload = None
                if _is_current_report_payload(payload):
                    continue
                removed_files.append(path.name)
                if not dry_run:
                    path.unlink(missing_ok=True)
                    path.with_suffix(".csv").unlink(missing_ok=True)

        stale_artifact_ids: list[int] = []
        for artifact in Artifact.objects.all().only("id", "uri"):
            uri = str(artifact.uri or "").strip()
            if uri and not Path(uri).exists():
                stale_artifact_ids.append(int(artifact.id))

        stale_run_ids: list[int] = []
        if stale_artifact_ids:
            for run in PipelineRun.objects.filter(artifacts__id__in=stale_artifact_ids).distinct():
                has_live_artifacts = run.artifacts.exclude(id__in=stale_artifact_ids).exists()
                if not has_live_artifacts and not run.job_runs.exists():
                    stale_run_ids.append(int(run.id))

        if not dry_run and stale_artifact_ids:
            Artifact.objects.filter(id__in=stale_artifact_ids).delete()
        if not dry_run and stale_run_ids:
            PipelineRun.objects.filter(id__in=stale_run_ids).delete()

        self.stdout.write(
            json.dumps(
                {
                    "dry_run": dry_run,
                    "removed_files": removed_files,
                    "removed_file_count": len(removed_files),
                    "stale_artifact_ids": stale_artifact_ids,
                    "stale_artifact_count": len(stale_artifact_ids),
                    "stale_run_ids": stale_run_ids,
                    "stale_run_count": len(stale_run_ids),
                },
                indent=2,
                sort_keys=True,
            )
        )
