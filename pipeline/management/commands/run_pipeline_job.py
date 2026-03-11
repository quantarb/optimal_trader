from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError

from pipeline.models import PipelineRun
from pipeline.run_support import serialize_pipeline_run
from pipeline.services import JOB_TYPES, execute_pipeline_run


class Command(BaseCommand):
    help = "Run a dependency-aware pipeline job and persist artifacts for downstream jobs."

    def add_arguments(self, parser):
        parser.add_argument("--job", required=True, choices=JOB_TYPES)
        parser.add_argument(
            "--mode",
            default=PipelineRun.Mode.STRICT,
            choices=[PipelineRun.Mode.STRICT, PipelineRun.Mode.AUTO_BUILD_MISSING],
        )
        parser.add_argument("--name", default="")
        parser.add_argument("--config", default="{}")
        parser.add_argument(
            "--input-artifact-ids",
            default="",
            help="Comma-separated artifact ids to use as explicit inputs.",
        )

    def handle(self, *args, **options):
        job = str(options["job"]).strip().lower()
        mode = str(options["mode"]).strip().lower()
        name = str(options["name"] or "").strip()

        try:
            config = json.loads(str(options["config"] or "{}"))
        except Exception as exc:
            raise CommandError(f"Invalid --config JSON: {exc}") from exc
        if not isinstance(config, dict):
            raise CommandError("--config must decode to a JSON object.")

        raw_ids = str(options["input_artifact_ids"] or "").strip()
        input_ids: list[int] = []
        if raw_ids:
            for token in raw_ids.split(","):
                t = token.strip()
                if not t:
                    continue
                try:
                    input_ids.append(int(t))
                except Exception as exc:
                    raise CommandError(f"Invalid artifact id: {t}") from exc

        run = PipelineRun.objects.create(
            name=name,
            requested_job=job,
            mode=mode,
            status=PipelineRun.Status.PENDING,
            config=config,
        )

        execute_pipeline_run(
            pipeline_run=run,
            target_job=job,
            mode=mode,
            config=config,
            input_artifact_ids=input_ids,
        )

        run.refresh_from_db()
        self.stdout.write(json.dumps(serialize_pipeline_run(run), indent=2, sort_keys=True))
