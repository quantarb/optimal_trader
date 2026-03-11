from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from pipeline.models import Artifact
from analysis.oracle_reports import build_oracle_trade_report, write_oracle_trade_report


class Command(BaseCommand):
    help = "Build an oracle-trade coverage, clustering, and feature-attribution report from saved prediction artifacts."

    def add_arguments(self, parser):
        parser.add_argument("--labels", type=int, required=True)
        parser.add_argument("--prediction-artifacts", required=True, help="Comma-separated prediction artifact ids to compare.")
        parser.add_argument("--market-situation-artifact", type=int, default=0)
        parser.add_argument("--selection-quantile", type=float, default=0.8)
        parser.add_argument("--top-clusters", type=int, default=20)
        parser.add_argument("--output-basename", default="oracle_trade_report")

    def handle(self, *args, **options):
        label_artifact = Artifact.objects.filter(pk=int(options["labels"]), artifact_type="LABELS").first()
        if label_artifact is None:
            raise CommandError(f"Label artifact #{options['labels']} was not found.")

        prediction_ids: list[int] = []
        for token in str(options["prediction_artifacts"] or "").split(","):
            token = str(token).strip()
            if not token:
                continue
            try:
                prediction_ids.append(int(token))
            except Exception as exc:
                raise CommandError(f"Invalid prediction artifact id: {token}") from exc
        if not prediction_ids:
            raise CommandError("At least one prediction artifact id is required.")

        prediction_artifacts = list(
            Artifact.objects.filter(
                id__in=prediction_ids,
                artifact_type__in=["CLASSIFIER_PREDICTIONS", "REGRESSOR_PREDICTIONS", "AUTOENCODER_SCORES", "MTL_PREDICTIONS"],
            ).order_by("id")
        )
        if len(prediction_artifacts) != len(prediction_ids):
            found_ids = {int(artifact.id) for artifact in prediction_artifacts}
            missing = [str(value) for value in prediction_ids if value not in found_ids]
            raise CommandError(f"Prediction artifact(s) not found or unsupported type: {', '.join(missing)}")

        payload = build_oracle_trade_report(
            label_artifact=label_artifact,
            prediction_artifacts=prediction_artifacts,
            market_situation_artifact=Artifact.objects.filter(
                pk=int(options["market_situation_artifact"] or 0),
                artifact_type="MARKET_SITUATION_CLUSTER",
            ).first()
            if int(options["market_situation_artifact"] or 0) > 0
            else Artifact.objects.filter(artifact_type="MARKET_SITUATION_CLUSTER").order_by("-created_at", "-id").first(),
            selection_quantile=float(options["selection_quantile"]),
            top_cluster_count=int(options["top_clusters"]),
        )
        output_path = write_oracle_trade_report(
            Path("data") / "pipeline_artifacts" / f"{str(options['output_basename']).strip()}.json",
            payload,
        )
        payload["report_path"] = str(output_path)
        self.stdout.write(self.style.SUCCESS(json.dumps(payload, indent=2, sort_keys=True, default=str)))
