from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError

from analysis.market_state import resolve_insight_artifacts
from pipeline.models import Artifact
from analysis.situation_clustering import (
    fit_market_situation_clusters,
    materialize_market_situation_cluster_artifact,
)


class Command(BaseCommand):
    help = "Build market situation clusters from oracle trade entry states and store them as artifacts."

    def add_arguments(self, parser):
        parser.add_argument("--strategy-artifact", type=int, default=0)
        parser.add_argument("--features", type=int, default=0)
        parser.add_argument("--labels", type=int, default=0)
        parser.add_argument("--prediction-artifacts", default="")
        parser.add_argument("--pca-components", type=int, default=8)
        parser.add_argument("--max-clusters", type=int, default=12)
        parser.add_argument("--min-cluster-size", type=int, default=25)
        parser.add_argument("--start-date", default="")
        parser.add_argument("--end-date", default="")
        parser.add_argument("--label-ks", default="")
        parser.add_argument("--min-abs-trade-return-pct", type=float, default=None)
        parser.add_argument("--output-basename", default="market_situation_clusters")

    def handle(self, *args, **options):
        prediction_ids: list[int] = []
        for token in str(options.get("prediction_artifacts") or "").split(","):
            token = str(token).strip()
            if not token:
                continue
            try:
                prediction_ids.append(int(token))
            except Exception as exc:
                raise CommandError(f"Invalid prediction artifact id: {token}") from exc

        artifacts = resolve_insight_artifacts(
            strategy_artifact_id=int(options.get("strategy_artifact") or 0),
            feature_artifact_id=int(options.get("features") or 0),
            label_artifact_id=int(options.get("labels") or 0),
            prediction_artifact_ids=prediction_ids,
        )
        if int(options.get("strategy_artifact") or 0) <= 0 and (
            int(options.get("features") or 0) > 0 or int(options.get("labels") or 0) > 0
        ):
            artifacts.strategy_artifact = None
        if artifacts.strategy_artifact is None and (artifacts.feature_artifact is None or artifacts.label_artifact is None):
            raise CommandError("Provide a strategy artifact or a feature + label artifact pair.")

        label_ks: list[int] = []
        for token in str(options.get("label_ks") or "").split(","):
            token = str(token).strip()
            if not token:
                continue
            try:
                label_ks.append(int(token))
            except Exception as exc:
                raise CommandError(f"Invalid label k value: {token}") from exc

        payload = fit_market_situation_clusters(
            strategy_artifact=artifacts.strategy_artifact,
            feature_artifact=artifacts.feature_artifact,
            label_artifact=artifacts.label_artifact,
            prediction_artifacts=artifacts.prediction_artifacts,
            pca_components=int(options.get("pca_components") or 0),
            max_clusters=int(options.get("max_clusters") or 12),
            min_cluster_size=int(options.get("min_cluster_size") or 25),
            start_date=str(options.get("start_date") or "").strip() or None,
            end_date=str(options.get("end_date") or "").strip() or None,
            label_ks=label_ks,
            min_abs_trade_return=(
                max(0.0, float(options["min_abs_trade_return_pct"]) / 100.0)
                if options.get("min_abs_trade_return_pct") not in (None, "")
                else None
            ),
        )
        artifact = materialize_market_situation_cluster_artifact(
            output_basename=str(options.get("output_basename") or "").strip(),
            clustering_payload=payload,
            strategy_artifact=artifacts.strategy_artifact,
            feature_artifact=artifacts.feature_artifact,
            label_artifact=artifacts.label_artifact,
            prediction_artifacts=artifacts.prediction_artifacts,
        )
        summary_path = str((artifact.metadata or {}).get("summary_json_uri") or "")
        summary_payload = json.loads(open(summary_path, "r", encoding="utf-8").read()) if summary_path else {}
        output = {
            "artifact_id": int(artifact.id),
            "artifact_type": str(artifact.artifact_type),
            "assignments_uri": str(artifact.uri),
            "summary_json_uri": summary_path,
            "summary": dict(summary_payload.get("summary") or artifact.content or {}),
        }
        self.stdout.write(json.dumps(output, indent=2, sort_keys=True, default=str))
