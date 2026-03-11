from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from analysis.diagnostics import build_diagnostic_report, write_diagnostic_report
from pipeline.models import Artifact


class Command(BaseCommand):
    help = "Build an explainable diagnostics report from saved pipeline artifacts."

    def add_arguments(self, parser):
        parser.add_argument("--labels", type=int, required=True)
        parser.add_argument("--classifier-predictions", type=int, required=True)
        parser.add_argument("--regressor-predictions", type=int, required=True)
        parser.add_argument("--autoencoder-scores", type=int, required=True)
        parser.add_argument("--strategy-dataset", type=int, default=0)
        parser.add_argument("--backtest-result", type=int, default=0)
        parser.add_argument("--output-basename", default="pipeline_diagnostics")
        parser.add_argument("--run-rl", action="store_true")
        parser.add_argument("--rl-train-split-date", default="2023-12-31")
        parser.add_argument("--rl-episodes", type=int, default=20)

    def handle(self, *args, **options):
        def get_artifact(artifact_id: int, artifact_type: str) -> Artifact:
            artifact = Artifact.objects.filter(pk=int(artifact_id), artifact_type=artifact_type).first()
            if artifact is None:
                raise CommandError(f"Artifact #{artifact_id} with type {artifact_type} was not found.")
            return artifact

        label_artifact = get_artifact(int(options["labels"]), "LABELS")
        classifier_predictions = get_artifact(int(options["classifier_predictions"]), "CLASSIFIER_PREDICTIONS")
        regressor_predictions = get_artifact(int(options["regressor_predictions"]), "REGRESSOR_PREDICTIONS")
        autoencoder_scores = get_artifact(int(options["autoencoder_scores"]), "AUTOENCODER_SCORES")
        strategy_artifact = None
        backtest_artifact = None
        if int(options["strategy_dataset"] or 0) > 0:
            strategy_artifact = get_artifact(int(options["strategy_dataset"]), "STRATEGY_DATASET")
        if int(options["backtest_result"] or 0) > 0:
            backtest_artifact = get_artifact(int(options["backtest_result"]), "BACKTEST_RESULT")

        payload = build_diagnostic_report(
            label_artifact=label_artifact,
            classifier_predictions_artifact=classifier_predictions,
            regressor_predictions_artifact=regressor_predictions,
            autoencoder_scores_artifact=autoencoder_scores,
            strategy_artifact=strategy_artifact,
            backtest_artifact=backtest_artifact,
            run_rl=bool(options["run_rl"]),
            rl_train_split_date=str(options["rl_train_split_date"]).strip(),
            rl_episodes=int(options["rl_episodes"]),
        )
        output_path = write_diagnostic_report(
            Path("data") / "pipeline_artifacts" / f"{str(options['output_basename']).strip()}.json",
            payload,
        )
        payload["report_path"] = str(output_path)
        self.stdout.write(self.style.SUCCESS(json.dumps(payload, indent=2, sort_keys=True, default=str)))
