from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

from django.test import TestCase

from pipeline.models import Artifact, PipelineRun, StrategyDefinition
from pipeline.services import execute_pipeline_run
from pipeline.test_support import ArtifactTestMixin


class StrategyRebalanceTests(ArtifactTestMixin, TestCase):
    def test_weekly_strategy_rebalances_on_first_available_day(self):
        weekly_definition = StrategyDefinition.objects.create(
            name="Weekly First-Day Test",
            slug="weekly-first-day-test",
            strategy_type="notebook_topk_v1",
            gate_quantile=0.0,
            top_k=1,
            rebalance_freq="W",
            gross_exposure=1.0,
            selection_side="long_only",
            signal_combination="multiply",
            config={
                "gate_quantile": 0.0,
                "top_k": 1,
                "rebalance_freq": "W",
                "gross_exposure": 1.0,
                "selection_side": "long_only",
            },
            is_active=True,
        )
        feature_run = PipelineRun.objects.create(
            name="weekly-first-day-feature-source",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_rows: list[dict[str, object]] = []
        prediction_rows: list[dict[str, object]] = []
        for offset in range(5):
            date_value = f"2024-01-0{offset + 1}"
            feature_rows.extend(
                [
                    {"date": date_value, "symbol": "AAPL", "close": 100.0 + offset, "ret_1": 0.02},
                    {"date": date_value, "symbol": "MSFT", "close": 95.0 + offset, "ret_1": 0.01},
                ]
            )
            prediction_rows.extend(
                [
                    {"date": date_value, "symbol": "AAPL", "prediction": 0.8 + offset * 0.01},
                    {"date": date_value, "symbol": "MSFT", "prediction": 0.3 + offset * 0.01},
                ]
            )
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="weekly_first_day_features",
            uri=self.write_csv(
                "weekly_first_day_features",
                ["date", "symbol", "close", "ret_1"],
                feature_rows,
            ),
            content={"rows": len(feature_rows)},
            metadata={},
        )
        prediction_run = PipelineRun.objects.create(
            name="weekly-first-day-predictions",
            requested_job="score_regressor",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        prediction_artifact = Artifact.objects.create(
            pipeline_run=prediction_run,
            artifact_type="REGRESSOR_PREDICTIONS",
            key="weekly_first_day_predictions",
            uri=self.write_csv(
                "weekly_first_day_predictions",
                ["date", "symbol", "prediction"],
                prediction_rows,
            ),
            content={"rows": len(prediction_rows)},
            metadata={},
        )

        strategy_run = PipelineRun.objects.create(
            name="weekly-first-day-strategy",
            requested_job="build_strategy_dataset",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
            config={
                "strategy_definition_id": weekly_definition.id,
                "prediction_artifact_ids": [prediction_artifact.id],
            },
        )
        with patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            strategy_artifact = execute_pipeline_run(
                pipeline_run=strategy_run,
                target_job="build_strategy_dataset",
                mode="strict",
                config=dict(strategy_run.config or {}),
                input_artifact_ids=[feature_artifact.id],
            )

        rows = list(csv.DictReader(Path(strategy_artifact.uri).open("r", encoding="utf-8", newline="")))
        first_day_rows = [row for row in rows if row["date"] == "2024-01-01"]
        last_day_rows = [row for row in rows if row["date"] == "2024-01-05"]
        self.assertEqual(sum(int(row["selected_on_rebalance"]) for row in first_day_rows), 1)
        self.assertEqual(sum(int(row["selected_on_rebalance"]) for row in last_day_rows), 0)
        first_day_aapl = next(row for row in first_day_rows if row["symbol"] == "AAPL")
        last_day_aapl = next(row for row in last_day_rows if row["symbol"] == "AAPL")
        self.assertGreater(float(first_day_aapl["target_weight"]), 0.0)
        self.assertGreater(float(last_day_aapl["target_weight"]), 0.0)
        self.assertEqual(float(next(row for row in last_day_rows if row["symbol"] == "MSFT")["target_weight"]), 0.0)
