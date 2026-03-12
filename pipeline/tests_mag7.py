import csv
import json
from pathlib import Path
from unittest.mock import patch
from io import StringIO

from django.core.management import call_command
from django.test import TestCase
from django.urls import reverse

from .cohort_runner import COHORT_SUMMARY_SCHEMA_VERSION, _expand_cluster_specialist_variants, run_model_cohort_backtests, run_walk_forward_model_cohort_backtests
from .direct_strategy_runner import run_walk_forward_direct_strategy_backtests
from .experiments import expand_model_cohort_configs
from analysis.feature_attribution import run_feature_family_attribution_suite
from .models import Artifact, PipelineRun, StrategyDefinition
from analysis.oracle_reports import build_oracle_trade_report
from .services import _stable_payload_hash, execute_pipeline_run
from .strategy_definitions import ensure_default_strategy_definitions
from .test_support import MAG7_SYMBOLS, Mag7FixtureMixin
from .research_suite import RESEARCH_REPORT_SCHEMA_VERSION, run_optimal_trade_research_suite


class Mag7PipelineSuiteTests(Mag7FixtureMixin, TestCase):
    def setUp(self):
        super().setUp()
        self.create_mag7_symbols()
        self.seed_mag7_price_history(days=5)

    def test_mag7_universe_run_persists_all_symbols(self):
        run = PipelineRun.objects.create(
            name="mag7-universe",
            requested_job="universe",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
        )

        with patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            artifact = execute_pipeline_run(
                pipeline_run=run,
                target_job="universe",
                mode="strict",
                config={"symbols": MAG7_SYMBOLS},
            )

        run.refresh_from_db()
        self.assertEqual(run.status, PipelineRun.Status.SUCCEEDED)
        self.assertEqual(artifact.artifact_type, "UNIVERSE")
        payload = Path(artifact.uri).read_text(encoding="utf-8")
        self.assertIn('"count": 7', payload)
        for symbol in MAG7_SYMBOLS:
            self.assertIn(symbol, payload)

    def test_mag7_features_run_builds_rows_for_every_symbol(self):
        universe_run = PipelineRun.objects.create(
            name="mag7-universe-source",
            requested_job="universe",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        universe_uri = self.write_json(
            "mag7_universe",
            {"symbols": MAG7_SYMBOLS, "count": len(MAG7_SYMBOLS)},
        )
        universe_artifact = Artifact.objects.create(
            pipeline_run=universe_run,
            artifact_type="UNIVERSE",
            key="mag7_universe",
            uri=universe_uri,
            content={"count": len(MAG7_SYMBOLS)},
            metadata={},
        )

        feature_run = PipelineRun.objects.create(
            name="mag7-features",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
        )

        with patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            feature_artifact = execute_pipeline_run(
                pipeline_run=feature_run,
                target_job="features",
                mode="strict",
                input_artifact_ids=[universe_artifact.id],
            )

        feature_run.refresh_from_db()
        self.assertEqual(feature_run.status, PipelineRun.Status.SUCCEEDED)
        self.assertEqual(feature_artifact.artifact_type, "FEATURES")
        self.assertEqual(feature_artifact.metadata["source_universe_artifact_id"], universe_artifact.id)
        self.assertIn("feature_family_columns", feature_artifact.metadata)
        self.assertIn("coverage_rows", feature_artifact.metadata)
        self.assertGreaterEqual(int(feature_artifact.content.get("feature_column_count") or 0), 1)
        with Path(feature_artifact.uri).open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), len(MAG7_SYMBOLS) * 5)
        self.assertEqual(sorted({row["symbol"] for row in rows}), sorted(MAG7_SYMBOLS))
        self.assert_rows_have_columns(
            rows,
            ["date", "symbol", "close", "ret_1", "sma_5", "sma_5_ratio", "vol_5"],
        )

    def test_mag7_predict_run_scores_every_symbol(self):
        feature_run = PipelineRun.objects.create(
            name="mag7-feature-source",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_rows = []
        for symbol_index, symbol in enumerate(MAG7_SYMBOLS, start=1):
            for offset in range(5):
                feature_rows.append(
                    {
                        "date": f"2024-01-0{offset + 1}",
                        "symbol": symbol,
                        "ret_1": round((symbol_index * 0.01) - 0.02 + (offset * 0.001), 6),
                    }
                )
        feature_uri = self.write_csv("mag7_features_for_predict", ["date", "symbol", "ret_1"], feature_rows)
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="mag7_features_for_predict",
            uri=feature_uri,
            content={"rows": len(feature_rows)},
            metadata={},
        )

        label_run = PipelineRun.objects.create(
            name="mag7-label-source",
            requested_job="labels",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        label_rows = []
        for symbol_index, symbol in enumerate(MAG7_SYMBOLS, start=1):
            for offset in range(5):
                label_rows.append(
                    {
                        "date": f"2024-01-0{offset + 1}",
                        "symbol": symbol,
                        "label": 1 if (symbol_index + offset) % 2 == 0 else 0,
                        "market_position": 1 if (symbol_index + offset) % 2 == 0 else -1,
                        "trade_return": round((symbol_index * 0.01) - 0.03 + (offset * 0.005), 6),
                        "hold_days": 1 + offset,
                        "side": "long" if (symbol_index + offset) % 2 == 0 else "short",
                        "freq": "YE",
                        "k": 1,
                        "entry_date": f"2024-01-0{offset + 1}",
                        "exit_date": f"2024-01-0{min(offset + 2, 5)}",
                        "entry_px": str(100 + symbol_index + offset),
                        "exit_px": str(101 + symbol_index + offset),
                        "ret_pct": "1.00%",
                    }
                )
        label_uri = self.write_csv(
            "mag7_labels_for_predict",
            [
                "date", "symbol", "label", "market_position", "trade_return", "hold_days",
                "side", "freq", "k", "entry_date", "exit_date", "entry_px", "exit_px", "ret_pct",
            ],
            label_rows,
        )
        label_artifact = Artifact.objects.create(
            pipeline_run=label_run,
            artifact_type="LABELS",
            key="mag7_labels_for_predict",
            uri=label_uri,
            content={"rows": len(label_rows)},
            metadata={},
        )

        model_run = PipelineRun.objects.create(
            name="mag7-model",
            requested_job="train",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
            config={
                "algorithm": "random_forest_regressor",
                "task_type": "regression",
                "target_col": "trade_return",
                "model_name": "mag7_predict_source_model",
            },
        )
        with patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            model_artifact = execute_pipeline_run(
                pipeline_run=model_run,
                target_job="train",
                mode="strict",
                config=dict(model_run.config or {}),
                input_artifact_ids=[label_artifact.id, feature_artifact.id],
            )

        predict_run = PipelineRun.objects.create(
            name="mag7-predict",
            requested_job="predict",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
        )

        with patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            prediction_artifact = execute_pipeline_run(
                pipeline_run=predict_run,
                target_job="predict",
                mode="strict",
                input_artifact_ids=[model_artifact.id, feature_artifact.id],
            )

        predict_run.refresh_from_db()
        self.assertEqual(predict_run.status, PipelineRun.Status.SUCCEEDED)
        with Path(prediction_artifact.uri).open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), len(MAG7_SYMBOLS) * 5)
        self.assertEqual(sorted({row["symbol"] for row in rows}), sorted(MAG7_SYMBOLS))
        self.assert_rows_have_columns(rows, ["date", "symbol", "prediction", "ret_1"])

    def test_mag7_full_chain_labels_features_train_predict_research(self):
        with patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            universe_run = PipelineRun.objects.create(
                name="mag7-full-universe",
                requested_job="universe",
                mode=PipelineRun.Mode.STRICT,
                status=PipelineRun.Status.PENDING,
            )
            universe_artifact = execute_pipeline_run(
                pipeline_run=universe_run,
                target_job="universe",
                mode="strict",
                config={"symbols": MAG7_SYMBOLS},
            )

            label_run = PipelineRun.objects.create(
                name="mag7-full-labels",
                requested_job="labels",
                mode=PipelineRun.Mode.STRICT,
                status=PipelineRun.Status.PENDING,
            )
            label_artifact = execute_pipeline_run(
                pipeline_run=label_run,
                target_job="labels",
                mode="strict",
                config={"k_params": {"YE": [1]}, "min_profit_pct": 0.0},
                input_artifact_ids=[universe_artifact.id],
            )

            feature_run = PipelineRun.objects.create(
                name="mag7-full-features",
                requested_job="features",
                mode=PipelineRun.Mode.STRICT,
                status=PipelineRun.Status.PENDING,
            )
            feature_artifact = execute_pipeline_run(
                pipeline_run=feature_run,
                target_job="features",
                mode="strict",
                input_artifact_ids=[universe_artifact.id],
            )

            train_run = PipelineRun.objects.create(
                name="mag7-full-train",
                requested_job="train",
                mode=PipelineRun.Mode.STRICT,
                status=PipelineRun.Status.PENDING,
                config={
                    "algorithm": "random_forest_regressor",
                    "task_type": "regression",
                    "target_col": "trade_return",
                    "model_name": "mag7_full_chain_regressor",
                },
            )
            model_artifact = execute_pipeline_run(
                pipeline_run=train_run,
                target_job="train",
                mode="strict",
                config=dict(train_run.config or {}),
                input_artifact_ids=[label_artifact.id, feature_artifact.id],
            )

            predict_run = PipelineRun.objects.create(
                name="mag7-full-predict",
                requested_job="predict",
                mode=PipelineRun.Mode.STRICT,
                status=PipelineRun.Status.PENDING,
            )
            prediction_artifact = execute_pipeline_run(
                pipeline_run=predict_run,
                target_job="predict",
                mode="strict",
                input_artifact_ids=[model_artifact.id, feature_artifact.id],
            )

        universe_run.refresh_from_db()
        label_run.refresh_from_db()
        feature_run.refresh_from_db()
        train_run.refresh_from_db()
        predict_run.refresh_from_db()

        self.assertEqual(universe_run.status, PipelineRun.Status.SUCCEEDED)
        self.assertEqual(label_run.status, PipelineRun.Status.SUCCEEDED)
        self.assertEqual(feature_run.status, PipelineRun.Status.SUCCEEDED)
        self.assertEqual(train_run.status, PipelineRun.Status.SUCCEEDED)
        self.assertEqual(predict_run.status, PipelineRun.Status.SUCCEEDED)

        self.assertEqual(label_artifact.metadata["source_universe_artifact_id"], universe_artifact.id)
        self.assertEqual(feature_artifact.metadata["source_universe_artifact_id"], universe_artifact.id)
        self.assertEqual(model_artifact.metadata["source_labels_artifact_id"], label_artifact.id)
        self.assertEqual(model_artifact.metadata["source_features_artifact_id"], feature_artifact.id)
        self.assertEqual(prediction_artifact.metadata["source_model_artifact_id"], model_artifact.id)
        self.assertEqual(prediction_artifact.metadata["source_features_artifact_id"], feature_artifact.id)

        with Path(label_artifact.uri).open("r", encoding="utf-8", newline="") as fh:
            label_rows = list(csv.DictReader(fh))
        with Path(feature_artifact.uri).open("r", encoding="utf-8", newline="") as fh:
            feature_rows = list(csv.DictReader(fh))
        with Path(prediction_artifact.uri).open("r", encoding="utf-8", newline="") as fh:
            prediction_rows = list(csv.DictReader(fh))

        self.assertGreater(len(label_rows), 0)
        self.assertEqual(len(feature_rows), len(MAG7_SYMBOLS) * 5)
        self.assertGreater(len(prediction_rows), 0)
        self.assertEqual(sorted({row["symbol"] for row in prediction_rows}), sorted(MAG7_SYMBOLS))
        self.assertTrue(any(row["symbol"] == "AAPL" for row in label_rows))
        self.assertTrue(any(row["symbol"] == "NVDA" for row in prediction_rows))
        self.assert_rows_have_columns(
            label_rows,
            [
                "date",
                "symbol",
                "label",
                "market_position",
                "trade_return",
                "hold_days",
                "side",
                "freq",
                "k",
                "entry_date",
                "exit_date",
                "entry_px",
                "exit_px",
                "ret_pct",
            ],
        )
        self.assert_rows_have_columns(
            feature_rows,
            ["date", "symbol", "close", "ret_1", "sma_5", "sma_5_ratio", "vol_5"],
        )
        self.assert_rows_have_columns(prediction_rows, ["date", "symbol", "prediction", "ret_1"])

        response = self.client.get(
            reverse("pipeline-symbol-research", args=["AAPL"]),
            {
                "label_artifact_id": label_artifact.id,
                "feature_artifact_id": feature_artifact.id,
                "prediction_artifact_id": prediction_artifact.id,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "AAPL Research Workspace")
        self.assertContains(response, f"#{label_artifact.id}")
        self.assertContains(response, f"#{prediction_artifact.id}")

    def test_mag7_train_can_use_prior_prediction_panel_as_feature_input(self):
        universe_run = PipelineRun.objects.create(
            name="mag7-extra-universe",
            requested_job="universe",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        universe_uri = self.write_json("mag7_extra_universe", {"symbols": MAG7_SYMBOLS, "count": len(MAG7_SYMBOLS)})
        universe_artifact = Artifact.objects.create(
            pipeline_run=universe_run,
            artifact_type="UNIVERSE",
            key="mag7_extra_universe",
            uri=universe_uri,
            content={"count": len(MAG7_SYMBOLS)},
            metadata={},
        )

        with patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            label_run = PipelineRun.objects.create(
                name="mag7-extra-labels",
                requested_job="labels",
                mode=PipelineRun.Mode.STRICT,
                status=PipelineRun.Status.PENDING,
            )
            label_artifact = execute_pipeline_run(
                pipeline_run=label_run,
                target_job="labels",
                mode="strict",
                config={"k_params": {"YE": [1]}, "min_profit_pct": 0.0},
                input_artifact_ids=[universe_artifact.id],
            )
            feature_run = PipelineRun.objects.create(
                name="mag7-extra-features",
                requested_job="features",
                mode=PipelineRun.Mode.STRICT,
                status=PipelineRun.Status.PENDING,
            )
            feature_artifact = execute_pipeline_run(
                pipeline_run=feature_run,
                target_job="features",
                mode="strict",
                input_artifact_ids=[universe_artifact.id],
            )

        prior_prediction_run = PipelineRun.objects.create(
            name="mag7-prior-predictions",
            requested_job="predict",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        prior_prediction_rows = []
        for symbol_index, symbol in enumerate(MAG7_SYMBOLS, start=1):
            for offset in range(5):
                prior_prediction_rows.append(
                    {
                        "date": f"2024-01-0{offset + 1}",
                        "symbol": symbol,
                        "prediction_score": round(0.4 + symbol_index * 0.03 + offset * 0.01, 6),
                        "prediction": 1 if symbol_index % 2 == 0 else 0,
                    }
                )
        prior_prediction_uri = self.write_csv(
            "mag7_prior_predictions",
            ["date", "symbol", "prediction_score", "prediction"],
            prior_prediction_rows,
        )
        prior_prediction_artifact = Artifact.objects.create(
            pipeline_run=prior_prediction_run,
            artifact_type="PREDICTIONS",
            key="mag7_prior_predictions",
            uri=prior_prediction_uri,
            content={"rows": len(prior_prediction_rows)},
            metadata={},
        )

        train_run = PipelineRun.objects.create(
            name="mag7-extra-train",
            requested_job="train",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
            config={
                "algorithm": "random_forest_regressor",
                "task_type": "regression",
                "target_col": "trade_return",
                "prediction_artifact_ids": [prior_prediction_artifact.id],
                "model_name": "mag7_reg_with_prior_preds",
            },
        )
        with patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            model_artifact = execute_pipeline_run(
                pipeline_run=train_run,
                target_job="train",
                mode="strict",
                config=dict(train_run.config or {}),
                input_artifact_ids=[label_artifact.id, feature_artifact.id],
            )

        train_run.refresh_from_db()
        self.assertEqual(train_run.status, PipelineRun.Status.SUCCEEDED)
        self.assertEqual(model_artifact.metadata["source_prediction_artifact_ids"], [prior_prediction_artifact.id])
        model_payload = Path(model_artifact.uri).read_text(encoding="utf-8")
        self.assertIn("mag7_reg_with_prior_preds", model_payload)
        self.assertIn(str(prior_prediction_artifact.id), model_payload)

    def test_explicit_regressor_jobs_can_build_strategy_dataset_and_backtest(self):
        strategy_definition = ensure_default_strategy_definitions()[0]
        feature_run = PipelineRun.objects.create(
            name="mag7-explicit-features",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_rows = []
        for symbol_index, symbol in enumerate(MAG7_SYMBOLS, start=1):
            for offset in range(5):
                feature_rows.append(
                    {
                        "date": f"2024-01-0{offset + 1}",
                        "symbol": symbol,
                        "close": round(100 + symbol_index + offset + 0.5, 4),
                        "ret_1": round((symbol_index * 0.01) - 0.02 + offset * 0.002, 6),
                        "sma_5": round(100 + symbol_index + offset, 4),
                        "sma_5_ratio": round(1.0 + symbol_index * 0.001, 6),
                        "vol_5": round(0.1 + offset * 0.01, 6),
                    }
                )
        feature_uri = self.write_csv(
            "mag7_explicit_features",
            ["date", "symbol", "close", "ret_1", "sma_5", "sma_5_ratio", "vol_5"],
            feature_rows,
        )
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="mag7_explicit_features",
            uri=feature_uri,
            content={"rows": len(feature_rows)},
            metadata={},
        )

        label_run = PipelineRun.objects.create(
            name="mag7-explicit-labels",
            requested_job="labels",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        label_rows = []
        for symbol_index, symbol in enumerate(MAG7_SYMBOLS, start=1):
            for offset in range(5):
                label_rows.append(
                    {
                        "date": f"2024-01-0{offset + 1}",
                        "symbol": symbol,
                        "label": 1 if (symbol_index + offset) % 2 == 0 else 0,
                        "market_position": 1 if (symbol_index + offset) % 2 == 0 else -1,
                        "trade_return": round((symbol_index * 0.015) - 0.04 + (offset * 0.01), 6),
                        "hold_days": 1 + offset,
                        "side": "long" if (symbol_index + offset) % 2 == 0 else "short",
                        "freq": "YE",
                        "k": 1,
                        "entry_date": f"2024-01-0{offset + 1}",
                        "exit_date": f"2024-01-0{min(offset + 2, 5)}",
                        "entry_px": str(100 + symbol_index + offset),
                        "exit_px": str(101 + symbol_index + offset),
                        "ret_pct": "1.00%",
                    }
                )
        label_uri = self.write_csv(
            "mag7_explicit_labels",
            [
                "date", "symbol", "label", "market_position", "trade_return", "hold_days",
                "side", "freq", "k", "entry_date", "exit_date", "entry_px", "exit_px", "ret_pct",
            ],
            label_rows,
        )
        label_artifact = Artifact.objects.create(
            pipeline_run=label_run,
            artifact_type="LABELS",
            key="mag7_explicit_labels",
            uri=label_uri,
            content={"rows": len(label_rows)},
            metadata={},
        )

        fit_run = PipelineRun.objects.create(
            name="mag7-fit-regressor",
            requested_job="fit_regressor",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
            config={"target_col": "trade_return", "model_name": "mag7_explicit_regressor"},
        )
        with patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            reg_model_artifact = execute_pipeline_run(
                pipeline_run=fit_run,
                target_job="fit_regressor",
                mode="strict",
                config=dict(fit_run.config or {}),
                input_artifact_ids=[feature_artifact.id, label_artifact.id],
            )

            score_run = PipelineRun.objects.create(
                name="mag7-score-regressor",
                requested_job="score_regressor",
                mode=PipelineRun.Mode.STRICT,
                status=PipelineRun.Status.PENDING,
                config={"label_artifact_id": label_artifact.id},
            )
            reg_scores_artifact = execute_pipeline_run(
                pipeline_run=score_run,
                target_job="score_regressor",
                mode="strict",
                config=dict(score_run.config or {}),
                input_artifact_ids=[reg_model_artifact.id, feature_artifact.id],
            )

            strategy_run = PipelineRun.objects.create(
                name="mag7-strategy-dataset",
                requested_job="build_strategy_dataset",
                mode=PipelineRun.Mode.STRICT,
                status=PipelineRun.Status.PENDING,
                config={
                    "strategy_definition_id": strategy_definition.id,
                    "prediction_artifact_ids": [reg_scores_artifact.id],
                    "label_artifact_id": label_artifact.id,
                },
            )
            strategy_artifact = execute_pipeline_run(
                pipeline_run=strategy_run,
                target_job="build_strategy_dataset",
                mode="strict",
                config=dict(strategy_run.config or {}),
                input_artifact_ids=[feature_artifact.id],
            )

            backtest_run = PipelineRun.objects.create(
                name="mag7-backtest",
                requested_job="backtest_strategy",
                mode=PipelineRun.Mode.STRICT,
                status=PipelineRun.Status.PENDING,
                config={"transaction_cost_bps": 10.0},
            )
            backtest_artifact = execute_pipeline_run(
                pipeline_run=backtest_run,
                target_job="backtest_strategy",
                mode="strict",
                config=dict(backtest_run.config or {}),
                input_artifact_ids=[strategy_artifact.id],
            )

        fit_run.refresh_from_db()
        score_run.refresh_from_db()
        strategy_run.refresh_from_db()
        backtest_run.refresh_from_db()
        self.assertEqual(fit_run.status, PipelineRun.Status.SUCCEEDED)
        self.assertEqual(score_run.status, PipelineRun.Status.SUCCEEDED)
        self.assertEqual(strategy_run.status, PipelineRun.Status.SUCCEEDED)
        self.assertEqual(backtest_run.status, PipelineRun.Status.SUCCEEDED)
        self.assertEqual(reg_model_artifact.artifact_type, "REGRESSOR_MODEL")
        self.assertEqual(reg_scores_artifact.artifact_type, "REGRESSOR_PREDICTIONS")
        self.assertEqual(strategy_artifact.artifact_type, "STRATEGY_DATASET")
        self.assertEqual(backtest_artifact.artifact_type, "BACKTEST_RESULT")
        self.assertEqual(strategy_artifact.metadata["strategy_definition_id"], strategy_definition.id)
        self.assertEqual(strategy_artifact.metadata["strategy_definition_name"], strategy_definition.name)
        strategy_rows = list(csv.DictReader(Path(strategy_artifact.uri).open("r", encoding="utf-8", newline="")))
        self.assert_rows_have_columns(
            strategy_rows,
            [
                "date",
                "symbol",
                "prob_buy",
                "ranking",
                "ae_familiarity",
                "combined_score",
                "strategy_score",
                "eligible",
                "selected_on_rebalance",
                "target_weight",
                "strategy_signal",
            ],
        )
        score_rows = list(csv.DictReader(Path(reg_scores_artifact.uri).open("r", encoding="utf-8", newline="")))
        self.assert_rows_have_columns(
            score_rows,
            ["date", "symbol", "prediction", "prediction_score", "raw_prediction", "signal_score"],
        )
        backtest_payload = Path(backtest_artifact.uri).read_text(encoding="utf-8")
        self.assertIn("realized_return", backtest_payload)
        self.assertIn("effective_weight", backtest_payload)
        self.assertIn("asset_return", backtest_payload)
        backtest_rows = list(csv.DictReader(Path(backtest_artifact.uri).open("r", encoding="utf-8", newline="")))
        selected_symbol = backtest_rows[0]["symbol"]

        model_detail = self.client.get(reverse("pipeline-artifact-detail", args=[reg_model_artifact.id]))
        strategy_detail = self.client.get(reverse("pipeline-strategy-detail", args=[strategy_artifact.id]))
        backtest_detail = self.client.get(reverse("pipeline-backtest-detail", args=[backtest_artifact.id]))

        self.assertEqual(model_detail.status_code, 200)
        self.assertEqual(strategy_detail.status_code, 200)
        self.assertEqual(backtest_detail.status_code, 200)
        self.assertContains(model_detail, "REGRESSOR_MODEL Artifact")
        self.assertContains(model_detail, "mag7_explicit_regressor")
        self.assertContains(strategy_detail, "Strategy Dataset")
        self.assertContains(strategy_detail, strategy_definition.name)
        self.assertContains(strategy_detail, f"/pipeline/research/symbol/AAPL/")
        self.assertContains(strategy_detail, "Selected Rebalance Snapshot")
        self.assertContains(strategy_detail, "Open raw strategy payload")
        self.assertContains(backtest_detail, "Backtest Result")
        self.assertContains(backtest_detail, "Final Equity")
        self.assertContains(backtest_detail, "points")
        self.assertContains(backtest_detail, "Turnover And Holdings")
        self.assertContains(backtest_detail, "Recent Daily Summary")
        self.assertContains(backtest_detail, "Top Contributors")
        self.assertContains(backtest_detail, "Open raw backtest payload")
        self.assertContains(backtest_detail, "Sharpe")
        self.assertContains(backtest_detail, "Profit Factor")

        filtered_backtest_detail = self.client.get(
            reverse("pipeline-backtest-detail", args=[backtest_artifact.id]),
            {"symbol": selected_symbol, "contrib_sort": "rows_desc"},
        )
        self.assertEqual(filtered_backtest_detail.status_code, 200)
        self.assertContains(filtered_backtest_detail, f'value="{selected_symbol}" selected')
        self.assertContains(filtered_backtest_detail, 'value="rows_desc" selected')

        research_response = self.client.get(
            reverse("pipeline-symbol-research", args=["AAPL"]),
            {
                "label_artifact_id": label_artifact.id,
                "feature_artifact_id": feature_artifact.id,
                "prediction_artifact_id": reg_scores_artifact.id,
                "strategy_artifact_id": strategy_artifact.id,
                "backtest_artifact_id": backtest_artifact.id,
            },
        )
        self.assertEqual(research_response.status_code, 200)
        self.assertContains(research_response, "Strategy Rows")
        self.assertContains(research_response, "Backtest Rows")

    def test_mag7_symbol_research_view_respects_selected_prediction_artifact(self):
        prediction_run_a = PipelineRun.objects.create(
            name="mag7-prediction-a",
            requested_job="predict",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        prediction_uri_a = self.write_csv(
            "mag7_predictions_a",
            ["date", "symbol", "prediction_score", "prediction"],
            [{"date": "2024-01-01", "symbol": "META", "prediction_score": 0.11, "prediction": 0}],
        )
        prediction_artifact_a = Artifact.objects.create(
            pipeline_run=prediction_run_a,
            artifact_type="PREDICTIONS",
            key="mag7_predictions_a",
            uri=prediction_uri_a,
            content={"rows": 1},
            metadata={},
        )

        prediction_run_b = PipelineRun.objects.create(
            name="mag7-prediction-b",
            requested_job="predict",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        prediction_uri_b = self.write_csv(
            "mag7_predictions_b",
            ["date", "symbol", "prediction_score", "prediction"],
            [{"date": "2024-01-01", "symbol": "META", "prediction_score": 0.93, "prediction": 1}],
        )
        prediction_artifact_b = Artifact.objects.create(
            pipeline_run=prediction_run_b,
            artifact_type="PREDICTIONS",
            key="mag7_predictions_b",
            uri=prediction_uri_b,
            content={"rows": 1},
            metadata={},
        )

        response_a = self.client.get(
            reverse("pipeline-symbol-research", args=["META"]),
            {"prediction_artifact_id": prediction_artifact_a.id},
        )
        response_b = self.client.get(
            reverse("pipeline-symbol-research", args=["META"]),
            {"prediction_artifact_id": prediction_artifact_b.id},
        )

        self.assertEqual(response_a.status_code, 200)
        self.assertEqual(response_b.status_code, 200)
        self.assertContains(response_a, "0.11")
        self.assertNotContains(response_a, "0.93")
        self.assertContains(response_b, "0.93")
        self.assertNotContains(response_b, "0.11")

    def test_mag7_symbol_research_view_can_overlay_multiple_prediction_artifacts(self):
        prediction_run_a = PipelineRun.objects.create(
            name="mag7-multi-prediction-a",
            requested_job="predict",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        prediction_uri_a = self.write_csv(
            "mag7_multi_predictions_a",
            ["date", "symbol", "prediction_score", "prediction"],
            [{"date": "2024-01-01", "symbol": "META", "prediction_score": 0.11, "prediction": 0}],
        )
        prediction_artifact_a = Artifact.objects.create(
            pipeline_run=prediction_run_a,
            artifact_type="PREDICTIONS",
            key="mag7_multi_predictions_a",
            uri=prediction_uri_a,
            content={"rows": 1},
            metadata={},
        )

        prediction_run_b = PipelineRun.objects.create(
            name="mag7-multi-prediction-b",
            requested_job="score_regressor",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        prediction_uri_b = self.write_csv(
            "mag7_multi_predictions_b",
            ["date", "symbol", "prediction_score", "prediction"],
            [{"date": "2024-01-01", "symbol": "META", "prediction_score": 0.93, "prediction": 1}],
        )
        prediction_artifact_b = Artifact.objects.create(
            pipeline_run=prediction_run_b,
            artifact_type="REGRESSOR_PREDICTIONS",
            key="mag7_multi_predictions_b",
            uri=prediction_uri_b,
            content={"rows": 1},
            metadata={},
        )

        response = self.client.get(
            reverse("pipeline-symbol-research", args=["META"]),
            [("prediction_artifact_id", prediction_artifact_a.id), ("prediction_artifact_id", prediction_artifact_b.id)],
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "0.11")
        self.assertContains(response, "0.93")
        self.assertContains(response, f"PREDICTIONS #{prediction_artifact_a.id}")
        self.assertContains(response, f"REGRESSOR_PREDICTIONS #{prediction_artifact_b.id}")

    def test_mag7_symbol_research_view_renders_selected_symbol(self):
        label_run = PipelineRun.objects.create(
            name="mag7-labels",
            requested_job="labels",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        label_rows = [
            {
                "date": "2024-01-01",
                "symbol": "NVDA",
                "label": 1,
                "market_position": 1,
                "trade_return": 0.07,
                "hold_days": 2,
                "side": "long",
                "freq": "YE",
                "k": 1,
                "entry_date": "2024-01-01",
                "exit_date": "2024-01-03",
                "entry_px": "250.0",
                "exit_px": "267.5",
                "ret_pct": "7.00%",
            }
        ]
        label_uri = self.write_csv(
            "mag7_labels_view",
            [
                "date", "symbol", "label", "market_position", "trade_return", "hold_days",
                "side", "freq", "k", "entry_date", "exit_date", "entry_px", "exit_px", "ret_pct",
            ],
            label_rows,
        )
        label_artifact = Artifact.objects.create(
            pipeline_run=label_run,
            artifact_type="LABELS",
            key="mag7_labels_view",
            uri=label_uri,
            content={"rows": 1},
            metadata={},
        )

        feature_run = PipelineRun.objects.create(
            name="mag7-features-view",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_uri = self.write_csv(
            "mag7_features_view",
            ["date", "symbol", "close", "ret_1"],
            [{"date": "2024-01-01", "symbol": "NVDA", "close": 250.75, "ret_1": 0.012}],
        )
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="mag7_features_view",
            uri=feature_uri,
            content={"rows": 1},
            metadata={},
        )

        prediction_run = PipelineRun.objects.create(
            name="mag7-predictions-view",
            requested_job="predict",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        prediction_uri = self.write_csv(
            "mag7_predictions_view",
            ["date", "symbol", "prediction_score", "prediction"],
            [{"date": "2024-01-01", "symbol": "NVDA", "prediction_score": 0.91, "prediction": 1}],
        )
        prediction_artifact = Artifact.objects.create(
            pipeline_run=prediction_run,
            artifact_type="PREDICTIONS",
            key="mag7_predictions_view",
            uri=prediction_uri,
            content={"rows": 1},
            metadata={},
        )

        response = self.client.get(
            reverse("pipeline-symbol-research", args=["NVDA"]),
            {
                "label_artifact_id": label_artifact.id,
                "feature_artifact_id": feature_artifact.id,
                "prediction_artifact_id": prediction_artifact.id,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "NVDA Research Workspace")
        self.assertContains(response, "7.00%")
        self.assertContains(response, "0.91")

    def test_strategy_lab_page_renders_recent_strategy_and_backtest_artifacts(self):
        strategy_definition = ensure_default_strategy_definitions()[0]
        strategy_run = PipelineRun.objects.create(
            name="mag7-strategy-lab-view",
            requested_job="build_strategy_dataset",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        strategy_uri = self.write_csv(
            "mag7_strategy_lab",
            ["date", "symbol", "strategy_score", "strategy_signal", "target_weight", "selected_on_rebalance"],
            [{"date": "2024-01-01", "symbol": "AAPL", "strategy_score": 0.42, "strategy_signal": 1, "target_weight": 0.5, "selected_on_rebalance": 1}],
        )
        strategy_artifact = Artifact.objects.create(
            pipeline_run=strategy_run,
            artifact_type="STRATEGY_DATASET",
            key="mag7_strategy_lab",
            uri=strategy_uri,
            content={"rows": 1, "symbols": 1, "selected_rows": 1, "dates": 1, "avg_daily_positions": 1.0},
            metadata={
                "source_features_artifact_id": 48,
                "source_label_artifact_id": 53,
                "source_prediction_artifact_ids": [50, 55],
                "strategy_definition_id": strategy_definition.id,
                "strategy_definition_name": strategy_definition.name,
                "strategy_config": {"gate_quantile": 0.5, "top_k": 20, "rebalance_freq": "W", "gross_exposure": 0.8},
            },
        )

        backtest_run = PipelineRun.objects.create(
            name="mag7-backtest-lab-view",
            requested_job="backtest_strategy",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        backtest_uri = self.write_csv(
            "mag7_backtest_lab",
            ["date", "symbol", "strategy_signal", "strategy_score", "target_weight", "effective_weight", "asset_return", "realized_return"],
            [{"date": "2024-01-01", "symbol": "AAPL", "strategy_signal": 1, "strategy_score": 0.42, "target_weight": 0.5, "effective_weight": 0.5, "asset_return": 0.07, "realized_return": 0.035}],
        )
        Artifact.objects.create(
            pipeline_run=backtest_run,
            artifact_type="BACKTEST_RESULT",
            key="mag7_backtest_lab",
            uri=backtest_uri,
            content={"trades": 1, "wins": 1, "losses": 0, "avg_return": 0.01, "cumulative_return": 0.01, "days": 1, "final_equity": 1.01, "max_drawdown": 0.0},
            metadata={"source_strategy_dataset_artifact_id": strategy_artifact.id, "equity_curve": [{"date": "2024-01-01", "equity": 1.01, "net_daily_return": 0.01}]},
        )

        response = self.client.get(reverse("pipeline-strategies"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Strategy Lab")
        self.assertContains(response, reverse("pipeline-strategy-detail", args=[strategy_artifact.id]))
        self.assertContains(response, reverse("pipeline-backtest-detail", args=[backtest_run.artifacts.first().id]))
        self.assertContains(response, "AAPL")
        self.assertContains(response, strategy_definition.name)

    def test_strategy_lab_shows_saved_strategy_definitions(self):
        definitions = ensure_default_strategy_definitions()
        response = self.client.get(reverse("pipeline-strategies"))
        self.assertEqual(response.status_code, 200)
        for definition in definitions:
            self.assertContains(response, definition.name)

    def test_strategy_definition_pages_create_and_edit_saved_definitions(self):
        list_response = self.client.get(reverse("pipeline-strategy-definitions"))
        self.assertEqual(list_response.status_code, 200)
        self.assertContains(list_response, "Strategy Definitions")
        self.assertContains(list_response, "Notebook Top-K Weekly")

        create_response = self.client.post(
            reverse("pipeline-strategy-definitions"),
            {
                "name": "Custom Weekly Top K",
                "slug": "custom-weekly-top-k",
                "strategy_type": "notebook_topk_v1",
                "description": "Custom test definition",
                "gate_quantile": "0.6",
                "top_k": "3",
                "rebalance_freq": "W",
                "gross_exposure": "0.7",
                "selection_side": "long_only",
                "signal_combination": "multiply",
                "action_source_field": "",
                "action_threshold": "0.0",
                "is_active": "on",
                "advanced_config_json": "{}",
            },
        )
        self.assertEqual(create_response.status_code, 200)
        self.assertContains(create_response, "Custom Weekly Top K")

        from .models import StrategyDefinition

        definition = StrategyDefinition.objects.get(slug="custom-weekly-top-k")
        self.assertEqual(definition.config["top_k"], 3)
        self.assertEqual(definition.top_k, 3)

        edit_response = self.client.post(
            reverse("pipeline-strategy-definition-edit", args=[definition.id]),
            {
                "name": "Custom Weekly Top K Edited",
                "slug": "custom-weekly-top-k",
                "strategy_type": "notebook_topk_v1",
                "description": "Edited definition",
                "gate_quantile": "0.55",
                "top_k": "4",
                "rebalance_freq": "W",
                "gross_exposure": "0.9",
                "selection_side": "long_only",
                "signal_combination": "multiply",
                "action_source_field": "",
                "action_threshold": "0.0",
                "is_active": "on",
                "advanced_config_json": "{}",
            },
        )
        self.assertEqual(edit_response.status_code, 200)
        self.assertContains(edit_response, "Saved changes")
        definition.refresh_from_db()
        self.assertEqual(definition.name, "Custom Weekly Top K Edited")
        self.assertEqual(definition.config["top_k"], 4)
        self.assertEqual(definition.top_k, 4)

    def test_rl_strategy_definition_can_build_signed_portfolio_targets(self):
        from .models import StrategyDefinition

        rl_definition = StrategyDefinition.objects.create(
            name="RL Policy Test",
            slug="rl-policy-test",
            strategy_type="rl_policy_v1",
            description="Direct-weight RL policy",
            gate_quantile=0.0,
            top_k=7,
            rebalance_freq="D",
            gross_exposure=1.0,
            selection_side="long_short",
            signal_combination="direct",
            action_source_field="strategy_score",
            action_threshold=0.05,
            config={
                "signal_combination": "direct",
                "action_source_field": "strategy_score",
                "selection_side": "long_short",
                "gross_exposure": 1.0,
                "rebalance_freq": "D",
                "action_threshold": 0.05,
            },
            is_active=True,
        )
        feature_run = PipelineRun.objects.create(
            name="rl-feature-source",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_uri = self.write_csv(
            "rl_strategy_features",
            ["date", "symbol", "close", "ret_1"],
            [
                {"date": "2024-01-01", "symbol": "AAPL", "close": 100.0, "ret_1": 0.01},
                {"date": "2024-01-01", "symbol": "MSFT", "close": 100.0, "ret_1": -0.01},
            ],
        )
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="rl_strategy_features",
            uri=feature_uri,
            content={"rows": 2},
            metadata={},
        )
        pred_run = PipelineRun.objects.create(
            name="rl-pred-source",
            requested_job="score_regressor",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        pred_uri = self.write_csv(
            "rl_strategy_preds",
            ["date", "symbol", "signal_score", "prediction", "raw_prediction"],
            [
                {"date": "2024-01-01", "symbol": "AAPL", "signal_score": 0.8, "prediction": 0.8, "raw_prediction": 0.8},
                {"date": "2024-01-01", "symbol": "MSFT", "signal_score": -0.6, "prediction": -0.6, "raw_prediction": -0.6},
            ],
        )
        pred_artifact = Artifact.objects.create(
            pipeline_run=pred_run,
            artifact_type="REGRESSOR_PREDICTIONS",
            key="rl_strategy_preds",
            uri=pred_uri,
            content={"rows": 2},
            metadata={},
        )
        strategy_run = PipelineRun.objects.create(
            name="rl-strategy",
            requested_job="build_strategy_dataset",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
            config={"strategy_definition_id": rl_definition.id, "prediction_artifact_ids": [pred_artifact.id]},
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
        self.assertEqual(len(rows), 2)
        by_symbol = {row["symbol"]: float(row["target_weight"]) for row in rows}
        self.assertGreater(by_symbol["AAPL"], 0.0)
        self.assertLess(by_symbol["MSFT"], 0.0)

    def test_direct_strategy_definition_supports_expression_and_sign_transform(self):
        from .models import StrategyDefinition

        direct_definition = StrategyDefinition.objects.create(
            name="Direct Expression Sign Test",
            slug="direct-expression-sign-test",
            strategy_type="notebook_topk_v1",
            gate_quantile=0.0,
            top_k=4,
            rebalance_freq="D",
            gross_exposure=1.0,
            selection_side="long_short",
            signal_combination="direct",
            action_source_field="",
            action_threshold=0.0,
            config={
                "signal_combination": "direct",
                "selection_side": "long_short",
                "gross_exposure": 1.0,
                "rebalance_freq": "D",
                "combined_score_expr": "(1.0 + px__ret_252_d) / (1.0 + px__ret_21_d) - 1.0",
                "action_transform": "sign",
            },
            is_active=True,
        )
        feature_run = PipelineRun.objects.create(
            name="direct-expression-feature-source",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_uri = self.write_csv(
            "direct_expression_features",
            ["date", "symbol", "close", "ret_1", "px__ret_252_d", "px__ret_21_d"],
            [
                {
                    "date": "2024-01-01",
                    "symbol": "AAPL",
                    "close": 100.0,
                    "ret_1": 0.01,
                    "px__ret_252_d": 0.24,
                    "px__ret_21_d": 0.02,
                },
                {
                    "date": "2024-01-01",
                    "symbol": "MSFT",
                    "close": 100.0,
                    "ret_1": -0.01,
                    "px__ret_252_d": -0.18,
                    "px__ret_21_d": 0.01,
                },
            ],
        )
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="direct_expression_features",
            uri=feature_uri,
            content={"rows": 2},
            metadata={},
        )
        strategy_run = PipelineRun.objects.create(
            name="direct-expression-strategy",
            requested_job="build_strategy_dataset",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
            config={"strategy_definition_id": direct_definition.id},
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
        by_symbol = {row["symbol"]: float(row["target_weight"]) for row in rows}
        self.assertAlmostEqual(by_symbol["AAPL"], 0.5, places=8)
        self.assertAlmostEqual(by_symbol["MSFT"], -0.5, places=8)
        self.assertEqual(strategy_artifact.metadata["strategy_config"]["action_transform"], "sign")

    def test_mean_signal_combination_uses_average_score(self):
        from .models import StrategyDefinition

        mean_definition = StrategyDefinition.objects.create(
            name="Mean Score Test",
            slug="mean-score-test",
            strategy_type="notebook_topk_v1",
            gate_quantile=0.0,
            top_k=2,
            rebalance_freq="D",
            gross_exposure=1.0,
            selection_side="long_only",
            signal_combination="mean",
            config={
                "signal_combination": "mean",
                "selection_side": "long_only",
                "rebalance_freq": "D",
                "gross_exposure": 1.0,
            },
            is_active=True,
        )
        feature_run = PipelineRun.objects.create(
            name="mean-feature-source",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_uri = self.write_csv(
            "mean_strategy_features",
            ["date", "symbol", "close", "ret_1"],
            [
                {"date": "2024-01-01", "symbol": "AAPL", "close": 100.0, "ret_1": 0.01},
                {"date": "2024-01-01", "symbol": "MSFT", "close": 100.0, "ret_1": 0.01},
            ],
        )
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="mean_strategy_features",
            uri=feature_uri,
            content={"rows": 2},
            metadata={},
        )

        clf_run = PipelineRun.objects.create(
            name="mean-clf-source",
            requested_job="score_classifier",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        clf_uri = self.write_csv(
            "mean_strategy_clf",
            ["date", "symbol", "prediction_score"],
            [
                {"date": "2024-01-01", "symbol": "AAPL", "prediction_score": 0.9},
                {"date": "2024-01-01", "symbol": "MSFT", "prediction_score": 0.3},
            ],
        )
        clf_artifact = Artifact.objects.create(
            pipeline_run=clf_run,
            artifact_type="CLASSIFIER_PREDICTIONS",
            key="mean_strategy_clf",
            uri=clf_uri,
            content={"rows": 2},
            metadata={},
        )

        reg_run = PipelineRun.objects.create(
            name="mean-reg-source",
            requested_job="score_regressor",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        reg_uri = self.write_csv(
            "mean_strategy_reg",
            ["date", "symbol", "prediction"],
            [
                {"date": "2024-01-01", "symbol": "AAPL", "prediction": 0.6},
                {"date": "2024-01-01", "symbol": "MSFT", "prediction": 0.9},
            ],
        )
        reg_artifact = Artifact.objects.create(
            pipeline_run=reg_run,
            artifact_type="REGRESSOR_PREDICTIONS",
            key="mean_strategy_reg",
            uri=reg_uri,
            content={"rows": 2},
            metadata={},
        )

        ae_run = PipelineRun.objects.create(
            name="mean-ae-source",
            requested_job="score_autoencoder",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        ae_uri = self.write_csv(
            "mean_strategy_ae",
            ["date", "symbol", "prediction_score"],
            [
                {"date": "2024-01-01", "symbol": "AAPL", "prediction_score": 0.3},
                {"date": "2024-01-01", "symbol": "MSFT", "prediction_score": 0.6},
            ],
        )
        ae_artifact = Artifact.objects.create(
            pipeline_run=ae_run,
            artifact_type="AUTOENCODER_SCORES",
            key="mean_strategy_ae",
            uri=ae_uri,
            content={"rows": 2},
            metadata={},
        )

        strategy_run = PipelineRun.objects.create(
            name="mean-strategy",
            requested_job="build_strategy_dataset",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
            config={
                "strategy_definition_id": mean_definition.id,
                "prediction_artifact_ids": [clf_artifact.id, reg_artifact.id, ae_artifact.id],
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
        by_symbol = {row["symbol"]: row for row in rows}
        self.assertAlmostEqual(float(by_symbol["AAPL"]["combined_score"]), 0.6, places=6)
        self.assertAlmostEqual(float(by_symbol["MSFT"]["combined_score"]), 0.6, places=6)
        self.assertEqual(strategy_artifact.metadata["score_logic"]["signal_combination"], "mean")

    def test_autoencoder_strategy_uses_prediction_score_not_raw_prediction(self):
        mean_definition = StrategyDefinition.objects.create(
            name="AE Score Only",
            slug="ae-score-only",
            strategy_type="notebook_topk_v1",
            description="Mean-combination strategy for AE score regression test.",
            signal_combination="mean",
            top_k=2,
            gate_quantile=0.0,
        )
        feature_run = PipelineRun.objects.create(
            name="ae-score-feature-source",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="ae_score_features",
            uri=self.write_csv("ae_score_features", ["date", "symbol", "ret_1"], [{"date": "2024-01-01", "symbol": "AAPL", "ret_1": 0.01}]),
            content={"rows": 1},
            metadata={},
        )
        clf_artifact = Artifact.objects.create(
            pipeline_run=PipelineRun.objects.create(name="ae-score-clf", requested_job="score_classifier", mode=PipelineRun.Mode.STRICT, status=PipelineRun.Status.SUCCEEDED),
            artifact_type="CLASSIFIER_PREDICTIONS",
            key="ae_score_clf",
            uri=self.write_csv("ae_score_clf", ["date", "symbol", "prediction_score"], [{"date": "2024-01-01", "symbol": "AAPL", "prediction_score": 0.9}]),
            content={"rows": 1},
            metadata={},
        )
        reg_artifact = Artifact.objects.create(
            pipeline_run=PipelineRun.objects.create(name="ae-score-reg", requested_job="score_regressor", mode=PipelineRun.Mode.STRICT, status=PipelineRun.Status.SUCCEEDED),
            artifact_type="REGRESSOR_PREDICTIONS",
            key="ae_score_reg",
            uri=self.write_csv("ae_score_reg", ["date", "symbol", "prediction"], [{"date": "2024-01-01", "symbol": "AAPL", "prediction": 0.6}]),
            content={"rows": 1},
            metadata={},
        )
        ae_artifact = Artifact.objects.create(
            pipeline_run=PipelineRun.objects.create(name="ae-score-ae", requested_job="score_autoencoder", mode=PipelineRun.Mode.STRICT, status=PipelineRun.Status.SUCCEEDED),
            artifact_type="AUTOENCODER_SCORES",
            key="ae_score_ae",
            uri=self.write_csv(
                "ae_score_ae",
                ["date", "symbol", "prediction_score", "prediction"],
                [{"date": "2024-01-01", "symbol": "AAPL", "prediction_score": 0.2, "prediction": 90.0}],
            ),
            content={"rows": 1},
            metadata={},
        )
        strategy_run = PipelineRun.objects.create(
            name="ae-score-strategy",
            requested_job="build_strategy_dataset",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
            config={"strategy_definition_id": mean_definition.id, "prediction_artifact_ids": [clf_artifact.id, reg_artifact.id, ae_artifact.id]},
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
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["ae_familiarity"]), 0.2, places=6)
        self.assertAlmostEqual(float(rows[0]["combined_score"]), (0.9 + 0.6 + 0.2) / 3.0, places=6)

    def test_model_cohort_expansion_supports_grouped_feature_families_and_grouped_k_buckets(self):
        feature_run = PipelineRun.objects.create(
            name="cohort-feature-source",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_uri = self.write_csv(
            "cohort_features",
            ["date", "symbol", "close", "sma_5", "is__revenue", "isg__revenue_growth", "evt__ae_eps"],
            [
                {"date": "2024-01-01", "symbol": "AAPL", "close": 100.0, "sma_5": 99.0, "is__revenue": 10.0, "isg__revenue_growth": 0.1, "evt__ae_eps": 2.0},
            ],
        )
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="cohort_features",
            uri=feature_uri,
            content={"rows": 1},
            metadata={},
        )
        label_run = PipelineRun.objects.create(
            name="cohort-label-source",
            requested_job="labels",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        label_uri = self.write_csv(
            "cohort_labels",
            ["date", "symbol", "label", "trade_return", "k"],
            [
                {"date": "2024-01-01", "symbol": "AAPL", "label": 1, "trade_return": 0.1, "k": 1},
                {"date": "2024-01-01", "symbol": "AAPL", "label": 1, "trade_return": 0.2, "k": 2},
                {"date": "2024-01-01", "symbol": "AAPL", "label": 1, "trade_return": 0.3, "k": 4},
                {"date": "2024-01-01", "symbol": "AAPL", "label": 1, "trade_return": 0.4, "k": 8},
            ],
        )
        label_artifact = Artifact.objects.create(
            pipeline_run=label_run,
            artifact_type="LABELS",
            key="cohort_labels",
            uri=label_uri,
            content={"rows": 4},
            metadata={},
        )

        variants = expand_model_cohort_configs(
            base_config={
                "model_name": "family_horizon_model",
                "feature_family_mode": "grouped_family",
                "feature_family_groups": [["income_statement", "income_statement_growth"], ["analyst_estimates"]],
                "label_horizon_mode": "grouped_k",
                "label_k_groups": [[1, 2], [4, 8]],
            },
            feature_artifact=feature_artifact,
            label_artifact=label_artifact,
        )

        self.assertEqual(len(variants), 4)
        variant_names = {variant["model_name"] for variant in variants}
        self.assertIn("family_horizon_model__income_statement+income_statement_growth__k1-2", variant_names)
        self.assertIn("family_horizon_model__analyst_estimates__k4-8", variant_names)
        target_variant = next(
            variant for variant in variants
            if variant["feature_families"] == ["income_statement", "income_statement_growth"] and variant["label_ks"] == [1, 2]
        )
        self.assertEqual(target_variant["feature_family"], "")
        self.assertIsNone(target_variant["label_k"])

    def test_fit_and_score_record_family_group_horizon_group_and_runtime_metadata(self):
        feature_run = PipelineRun.objects.create(
            name="timing-feature-source",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_rows = []
        for symbol in ["AAPL", "MSFT"]:
            feature_rows.extend(
                [
                    {
                        "date": "2024-01-01",
                        "symbol": symbol,
                        "close": 100.0,
                        "sma_5": 99.0,
                        "evt__ae_eps": 1.5,
                        "is__revenue": 10.0,
                        "isg__revenue_growth": 0.2,
                    },
                    {
                        "date": "2024-01-02",
                        "symbol": symbol,
                        "close": 101.0,
                        "sma_5": 100.0,
                        "evt__ae_eps": 1.6,
                        "is__revenue": 10.5,
                        "isg__revenue_growth": 0.25,
                    },
                ]
            )
        feature_uri = self.write_csv(
            "timing_features",
            ["date", "symbol", "close", "sma_5", "evt__ae_eps", "is__revenue", "isg__revenue_growth"],
            feature_rows,
        )
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="timing_features",
            uri=feature_uri,
            content={"rows": len(feature_rows)},
            metadata={},
        )
        label_run = PipelineRun.objects.create(
            name="timing-label-source",
            requested_job="labels",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        label_rows = []
        for symbol in ["AAPL", "MSFT"]:
            label_rows.extend(
                [
                    {"date": "2024-01-01", "symbol": symbol, "label": 1, "market_position": 1, "trade_return": 0.12, "hold_days": 2, "side": "long", "freq": "YE", "k": 1},
                    {"date": "2024-01-02", "symbol": symbol, "label": 1, "market_position": 1, "trade_return": 0.15, "hold_days": 3, "side": "long", "freq": "YE", "k": 2},
                    {"date": "2024-01-02", "symbol": symbol, "label": 0, "market_position": -1, "trade_return": 0.03, "hold_days": 1, "side": "short", "freq": "YE", "k": 8},
                ]
            )
        label_uri = self.write_csv(
            "timing_labels",
            ["date", "symbol", "label", "market_position", "trade_return", "hold_days", "side", "freq", "k"],
            label_rows,
        )
        label_artifact = Artifact.objects.create(
            pipeline_run=label_run,
            artifact_type="LABELS",
            key="timing_labels",
            uri=label_uri,
            content={"rows": len(label_rows)},
            metadata={},
        )

        fit_run = PipelineRun.objects.create(
            name="timing-fit-regressor",
            requested_job="fit_regressor",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
            config={
                "target_col": "trade_return",
                "model_name": "timing_grouped_regressor",
                "feature_families": ["income_statement", "income_statement_growth"],
                "label_ks": [1, 2],
                "split_ratio": 1.0,
            },
        )
        with patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            model_artifact = execute_pipeline_run(
                pipeline_run=fit_run,
                target_job="fit_regressor",
                mode="strict",
                config=dict(fit_run.config or {}),
                input_artifact_ids=[feature_artifact.id, label_artifact.id],
            )

        fit_run.refresh_from_db()
        self.assertEqual(fit_run.status, PipelineRun.Status.SUCCEEDED)
        self.assertEqual(model_artifact.metadata["feature_families"], ["income_statement", "income_statement_growth"])
        self.assertEqual(model_artifact.metadata["label_ks"], [1, 2])
        self.assertEqual(model_artifact.metadata["feature_family_columns"], ["is__revenue", "isg__revenue_growth"])
        self.assertGreaterEqual(float(model_artifact.metadata["dataset_build_seconds"]), 0.0)
        self.assertGreaterEqual(float(model_artifact.metadata["fit_seconds"]), 0.0)
        self.assertGreaterEqual(float(model_artifact.metadata["train_prediction_seconds"]), 0.0)
        self.assertGreater(int(model_artifact.metadata["coverage_rows"]), 0)
        self.assertTrue(model_artifact.metadata["coverage_start_date"])
        self.assertTrue(model_artifact.metadata["coverage_end_date"])
        self.assertIn("job_duration_seconds", model_artifact.metadata)

        score_run = PipelineRun.objects.create(
            name="timing-score-regressor",
            requested_job="score_regressor",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
            config={"label_artifact_id": label_artifact.id},
        )
        with patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            prediction_artifact = execute_pipeline_run(
                pipeline_run=score_run,
                target_job="score_regressor",
                mode="strict",
                config=dict(score_run.config or {}),
                input_artifact_ids=[model_artifact.id, feature_artifact.id],
            )

        score_run.refresh_from_db()
        self.assertEqual(score_run.status, PipelineRun.Status.SUCCEEDED)
        self.assertEqual(prediction_artifact.metadata["feature_families"], ["income_statement", "income_statement_growth"])
        self.assertEqual(prediction_artifact.metadata["label_ks"], [1, 2])
        self.assertGreaterEqual(float(prediction_artifact.metadata["score_seconds"]), 0.0)
        self.assertIn("job_duration_seconds", prediction_artifact.metadata)

    def test_strategy_and_backtest_artifacts_record_runtime_metadata(self):
        strategy_definition = ensure_default_strategy_definitions()[0]
        feature_run = PipelineRun.objects.create(
            name="runtime-feature-source",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_uri = self.write_csv(
            "runtime_features",
            ["date", "symbol", "close", "ret_1"],
            [
                {"date": "2024-01-01", "symbol": "AAPL", "close": 100.0, "ret_1": 0.02},
                {"date": "2024-01-01", "symbol": "MSFT", "close": 100.0, "ret_1": -0.01},
                {"date": "2024-01-08", "symbol": "AAPL", "close": 102.0, "ret_1": 0.03},
                {"date": "2024-01-08", "symbol": "MSFT", "close": 99.0, "ret_1": 0.01},
            ],
        )
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="runtime_features",
            uri=feature_uri,
            content={"rows": 4},
            metadata={},
        )
        pred_run = PipelineRun.objects.create(
            name="runtime-pred-source",
            requested_job="score_regressor",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        pred_uri = self.write_csv(
            "runtime_preds",
            ["date", "symbol", "signal_score", "prediction", "raw_prediction"],
            [
                {"date": "2024-01-01", "symbol": "AAPL", "signal_score": 0.8, "prediction": 0.8, "raw_prediction": 0.8},
                {"date": "2024-01-01", "symbol": "MSFT", "signal_score": 0.2, "prediction": 0.2, "raw_prediction": 0.2},
                {"date": "2024-01-08", "symbol": "AAPL", "signal_score": 0.9, "prediction": 0.9, "raw_prediction": 0.9},
                {"date": "2024-01-08", "symbol": "MSFT", "signal_score": 0.1, "prediction": 0.1, "raw_prediction": 0.1},
            ],
        )
        pred_artifact = Artifact.objects.create(
            pipeline_run=pred_run,
            artifact_type="REGRESSOR_PREDICTIONS",
            key="runtime_preds",
            uri=pred_uri,
            content={"rows": 4},
            metadata={},
        )

        strategy_run = PipelineRun.objects.create(
            name="runtime-strategy",
            requested_job="build_strategy_dataset",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
            config={"strategy_definition_id": strategy_definition.id, "prediction_artifact_ids": [pred_artifact.id]},
        )
        with patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            strategy_artifact = execute_pipeline_run(
                pipeline_run=strategy_run,
                target_job="build_strategy_dataset",
                mode="strict",
                config=dict(strategy_run.config or {}),
                input_artifact_ids=[feature_artifact.id],
            )

        self.assertGreaterEqual(float(strategy_artifact.metadata["strategy_build_seconds"]), 0.0)
        self.assertIn("job_duration_seconds", strategy_artifact.metadata)

        backtest_run = PipelineRun.objects.create(
            name="runtime-backtest",
            requested_job="backtest_strategy",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
            config={"transaction_cost_bps": 10.0},
        )
        with patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            backtest_artifact = execute_pipeline_run(
                pipeline_run=backtest_run,
                target_job="backtest_strategy",
                mode="strict",
                config=dict(backtest_run.config or {}),
                input_artifact_ids=[strategy_artifact.id],
            )

        self.assertGreaterEqual(float(backtest_artifact.metadata["backtest_seconds"]), 0.0)
        self.assertIn("job_duration_seconds", backtest_artifact.metadata)

    def test_cohort_runner_writes_variant_comparison_summary(self):
        feature_run = PipelineRun.objects.create(
            name="cohort-runner-feature-source",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_rows = []
        label_rows = []
        for symbol_index, symbol in enumerate(MAG7_SYMBOLS, start=1):
            for offset in range(5):
                date_value = f"2024-01-0{offset + 1}"
                feature_rows.append(
                    {
                        "date": date_value,
                        "symbol": symbol,
                        "close": 100.0 + symbol_index + offset,
                        "sma_5": 99.0 + symbol_index + offset,
                        "is__revenue": 10.0 + symbol_index + offset,
                        "isg__revenue_growth": 0.1 + offset * 0.01,
                        "ret_1": 0.01 * symbol_index,
                    }
                )
                label_rows.append(
                    {
                        "date": date_value,
                        "symbol": symbol,
                        "label": 1 if (symbol_index + offset) % 2 == 0 else 0,
                        "market_position": 1 if (symbol_index + offset) % 2 == 0 else -1,
                        "trade_return": 0.02 * symbol_index + 0.01 * offset,
                        "hold_days": 1 + offset,
                        "side": "long" if (symbol_index + offset) % 2 == 0 else "short",
                        "freq": "YE",
                        "k": 1 if offset % 2 == 0 else 2,
                    }
                )
        feature_uri = self.write_csv(
            "cohort_runner_features",
            ["date", "symbol", "close", "sma_5", "is__revenue", "isg__revenue_growth", "ret_1"],
            feature_rows,
        )
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="cohort_runner_features",
            uri=feature_uri,
            content={"rows": len(feature_rows)},
            metadata={},
        )
        label_run = PipelineRun.objects.create(
            name="cohort-runner-label-source",
            requested_job="labels",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        label_uri = self.write_csv(
            "cohort_runner_labels",
            ["date", "symbol", "label", "market_position", "trade_return", "hold_days", "side", "freq", "k"],
            label_rows,
        )
        label_artifact = Artifact.objects.create(
            pipeline_run=label_run,
            artifact_type="LABELS",
            key="cohort_runner_labels",
            uri=label_uri,
            content={"rows": len(label_rows)},
            metadata={},
        )

        with patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            summary = run_model_cohort_backtests(
                symbols=MAG7_SYMBOLS,
                fit_job="fit_regressor",
                base_model_config={
                    "model_name": "mag7_test_cohort",
                    "split_ratio": 1.0,
                    "feature_family_mode": "grouped_family",
                    "feature_family_groups": [["prices_div_adj"], ["income_statement", "income_statement_growth"]],
                    "label_horizon_mode": "grouped_k",
                    "label_k_groups": [[1], [2]],
                },
                train_end_date="2024-01-03",
                backtest_start_date="2024-01-04",
                backtest_end_date="2024-01-05",
                feature_artifact=feature_artifact,
                label_artifact=label_artifact,
                strategy_definition_name="MAG7 Test Cohort Strategy",
                strategy_definition_slug="mag7-test-cohort-strategy",
                strategy_config={
                    "gate_quantile": 0.5,
                    "top_k": 3,
                    "rebalance_freq": "W",
                    "gross_exposure": 0.8,
                    "selection_side": "long_only",
                },
                transaction_cost_bps=10.0,
                output_basename="mag7_test_cohort_summary",
            )

        self.assertIn("summary_rows", summary)
        self.assertEqual(len(summary["summary_rows"]), 4)
        json_path = Path(summary["summary_json_path"])
        csv_path = Path(summary["summary_csv_path"])
        self.assertTrue(json_path.exists())
        self.assertTrue(csv_path.exists())

        row = summary["summary_rows"][0]
        self.assertIn("feature_families", row)
        self.assertIn("label_ks", row)
        self.assertIn("fit_seconds", row)
        self.assertIn("backtest_seconds", row)
        self.assertIn("final_equity", row)
        self.assertIn("benchmark_final_equity", row)
        self.assertIn("excess_cumulative_return", row)
        self.assertGreaterEqual(float(row["fit_seconds"]), 0.0)
        self.assertGreaterEqual(float(row["backtest_seconds"]), 0.0)

        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 4)
        self.assertTrue(any("income_statement" in row["feature_families"] for row in rows))

    def test_walk_forward_cohort_runner_writes_aggregate_summary(self):
        feature_run = PipelineRun.objects.create(
            name="wf-feature-source",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        label_run = PipelineRun.objects.create(
            name="wf-label-source",
            requested_job="labels",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_rows = []
        label_rows = []
        for symbol_index, symbol in enumerate(MAG7_SYMBOLS[:3], start=1):
            for offset in range(6):
                date_value = f"2024-01-0{offset + 1}"
                feature_rows.append(
                    {
                        "date": date_value,
                        "symbol": symbol,
                        "close": 100.0 + symbol_index + offset,
                        "sma_5": 99.0 + symbol_index + offset,
                        "ret_1": 0.01 * symbol_index,
                    }
                )
                label_rows.append(
                    {
                        "date": date_value,
                        "symbol": symbol,
                        "label": 1,
                        "market_position": 1,
                        "trade_return": 0.01 * symbol_index + 0.005 * offset,
                        "hold_days": 1,
                        "side": "long",
                        "freq": "YE",
                        "k": 1,
                    }
                )
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="wf_features",
            uri=self.write_csv("wf_features", ["date", "symbol", "close", "sma_5", "ret_1"], feature_rows),
            content={"rows": len(feature_rows)},
            metadata={},
        )
        label_artifact = Artifact.objects.create(
            pipeline_run=label_run,
            artifact_type="LABELS",
            key="wf_labels",
            uri=self.write_csv("wf_labels", ["date", "symbol", "label", "market_position", "trade_return", "hold_days", "side", "freq", "k"], label_rows),
            content={"rows": len(label_rows)},
            metadata={},
        )

        with patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            summary = run_walk_forward_model_cohort_backtests(
                symbols=MAG7_SYMBOLS[:3],
                fit_job="fit_regressor",
                base_model_config={
                    "model_name": "wf_test",
                    "split_ratio": 1.0,
                    "feature_family_mode": "all_features",
                    "label_horizon_mode": "grouped_k",
                    "label_k_groups": [[1]],
                },
                folds=[
                    {"name": "fold_1", "train_end_date": "2024-01-02", "backtest_start_date": "2024-01-03", "backtest_end_date": "2024-01-04"},
                    {"name": "fold_2", "train_end_date": "2024-01-03", "backtest_start_date": "2024-01-05", "backtest_end_date": "2024-01-06"},
                ],
                feature_artifact=feature_artifact,
                label_artifact=label_artifact,
                strategy_config={
                    "gate_quantile": 0.0,
                    "top_k": 2,
                    "rebalance_freq": "D",
                    "gross_exposure": 1.0,
                    "selection_side": "long_only",
                },
                validation_config={
                    "min_trained_rows": 1,
                    "min_rows_scored": 1,
                    "min_selected_rows": 1,
                    "min_trades": 1,
                    "min_benchmark_days": 1,
                    "min_valid_fold_rate": 0.5,
                },
                output_basename="wf_test_summary",
            )

        self.assertEqual(len(summary["folds"]), 2)
        self.assertEqual(len(summary["aggregate_rows"]), 1)
        aggregate = summary["aggregate_rows"][0]
        self.assertEqual(aggregate["fold_count"], 2)
        self.assertIn("benchmark_walk_forward_final_equity", aggregate)
        self.assertIn("walk_forward_excess_cumulative_return", aggregate)
        self.assertIn("passed_stability_gates", aggregate)
        self.assertIn("mean_fold_excess_cumulative_return", aggregate)
        self.assertIn("fold_excess_cumulative_return_std", aggregate)
        self.assertIn("strategy_artifact_id", aggregate)

    def test_walk_forward_direct_strategy_runner_writes_aggregate_summary(self):
        feature_run = PipelineRun.objects.create(
            name="wf-direct-feature-source",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_rows = []
        for date_value, aapl_ret, msft_ret in [
            ("2024-01-02", 0.01, -0.01),
            ("2024-01-03", 0.02, -0.02),
            ("2024-01-04", 0.03, -0.01),
            ("2024-01-05", 0.01, -0.03),
            ("2024-01-08", 0.02, -0.01),
            ("2024-01-09", 0.01, -0.02),
        ]:
            feature_rows.extend(
                [
                    {
                        "date": date_value,
                        "symbol": "AAPL",
                        "close": 100.0,
                        "ret_1": aapl_ret,
                        "px__ret_252_d": 0.20,
                        "px__ret_21_d": 0.01,
                    },
                    {
                        "date": date_value,
                        "symbol": "MSFT",
                        "close": 100.0,
                        "ret_1": msft_ret,
                        "px__ret_252_d": -0.15,
                        "px__ret_21_d": 0.01,
                    },
                ]
            )
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="wf_direct_features",
            uri=self.write_csv(
                "wf_direct_features",
                ["date", "symbol", "close", "ret_1", "px__ret_252_d", "px__ret_21_d"],
                feature_rows,
            ),
            content={"rows": len(feature_rows)},
            metadata={"feature_family_columns": {"prices_div_adj": ["close", "ret_1", "px__ret_252_d", "px__ret_21_d"]}},
        )

        with patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            summary = run_walk_forward_direct_strategy_backtests(
                symbols=["AAPL", "MSFT"],
                folds=[
                    {"name": "fold_1", "train_end_date": "2024-01-03", "backtest_start_date": "2024-01-04", "backtest_end_date": "2024-01-05"},
                    {"name": "fold_2", "train_end_date": "2024-01-05", "backtest_start_date": "2024-01-08", "backtest_end_date": "2024-01-09"},
                ],
                feature_artifact=feature_artifact,
                strategy_config={
                    "rebalance_freq": "D",
                    "gross_exposure": 1.0,
                    "selection_side": "long_short",
                    "signal_combination": "direct",
                    "combined_score_expr": "(1.0 + px__ret_252_d) / (1.0 + px__ret_21_d) - 1.0",
                    "action_transform": "sign",
                },
                validation_config={
                    "min_trained_rows": 1,
                    "min_rows_scored": 1,
                    "min_selected_rows": 1,
                    "min_trades": 1,
                    "min_benchmark_days": 1,
                    "min_valid_fold_rate": 0.5,
                },
                backtest_config={
                    "fee_bps": 0.0,
                    "slippage_bps": 0.0,
                    "execution_delay_days": 1,
                },
                output_basename="wf_direct_strategy_summary",
            )

        self.assertEqual(len(summary["folds"]), 2)
        self.assertEqual(len(summary["aggregate_rows"]), 1)
        aggregate = summary["aggregate_rows"][0]
        self.assertEqual(aggregate["fold_count"], 2)
        self.assertIn("walk_forward_cumulative_return", aggregate)
        self.assertIn("walk_forward_excess_cumulative_return", aggregate)
        self.assertIn("passed_stability_gates", aggregate)
        self.assertIn("strategy_artifact_id", aggregate)
        self.assertTrue(any("sharpe" in row for row in summary["summary_rows"]))
        self.assertIn("walk_forward_metrics", summary)
        self.assertGreater(summary["walk_forward_metrics"]["days"], 0)
        self.assertEqual(
            summary["walk_forward_metrics"]["trade_count"],
            sum(int(row.get("trades") or 0) for row in summary["summary_rows"]),
        )

    def test_time_series_momentum_report_includes_walk_forward_metrics(self):
        from pipeline.management.commands.run_time_series_momentum_research import _write_report

        report_path = self.temp_path / "time_series_momentum_report.md"
        _write_report(
            report_path=report_path,
            payload={
                "aggregate_rows": [
                    {
                        "fold_count": 2,
                        "walk_forward_cumulative_return": 0.12,
                        "walk_forward_max_drawdown": -0.08,
                        "walk_forward_excess_cumulative_return": 0.03,
                        "mean_fold_excess_cumulative_return": 0.01,
                    }
                ],
                "summary_rows": [
                    {"fold_name": "wf_2024", "cumulative_return": 0.05, "sharpe": 0.8, "trades": 10},
                    {"fold_name": "wf_2025", "cumulative_return": -0.01, "sharpe": -0.2, "trades": 12},
                ],
                "folds": [
                    {"backtest_start_date": "2024-01-01", "backtest_end_date": "2024-12-31"},
                    {"backtest_start_date": "2025-01-01", "backtest_end_date": "2025-12-31"},
                ],
                "walk_forward_metrics": {
                    "start_date": "2024-01-02",
                    "end_date": "2025-12-31",
                    "sharpe": 0.42,
                    "total_return": 0.12,
                    "final_equity": 1.12,
                    "max_drawdown": -0.09,
                    "avg_turnover": 0.03,
                    "total_turnover": 5.4,
                    "trade_count": 22,
                },
            },
            available_symbols=["AAPL", "MSFT"],
            missing_symbols=[],
            strategy_config={"rebalance_freq": "M", "gross_exposure": 1.0},
            backtest_config={
                "fee_bps": 2.0,
                "slippage_bps": 8.0,
                "short_borrow_bps_annual": 25.0,
                "execution_delay_days": 1,
            },
        )

        report_text = report_path.read_text(encoding="utf-8")
        self.assertIn("Walk-forward Sharpe ratio: 0.420", report_text)
        self.assertIn("Walk-forward total return: 12.00%", report_text)
        self.assertIn("Trade count: 22", report_text)
        self.assertIn("Train/test split: yearly walk-forward validation", report_text)

    def test_fit_model_respects_trade_return_and_hold_filters(self):
        feature_run = PipelineRun.objects.create(
            name="filter-fit-feature-source",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        label_run = PipelineRun.objects.create(
            name="filter-fit-label-source",
            requested_job="labels",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="filter_fit_features",
            uri=self.write_csv(
                "filter_fit_features",
                ["date", "symbol", "close", "ret_1"],
                [
                    {"date": "2024-01-01", "symbol": "AAPL", "close": 100.0, "ret_1": 0.01},
                    {"date": "2024-01-02", "symbol": "AAPL", "close": 101.0, "ret_1": 0.01},
                    {"date": "2024-01-01", "symbol": "MSFT", "close": 100.0, "ret_1": 0.01},
                    {"date": "2024-01-02", "symbol": "MSFT", "close": 101.0, "ret_1": 0.01},
                ],
            ),
            content={"rows": 4},
            metadata={},
        )
        label_artifact = Artifact.objects.create(
            pipeline_run=label_run,
            artifact_type="LABELS",
            key="filter_fit_labels",
            uri=self.write_csv(
                "filter_fit_labels",
                ["date", "symbol", "label", "market_position", "trade_return", "hold_days", "side", "freq", "k"],
                [
                    {"date": "2024-01-01", "symbol": "AAPL", "label": 1, "market_position": 1, "trade_return": 0.15, "hold_days": 10, "side": "long", "freq": "YE", "k": 1},
                    {"date": "2024-01-01", "symbol": "MSFT", "label": 1, "market_position": 1, "trade_return": 0.03, "hold_days": 120, "side": "long", "freq": "YE", "k": 1},
                ],
            ),
            content={"rows": 2},
            metadata={},
        )
        fit_run = PipelineRun.objects.create(
            name="filter-fit-run",
            requested_job="fit_regressor",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
            config={
                "target_col": "trade_return",
                "split_ratio": 1.0,
                "model_name": "filter_fit_model",
                "min_abs_trade_return_pct": 8.0,
                "max_hold_days": 30,
                "sample_weight_mode": "trade_return_abs",
            },
        )
        with patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            model_artifact = execute_pipeline_run(
                pipeline_run=fit_run,
                target_job="fit_regressor",
                mode="strict",
                config=dict(fit_run.config or {}),
                input_artifact_ids=[feature_artifact.id, label_artifact.id],
            )
        self.assertEqual(model_artifact.content["trained_rows"], 1)
        self.assertEqual(model_artifact.metadata["label_rows_before_trade_filters"], 2)
        self.assertEqual(model_artifact.metadata["label_rows_after_filters"], 1)
        self.assertEqual(model_artifact.metadata["sample_weight_mode"], "trade_return_abs")

    def test_optimal_trade_research_suite_writes_leaderboard(self):
        feature_run = PipelineRun.objects.create(
            name="suite-feature-source",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        label_run = PipelineRun.objects.create(
            name="suite-label-source",
            requested_job="labels",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_rows = []
        label_rows = []
        for symbol_index, symbol in enumerate(MAG7_SYMBOLS[:3], start=1):
            for offset in range(20):
                date_value = f"2024-01-{offset + 1:02d}"
                feature_rows.append(
                    {
                        "date": date_value,
                        "symbol": symbol,
                        "close": 100.0 + symbol_index + offset,
                        "ret_1": 0.01 * symbol_index,
                        "is__revenue": 10.0 + symbol_index + offset,
                        "isg__revenue_growth": 0.1 + offset * 0.01,
                        "evt__ae_revision": 0.05 * symbol_index,
                    }
                )
                label_rows.append(
                    {
                        "date": date_value,
                        "symbol": symbol,
                        "label": 1 if (offset + symbol_index) % 2 == 0 else 0,
                        "market_position": 1 if (offset + symbol_index) % 2 == 0 else -1,
                        "trade_return": 0.12 + 0.01 * symbol_index,
                        "hold_days": 10,
                        "side": "long" if (offset + symbol_index) % 2 == 0 else "short",
                        "freq": "YE",
                        "k": 1 if offset % 2 == 0 else 2,
                    }
                )
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="suite_features",
            uri=self.write_csv("suite_features", ["date", "symbol", "close", "ret_1", "is__revenue", "isg__revenue_growth", "evt__ae_revision"], feature_rows),
            content={"rows": len(feature_rows)},
            metadata={},
        )
        label_artifact = Artifact.objects.create(
            pipeline_run=label_run,
            artifact_type="LABELS",
            key="suite_labels",
            uri=self.write_csv("suite_labels", ["date", "symbol", "label", "market_position", "trade_return", "hold_days", "side", "freq", "k"], label_rows),
            content={"rows": len(label_rows)},
            metadata={},
        )
        with patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            payload = run_optimal_trade_research_suite(
                symbols=MAG7_SYMBOLS[:3],
                folds=[
                    {"name": "wf_2024", "train_end_date": "2024-01-10", "backtest_start_date": "2024-01-11", "backtest_end_date": "2024-01-20"},
                ],
                min_profit_pct=0.0,
                transaction_cost_bps=10.0,
                profile_name="small_universe_fast",
                validation_config_override={
                    "min_trained_rows": 5,
                    "min_rows_scored": 5,
                    "min_selected_rows": 2,
                    "min_trades": 2,
                    "min_benchmark_days": 2,
                    "min_valid_fold_rate": 0.5,
                },
                label_artifact=label_artifact,
                feature_artifact=feature_artifact,
                output_basename="test_optimal_trade_suite",
            )
        self.assertIn("leaderboard_rows", payload)
        self.assertGreaterEqual(len(payload["leaderboard_rows"]), 1)
        first = payload["leaderboard_rows"][0]
        self.assertIn("suite_name", first)
        self.assertIn("walk_forward_excess_cumulative_return", first)
        self.assertIn("mean_fold_excess_cumulative_return", first)
        self.assertEqual(payload["research_profile"]["name"], "small_universe_fast")

    def test_optimal_trade_research_suite_resume_uses_cached_summary(self):
        cached_path = Path("data/pipeline_artifacts/test_resume_suite.json")
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        cached_path.write_text(
            json.dumps(
                {
                    "schema_version": RESEARCH_REPORT_SCHEMA_VERSION,
                    "leaderboard_rows": [{"variant_name": "cached_variant"}],
                    "suite_outputs": [],
                    "report_summary": {"leaderboard_count": 1},
                }
            ),
            encoding="utf-8",
        )
        with patch("pipeline.research_suite.run_walk_forward_model_cohort_backtests") as runner:
            payload = run_optimal_trade_research_suite(
                symbols=MAG7_SYMBOLS[:2],
                folds=[
                    {"name": "wf_2024", "train_end_date": "2023-12-31", "backtest_start_date": "2024-01-01", "backtest_end_date": "2024-12-31"},
                ],
                min_profit_pct=12.0,
                transaction_cost_bps=10.0,
                output_basename="test_resume_suite",
                resume_existing=True,
            )
        self.assertEqual(payload["leaderboard_rows"][0]["variant_name"], "cached_variant")
        runner.assert_not_called()

    def test_optimal_trade_research_suite_ignores_stale_cached_summary(self):
        cached_path = Path("data/pipeline_artifacts/test_resume_suite_stale.json")
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        cached_path.write_text(
            json.dumps(
                {
                    "leaderboard_rows": [{"variant_name": "stale_variant"}],
                    "suite_outputs": [],
                    "report_summary": {"leaderboard_count": 1},
                }
            ),
            encoding="utf-8",
        )
        with patch("pipeline.research_suite.run_walk_forward_model_cohort_backtests") as runner:
            runner.return_value = {
                "schema_version": COHORT_SUMMARY_SCHEMA_VERSION,
                "folds": [],
                "aggregate_rows": [],
                "summary_rows": [],
            }
            payload = run_optimal_trade_research_suite(
                symbols=MAG7_SYMBOLS[:2],
                folds=[
                    {"name": "wf_2024", "train_end_date": "2023-12-31", "backtest_start_date": "2024-01-01", "backtest_end_date": "2024-12-31"},
                ],
                min_profit_pct=12.0,
                transaction_cost_bps=10.0,
                output_basename="test_resume_suite_stale",
                resume_existing=True,
            )
        self.assertEqual(payload["schema_version"], RESEARCH_REPORT_SCHEMA_VERSION)
        runner.assert_called()

    def test_run_model_cohort_backtests_reuses_cached_base_artifacts(self):
        symbols = MAG7_SYMBOLS[:2]
        universe_run = PipelineRun.objects.create(
            name="cached-universe-run",
            requested_job="universe",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        universe_artifact = Artifact.objects.create(
            pipeline_run=universe_run,
            artifact_type="UNIVERSE",
            key="cached_universe",
            uri=self.write_json("cached_universe", {"symbols": symbols, "count": len(symbols), "filters": {}}),
            content={"count": len(symbols)},
            metadata={"universe_cache_key": _stable_payload_hash({"symbols": symbols, "filters": {}})},
        )
        label_run = PipelineRun.objects.create(
            name="cached-label-run",
            requested_job="labels",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        label_artifact = Artifact.objects.create(
            pipeline_run=label_run,
            artifact_type="LABELS",
            key="cached_labels",
            uri=self.write_csv(
                "cached_labels",
                ["date", "symbol", "label", "market_position", "trade_return", "hold_days", "side", "freq", "k"],
                [{"date": "2024-01-01", "symbol": symbols[0], "label": 1, "market_position": 1, "trade_return": 0.1, "hold_days": 5, "side": "long", "freq": "YE", "k": 1}],
            ),
            content={"rows": 1},
            metadata={
                "source_universe_artifact_id": universe_artifact.id,
                "labels_cache_key": _stable_payload_hash(
                    {
                        "source_universe_artifact_id": int(universe_artifact.id),
                        "symbols": symbols,
                        "k_params": {"YE": [1]},
                        "min_profit_decimal": 0.1,
                        "buy_col": "adj_high",
                        "sell_col": "adj_low",
                        "short_col": "adj_low",
                        "cover_col": "adj_high",
                        "dedup_mode": "exact",
                    }
                ),
            },
        )
        feature_run = PipelineRun.objects.create(
            name="cached-feature-run",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="cached_features",
            uri=self.write_csv(
                "cached_features",
                ["date", "symbol", "px__adj_close"],
                [{"date": "2024-01-01", "symbol": symbols[0], "px__adj_close": 100.0}],
            ),
            content={"rows": 1},
            metadata={
                "source_universe_artifact_id": universe_artifact.id,
                "features_cache_key": _stable_payload_hash(
                    {
                        "source_universe_artifact_id": int(universe_artifact.id),
                        "symbols": symbols,
                        "feature_config": {"include_price_technicals": True},
                    }
                ),
            },
        )

        recorded_jobs: list[str] = []

        def fake_run_pipeline_job(*, name, requested_job, config=None, input_ids=None):
            recorded_jobs.append(str(requested_job))
            self.assertNotIn(requested_job, {"universe", "labels", "features"})
            run = PipelineRun.objects.create(
                name=name,
                requested_job=requested_job,
                mode=PipelineRun.Mode.STRICT,
                status=PipelineRun.Status.SUCCEEDED,
            )
            if requested_job == "fit_classifier":
                return Artifact.objects.create(
                    pipeline_run=run,
                    artifact_type="CLASSIFIER_MODEL",
                    key=f"{requested_job}_{len(recorded_jobs)}",
                    uri=self.write_json(f"{requested_job}_{len(recorded_jobs)}", {"model": "ok"}),
                    content={"trained_rows": 10},
                    metadata={
                        "dataset_build_seconds": 1.0,
                        "fit_seconds": 2.0,
                        "feature_families": ["prices_div_adj"],
                        "label_ks": [1],
                    },
                )
            if requested_job == "score_classifier":
                return Artifact.objects.create(
                    pipeline_run=run,
                    artifact_type="CLASSIFIER_PREDICTIONS",
                    key=f"{requested_job}_{len(recorded_jobs)}",
                    uri=self.write_csv(f"{requested_job}_{len(recorded_jobs)}", ["date", "symbol", "signal_score"], [{"date": "2024-01-11", "symbol": symbols[0], "signal_score": 0.8}]),
                    content={"rows": 1},
                    metadata={"score_seconds": 0.5, "rows_scored": 8},
                )
            if requested_job == "build_strategy_dataset":
                return Artifact.objects.create(
                    pipeline_run=run,
                    artifact_type="STRATEGY_DATASET",
                    key=f"{requested_job}_{len(recorded_jobs)}",
                    uri=self.write_csv(
                        f"{requested_job}_{len(recorded_jobs)}",
                        ["date", "symbol", "ret_1", "strategy_signal", "target_weight"],
                        [
                            {"date": "2024-01-11", "symbol": symbols[0], "ret_1": 0.01, "strategy_signal": 1, "target_weight": 0.5},
                            {"date": "2024-01-12", "symbol": symbols[1], "ret_1": 0.02, "strategy_signal": 1, "target_weight": 0.5},
                        ],
                    ),
                    content={"selected_rows": 4},
                    metadata={"strategy_build_seconds": 0.2},
                )
            if requested_job == "backtest_strategy":
                return Artifact.objects.create(
                    pipeline_run=run,
                    artifact_type="BACKTEST_RESULT",
                    key=f"{requested_job}_{len(recorded_jobs)}",
                    uri=self.write_csv(
                        f"{requested_job}_{len(recorded_jobs)}",
                        ["date", "equity"],
                        [{"date": "2024-01-11", "equity": 1.0}],
                    ),
                    content={"final_equity": 1.1, "cumulative_return": 0.1, "max_drawdown": -0.05, "trades": 4},
                    metadata={"backtest_seconds": 0.3, "backtest_config": {}},
                )
            raise AssertionError(f"Unexpected job {requested_job}")

        with patch("pipeline.cohort_runner.expand_model_cohort_configs", return_value=[{"model_name": "reuse_variant"}]), patch(
            "pipeline.cohort_runner._run_pipeline_job",
            side_effect=fake_run_pipeline_job,
        ):
            payload = run_model_cohort_backtests(
                symbols=symbols,
                fit_job="fit_classifier",
                base_model_config={"model_name": "reuse_variant", "min_profit_pct": 10.0, "label_ks": [1]},
                train_end_date="2024-01-10",
                backtest_start_date="2024-01-11",
                backtest_end_date="2024-01-20",
                feature_config={"include_price_technicals": True},
                validation_config={"min_trained_rows": 1, "min_rows_scored": 1, "min_selected_rows": 1, "min_trades": 1, "min_benchmark_days": 1},
                output_basename="test_cached_base_artifacts",
            )
        self.assertEqual(payload["base_artifacts"]["universe"], universe_artifact.id)
        self.assertEqual(payload["base_artifacts"]["labels"], label_artifact.id)
        self.assertEqual(payload["base_artifacts"]["features"], feature_artifact.id)
        self.assertEqual(recorded_jobs, ["fit_classifier", "score_classifier", "build_strategy_dataset", "backtest_strategy"])

    def test_cohort_comparison_page_renders_saved_summary(self):
        run = PipelineRun.objects.create(
            name="cohort-page-source",
            requested_job="backtest_strategy",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        Artifact.objects.create(
            id=103,
            pipeline_run=run,
            artifact_type="LABELS",
            key="cohort-page-labels",
            uri=self.write_csv("cohort_page_labels", ["date", "symbol"], [{"date": "2024-01-01", "symbol": "AAPL"}]),
            content={"rows": 1},
            metadata={},
        )
        Artifact.objects.create(
            id=104,
            pipeline_run=run,
            artifact_type="FEATURES",
            key="cohort-page-features",
            uri=self.write_csv("cohort_page_features", ["date", "symbol"], [{"date": "2024-01-01", "symbol": "AAPL"}]),
            content={"rows": 1},
            metadata={},
        )
        Artifact.objects.create(
            id=109,
            pipeline_run=run,
            artifact_type="REGRESSOR_PREDICTIONS",
            key="cohort-page-preds",
            uri=self.write_csv("cohort_page_preds", ["date", "symbol", "signal_score"], [{"date": "2024-01-01", "symbol": "AAPL", "signal_score": 0.5}]),
            content={"rows": 1},
            metadata={},
        )
        Artifact.objects.create(
            id=111,
            pipeline_run=run,
            artifact_type="STRATEGY_DATASET",
            key="cohort-page-strategy",
            uri=self.write_csv("cohort_page_strategy", ["date", "symbol", "strategy_signal", "target_weight"], [{"date": "2024-01-01", "symbol": "AAPL", "strategy_signal": 1, "target_weight": 0.5}]),
            content={"rows": 1},
            metadata={},
        )
        Artifact.objects.create(
            id=112,
            pipeline_run=run,
            artifact_type="BACKTEST_RESULT",
            key="cohort-page-backtest",
            uri=self.write_csv("cohort_page_backtest", ["date", "symbol", "strategy_signal", "target_weight", "effective_weight", "asset_return", "realized_return", "strategy_score"], [{"date": "2024-01-01", "symbol": "AAPL", "strategy_signal": 1, "target_weight": 0.5, "effective_weight": 0.5, "asset_return": 0.01, "realized_return": 0.005, "strategy_score": 0.5}]),
            content={"trades": 1},
            metadata={},
        )
        summary_path = self.temp_path / "mag7_cohort_backtest_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": RESEARCH_REPORT_SCHEMA_VERSION,
                    "base_artifacts": {"labels": 103, "features": 104},
                    "research_profile": {"name": "broad_universe_long_history"},
                    "report_summary": {
                        "research_profile": "broad_universe_long_history",
                        "leaderboard_count": 1,
                        "rejected_count": 0,
                        "best_variant_name": "income_statement+k1-2",
                        "best_mean_fold_excess_cumulative_return": 0.08,
                        "best_fold_excess_cumulative_return_std": 0.02,
                    },
                    "leaderboard_rows": [
                        {
                            "variant_name": "income_statement+k1-2",
                            "suite_name": "classifier_multiply",
                            "feature_families": ["income_statement", "income_statement_growth"],
                            "label_ks": [1, 2],
                            "mean_fold_excess_cumulative_return": 0.08,
                            "walk_forward_excess_cumulative_return": 0.11,
                            "walk_forward_final_equity": 1.50,
                            "walk_forward_max_drawdown": -0.20,
                            "valid_fold_rate": 1.0,
                            "fold_count": 2,
                            "valid_fold_count": 2,
                            "avg_fit_seconds": 1.2,
                            "prediction_artifact_id": 109,
                            "strategy_artifact_id": 111,
                            "backtest_artifact_id": 112,
                        }
                    ],
                    "rejected_rows": [],
                }
            ),
            encoding="utf-8",
        )
        with patch("pipeline.views.ARTIFACT_DIR", self.temp_path):
            response = self.client.get(reverse("pipeline-cohorts"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Cohort Comparisons")
        self.assertContains(response, "income_statement+k1-2")
        self.assertContains(response, "Summary Files")
        self.assertContains(response, "Mean Fold Excess")
        self.assertContains(response, "Showing 1 of 1 variants")
        self.assertContains(response, "/pipeline/strategies/111/")
        self.assertContains(response, "/pipeline/backtests/112/")
        self.assertContains(response, "prediction_artifact_id=109")

        with patch("pipeline.views.ARTIFACT_DIR", self.temp_path):
            filtered_response = self.client.get(reverse("pipeline-cohorts"), {"family": "income_statement", "k": "2", "sort": "fit_seconds"})
        self.assertEqual(filtered_response.status_code, 200)
        self.assertContains(filtered_response, 'value="income_statement" selected')
        self.assertContains(filtered_response, 'value="2" selected')
        self.assertContains(filtered_response, 'value="fit_seconds" selected')

    def test_cohort_comparison_page_hides_stale_links(self):
        summary_path = self.temp_path / "mag7_stale_cohort_backtest_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": RESEARCH_REPORT_SCHEMA_VERSION,
                    "base_artifacts": {"labels": 103, "features": 104},
                    "research_profile": {"name": "small_universe_fast"},
                    "report_summary": {
                        "research_profile": "small_universe_fast",
                        "leaderboard_count": 1,
                        "rejected_count": 0,
                    },
                    "leaderboard_rows": [
                        {
                            "variant_name": "stale-variant",
                            "feature_families": ["prices_div_adj"],
                            "label_ks": [1, 2],
                            "prediction_artifact_id": 9,
                            "strategy_artifact_id": 11,
                            "backtest_artifact_id": 14,
                            "mean_fold_excess_cumulative_return": 0.01,
                            "walk_forward_excess_cumulative_return": 0.02,
                            "walk_forward_final_equity": 1.2,
                            "walk_forward_max_drawdown": -0.1,
                        }
                    ],
                    "rejected_rows": [],
                }
            ),
            encoding="utf-8",
        )
        with patch("pipeline.views.ARTIFACT_DIR", self.temp_path):
            response = self.client.get(reverse("pipeline-cohorts"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Stale: prediction #9, strategy #11, backtest #14")
        self.assertNotContains(response, "/pipeline/backtests/14/")

    def test_cohort_comparison_page_renders_report_summary_and_rejections(self):
        summary_path = self.temp_path / "research_suite_report.json"
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": RESEARCH_REPORT_SCHEMA_VERSION,
                    "research_profile": {"name": "broad_universe_long_history"},
                    "report_summary": {
                        "research_profile": "broad_universe_long_history",
                        "leaderboard_count": 1,
                        "rejected_count": 1,
                        "best_mean_fold_excess_cumulative_return": 0.12,
                        "best_fold_excess_cumulative_return_std": 0.03,
                    },
                    "leaderboard_rows": [
                        {
                            "variant_name": "stable_bundle",
                            "suite_name": "classifier_multiply",
                            "feature_families": ["prices_div_adj", "income_statement"],
                            "label_ks": [1, 2],
                            "walk_forward_final_equity": 1.25,
                            "walk_forward_cumulative_return": 0.25,
                            "walk_forward_excess_cumulative_return": 0.10,
                            "walk_forward_max_drawdown": -0.15,
                            "mean_fold_excess_cumulative_return": 0.05,
                            "fold_excess_cumulative_return_std": 0.02,
                            "valid_fold_rate": 1.0,
                            "fold_count": 2,
                            "valid_fold_count": 2,
                            "avg_fit_seconds": 1.0,
                            "avg_backtest_seconds": 0.2,
                            "passed_stability_gates": True,
                        }
                    ],
                    "rejected_rows": [
                        {
                            "variant_name": "unstable_bundle",
                            "mean_fold_excess_cumulative_return": 0.04,
                            "fold_excess_cumulative_return_std": 0.5,
                            "valid_fold_rate": 0.5,
                            "stability_gate_reasons": ["fold_excess_std_above_max"],
                            "passed_stability_gates": False,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        with patch("pipeline.views.ARTIFACT_DIR", self.temp_path):
            response = self.client.get(reverse("pipeline-cohorts"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Rejected Variants")
        self.assertContains(response, "stable_bundle")
        self.assertContains(response, "unstable_bundle")
        self.assertContains(response, "fold_excess_std_above_max")

    def test_research_reports_page_renders_saved_report(self):
        summary_path = self.temp_path / "research_suite_report.json"
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": RESEARCH_REPORT_SCHEMA_VERSION,
                    "research_profile": {"name": "long_history"},
                    "report_summary": {
                        "research_profile": "long_history",
                        "leaderboard_count": 1,
                        "rejected_count": 0,
                        "best_mean_fold_excess_cumulative_return": 0.07,
                        "best_fold_excess_cumulative_return_std": 0.01,
                        "runtime_summary": {
                            "slowest_stage": "backtest",
                            "slowest_stage_seconds": 12.5,
                            "mean_variant_runtime_seconds": 14.0,
                            "stage_totals_seconds": {"dataset_build": 3.0, "backtest": 12.5},
                        },
                        "recommendations": ["Backtests remain the slowest stage."],
                    },
                    "leaderboard_rows": [
                        {
                            "variant_name": "macro_rates_bundle",
                            "suite_name": "regression_mean",
                            "walk_forward_final_equity": 1.3,
                            "walk_forward_excess_cumulative_return": 0.11,
                            "mean_fold_excess_cumulative_return": 0.06,
                            "fold_excess_cumulative_return_std": 0.01,
                            "valid_fold_rate": 1.0,
                            "passed_stability_gates": True,
                        }
                    ],
                    "rejected_rows": [],
                }
            ),
            encoding="utf-8",
        )
        with patch("pipeline.views.ARTIFACT_DIR", self.temp_path):
            response = self.client.get(reverse("pipeline-research-reports"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Research Reports")
        self.assertContains(response, "macro_rates_bundle")
        self.assertContains(response, "long_history")
        self.assertContains(response, "Slowest Stage")
        self.assertContains(response, "backtest")

    def test_backtest_strategy_applies_execution_delay_and_price_filter(self):
        strategy_run = PipelineRun.objects.create(
            name="realistic-backtest-source",
            requested_job="build_strategy_dataset",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        strategy_artifact = Artifact.objects.create(
            pipeline_run=strategy_run,
            artifact_type="STRATEGY_DATASET",
            key="realistic_backtest_source",
            uri=self.write_csv(
                "realistic_backtest_source",
                ["date", "symbol", "strategy_signal", "strategy_score", "target_weight", "ret_1", "close", "volume"],
                [
                    {"date": "2024-01-01", "symbol": "AAPL", "strategy_signal": 1, "strategy_score": 0.9, "target_weight": 1.0, "ret_1": 0.10, "close": 10.0, "volume": 1000},
                    {"date": "2024-01-02", "symbol": "AAPL", "strategy_signal": 1, "strategy_score": 0.9, "target_weight": 1.0, "ret_1": 0.20, "close": 10.0, "volume": 1000},
                    {"date": "2024-01-01", "symbol": "MSFT", "strategy_signal": 1, "strategy_score": 0.8, "target_weight": 1.0, "ret_1": 0.30, "close": 4.0, "volume": 1000},
                    {"date": "2024-01-02", "symbol": "MSFT", "strategy_signal": 1, "strategy_score": 0.8, "target_weight": 1.0, "ret_1": 0.40, "close": 4.0, "volume": 1000},
                ],
            ),
            content={"rows": 4},
            metadata={},
        )
        backtest_run = PipelineRun.objects.create(
            name="realistic-backtest",
            requested_job="backtest_strategy",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
            config={"execution_delay_days": 1, "min_price": 5.0, "fee_bps": 0.0, "slippage_bps": 0.0},
        )
        with patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            backtest_artifact = execute_pipeline_run(
                pipeline_run=backtest_run,
                target_job="backtest_strategy",
                mode="strict",
                config=dict(backtest_run.config or {}),
                input_artifact_ids=[strategy_artifact.id],
            )
        payload = dict(backtest_artifact.content or {})
        self.assertEqual(payload["days"], 2)
        self.assertAlmostEqual(float(payload["final_equity"]), 1.2, places=6)
        self.assertEqual(backtest_artifact.metadata["backtest_config"]["execution_delay_days"], 1)
        rows = list(csv.DictReader(Path(backtest_artifact.uri).open("r", encoding="utf-8", newline="")))
        self.assertTrue(all(row["symbol"] == "AAPL" for row in rows))

    def test_backtest_strategy_uses_prefixed_dollar_volume_for_liquidity_filter(self):
        strategy_run = PipelineRun.objects.create(
            name="prefixed-liquidity-source",
            requested_job="build_strategy_dataset",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        strategy_artifact = Artifact.objects.create(
            pipeline_run=strategy_run,
            artifact_type="STRATEGY_DATASET",
            key="prefixed_liquidity_source",
            uri=self.write_csv(
                "prefixed_liquidity_source",
                ["date", "symbol", "strategy_signal", "strategy_score", "target_weight", "ret_1", "close", "px__dollar_vol"],
                [
                    {"date": "2024-01-01", "symbol": "AAPL", "strategy_signal": 1, "strategy_score": 0.9, "target_weight": 1.0, "ret_1": 0.00, "close": 10.0, "px__dollar_vol": 20000000.0},
                    {"date": "2024-01-02", "symbol": "AAPL", "strategy_signal": 1, "strategy_score": 0.9, "target_weight": 1.0, "ret_1": 0.10, "close": 10.0, "px__dollar_vol": 20000000.0},
                ],
            ),
            content={"rows": 2},
            metadata={},
        )
        backtest_run = PipelineRun.objects.create(
            name="prefixed-liquidity-backtest",
            requested_job="backtest_strategy",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
            config={"execution_delay_days": 1, "min_dollar_volume": 10000000.0, "fee_bps": 0.0, "slippage_bps": 0.0},
        )
        with patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            backtest_artifact = execute_pipeline_run(
                pipeline_run=backtest_run,
                target_job="backtest_strategy",
                mode="strict",
                config=dict(backtest_run.config or {}),
                input_artifact_ids=[strategy_artifact.id],
            )
        payload = dict(backtest_artifact.content or {})
        self.assertEqual(payload["days"], 2)
        self.assertAlmostEqual(float(payload["final_equity"]), 1.1, places=6)

    def test_diagnostic_reports_page_renders_saved_report(self):
        summary_path = self.temp_path / "mag7_diagnostic_report.json"
        summary_path.write_text(
            json.dumps(
                {
                    "kind": "diagnostic_report",
                    "observations": ["Observed concentrated alpha."],
                    "recommendations": ["Increase breadth before promotion."],
                    "candidate_rule": {"rows": 18, "win_rate": 1.0, "prob_buy_gte": 0.93, "pred_rf_reg_gte": 0.46, "ae_familiarity_gte": 0.27},
                    "backtest_summary": {"final_equity": 1.8},
                    "backtest_config": {"execution_delay_days": 1, "fee_bps": 5.0, "slippage_bps": 10.0, "min_price": 5.0, "min_dollar_volume": 10000000.0},
                    "best_rl_result": {"combined_total_return_pct": 25.0},
                    "ae_signal_bug_check": {"raw_autoencoder_familiarity_median": 0.2, "strategy_dataset_ae_familiarity_median": 0.3},
                    "prediction_quantiles": {
                        "combined_rank_mean": [{"bucket": "(0.8,1.0]", "rows": 10, "win_rate": 1.0, "avg_trade_return": 0.5}],
                        "regressor_trade_return": [{"bucket": "(0.8,1.0]", "rows": 10, "win_rate": 1.0, "avg_trade_return": 0.4}],
                    },
                    "rl_results": [{"algorithm": "ppo", "eligibility_quantile": 0.5, "max_stocks_per_day": 3, "combined_total_return_pct": 25.0, "combined_sharpe": 1.2, "combined_max_drawdown_pct": -20.0, "executed_buys": 10}],
                }
            ),
            encoding="utf-8",
        )
        with patch("pipeline.views.ARTIFACT_DIR", self.temp_path):
            response = self.client.get(reverse("pipeline-diagnostic-reports"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Diagnostics That Tell You What To Try Next")
        self.assertContains(response, "Increase breadth before promotion.")
        self.assertContains(response, "Observed concentrated alpha.")
        self.assertContains(response, "25.0%")

    def test_build_oracle_trade_report_summarizes_models_clusters_and_families(self):
        label_run = PipelineRun.objects.create(
            name="oracle-labels",
            requested_job="labels",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        label_artifact = Artifact.objects.create(
            pipeline_run=label_run,
            artifact_type="LABELS",
            key="oracle_labels",
            uri=self.write_csv(
                "oracle_labels",
                ["date", "symbol", "trade_return", "hold_days", "side", "freq", "k"],
                [
                    {"date": "2024-01-01", "symbol": "AAPL", "trade_return": 0.10, "hold_days": 5, "side": "long", "freq": "YE", "k": 1},
                    {"date": "2024-01-02", "symbol": "MSFT", "trade_return": 0.25, "hold_days": 18, "side": "long", "freq": "YE", "k": 2},
                    {"date": "2024-01-03", "symbol": "NVDA", "trade_return": 0.40, "hold_days": 60, "side": "long", "freq": "YE", "k": 4},
                ],
            ),
            content={"rows": 3},
            metadata={},
        )
        model_run = PipelineRun.objects.create(
            name="oracle-model",
            requested_job="fit_classifier",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        model_artifact = Artifact.objects.create(
            pipeline_run=model_run,
            artifact_type="CLASSIFIER_MODEL",
            key="oracle_model",
            uri=self.write_json("oracle_model", {"model": "ok"}),
            content={"name": "oracle_clf"},
            metadata={
                "feature_families": ["prices_div_adj", "income_statement"],
                "oracle_cluster_scope": "specialist",
                "oracle_cluster_keys": ["long|YE|k=1|hold_1_10|(0, 1]"],
            },
        )
        prediction_run = PipelineRun.objects.create(
            name="oracle-preds",
            requested_job="score_classifier",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        prediction_artifact = Artifact.objects.create(
            pipeline_run=prediction_run,
            artifact_type="CLASSIFIER_PREDICTIONS",
            key="oracle_preds",
            uri=self.write_csv(
                "oracle_preds",
                ["date", "symbol", "signal_score"],
                [
                    {"date": "2024-01-01", "symbol": "AAPL", "signal_score": 0.95},
                    {"date": "2024-01-02", "symbol": "MSFT", "signal_score": 0.80},
                    {"date": "2024-01-03", "symbol": "NVDA", "signal_score": 0.40},
                ],
            ),
            content={"rows": 3},
            metadata={"source_model_artifact_id": model_artifact.id},
        )

        payload = build_oracle_trade_report(
            label_artifact=label_artifact,
            prediction_artifacts=[prediction_artifact],
            selection_quantile=0.5,
            top_cluster_count=5,
        )
        self.assertEqual(payload["kind"], "oracle_trade_report")
        self.assertEqual(payload["oracle_summary"]["oracle_rows"], 3)
        self.assertEqual(len(payload["model_rows"]), 1)
        self.assertGreaterEqual(len(payload["cluster_rows"]), 1)
        self.assertGreaterEqual(len(payload["missed_cluster_rows"]), 1)
        self.assertEqual(payload["model_rows"][0]["oracle_cluster_scope"], "specialist")
        self.assertTrue(any(row["family_name"] == "prices_div_adj + income_statement" for row in payload["feature_family_rows"]))

    def test_expand_cluster_specialist_variants_adds_specialist_configs(self):
        label_run = PipelineRun.objects.create(
            name="cluster-expand-labels",
            requested_job="labels",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        label_artifact = Artifact.objects.create(
            pipeline_run=label_run,
            artifact_type="LABELS",
            key="cluster_expand_labels",
            uri=self.write_csv(
                "cluster_expand_labels",
                ["date", "symbol", "trade_return", "hold_days", "side", "freq", "k"],
                [
                    {"date": "2024-01-01", "symbol": "AAPL", "trade_return": 0.10, "hold_days": 5, "side": "long", "freq": "YE", "k": 1},
                    {"date": "2024-01-02", "symbol": "MSFT", "trade_return": 0.20, "hold_days": 15, "side": "long", "freq": "YE", "k": 2},
                    {"date": "2024-01-03", "symbol": "NVDA", "trade_return": 0.35, "hold_days": 40, "side": "long", "freq": "YE", "k": 4},
                ],
            ),
            content={"rows": 3},
            metadata={},
        )

        variants = _expand_cluster_specialist_variants(
            variant_configs=[{"model_name": "oracle_mtl__prices_div_adj", "feature_families": ["prices_div_adj"]}],
            base_model_config={"oracle_cluster_mode": "top_clusters", "oracle_cluster_top_n": 2, "oracle_cluster_min_rows": 1, "include_cluster_generalist": True},
            label_artifact=label_artifact,
            train_end_date="2024-12-31",
            fit_job="fit_mtl",
        )
        self.assertEqual(len(variants), 3)
        self.assertEqual(variants[0]["oracle_cluster_scope"], "generalist")
        self.assertEqual(variants[0]["oracle_cluster_keys"], [])
        self.assertTrue(all(variant["oracle_cluster_scope"] == "specialist" for variant in variants[1:]))
        self.assertTrue(all(variant["oracle_cluster_keys"] for variant in variants[1:]))

    def test_oracle_reports_page_renders_saved_report(self):
        summary_path = self.temp_path / "oracle_trade_report.json"
        summary_path.write_text(
            json.dumps(
                {
                    "kind": "oracle_trade_report",
                    "oracle_summary": {
                        "oracle_rows": 12,
                        "symbols": 4,
                        "clusters": 3,
                        "median_trade_return": 0.18,
                        "avg_hold_days": 21.0,
                    },
                    "observations": ["Oracle label set contains 12 recoverable trade rows."],
                    "recommendations": ["Best coverage currently comes from 'prices_div_adj'."],
                    "model_rows": [
                        {
                            "artifact_id": 101,
                            "model_name": "oracle_clf",
                            "feature_family_signature": "prices_div_adj",
                            "oracle_cluster_scope": "specialist",
                            "oracle_cluster_keys": ["long|YE|k=1|hold_1_10|(0, 1]"],
                            "selected_rows": 4,
                            "oracle_recall": 0.33,
                            "cluster_coverage_rate": 0.67,
                            "selected_avg_trade_return": 0.22,
                        }
                    ],
                    "cluster_rows": [
                        {
                            "cluster_key": "long|YE|k=1|hold_1_10|(0, 1]",
                            "oracle_rows": 4,
                            "avg_trade_return": 0.22,
                            "avg_hold_days": 8.0,
                            "best_model_name": "oracle_clf",
                            "best_cluster_recall": 0.50,
                        }
                    ],
                    "missed_cluster_rows": [
                        {
                            "cluster_key": "long|YE|k=4|hold_31_90|(2, 3]",
                            "oracle_rows": 3,
                            "best_cluster_recall": 0.0,
                            "miss_rate": 1.0,
                        }
                    ],
                    "model_overlap_rows": [
                        {
                            "left_model_name": "oracle_clf",
                            "right_model_name": "oracle_reg",
                            "shared_selected_rows": 2,
                            "union_selected_rows": 4,
                            "jaccard_overlap": 0.50,
                        }
                    ],
                    "feature_family_rows": [
                        {
                            "family_kind": "family",
                            "family_name": "prices_div_adj",
                            "models": 1,
                            "avg_oracle_recall": 0.33,
                            "avg_cluster_coverage_rate": 0.67,
                            "avg_selected_trade_return": 0.22,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        with patch("pipeline.views.ARTIFACT_DIR", self.temp_path):
            response = self.client.get(reverse("pipeline-oracle-reports"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Which Models Recover Which Optimal Trades")
        self.assertContains(response, "oracle_clf")
        self.assertContains(response, "Price / Technical")
        self.assertContains(response, "Best-Recovered Trade Subtypes")
        self.assertContains(response, "Largest Oracle Gaps")
        self.assertContains(response, "Redundant Selections")

    def test_run_oracle_trade_report_command_writes_report(self):
        label_run = PipelineRun.objects.create(
            name="oracle-command-labels",
            requested_job="labels",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        label_artifact = Artifact.objects.create(
            pipeline_run=label_run,
            artifact_type="LABELS",
            key="oracle_command_labels",
            uri=self.write_csv(
                "oracle_command_labels",
                ["date", "symbol", "trade_return", "hold_days", "side", "freq", "k"],
                [
                    {"date": "2024-01-01", "symbol": "AAPL", "trade_return": 0.10, "hold_days": 5, "side": "long", "freq": "YE", "k": 1},
                    {"date": "2024-01-02", "symbol": "MSFT", "trade_return": 0.20, "hold_days": 15, "side": "long", "freq": "YE", "k": 2},
                ],
            ),
            content={"rows": 2},
            metadata={},
        )
        model_run = PipelineRun.objects.create(
            name="oracle-command-model",
            requested_job="fit_regressor",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        model_artifact = Artifact.objects.create(
            pipeline_run=model_run,
            artifact_type="REGRESSOR_MODEL",
            key="oracle_command_model",
            uri=self.write_json("oracle_command_model", {"model": "ok"}),
            content={"name": "oracle_reg"},
            metadata={"feature_families": ["prices_div_adj", "economic_indicators"]},
        )
        prediction_run = PipelineRun.objects.create(
            name="oracle-command-preds",
            requested_job="score_regressor",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        prediction_artifact = Artifact.objects.create(
            pipeline_run=prediction_run,
            artifact_type="REGRESSOR_PREDICTIONS",
            key="oracle_command_preds",
            uri=self.write_csv(
                "oracle_command_preds",
                ["date", "symbol", "prediction"],
                [
                    {"date": "2024-01-01", "symbol": "AAPL", "prediction": 0.15},
                    {"date": "2024-01-02", "symbol": "MSFT", "prediction": 0.25},
                ],
            ),
            content={"rows": 2},
            metadata={"source_model_artifact_id": model_artifact.id},
        )
        out = StringIO()
        call_command(
            "run_oracle_trade_report",
            labels=label_artifact.id,
            prediction_artifacts=str(prediction_artifact.id),
            output_basename="test_oracle_trade_report",
            stdout=out,
        )
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["kind"], "oracle_trade_report")
        self.assertTrue(Path(payload["report_path"]).exists())

    def test_run_oracle_trade_report_command_accepts_mtl_predictions(self):
        label_run = PipelineRun.objects.create(
            name="oracle-command-labels-mtl",
            requested_job="labels",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        label_artifact = Artifact.objects.create(
            pipeline_run=label_run,
            artifact_type="LABELS",
            key="oracle_command_labels_mtl",
            uri=self.write_csv(
                "oracle_command_labels_mtl",
                ["date", "symbol", "trade_return", "hold_days", "side", "freq", "k"],
                [
                    {"date": "2024-01-01", "symbol": "AAPL", "trade_return": 0.10, "hold_days": 5, "side": "long", "freq": "YE", "k": 1},
                    {"date": "2024-01-02", "symbol": "MSFT", "trade_return": 0.20, "hold_days": 15, "side": "long", "freq": "YE", "k": 2},
                ],
            ),
            content={"rows": 2},
            metadata={},
        )
        model_run = PipelineRun.objects.create(
            name="oracle-command-model-mtl",
            requested_job="fit_mtl",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        model_artifact = Artifact.objects.create(
            pipeline_run=model_run,
            artifact_type="MTL_MODEL",
            key="oracle_command_model_mtl",
            uri=self.write_json("oracle_command_model_mtl", {"model": "ok"}),
            content={"name": "oracle_mtl"},
            metadata={"feature_families": ["prices_div_adj", "income_statement", "economic_indicators"]},
        )
        prediction_run = PipelineRun.objects.create(
            name="oracle-command-preds-mtl",
            requested_job="score_mtl",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        prediction_artifact = Artifact.objects.create(
            pipeline_run=prediction_run,
            artifact_type="MTL_PREDICTIONS",
            key="oracle_command_preds_mtl",
            uri=self.write_csv(
                "oracle_command_preds_mtl",
                ["date", "symbol", "prediction_score", "mtl_prob_buy", "mtl_trade_return"],
                [
                    {"date": "2024-01-01", "symbol": "AAPL", "prediction_score": 0.91, "mtl_prob_buy": 0.91, "mtl_trade_return": 0.12},
                    {"date": "2024-01-02", "symbol": "MSFT", "prediction_score": 0.88, "mtl_prob_buy": 0.88, "mtl_trade_return": 0.25},
                ],
            ),
            content={"rows": 2},
            metadata={"source_model_artifact_id": model_artifact.id},
        )
        out = StringIO()
        call_command(
            "run_oracle_trade_report",
            labels=label_artifact.id,
            prediction_artifacts=str(prediction_artifact.id),
            output_basename="test_oracle_trade_report_mtl",
            stdout=out,
        )
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["kind"], "oracle_trade_report")
        self.assertEqual(payload["model_rows"][0]["artifact_type"], "MTL_PREDICTIONS")

    def test_run_feature_family_attribution_suite_writes_oracle_lifts(self):
        label_run = PipelineRun.objects.create(
            name="attr-labels",
            requested_job="labels",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        label_artifact = Artifact.objects.create(
            pipeline_run=label_run,
            artifact_type="LABELS",
            key="attr_labels",
            uri=self.write_csv(
                "attr_labels",
                ["date", "symbol", "trade_return", "hold_days", "side", "freq", "k"],
                [
                    {"date": "2024-01-01", "symbol": "AAPL", "trade_return": 0.12, "hold_days": 5, "side": "long", "freq": "YE", "k": 1},
                    {"date": "2024-01-02", "symbol": "MSFT", "trade_return": 0.22, "hold_days": 15, "side": "long", "freq": "YE", "k": 2},
                    {"date": "2024-01-03", "symbol": "NVDA", "trade_return": 0.30, "hold_days": 35, "side": "long", "freq": "YE", "k": 4},
                ],
            ),
            content={"rows": 3},
            metadata={},
        )

        baseline_model_run = PipelineRun.objects.create(
            name="attr-model-baseline",
            requested_job="fit_classifier",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        baseline_model_artifact = Artifact.objects.create(
            pipeline_run=baseline_model_run,
            artifact_type="CLASSIFIER_MODEL",
            key="attr_model_baseline",
            uri=self.write_json("attr_model_baseline", {"model": "ok"}),
            content={"name": "oracle_attr_clf__prices_div_adj__k1-2-4-8"},
            metadata={"feature_families": ["prices_div_adj"]},
        )
        rich_model_run = PipelineRun.objects.create(
            name="attr-model-rich",
            requested_job="fit_classifier",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        rich_model_artifact = Artifact.objects.create(
            pipeline_run=rich_model_run,
            artifact_type="CLASSIFIER_MODEL",
            key="attr_model_rich",
            uri=self.write_json("attr_model_rich", {"model": "ok"}),
            content={"name": "oracle_attr_clf__prices_div_adj+income_statement+economic_indicators__k1-2-4-8"},
            metadata={"feature_families": ["prices_div_adj", "income_statement", "economic_indicators"]},
        )

        baseline_prediction_run = PipelineRun.objects.create(
            name="attr-preds-baseline",
            requested_job="score_classifier",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        baseline_prediction_artifact = Artifact.objects.create(
            pipeline_run=baseline_prediction_run,
            artifact_type="CLASSIFIER_PREDICTIONS",
            key="attr_preds_baseline",
            uri=self.write_csv(
                "attr_preds_baseline",
                ["date", "symbol", "signal_score"],
                [
                    {"date": "2024-01-01", "symbol": "AAPL", "signal_score": 0.95},
                    {"date": "2024-01-02", "symbol": "MSFT", "signal_score": 0.35},
                    {"date": "2024-01-03", "symbol": "NVDA", "signal_score": 0.20},
                ],
            ),
            content={"rows": 3},
            metadata={"source_model_artifact_id": baseline_model_artifact.id},
        )
        rich_prediction_run = PipelineRun.objects.create(
            name="attr-preds-rich",
            requested_job="score_classifier",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        rich_prediction_artifact = Artifact.objects.create(
            pipeline_run=rich_prediction_run,
            artifact_type="CLASSIFIER_PREDICTIONS",
            key="attr_preds_rich",
            uri=self.write_csv(
                "attr_preds_rich",
                ["date", "symbol", "signal_score"],
                [
                    {"date": "2024-01-01", "symbol": "AAPL", "signal_score": 0.93},
                    {"date": "2024-01-02", "symbol": "MSFT", "signal_score": 0.88},
                    {"date": "2024-01-03", "symbol": "NVDA", "signal_score": 0.85},
                ],
            ),
            content={"rows": 3},
            metadata={"source_model_artifact_id": rich_model_artifact.id},
        )

        fake_cohort_payload = {
            "schema_version": COHORT_SUMMARY_SCHEMA_VERSION,
            "base_artifacts": {"labels": label_artifact.id, "features": 0, "universe": 0},
            "summary_rows": [
                {"variant_name": "oracle_attr_clf__prices_div_adj__k1-2-4-8", "prediction_artifact_id": baseline_prediction_artifact.id},
                {"variant_name": "oracle_attr_clf__prices_div_adj+income_statement+economic_indicators__k1-2-4-8", "prediction_artifact_id": rich_prediction_artifact.id},
            ],
            "aggregate_rows": [
                {
                    "variant_name": "oracle_attr_clf__prices_div_adj__k1-2-4-8",
                    "feature_families": ["prices_div_adj"],
                    "mean_fold_excess_cumulative_return": 0.04,
                    "walk_forward_excess_cumulative_return": 0.05,
                    "passed_stability_gates": True,
                },
                {
                    "variant_name": "oracle_attr_clf__prices_div_adj+income_statement+economic_indicators__k1-2-4-8",
                    "feature_families": ["prices_div_adj", "income_statement", "economic_indicators"],
                    "mean_fold_excess_cumulative_return": 0.07,
                    "walk_forward_excess_cumulative_return": 0.09,
                    "passed_stability_gates": True,
                },
            ],
            "summary_json_path": str(self.temp_path / "attr_cohort.json"),
        }

        with patch("analysis.feature_attribution.run_walk_forward_model_cohort_backtests", return_value=fake_cohort_payload):
            payload = run_feature_family_attribution_suite(
                symbols=MAG7_SYMBOLS[:3],
                folds=[{"name": "wf_2024", "train_end_date": "2023-12-31", "backtest_start_date": "2024-01-01", "backtest_end_date": "2024-12-31"}],
                fit_job="fit_classifier",
                base_model_config={"model_name": "oracle_attr_clf"},
                feature_family_groups=[["prices_div_adj"], ["prices_div_adj", "income_statement", "economic_indicators"]],
                label_artifact=label_artifact,
                output_basename="test_feature_attr_suite",
            )

        self.assertEqual(payload["kind"], "feature_attribution_report")
        self.assertEqual(payload["summary"]["best_feature_family_signature"], "prices_div_adj + income_statement + economic_indicators")
        self.assertGreater(len(payload["rows"]), 0)
        self.assertGreaterEqual(payload["rows"][0]["oracle_cluster_coverage_rate"], payload["rows"][-1]["oracle_cluster_coverage_rate"])
        self.assertGreaterEqual(payload["marginal_rows"][0]["delta_oracle_cluster_coverage_rate"], 0.0)

    def test_feature_attribution_reports_page_renders_saved_report(self):
        summary_path = self.temp_path / "feature_attribution_report.json"
        summary_path.write_text(
            json.dumps(
                {
                    "kind": "feature_attribution_report",
                    "summary": {
                        "fit_job": "fit_mtl",
                        "variant_count": 2,
                        "baseline_signature": "prices_div_adj",
                        "best_feature_family_signature": "prices_div_adj + income_statement + economic_indicators",
                        "best_oracle_recall": 0.66,
                        "best_oracle_cluster_coverage_rate": 0.75,
                        "best_mean_fold_excess_cumulative_return": 0.09,
                    },
                    "recommendations": ["Promote the richer bundle into the main suite."],
                    "rows": [
                        {
                            "variant_name": "oracle_attr_mtl__prices_div_adj",
                            "fit_job": "fit_mtl",
                            "feature_family_signature": "prices_div_adj",
                            "oracle_recall": 0.33,
                            "oracle_cluster_coverage_rate": 0.40,
                            "mean_fold_excess_cumulative_return": 0.02,
                            "walk_forward_excess_cumulative_return": 0.03,
                        },
                        {
                            "variant_name": "oracle_attr_mtl__prices_div_adj+income_statement+economic_indicators",
                            "fit_job": "fit_mtl",
                            "feature_family_signature": "prices_div_adj + income_statement + economic_indicators",
                            "oracle_recall": 0.66,
                            "oracle_cluster_coverage_rate": 0.75,
                            "mean_fold_excess_cumulative_return": 0.09,
                            "walk_forward_excess_cumulative_return": 0.12,
                        },
                    ],
                    "marginal_rows": [
                        {
                            "feature_family_signature": "prices_div_adj + income_statement + economic_indicators",
                            "baseline_signature": "prices_div_adj",
                            "delta_oracle_recall": 0.33,
                            "delta_oracle_cluster_coverage_rate": 0.35,
                            "delta_mean_fold_excess_cumulative_return": 0.07,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        with patch("pipeline.views.ARTIFACT_DIR", self.temp_path):
            response = self.client.get(reverse("pipeline-feature-attribution-reports"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Which Feature Bundles Expand Recoverable Oracle Coverage")
        self.assertContains(response, "Price / Technical + Income Statement + Economic Indicators")
        self.assertContains(response, "Promote the richer bundle into the main suite.")
