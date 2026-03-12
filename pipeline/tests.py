import csv
import json
import tempfile
from datetime import timedelta
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from django.core.management import call_command
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from fmp.models import Symbol, SymbolSectionHistorical
from ml.models import ModelArtifact
from ml.multitask import derive_oracle_cluster_labels

from analysis.analog_reasoning import summarize_same_symbol_analogs
from analysis.cluster_outcomes import compute_cluster_outcome_stats
from .feature_presentation import render_feature, render_feature_family_signature, serialize_features_for_embedding
from analysis.familiarity_reasoning import summarize_familiarity, summarize_novelty_risk
from analysis.historical_outcomes import aggregate_outcome_distribution, enrich_similarity_matches_with_outcomes
from analysis.historical_situation_search import (
    build_historical_situation_search_bundle,
    search_market_state_neighbors,
    summarize_historical_outcomes,
)
from analysis.insight_composer import build_portfolio_insight, build_stock_insight
from analysis.llm_prompt_builder import build_stock_insight_prompt
from analysis.market_state import compute_market_state_embedding, load_market_state_frame, resolve_price_column
from analysis.market_insight_schema import build_market_insight_input, build_portfolio_insight_input
from analysis.model_reasoning import detect_signal_conflicts
from .models import Artifact, JobRun, PipelineRun
from analysis.opportunity_scoring import compute_opportunity_summary
from analysis.oracle_state_dataset import build_oracle_state_dataset
from analysis.research import (
    artifact_rows_for_symbol,
    build_price_chart_context,
    resolve_artifact,
)
from .services import execute_pipeline_run
from analysis.similarity_engine import build_similarity_index, find_similar_market_states
from analysis.state_representations import append_embedding_features, build_market_state_representation, export_embedding_features
from analysis.state_embedding import compute_state_embedding as compute_cluster_state_embedding
from analysis.situation_clustering import fit_market_situation_clusters, materialize_market_situation_cluster_artifact
from analysis.situation_similarity import find_nearest_clusters, find_similar_historical_states, load_market_situation_cluster_artifact
from . import service_runtime
from .artifact_support import _load_artifact_preview_rows
from .progress import ProgressReporter, progress_from_config
from .run_support import serialize_pipeline_run
from .test_support import Mag7FixtureMixin
from .universe_selection import resolve_market_cap_tier_symbols, resolve_symbol_universe
from .views import _build_equity_curve_context


class PipelineRunTests(TestCase):
    def setUp(self):
        Symbol.objects.create(symbol="AAPL")
        Symbol.objects.create(symbol="MSFT")

    def test_strict_labels_requires_universe_input(self):
        with self.assertRaises(Exception):
            call_command("run_pipeline_job", job="labels", mode="strict")

    def test_auto_build_can_run_features_with_dependencies(self):
        call_command("run_pipeline_job", job="features", mode="auto_build_missing")
        run = PipelineRun.objects.order_by("-created_at", "-id").first()
        self.assertIsNotNone(run)
        self.assertEqual(run.status, PipelineRun.Status.SUCCEEDED)
        types = list(run.artifacts.values_list("artifact_type", flat=True))
        self.assertIn("UNIVERSE", types)
        self.assertIn("FEATURES", types)
        feature_artifact = run.artifacts.filter(artifact_type="FEATURES").order_by("-created_at", "-id").first()
        self.assertIsNotNone(feature_artifact)
        self.assertIn("rows", feature_artifact.content)

    def test_strict_partial_run_using_existing_universe_artifact(self):
        call_command("run_pipeline_job", job="universe", mode="strict")
        universe_artifact = Artifact.objects.filter(artifact_type="UNIVERSE").order_by("-id").first()
        self.assertIsNotNone(universe_artifact)
        call_command(
            "run_pipeline_job",
            job="labels",
            mode="strict",
            input_artifact_ids=str(universe_artifact.id),
        )
        latest = PipelineRun.objects.order_by("-created_at", "-id").first()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.requested_job, "labels")
        self.assertEqual(latest.status, PipelineRun.Status.SUCCEEDED)

    def test_progress_reporter_persists_eta_and_legacy_symbol_fields(self):
        started_at = timezone.now() - timedelta(seconds=20)
        pipeline_run = PipelineRun.objects.create(
            name="labels-progress",
            requested_job="labels",
            status=PipelineRun.Status.RUNNING,
            started_at=started_at,
        )
        job_run = JobRun.objects.create(
            pipeline_run=pipeline_run,
            job_type="labels",
            status=JobRun.Status.RUNNING,
            started_at=started_at,
        )
        reporter = ProgressReporter(pipeline_run=pipeline_run, job_run=job_run, throttle_seconds=0.0)
        reporter.update(
            phase="build_labels",
            phase_label="Generate oracle labels",
            phase_index=1,
            phase_total=1,
            unit_label="symbols",
            total_units=100,
            completed_units=40,
            current_item="AAPL",
            force=True,
        )

        pipeline_run.refresh_from_db()
        job_run.refresh_from_db()
        progress = progress_from_config(job_run.config)
        self.assertEqual(job_run.config["progress_total_symbols"], 100)
        self.assertEqual(job_run.config["progress_completed_symbols"], 40)
        self.assertEqual(job_run.config["progress_current_symbol"], "AAPL")
        self.assertEqual(progress["unit_label"], "symbols")
        self.assertGreater(float(progress["items_per_second"] or 0.0), 0.0)
        self.assertGreater(float(progress["eta_seconds"] or 0.0), 0.0)
        self.assertEqual(progress.get("eta_basis"), "throughput")

        serialized = serialize_pipeline_run(pipeline_run)
        self.assertEqual(serialized["progress"]["current_item"], "AAPL")
        self.assertEqual(serialized["job_runs"][0]["progress"]["total_units"], 100)

    def test_progress_reporter_can_estimate_eta_from_phase_progress(self):
        started_at = timezone.now() - timedelta(seconds=30)
        pipeline_run = PipelineRun.objects.create(
            name="fit-progress",
            requested_job="fit_classifier",
            status=PipelineRun.Status.RUNNING,
            started_at=started_at,
        )
        job_run = JobRun.objects.create(
            pipeline_run=pipeline_run,
            job_type="fit_classifier",
            status=JobRun.Status.RUNNING,
            started_at=started_at,
        )
        reporter = ProgressReporter(pipeline_run=pipeline_run, job_run=job_run, throttle_seconds=0.0)
        reporter.update(
            phase="fit_model",
            phase_label="Fit model",
            phase_index=2,
            phase_total=4,
            force=True,
        )

        job_run.refresh_from_db()
        progress = progress_from_config(job_run.config)
        self.assertEqual(progress["phase"], "fit_model")
        self.assertEqual(float(progress["overall_percent_complete"]), 25.0)
        self.assertGreater(float(progress["eta_seconds"] or 0.0), 0.0)
        self.assertEqual(progress.get("eta_basis"), "overall_progress")

    def test_pipeline_run_list_includes_progress_payload(self):
        started_at = timezone.now() - timedelta(seconds=15)
        pipeline_run = PipelineRun.objects.create(
            name="features-progress",
            requested_job="features",
            status=PipelineRun.Status.RUNNING,
            started_at=started_at,
        )
        reporter = ProgressReporter(pipeline_run=pipeline_run, throttle_seconds=0.0)
        reporter.update(
            phase="build_features",
            phase_label="Build feature panel",
            phase_index=1,
            phase_total=1,
            unit_label="symbols",
            total_units=50,
            completed_units=10,
            current_item="MSFT",
            force=True,
        )

        response = self.client.get(reverse("pipeline-run-list"), {"summary": "1"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        row = next((item for item in payload["runs"] if int(item["pipeline_run_id"]) == int(pipeline_run.id)), None)
        self.assertIsNotNone(row)
        self.assertEqual(row["progress"]["phase"], "build_features")
        self.assertEqual(int(row["progress"]["completed_units"]), 10)
        self.assertEqual(str(row["progress"]["current_item"]), "MSFT")


class ArtifactStorageRuntimeTests(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.pipeline_run = PipelineRun.objects.create(
            name="artifact-storage-tests",
            requested_job="features",
            status=PipelineRun.Status.SUCCEEDED,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_frame_artifact_round_trip_and_preview_use_storage_metadata(self):
        frame = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "aapl", "value": 1.25},
                {"date": "2024-01-03", "symbol": "msft", "value": 2.50},
            ]
        )
        with patch.object(service_runtime, "ARTIFACT_DIR", self.temp_path):
            stored = service_runtime.write_frame_artifact(
                "feature_panel_sample",
                frame=frame,
                fieldnames=["date", "symbol", "value"],
            )

        artifact = Artifact.objects.create(
            pipeline_run=self.pipeline_run,
            artifact_type="FEATURES",
            key="feature-panel-sample",
            uri=stored.uri,
            content={"rows": 2},
            metadata=stored.storage_metadata(),
            payload_hash="frame-artifact-test",
        )

        loaded = service_runtime.read_frame_artifact(artifact)
        self.assertEqual(stored.storage_format, "csv")
        self.assertEqual(stored.storage_kind, "frame")
        self.assertEqual(stored.row_count, 2)
        self.assertEqual(stored.columns, ["date", "symbol", "value"])
        self.assertEqual(service_runtime.infer_artifact_storage_format(artifact), "csv")
        self.assertEqual(loaded["symbol"].tolist(), ["AAPL", "MSFT"])
        self.assertEqual(loaded["date"].dt.strftime("%Y-%m-%d").tolist(), ["2024-01-02", "2024-01-03"])

        preview_rows, _content_payload = _load_artifact_preview_rows(artifact, limit=1)
        self.assertEqual(len(preview_rows), 1)
        self.assertEqual(preview_rows[0]["symbol"], "aapl")

    def test_payload_artifact_round_trip_uses_json_storage_metadata(self):
        payload = {"symbols": ["AAPL", "MSFT"], "count": 2}
        with patch.object(service_runtime, "ARTIFACT_DIR", self.temp_path):
            stored = service_runtime.write_payload_artifact("universe_sample", payload)

        artifact = Artifact.objects.create(
            pipeline_run=self.pipeline_run,
            artifact_type="UNIVERSE",
            key="universe-sample",
            uri=stored.uri,
            content={"count": 2},
            metadata=stored.storage_metadata(),
            payload_hash="payload-artifact-test",
        )

        loaded = service_runtime.read_json_artifact(artifact)
        self.assertEqual(service_runtime.infer_artifact_storage_format(artifact), "json")
        self.assertEqual(loaded["symbols"], ["AAPL", "MSFT"])
        self.assertEqual(artifact.metadata["storage_kind"], "payload")


class CleanupPipelineHistoryTests(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_model_artifact(self, *, name: str, created_at, predictions_name: str = "") -> ModelArtifact:
        model = ModelArtifact(
            name=name,
            version=1,
            framework="sklearn",
            task_type="classification",
            target_col="label",
            feature_cols=["f1"],
            metadata={},
        )
        if predictions_name:
            predictions_path = self.temp_path / predictions_name
            predictions_path.write_text("date,symbol,prediction\n", encoding="utf-8")
            model.metadata = {"predictions_uri": str(predictions_path)}
        model.set_artifact({"name": name})
        model.save()
        ModelArtifact.objects.filter(pk=model.pk).update(created_at=created_at, updated_at=created_at)
        model.refresh_from_db()
        return model

    def _create_pipeline_run_with_artifact(
        self,
        *,
        requested_job: str,
        artifact_type: str,
        created_at,
        saved_model: ModelArtifact | None = None,
        file_suffix: str = "json",
    ) -> tuple[PipelineRun, Artifact]:
        run = PipelineRun.objects.create(
            name=f"{requested_job}-{artifact_type}",
            requested_job=requested_job,
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
            started_at=created_at,
            finished_at=created_at,
        )
        PipelineRun.objects.filter(pk=run.pk).update(created_at=created_at, updated_at=created_at)
        run.refresh_from_db()

        artifact_path = self.temp_path / f"{artifact_type.lower()}_{run.id}.{file_suffix}"
        artifact_path.write_text("{}", encoding="utf-8")
        content: dict[str, object] = {}
        metadata: dict[str, object] = {}
        if saved_model is not None:
            content["model_artifact_id"] = int(saved_model.id)
            metadata["saved_model_artifact_id"] = int(saved_model.id)
        artifact = Artifact.objects.create(
            pipeline_run=run,
            artifact_type=artifact_type,
            key=f"{artifact_type.lower()}_{run.id}",
            uri=str(artifact_path),
            content=content,
            metadata=metadata,
        )
        Artifact.objects.filter(pk=artifact.pk).update(created_at=created_at)
        artifact.refresh_from_db()
        return run, artifact

    def test_cleanup_pipeline_history_dry_run_leaves_rows_intact(self):
        now = timezone.now()
        old_created_at = now - timedelta(days=2)
        recent_created_at = now
        old_model = self._create_model_artifact(name="cleanup_old_model", created_at=old_created_at, predictions_name="old_predictions.csv")
        recent_model = self._create_model_artifact(name="cleanup_recent_model", created_at=recent_created_at, predictions_name="recent_predictions.csv")
        old_run, old_artifact = self._create_pipeline_run_with_artifact(
            requested_job="fit_classifier",
            artifact_type="CLASSIFIER_MODEL",
            created_at=old_created_at,
            saved_model=old_model,
        )
        recent_run, recent_artifact = self._create_pipeline_run_with_artifact(
            requested_job="fit_classifier",
            artifact_type="CLASSIFIER_MODEL",
            created_at=recent_created_at,
            saved_model=recent_model,
        )

        out = StringIO()
        with patch("pipeline.run_support.runtime.ARTIFACT_DIR", self.temp_path), patch(
            "pipeline.management.commands.cleanup_pipeline_history.runtime.ARTIFACT_DIR",
            self.temp_path,
        ):
            call_command(
                "cleanup_pipeline_history",
                before_date=now.date().isoformat(),
                keep_latest_per_job=1,
                dry_run=True,
                stdout=out,
            )
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["delete_pipeline_run_count"], 1)
        self.assertIn(int(old_run.id), payload["delete_pipeline_run_ids_preview"])
        self.assertTrue(PipelineRun.objects.filter(pk=old_run.pk).exists())
        self.assertTrue(Artifact.objects.filter(pk=old_artifact.pk).exists())
        self.assertTrue(ModelArtifact.objects.filter(pk=old_model.pk).exists())
        self.assertTrue(Path(old_artifact.uri).exists())
        self.assertTrue(Path((old_model.metadata or {})["predictions_uri"]).exists())
        self.assertTrue(PipelineRun.objects.filter(pk=recent_run.pk).exists())
        self.assertTrue(Artifact.objects.filter(pk=recent_artifact.pk).exists())
        self.assertTrue(ModelArtifact.objects.filter(pk=recent_model.pk).exists())

    def test_cleanup_pipeline_history_deletes_old_runs_and_orphan_models(self):
        now = timezone.now()
        old_created_at = now - timedelta(days=2)
        recent_created_at = now
        old_model = self._create_model_artifact(name="cleanup_delete_old_model", created_at=old_created_at, predictions_name="delete_old_predictions.csv")
        recent_model = self._create_model_artifact(name="cleanup_keep_recent_model", created_at=recent_created_at, predictions_name="keep_recent_predictions.csv")
        orphan_model = self._create_model_artifact(name="cleanup_orphan_model", created_at=old_created_at, predictions_name="orphan_predictions.csv")
        old_run, old_artifact = self._create_pipeline_run_with_artifact(
            requested_job="fit_classifier",
            artifact_type="CLASSIFIER_MODEL",
            created_at=old_created_at,
            saved_model=old_model,
        )
        recent_run, recent_artifact = self._create_pipeline_run_with_artifact(
            requested_job="fit_classifier",
            artifact_type="CLASSIFIER_MODEL",
            created_at=recent_created_at,
            saved_model=recent_model,
        )

        with patch("pipeline.run_support.runtime.ARTIFACT_DIR", self.temp_path), patch(
            "pipeline.management.commands.cleanup_pipeline_history.runtime.ARTIFACT_DIR",
            self.temp_path,
        ):
            call_command(
                "cleanup_pipeline_history",
                before_date=now.date().isoformat(),
                keep_latest_per_job=1,
            )

        self.assertFalse(PipelineRun.objects.filter(pk=old_run.pk).exists())
        self.assertFalse(Artifact.objects.filter(pk=old_artifact.pk).exists())
        self.assertFalse(ModelArtifact.objects.filter(pk=old_model.pk).exists())
        self.assertFalse(ModelArtifact.objects.filter(pk=orphan_model.pk).exists())
        self.assertFalse(Path(old_artifact.uri).exists())
        self.assertFalse(Path((old_model.metadata or {})["predictions_uri"]).exists())
        self.assertFalse(Path((orphan_model.metadata or {})["predictions_uri"]).exists())
        self.assertTrue(PipelineRun.objects.filter(pk=recent_run.pk).exists())
        self.assertTrue(Artifact.objects.filter(pk=recent_artifact.pk).exists())
        self.assertTrue(ModelArtifact.objects.filter(pk=recent_model.pk).exists())
        self.assertTrue(Path(recent_artifact.uri).exists())
        self.assertTrue(Path((recent_model.metadata or {})["predictions_uri"]).exists())


class PipelineResearchTests(TestCase):
    def setUp(self):
        self.symbol = Symbol.objects.create(symbol="AAPL", company_name="Apple Inc.")
        SymbolSectionHistorical.objects.create(
            symbol=self.symbol,
            section_key="prices_div_adj",
            record_key="2024-01-01",
            record_date="2024-01-01",
            payload={
                "date": "2024-01-01",
                "adjOpen": 100.0,
                "adjHigh": 101.0,
                "adjLow": 99.5,
                "adjClose": 100.5,
                "volume": 1_000_000,
            },
        )
        SymbolSectionHistorical.objects.create(
            symbol=self.symbol,
            section_key="prices_div_adj",
            record_key="2024-01-02",
            record_date="2024-01-02",
            payload={
                "date": "2024-01-02",
                "adjOpen": 100.5,
                "adjHigh": 102.0,
                "adjLow": 100.0,
                "adjClose": 101.5,
                "volume": 1_200_000,
            },
        )
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_csv(self, name: str, fieldnames: list[str], rows: list[dict]) -> str:
        path = self.temp_path / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return str(path)

    def _write_json(self, name: str, payload: dict) -> str:
        path = self.temp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def _build_insight_strategy_artifact(self, name: str = "insight_strategy_fixture") -> Artifact:
        run = PipelineRun.objects.create(
            name=f"{name}-run",
            requested_job="build_strategy_dataset",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        rows: list[dict[str, object]] = []
        dates = pd.date_range("2023-01-02", periods=240, freq="B")
        symbol_specs = [("AAPL", 118.0, 1.0), ("MSFT", 205.0, 0.75), ("NVDA", 150.0, 1.25)]
        for sym_idx, (symbol, base_price, growth_mult) in enumerate(symbol_specs):
            for idx, date in enumerate(dates):
                trend = 1.0 + (0.0023 + sym_idx * 0.0004) * idx
                wave = (((idx + sym_idx * 7) % 18) - 9) / 260.0
                close = round(base_price * trend * (1.0 + wave), 4)
                rows.append(
                    {
                        "date": str(date.date()),
                        "symbol": symbol,
                        "close": close,
                        "ret_1": round((((idx + sym_idx) % 9) - 4) / 120.0, 6),
                        "prob_buy": round(min(0.98, 0.42 + 0.0016 * idx + sym_idx * 0.07), 6),
                        "ranking": round(0.18 + 0.0011 * idx + sym_idx * 0.03, 6),
                        "ae_familiarity": round(min(0.97, 0.35 + 0.00125 * idx + (2 - sym_idx) * 0.04), 6),
                        "combined_score": round(0.22 + 0.0014 * idx + sym_idx * 0.05, 6),
                        "is__revenue": round(65_000 + idx * 42 * growth_mult + sym_idx * 2_500, 4),
                        "isg__revenue_growth": round(0.08 + 0.00045 * idx + sym_idx * 0.01, 6),
                        "evt__ae_revision": round(0.10 + 0.0006 * idx + sym_idx * 0.008, 6),
                        "econ__liquidity_proxy": round(0.55 + ((idx % 24) - 12) / 80.0, 6),
                        "tr__2y": round(0.024 + ((idx % 30) - 15) / 1800.0, 6),
                        "trade_return": round(0.09 + 0.00095 * idx + sym_idx * 0.028, 6),
                        "hold_days": 25 + (idx % 55),
                        "side": "long",
                        "freq": "YE",
                        "k": [1, 2, 4, 8][idx % 4],
                    }
                )
        strategy_uri = self._write_csv(
            name,
            [
                "date",
                "symbol",
                "close",
                "ret_1",
                "prob_buy",
                "ranking",
                "ae_familiarity",
                "combined_score",
                "is__revenue",
                "isg__revenue_growth",
                "evt__ae_revision",
                "econ__liquidity_proxy",
                "tr__2y",
                "trade_return",
                "hold_days",
                "side",
                "freq",
                "k",
            ],
            rows,
        )
        return Artifact.objects.create(
            pipeline_run=run,
            artifact_type="STRATEGY_DATASET",
            key=name,
            uri=strategy_uri,
            content={"rows": len(rows)},
            metadata={},
        )

    def _build_market_situation_artifact(self, strategy_artifact: Artifact, name: str = "market_situation_fixture") -> Artifact:
        with patch("analysis.situation_clustering.ARTIFACT_DIR", self.temp_path):
            payload = fit_market_situation_clusters(
                strategy_artifact=strategy_artifact,
                pca_components=4,
                max_clusters=6,
                min_cluster_size=20,
            )
            return materialize_market_situation_cluster_artifact(
                output_basename=name,
                clustering_payload=payload,
                strategy_artifact=strategy_artifact,
            )

    def _build_sample_market_insight_input(self):
        frame = pd.DataFrame(
            [
                {
                    "date": "2025-03-10",
                    "symbol": "NVDA",
                    "ev_dividedby_ebitda": 13.482193,
                    "revenue_growth": 0.2214,
                    "ret_20d": 0.1294,
                    "eps_revision_30d": 0.08,
                    "ranking": 0.92,
                    "prob_buy": 0.87,
                    "combined_score": 0.91,
                    "ae_familiarity": 0.58,
                },
                {
                    "date": "2025-03-03",
                    "symbol": "NVDA",
                    "ev_dividedby_ebitda": 10.0,
                    "revenue_growth": 0.12,
                    "ret_20d": 0.03,
                    "eps_revision_30d": 0.01,
                    "ranking": 0.45,
                    "prob_buy": 0.51,
                    "combined_score": 0.47,
                    "ae_familiarity": 0.22,
                },
                {
                    "date": "2025-02-10",
                    "symbol": "AMD",
                    "ev_dividedby_ebitda": 11.5,
                    "revenue_growth": 0.16,
                    "ret_20d": 0.06,
                    "eps_revision_30d": 0.02,
                    "ranking": 0.55,
                    "prob_buy": 0.61,
                    "combined_score": 0.58,
                    "ae_familiarity": 0.35,
                },
            ]
        )
        row = dict(frame.iloc[0])
        return build_market_insight_input(
            symbol="NVDA",
            as_of_date="2025-03-10",
            row=row,
            state_frame=frame,
            feature_family_map={
                "Ratios": ["ev_dividedby_ebitda"],
                "Income Statement Growth": ["revenue_growth"],
                "Price / Technical": ["ret_20d"],
                "Analyst Estimates": ["eps_revision_30d"],
                "Model Signals": ["ranking", "prob_buy", "combined_score"],
                "Novelty": ["ae_familiarity"],
            },
            same_symbol_analogs=[
                {
                    "symbol": "NVDA",
                    "date": "2023-01-06",
                    "similarity_score": 91.0,
                    "match_type": "same_symbol",
                    "return_20d": 0.18,
                    "return_60d": 0.44,
                    "return_180d": 0.62,
                    "cluster_description": "Momentum Growth Regime",
                    "explanations": [{"explanation": "momentum features"}],
                },
                {
                    "symbol": "NVDA",
                    "date": "2019-09-03",
                    "similarity_score": 89.0,
                    "match_type": "same_symbol",
                    "return_20d": 0.09,
                    "return_60d": 0.21,
                    "return_180d": 0.34,
                    "cluster_description": "Momentum Growth Regime",
                    "explanations": [{"explanation": "analyst estimate trends"}],
                },
            ],
            cross_symbol_analogs=[
                {
                    "symbol": "AMD",
                    "date": "2016-04-12",
                    "similarity_score": 88.0,
                    "match_type": "cross_symbol",
                    "return_20d": 0.15,
                    "return_60d": 0.37,
                    "return_180d": 0.49,
                    "cluster_description": "High Growth Momentum",
                    "explanations": [{"explanation": "earnings revision features"}],
                },
                {
                    "symbol": "TSLA",
                    "date": "2020-05-18",
                    "similarity_score": 86.0,
                    "match_type": "cross_symbol",
                    "return_20d": -0.02,
                    "return_60d": 0.52,
                    "return_180d": 0.77,
                    "cluster_description": "High Growth Momentum",
                    "explanations": [{"explanation": "momentum features"}],
                },
            ],
            analog_outcome_summary={
                "primary_horizon_days": 60,
                "median_return": 0.33,
                "mean_return": 0.39,
                "win_rate": 0.75,
                "worst_case": -0.12,
                "best_case": 0.54,
                "tail_risk": -0.03,
                "avg_drawdown": -0.08,
                "avg_volatility": 0.03,
                "sample_size": 4,
                "horizon_rows": [
                    {"horizon_days": 20, "median_return": 0.12, "mean_return": 0.10, "win_rate": 0.75, "worst_case": -0.02, "best_case": 0.18, "tail_risk": 0.0, "avg_drawdown": -0.05, "avg_volatility": 0.02, "sample_size": 4},
                    {"horizon_days": 60, "median_return": 0.33, "mean_return": 0.39, "win_rate": 0.75, "worst_case": -0.12, "best_case": 0.54, "tail_risk": -0.03, "avg_drawdown": -0.08, "avg_volatility": 0.03, "sample_size": 4},
                ],
            },
            same_symbol_outcome_summary={
                "primary_horizon_days": 60,
                "median_return": 0.325,
                "mean_return": 0.325,
                "win_rate": 1.0,
                "worst_case": 0.21,
                "best_case": 0.44,
                "tail_risk": 0.21,
                "avg_drawdown": -0.04,
                "avg_volatility": 0.02,
                "sample_size": 2,
                "horizon_rows": [],
            },
            cross_symbol_outcome_summary={
                "primary_horizon_days": 60,
                "median_return": 0.445,
                "mean_return": 0.445,
                "win_rate": 1.0,
                "worst_case": 0.37,
                "best_case": 0.52,
                "tail_risk": 0.37,
                "avg_drawdown": -0.10,
                "avg_volatility": 0.04,
                "sample_size": 2,
                "horizon_rows": [],
            },
            opportunity={
                "opportunity_score": 82.0,
                "confidence_score": 68.0,
                "confidence_label": "Medium-High",
                "market_familiarity_score": 58.0,
                "market_familiarity_label": "Medium-High",
                "risk_score": 31.0,
                "risk_indicator": "Moderate",
            },
            current_cluster={
                "cluster_id": "17",
                "description": "Momentum Growth Regime",
                "similarity_score_pct": 84.0,
                "feature_signature": ["prices_div_adj", "analyst_estimates"],
                "outcome_statistics": {"median_return": 0.19, "win_rate": 0.68},
                "example_historical_dates": [{"date": "2017-05-12", "symbol": "NVDA"}],
            },
            nearest_clusters=[{"cluster_id": "17", "description": "Momentum Growth Regime", "similarity_score_pct": 84.0}],
            optional_notes={"source": "unit_test"},
        )

    def test_research_helpers_filter_rows_and_build_price_context(self):
        run = PipelineRun.objects.create(
            name="research-fixture",
            requested_job="predict",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        prediction_uri = self._write_csv(
            "predictions_fixture",
            ["date", "symbol", "prediction_score", "prediction"],
            [
                {"date": "2024-01-01", "symbol": "AAPL", "prediction_score": 0.8, "prediction": 1},
                {"date": "2024-01-01", "symbol": "MSFT", "prediction_score": 0.2, "prediction": 0},
            ],
        )
        artifact = Artifact.objects.create(
            pipeline_run=run,
            artifact_type="PREDICTIONS",
            key="predictions_fixture",
            uri=prediction_uri,
            content={"rows": 2},
            metadata={},
        )

        rows = artifact_rows_for_symbol(artifact, "AAPL")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], "AAPL")

        latest = resolve_artifact(artifact_type="PREDICTIONS")
        self.assertIsNotNone(latest)
        self.assertEqual(latest.id, artifact.id)

        chart = build_price_chart_context("AAPL")
        self.assertEqual(chart["price_points_count"], 2)
        self.assertIn("2024-01-01", chart["labels_json"])

    def test_feature_presentation_registry_renders_canonical_values(self):
        self.assertEqual(render_feature("ev_dividedby_ebitda", 13.482193), "EV / EBITDA: 13.48")
        self.assertEqual(render_feature("revenue_growth", 0.2214), "Revenue Growth: 22.14%")
        self.assertEqual(render_feature("ret_5d", 0.0328), "Return (5D): 3.28%")
        self.assertEqual(
            render_feature_family_signature("prices_div_adj + income_statement"),
            "Price / Technical + Income Statement",
        )

    def test_serialize_features_for_embedding_groups_canonical_features(self):
        rendered = serialize_features_for_embedding(
            {
                "prices_div_adj": {"ret_5d": 0.0328, "ret_20d": 0.1294},
                "income_statement_growth": {"revenue_growth": 0.2214},
                "ratios": {"ev_dividedby_ebitda": 13.482193},
            }
        )
        self.assertIn("Price / Technical", rendered)
        self.assertIn("Return (5D): 3.28%", rendered)
        self.assertIn("Income Statement Growth", rendered)
        self.assertIn("Revenue Growth: 22.14%", rendered)
        self.assertIn("Ratios", rendered)
        self.assertIn("EV / EBITDA: 13.48", rendered)

    def test_reasoning_input_preserves_canonical_feature_names(self):
        insight_input = self._build_sample_market_insight_input()
        flat_names = [feature.display_name for rows in insight_input.canonical_features.values() for feature in rows]
        self.assertIn("EV / EBITDA", flat_names)
        self.assertIn("Revenue Growth", flat_names)
        self.assertIn("Return (20D)", flat_names)

    def test_analog_summaries_are_grounded_in_provided_inputs(self):
        insight_input = self._build_sample_market_insight_input()
        summary = summarize_same_symbol_analogs(insight_input.same_symbol_analogs, primary_horizon_days=60)
        self.assertEqual(summary["analog_count"], 2)
        self.assertEqual(summary["positive_count"], 2)
        self.assertIn("Same-symbol analogs were favorable", summary["summary_lines"][0])

    def test_model_conflict_detection_flags_unfamiliar_supportive_setup(self):
        insight_input = self._build_sample_market_insight_input()
        low_familiarity = insight_input.familiarity_signals.__class__(
            market_familiarity_score=30.0,
            market_familiarity_label="Low",
            confidence_score=72.0,
            confidence_label="High",
            ae_familiarity_raw=0.12,
            analog_density=2.0,
            mean_similarity=0.61,
        )
        conflicts = detect_signal_conflicts(
            insight_input.model_scores,
            {"positive_rate": 0.75},
            low_familiarity,
        )
        kinds = [item["kind"] for item in conflicts["evidence"]["conflicts"]]
        self.assertIn("favorable_but_unfamiliar", kinds)
        self.assertIn("confidence_vs_familiarity", kinds)

    def test_familiarity_reasoning_behaves_correctly(self):
        summary = summarize_familiarity(
            self._build_sample_market_insight_input().familiarity_signals.__class__(
                market_familiarity_score=82.0,
                market_familiarity_label="High",
                confidence_score=70.0,
                confidence_label="High",
                ae_familiarity_raw=0.72,
                analog_density=9.0,
                mean_similarity=0.84,
            )
        )
        novelty = summarize_novelty_risk(
            self._build_sample_market_insight_input().familiarity_signals.__class__(
                market_familiarity_score=28.0,
                market_familiarity_label="Low",
                confidence_score=41.0,
                confidence_label="Low",
                ae_familiarity_raw=0.05,
                analog_density=2.0,
                mean_similarity=0.33,
            )
        )
        self.assertIn("appears familiar", summary["summary_lines"][0])
        self.assertIn("Novelty risk is elevated", novelty["summary_lines"][0])

    def test_stock_insight_output_is_well_formed_without_llm(self):
        insight_input = self._build_sample_market_insight_input()
        stock_insight = build_stock_insight(insight_input, mode="deterministic")
        payload = stock_insight.to_dict()
        self.assertTrue(payload["headline"])
        self.assertTrue(payload["summary"])
        self.assertTrue(payload["key_drivers"])
        self.assertTrue(payload["historical_context"])
        self.assertEqual(payload["llm_prompt"], "")
        self.assertEqual(payload["mode"], "deterministic")

    def test_portfolio_insight_output_is_well_formed(self):
        portfolio_input = build_portfolio_insight_input(
            symbols=["NVDA", "MSFT", "AAPL"],
            as_of_date="2025-03-10",
            rows=[
                {"symbol": "NVDA", "opportunity_score": 82.0, "confidence_score": 68.0, "market_familiarity_score": 58.0, "risk_indicator": "Moderate", "cluster_id": "17", "cluster_description": "Momentum Growth Regime"},
                {"symbol": "MSFT", "opportunity_score": 74.0, "confidence_score": 70.0, "market_familiarity_score": 63.0, "risk_indicator": "Low", "cluster_id": "12", "cluster_description": "Quality Compounder"},
                {"symbol": "AAPL", "opportunity_score": 41.0, "confidence_score": 49.0, "market_familiarity_score": 39.0, "risk_indicator": "Elevated", "cluster_id": "3", "cluster_description": "Crowded Weakness"},
            ],
            cluster_exposure_rows=[
                {"cluster_id": "17", "cluster_description": "Momentum Growth Regime", "exposure_pct": 66.7},
                {"cluster_id": "3", "cluster_description": "Crowded Weakness", "exposure_pct": 33.3},
            ],
            portfolio_score=65.7,
            regime_similarity_score=62.3,
            risk_concentration_score=58.1,
        )
        portfolio_insight = build_portfolio_insight(portfolio_input, mode="deterministic")
        payload = portfolio_insight.to_dict()
        self.assertTrue(payload["headline"])
        self.assertEqual(payload["strongest_holdings"][0], "NVDA")
        self.assertIn("Momentum Growth Regime", payload["concentration_summary"][0])

    def test_llm_prompt_builder_uses_only_structured_evidence(self):
        insight_input = self._build_sample_market_insight_input()
        stock_insight = build_stock_insight(insight_input, mode="deterministic")
        prompt = build_stock_insight_prompt(insight_input, stock_insight)
        self.assertIn("EV / EBITDA: 13.48", prompt)
        self.assertIn("Revenue Growth: 22.14%", prompt)
        self.assertIn("Context", prompt)
        self.assertNotIn("ev_dividedby_ebitda", prompt)

    def test_strict_score_classifier_run_materializes_predictions_artifact(self):
        feature_run = PipelineRun.objects.create(
            name="feature-source",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_uri = self._write_csv(
            "features_fixture",
            ["date", "symbol", "close", "ret_1"],
            [
                {"date": "2024-01-01", "symbol": "AAPL", "close": 100.0, "ret_1": 0.10},
                {"date": "2024-01-02", "symbol": "AAPL", "close": 101.0, "ret_1": -0.20},
                {"date": "2024-01-03", "symbol": "AAPL", "close": 102.0, "ret_1": 0.05},
                {"date": "2024-01-04", "symbol": "AAPL", "close": 101.5, "ret_1": -0.03},
            ],
        )
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="features_fixture",
            uri=feature_uri,
            content={"rows": 2},
            metadata={},
        )

        label_run = PipelineRun.objects.create(
            name="label-source",
            requested_job="labels",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        label_uri = self._write_csv(
            "labels_fixture_predict",
            ["date", "symbol", "label", "market_position", "trade_return", "hold_days", "side", "freq", "k"],
            [
                {"date": "2024-01-01", "symbol": "AAPL", "label": 1, "market_position": 1, "trade_return": 0.10, "hold_days": 1, "side": "long", "freq": "W", "k": 1},
                {"date": "2024-01-02", "symbol": "AAPL", "label": 0, "market_position": 0, "trade_return": -0.20, "hold_days": 1, "side": "short", "freq": "W", "k": 1},
                {"date": "2024-01-03", "symbol": "AAPL", "label": 1, "market_position": 1, "trade_return": 0.05, "hold_days": 1, "side": "long", "freq": "W", "k": 1},
                {"date": "2024-01-04", "symbol": "AAPL", "label": 0, "market_position": 0, "trade_return": -0.03, "hold_days": 1, "side": "short", "freq": "W", "k": 1},
            ],
        )
        label_artifact = Artifact.objects.create(
            pipeline_run=label_run,
            artifact_type="LABELS",
            key="labels_fixture_predict",
            uri=label_uri,
            content={"rows": 4},
            metadata={},
        )

        fit_run = PipelineRun.objects.create(
            name="fit-classifier-fixture",
            requested_job="fit_classifier",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
            config={"params": {"n_estimators": 5}, "split_ratio": 0.5},
        )
        model_artifact = execute_pipeline_run(
            pipeline_run=fit_run,
            target_job="fit_classifier",
            mode="strict",
            config=dict(fit_run.config or {}),
            input_artifact_ids=[feature_artifact.id, label_artifact.id],
        )

        predict_run = PipelineRun.objects.create(
            name="score-classifier-fixture",
            requested_job="score_classifier",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
        )

        with patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            output_artifact = execute_pipeline_run(
                pipeline_run=predict_run,
                target_job="score_classifier",
                mode="strict",
                input_artifact_ids=[model_artifact.id, feature_artifact.id],
            )

        predict_run.refresh_from_db()
        self.assertEqual(predict_run.status, PipelineRun.Status.SUCCEEDED)
        self.assertEqual(output_artifact.artifact_type, "CLASSIFIER_PREDICTIONS")
        self.assertEqual(output_artifact.metadata["source_model_artifact_id"], model_artifact.id)
        self.assertEqual(output_artifact.metadata["source_features_artifact_id"], feature_artifact.id)
        with Path(output_artifact.uri).open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["symbol"], "AAPL")
        self.assertIn("prediction_score", rows[0])
        self.assertIn("raw_prediction", rows[0])
        self.assertIn("signal_score", rows[0])
        self.assertIn("predicted_class", rows[0])

    def test_pipeline_job_catalog_hides_legacy_jobs(self):
        response = self.client.get(reverse("pipeline-job-catalog"))
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        job_types = [row["job_type"] for row in payload["jobs"]]
        self.assertEqual(
            job_types,
            [
                "universe",
                "labels",
                "features",
                "fit_classifier",
                "fit_regressor",
                "fit_mtl",
                "score_classifier",
                "score_regressor",
                "score_mtl",
                "build_strategy_dataset",
                "backtest_strategy",
            ],
        )
        self.assertNotIn("train", job_types)


    def test_pipeline_lab_hides_legacy_model_jobs(self):
        response = self.client.get(reverse("pipeline-lab"))
        self.assertEqual(response.status_code, 200)
        html = response.content.decode()
        self.assertIn("Fit Classifier", html)
        self.assertIn("Fit Regressor", html)
        self.assertIn("Fit Multi-Task", html)
        self.assertNotIn("Fit Autoencoder", html)
        self.assertIn("Score Classifier", html)
        self.assertIn("Score Regressor", html)
        self.assertIn("Score Multi-Task", html)
        self.assertNotIn("Score Autoencoder", html)

    def test_symbol_research_view_loads_saved_artifacts(self):
        label_run = PipelineRun.objects.create(
            name="label-source",
            requested_job="labels",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        label_uri = self._write_csv(
            "labels_fixture",
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
            [
                {
                    "date": "2024-01-01",
                    "symbol": "AAPL",
                    "label": 1,
                    "market_position": 1,
                    "trade_return": 0.05,
                    "hold_days": 1,
                    "side": "long",
                    "freq": "YE",
                    "k": 1,
                    "entry_date": "2024-01-01",
                    "exit_date": "2024-01-02",
                    "entry_px": "100.0",
                    "exit_px": "105.0",
                    "ret_pct": "5.00%",
                }
            ],
        )
        label_artifact = Artifact.objects.create(
            pipeline_run=label_run,
            artifact_type="LABELS",
            key="labels_fixture",
            uri=label_uri,
            content={"rows": 1},
            metadata={},
        )

        feature_run = PipelineRun.objects.create(
            name="feature-source",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_uri = self._write_csv(
            "features_fixture",
            ["date", "symbol", "close", "ret_1", "ev_dividedby_ebitda", "revenue_growth"],
            [{"date": "2024-01-01", "symbol": "AAPL", "close": 100.5, "ret_1": 0.01, "ev_dividedby_ebitda": 12.3, "revenue_growth": 0.22}],
        )
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="features_fixture_view",
            uri=feature_uri,
            content={"rows": 1},
            metadata={},
        )

        prediction_run = PipelineRun.objects.create(
            name="prediction-source",
            requested_job="predict",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        prediction_uri = self._write_csv(
            "predictions_fixture_view",
            ["date", "symbol", "prediction_score", "prediction"],
            [{"date": "2024-01-01", "symbol": "AAPL", "prediction_score": 0.82, "prediction": 1}],
        )
        prediction_artifact = Artifact.objects.create(
            pipeline_run=prediction_run,
            artifact_type="PREDICTIONS",
            key="predictions_fixture_view",
            uri=prediction_uri,
            content={"rows": 1},
            metadata={},
        )

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
        self.assertContains(response, f"#{feature_artifact.id}")
        self.assertContains(response, f"#{prediction_artifact.id}")
        self.assertContains(response, "0.82")
        self.assertContains(response, "EV / EBITDA")
        self.assertContains(response, "Revenue Growth")

    def test_backtest_equity_curve_context_falls_back_to_csv_rows(self):
        backtest_run = PipelineRun.objects.create(
            name="backtest-source",
            requested_job="backtest_strategy",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        backtest_uri = self._write_csv(
            "backtest_rows_fixture",
            [
                "date",
                "symbol",
                "strategy_signal",
                "strategy_score",
                "target_weight",
                "effective_weight",
                "asset_return",
                "gross_exposure",
                "realized_return",
                "turnover",
                "turnover_cost",
            ],
            [
                {
                    "date": "2024-01-01",
                    "symbol": "AAPL",
                    "strategy_signal": 1,
                    "strategy_score": 0.5,
                    "target_weight": 0.8,
                    "effective_weight": 0.0,
                    "asset_return": 0.01,
                    "gross_exposure": 0.8,
                    "realized_return": 0.0,
                    "turnover": 0.8,
                    "turnover_cost": 0.0008,
                },
                {
                    "date": "2024-01-02",
                    "symbol": "AAPL",
                    "strategy_signal": 1,
                    "strategy_score": 0.4,
                    "target_weight": 0.8,
                    "effective_weight": 0.8,
                    "asset_return": 0.02,
                    "gross_exposure": 0.8,
                    "realized_return": 0.016,
                    "turnover": 0.0,
                    "turnover_cost": 0.0,
                },
            ],
        )
        artifact = Artifact.objects.create(
            pipeline_run=backtest_run,
            artifact_type="BACKTEST_RESULT",
            key="backtest_rows_fixture",
            uri=backtest_uri,
            content={},
            metadata={},
        )
        context = _build_equity_curve_context(artifact)
        self.assertEqual(context["equity_curve_count"], 2)
        self.assertIn("2024-01-01", context["equity_curve_points_json"])
        self.assertIn("2024-01-02", context["equity_curve_points_json"])

    def test_strategy_detail_summarizes_large_payloads(self):
        strategy_run = PipelineRun.objects.create(
            name="strategy-large-payload",
            requested_job="build_strategy_dataset",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        strategy_uri = self._write_csv(
            "strategy_large_payload",
            ["date", "symbol", "strategy_score", "selected_on_rebalance", "target_weight"],
            [{"date": "2024-01-01", "symbol": "AAPL", "strategy_score": 0.7, "selected_on_rebalance": 1, "target_weight": 0.5}],
        )
        artifact = Artifact.objects.create(
            pipeline_run=strategy_run,
            artifact_type="STRATEGY_DATASET",
            key="strategy_large_payload",
            uri=strategy_uri,
            content={
                "rows": 1,
                "symbols": 1,
                "selected_rows": 1,
                "dates": 1,
                "feature_cols": [f"feature_{idx}" for idx in range(200)],
            },
            metadata={"strategy_config": {"top_k": 3}},
        )
        response = self.client.get(reverse("pipeline-strategy-detail", args=[artifact.id]))
        self.assertEqual(response.status_code, 200)
        html = response.content.decode()
        self.assertIn("feature_col_count", html)
        self.assertIn("feature_cols_preview", html)
        self.assertNotIn('"feature_cols": [', html)

    def test_backtest_detail_uses_inline_chart_renderer(self):
        backtest_run = PipelineRun.objects.create(
            name="backtest-inline-chart",
            requested_job="backtest_strategy",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        artifact = Artifact.objects.create(
            pipeline_run=backtest_run,
            artifact_type="BACKTEST_RESULT",
            key="backtest_inline_chart",
            uri=self._write_csv(
                "backtest_inline_chart",
                ["date", "symbol", "strategy_signal", "strategy_score", "target_weight", "effective_weight", "asset_return", "realized_return", "turnover"],
                [
                    {"date": "2024-01-01", "symbol": "AAPL", "strategy_signal": 1, "strategy_score": 0.5, "target_weight": 0.8, "effective_weight": 0.0, "asset_return": 0.01, "realized_return": 0.0, "turnover": 0.8},
                    {"date": "2024-01-02", "symbol": "AAPL", "strategy_signal": 1, "strategy_score": 0.4, "target_weight": 0.8, "effective_weight": 0.8, "asset_return": 0.02, "realized_return": 0.016, "turnover": 0.0},
                ],
            ),
            content={
                "daily_rows": [
                    {"date": "2024-01-01", "equity": 1.0, "net_daily_return": 0.0, "turnover": 0.8, "positions": 0},
                    {"date": "2024-01-02", "equity": 1.016, "net_daily_return": 0.016, "turnover": 0.0, "positions": 1},
                ],
                "cumulative_return": 0.016,
                "max_drawdown": 0.0,
            },
            metadata={},
        )
        response = self.client.get(reverse("pipeline-backtest-detail", args=[artifact.id]))
        self.assertEqual(response.status_code, 200)
        html = response.content.decode()
        self.assertIn("renderCompositeChart('equity-chart'", html)
        self.assertNotIn("Highcharts.chart", html)

    def test_fit_and_score_mtl_pipeline_jobs_materialize_multitask_predictions(self):
        feature_run = PipelineRun.objects.create(
            name="feature-source-mtl",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_uri = self._write_csv(
            "features_fixture_mtl",
            ["date", "symbol", "close", "ret_1", "vol_5"],
            [
                {"date": "2024-01-01", "symbol": "AAPL", "close": 100.0, "ret_1": 0.10, "vol_5": 0.20},
                {"date": "2024-01-02", "symbol": "AAPL", "close": 101.0, "ret_1": -0.05, "vol_5": 0.18},
                {"date": "2024-01-03", "symbol": "AAPL", "close": 102.0, "ret_1": 0.06, "vol_5": 0.22},
                {"date": "2024-01-04", "symbol": "AAPL", "close": 99.0, "ret_1": -0.08, "vol_5": 0.25},
                {"date": "2024-01-05", "symbol": "AAPL", "close": 104.0, "ret_1": 0.12, "vol_5": 0.24},
                {"date": "2024-01-06", "symbol": "AAPL", "close": 98.0, "ret_1": -0.09, "vol_5": 0.27},
                {"date": "2024-01-07", "symbol": "AAPL", "close": 105.0, "ret_1": 0.11, "vol_5": 0.21},
                {"date": "2024-01-08", "symbol": "AAPL", "close": 97.0, "ret_1": -0.10, "vol_5": 0.29},
            ],
        )
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="features_fixture_mtl",
            uri=feature_uri,
            content={"rows": 8},
            metadata={},
        )

        label_run = PipelineRun.objects.create(
            name="label-source-mtl",
            requested_job="labels",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        label_uri = self._write_csv(
            "labels_fixture_mtl",
            ["date", "symbol", "label", "market_position", "trade_return", "hold_days", "side", "freq", "k"],
            [
                {"date": "2024-01-01", "symbol": "AAPL", "label": 1, "market_position": 1, "trade_return": 0.12, "hold_days": 5, "side": "long", "freq": "YE", "k": 1},
                {"date": "2024-01-02", "symbol": "AAPL", "label": 0, "market_position": -1, "trade_return": -0.06, "hold_days": 7, "side": "short", "freq": "YE", "k": 1},
                {"date": "2024-01-03", "symbol": "AAPL", "label": 1, "market_position": 1, "trade_return": 0.08, "hold_days": 12, "side": "long", "freq": "YE", "k": 2},
                {"date": "2024-01-04", "symbol": "AAPL", "label": 0, "market_position": -1, "trade_return": -0.04, "hold_days": 18, "side": "short", "freq": "YE", "k": 2},
                {"date": "2024-01-05", "symbol": "AAPL", "label": 1, "market_position": 1, "trade_return": 0.20, "hold_days": 40, "side": "long", "freq": "YE", "k": 4},
                {"date": "2024-01-06", "symbol": "AAPL", "label": 0, "market_position": -1, "trade_return": -0.05, "hold_days": 35, "side": "short", "freq": "YE", "k": 4},
                {"date": "2024-01-07", "symbol": "AAPL", "label": 1, "market_position": 1, "trade_return": 0.30, "hold_days": 80, "side": "long", "freq": "YE", "k": 8},
                {"date": "2024-01-08", "symbol": "AAPL", "label": 0, "market_position": -1, "trade_return": -0.09, "hold_days": 90, "side": "short", "freq": "YE", "k": 8},
            ],
        )
        label_artifact = Artifact.objects.create(
            pipeline_run=label_run,
            artifact_type="LABELS",
            key="labels_fixture_mtl",
            uri=label_uri,
            content={"rows": 8},
            metadata={},
        )

        fit_run = PipelineRun.objects.create(
            name="fit-mtl-fixture",
            requested_job="fit_mtl",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
            config={"model_name": "mtl_fixture_model", "split_ratio": 0.5, "params": {"n_estimators": 8}},
        )
        model_artifact = execute_pipeline_run(
            pipeline_run=fit_run,
            target_job="fit_mtl",
            mode="strict",
            config=dict(fit_run.config or {}),
            input_artifact_ids=[feature_artifact.id, label_artifact.id],
        )

        score_run = PipelineRun.objects.create(
            name="score-mtl-fixture",
            requested_job="score_mtl",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
        )
        with patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            output_artifact = execute_pipeline_run(
                pipeline_run=score_run,
                target_job="score_mtl",
                mode="strict",
                config={"label_artifact_id": label_artifact.id},
                input_artifact_ids=[model_artifact.id, feature_artifact.id],
            )

        self.assertEqual(output_artifact.artifact_type, "MTL_PREDICTIONS")
        with Path(output_artifact.uri).open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 8)
        self.assertIn("mtl_prob_buy", rows[0])
        self.assertIn("mtl_trade_return", rows[0])
        self.assertIn("mtl_hold_days", rows[0])
        if "mtl_cluster_key" not in rows[0]:
            self.assertEqual((model_artifact.content.get("metrics") or {}).get("cluster", {}).get("status"), "skipped")
        self.assertIn("prediction", rows[0])
        self.assertIn("prediction_score", rows[0])
        self.assertIn("raw_prediction", rows[0])
        self.assertIn("signal_score", rows[0])

    def test_fit_mtl_pipeline_job_accepts_oracle_cluster_filters(self):
        feature_run = PipelineRun.objects.create(
            name="feature-source-mtl-cluster",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="features_fixture_mtl_cluster",
            uri=self._write_csv(
                "features_fixture_mtl_cluster",
                ["date", "symbol", "close", "ret_1", "vol_5"],
                [
                    {"date": "2024-01-01", "symbol": "AAPL", "close": 100.0, "ret_1": 0.10, "vol_5": 0.20},
                    {"date": "2024-01-02", "symbol": "AAPL", "close": 101.0, "ret_1": 0.06, "vol_5": 0.18},
                    {"date": "2024-01-03", "symbol": "AAPL", "close": 102.0, "ret_1": -0.04, "vol_5": 0.21},
                    {"date": "2024-01-04", "symbol": "AAPL", "close": 103.0, "ret_1": -0.08, "vol_5": 0.23},
                ],
            ),
            content={"rows": 4},
            metadata={},
        )
        label_rows = [
            {"date": "2024-01-01", "symbol": "AAPL", "label": 1, "market_position": 1, "trade_return": 0.10, "hold_days": 5, "side": "long", "freq": "YE", "k": 1},
            {"date": "2024-01-02", "symbol": "AAPL", "label": 1, "market_position": 1, "trade_return": 0.10, "hold_days": 5, "side": "long", "freq": "YE", "k": 1},
            {"date": "2024-01-03", "symbol": "AAPL", "label": 0, "market_position": -1, "trade_return": -0.03, "hold_days": 40, "side": "short", "freq": "YE", "k": 4},
            {"date": "2024-01-04", "symbol": "AAPL", "label": 0, "market_position": -1, "trade_return": -0.05, "hold_days": 60, "side": "short", "freq": "YE", "k": 4},
        ]
        target_cluster_key = str(derive_oracle_cluster_labels(pd.DataFrame(label_rows)).iloc[0])
        label_run = PipelineRun.objects.create(
            name="label-source-mtl-cluster",
            requested_job="labels",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        label_artifact = Artifact.objects.create(
            pipeline_run=label_run,
            artifact_type="LABELS",
            key="labels_fixture_mtl_cluster",
            uri=self._write_csv(
                "labels_fixture_mtl_cluster",
                ["date", "symbol", "label", "market_position", "trade_return", "hold_days", "side", "freq", "k"],
                label_rows,
            ),
            content={"rows": 4},
            metadata={},
        )

        fit_run = PipelineRun.objects.create(
            name="fit-mtl-cluster-fixture",
            requested_job="fit_mtl",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
            config={"model_name": "mtl_cluster_model", "split_ratio": 0.5, "oracle_cluster_keys": [target_cluster_key]},
        )
        model_artifact = execute_pipeline_run(
            pipeline_run=fit_run,
            target_job="fit_mtl",
            mode="strict",
            config=dict(fit_run.config or {}),
            input_artifact_ids=[feature_artifact.id, label_artifact.id],
        )

        self.assertEqual(model_artifact.artifact_type, "MTL_MODEL")
        self.assertEqual(model_artifact.content["oracle_cluster_scope"], "specialist")
        self.assertEqual(model_artifact.content["oracle_cluster_keys"], [target_cluster_key])
        self.assertEqual(model_artifact.metadata["oracle_cluster_scope"], "specialist")
        self.assertEqual(model_artifact.metadata["oracle_cluster_keys"], [target_cluster_key])
        self.assertLess(int(model_artifact.content["trained_rows"]), 4)

    def test_symbol_research_view_supports_mtl_prediction_artifacts(self):
        feature_run = PipelineRun.objects.create(
            name="feature-source-mtl-view",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="features_fixture_mtl_view",
            uri=self._write_csv(
                "features_fixture_mtl_view",
                ["date", "symbol", "close", "ret_1"],
                [{"date": "2024-01-01", "symbol": "AAPL", "close": 100.5, "ret_1": 0.01}],
            ),
            content={"rows": 1},
            metadata={},
        )
        prediction_run = PipelineRun.objects.create(
            name="prediction-source-mtl-view",
            requested_job="score_mtl",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        prediction_artifact = Artifact.objects.create(
            pipeline_run=prediction_run,
            artifact_type="MTL_PREDICTIONS",
            key="predictions_fixture_mtl_view",
            uri=self._write_csv(
                "predictions_fixture_mtl_view",
                ["date", "symbol", "prediction_score", "prediction", "mtl_prob_buy", "mtl_trade_return"],
                [{"date": "2024-01-01", "symbol": "AAPL", "prediction_score": 0.84, "prediction": 0.11, "mtl_prob_buy": 0.84, "mtl_trade_return": 0.11}],
            ),
            content={"rows": 1},
            metadata={"source_features_artifact_id": feature_artifact.id},
        )

        response = self.client.get(
            reverse("pipeline-symbol-research", args=["AAPL"]),
            {
                "feature_artifact_id": feature_artifact.id,
                "prediction_artifact_id": prediction_artifact.id,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "AAPL Research Workspace")
        self.assertContains(response, "0.84")

    def test_rl_train_job_materializes_policy_result(self):
        strategy_run = PipelineRun.objects.create(
            name="strategy-source-rl",
            requested_job="build_strategy_dataset",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        strategy_artifact = Artifact.objects.create(
            pipeline_run=strategy_run,
            artifact_type="STRATEGY_DATASET",
            key="strategy_fixture_rl",
            uri=self._write_csv(
                "strategy_fixture_rl",
                ["date", "symbol", "prob_buy", "ranking", "ae_familiarity", "close"],
                [
                    {"date": "2024-01-01", "symbol": "AAPL", "prob_buy": 0.80, "ranking": 0.12, "ae_familiarity": 0.60, "close": 100.0},
                    {"date": "2024-01-01", "symbol": "MSFT", "prob_buy": 0.70, "ranking": 0.05, "ae_familiarity": 0.55, "close": 200.0},
                    {"date": "2024-01-02", "symbol": "AAPL", "prob_buy": 0.82, "ranking": 0.15, "ae_familiarity": 0.62, "close": 101.0},
                    {"date": "2024-01-02", "symbol": "MSFT", "prob_buy": 0.68, "ranking": 0.04, "ae_familiarity": 0.54, "close": 201.0},
                    {"date": "2024-01-03", "symbol": "AAPL", "prob_buy": 0.85, "ranking": 0.18, "ae_familiarity": 0.65, "close": 102.0},
                    {"date": "2024-01-03", "symbol": "MSFT", "prob_buy": 0.72, "ranking": 0.06, "ae_familiarity": 0.56, "close": 202.0},
                ],
            ),
            content={"rows": 6},
            metadata={},
        )

        rl_summary_df = __import__("pandas").DataFrame(
            [
                {
                    "mode": "rl_agent_ppo_framework_backtest",
                    "years": "2024-2024",
                    "combined_total_return_pct": 24.5,
                    "combined_sharpe": 1.2,
                    "combined_max_drawdown_pct": -11.0,
                }
            ]
        )
        rl_yearly_df = __import__("pandas").DataFrame(
            [{"mode": "rl_agent_ppo_framework_backtest", "test_year": 2024, "total_return_pct": 24.5, "sharpe": 1.2, "max_drawdown_pct": -11.0}]
        )
        trade_log_df = __import__("pandas").DataFrame(
            [{"date": pd.Timestamp("2024-01-02"), "symbol": "AAPL", "side": "buy", "price": 101.0, "shares": 10.0}]
        )
        fake_result = {
            "rl_summary_df": rl_summary_df,
            "rl_yearly_df": rl_yearly_df,
            "executed_action_counts": pd.Series({"buy": 3, "sell": 1}),
            "action_counts": pd.Series({"hold": 4, "buy": 3, "sell": 1}),
            "trade_log": trade_log_df,
        }

        rl_run = PipelineRun.objects.create(
            name="rl-train-fixture",
            requested_job="rl_train",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
        )
        with patch("pipeline.services.run_ppo_workflow", return_value=fake_result) as mocked_runner, patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            rl_artifact = execute_pipeline_run(
                pipeline_run=rl_run,
                target_job="rl_train",
                mode="strict",
                config={"algorithm": "ppo", "train_split_date": "2024-01-01", "eval_years": [2024], "episodes": 2},
                input_artifact_ids=[strategy_artifact.id],
            )

        self.assertEqual(rl_artifact.artifact_type, "RL_POLICY_RESULT")
        self.assertEqual(rl_artifact.metadata["source_strategy_dataset_artifact_id"], strategy_artifact.id)
        self.assertEqual(rl_artifact.content["executed_buys"], 3)
        self.assertEqual(rl_artifact.content["executed_sells"], 1)
        self.assertAlmostEqual(float(rl_artifact.content["combined_total_return_pct"]), 24.5)
        runner_kwargs = mocked_runner.call_args.kwargs
        self.assertEqual(list(runner_kwargs["bt_panel"].index.names), ["date", "symbol"])
        self.assertIn("pred_rf_reg", runner_kwargs["bt_panel"].columns)
        payload = json.loads(Path(rl_artifact.uri).read_text(encoding="utf-8"))
        self.assertEqual(payload["executed_action_counts"]["buy"], 3)
        self.assertEqual(len(payload["summary_rows"]), 1)
        self.assertEqual(len(payload["trade_log_preview"]), 1)

    def test_rl_policy_reports_page_renders_saved_artifact(self):
        rl_run = PipelineRun.objects.create(
            name="rl-policy-page",
            requested_job="rl_train",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        artifact = Artifact.objects.create(
            pipeline_run=rl_run,
            artifact_type="RL_POLICY_RESULT",
            key="rl_policy_page",
            uri=self._write_json(
                "rl_policy_page",
                {
                    "algorithm": "ppo",
                    "train_split_date": "2024-12-31",
                    "eval_years": [2025],
                    "summary_rows": [
                        {
                            "mode": "rl_agent_ppo_framework_backtest",
                            "years": "2025-2025",
                            "combined_total_return_pct": 18.5,
                            "combined_sharpe": 1.1,
                            "combined_max_drawdown_pct": -9.2,
                        }
                    ],
                    "yearly_rows": [
                        {
                            "test_year": 2025,
                            "total_return_pct": 18.5,
                            "sharpe": 1.1,
                            "max_drawdown_pct": -9.2,
                        }
                    ],
                    "executed_action_counts": {"buy": 4, "sell": 2},
                    "action_counts": {"hold": 10, "buy": 4, "sell": 2},
                    "trade_log_preview": [{"date": "2025-01-02", "symbol": "AAPL", "side": "buy", "price": 101.0, "shares": 10.0}],
                },
            ),
            content={"combined_total_return_pct": 18.5},
            metadata={},
        )
        response = self.client.get(reverse("pipeline-rl-policy-reports"), {"artifact_id": artifact.id})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Policy Learning On Opportunity State")
        self.assertContains(response, "rl_agent_ppo_framework_backtest")
        self.assertContains(response, "AAPL")

    def test_similarity_search_returns_historical_states(self):
        strategy_artifact = self._build_insight_strategy_artifact()
        frame, meta = load_market_state_frame(
            strategy_artifact=strategy_artifact,
            feature_artifact=None,
            label_artifact=None,
            prediction_artifacts=(),
        )
        embedding = compute_market_state_embedding(
            symbol="AAPL",
            strategy_artifact=strategy_artifact,
            feature_artifact=None,
            label_artifact=None,
            prediction_artifacts=(),
        )
        index = build_similarity_index(frame, meta["embedding_columns"])
        matches = find_similar_market_states(
            embedding["embedding_vector"],
            index,
            k=6,
            query_date=embedding["date"],
            exclude_symbol="AAPL",
            exclude_date=embedding["date"],
        )
        self.assertEqual(len(matches), 6)
        self.assertTrue(all(row["similarity_score"] >= 0 for row in matches))
        self.assertTrue(all(str(row["date"]) < embedding["date"] for row in matches))
        self.assertTrue(all(row["symbol"] in {"AAPL", "MSFT", "NVDA"} for row in matches))

    def test_market_state_representations_export_embedding_features(self):
        strategy_artifact = self._build_insight_strategy_artifact("representation_fixture")
        frame, meta = load_market_state_frame(
            strategy_artifact=strategy_artifact,
            feature_artifact=None,
            label_artifact=None,
            prediction_artifacts=(),
        )
        row = frame[frame["symbol"] == "AAPL"].sort_values("date").iloc[-1].to_dict()
        numeric = build_market_state_representation(
            row,
            method="numeric",
            feature_columns=meta["embedding_columns"],
            feature_family_map=meta["feature_family_map"],
        )
        text_embedding = build_market_state_representation(
            row,
            method="text_embedding",
            feature_columns=meta["embedding_columns"],
            feature_family_map=meta["feature_family_map"],
        )
        hybrid = build_market_state_representation(
            row,
            method="hybrid",
            feature_columns=meta["embedding_columns"],
            feature_family_map=meta["feature_family_map"],
        )
        self.assertGreater(len(numeric.numeric_vector), 0)
        self.assertGreater(len(text_embedding.embedding_vector), 0)
        self.assertEqual(len(hybrid.vector), len(hybrid.numeric_vector) + len(hybrid.embedding_vector))

        embedding_features = export_embedding_features(
            frame.head(6),
            feature_family_map=meta["feature_family_map"],
            feature_columns=meta["embedding_columns"],
        )
        augmented = append_embedding_features(
            frame.head(6)[meta["embedding_columns"]].apply(pd.to_numeric, errors="coerce").fillna(0.0),
            embedding_features.drop(columns=["date", "symbol"]),
        )
        self.assertEqual(len(embedding_features), 6)
        self.assertGreater(augmented.shape[1], len(meta["embedding_columns"]))

    def test_historical_situation_search_supports_same_cross_and_hybrid_modes(self):
        strategy_artifact = self._build_insight_strategy_artifact("hybrid_search_fixture")
        frame, meta = load_market_state_frame(
            strategy_artifact=strategy_artifact,
            feature_artifact=None,
            label_artifact=None,
            prediction_artifacts=(),
        )
        bundle = build_historical_situation_search_bundle(
            frame,
            feature_columns=meta["embedding_columns"],
            feature_family_map=meta["feature_family_map"],
        )
        row = frame[frame["symbol"] == "AAPL"].sort_values("date").iloc[170].to_dict()
        query_date = str(pd.Timestamp(row["date"]).date())
        same_rows = search_market_state_neighbors(
            row,
            bundle,
            method="numeric",
            top_k=5,
            search_mode="same_symbol",
            query_symbol="AAPL",
            query_date=query_date,
        )
        cross_rows = search_market_state_neighbors(
            row,
            bundle,
            method="text_embedding",
            top_k=5,
            search_mode="cross_symbol",
            query_symbol="AAPL",
            query_date=query_date,
        )
        mixed_rows = search_market_state_neighbors(
            row,
            bundle,
            method="hybrid",
            top_k=6,
            search_mode="mixed",
            query_symbol="AAPL",
            query_date=query_date,
        )
        self.assertTrue(same_rows)
        self.assertTrue(cross_rows)
        self.assertTrue(mixed_rows)
        self.assertTrue(all(item["symbol"] == "AAPL" for item in same_rows))
        self.assertTrue(all(item["symbol"] != "AAPL" for item in cross_rows))
        self.assertTrue(all(item["match_type"] in {"same_symbol", "cross_symbol"} for item in mixed_rows))
        self.assertTrue(any(item["explanations"] for item in mixed_rows))

        outcomes = summarize_historical_outcomes(mixed_rows, frame, price_col=resolve_price_column(frame))
        self.assertGreater(len(outcomes["matches"]), 0)
        self.assertEqual(outcomes["summary"]["primary_horizon_days"], 60)
        self.assertTrue(any(int(item["horizon_days"]) == 5 for item in outcomes["summary"]["horizon_rows"]))

    def test_oracle_state_dataset_builds_correctly(self):
        strategy_artifact = self._build_insight_strategy_artifact("oracle_state_dataset_fixture")
        oracle_df, meta = build_oracle_state_dataset(strategy_artifact=strategy_artifact)
        self.assertGreater(len(oracle_df), 0)
        self.assertIn("macro_liquidity_regime", oracle_df.columns)
        self.assertIn("macro_rate_regime", oracle_df.columns)
        self.assertIn("price_momentum_regime", oracle_df.columns)
        self.assertEqual(meta["strategy_artifact_id"], strategy_artifact.id)

    def test_market_situation_clusters_are_generated(self):
        strategy_artifact = self._build_insight_strategy_artifact("market_cluster_fixture")
        payload = fit_market_situation_clusters(
            strategy_artifact=strategy_artifact,
            pca_components=4,
            max_clusters=5,
            min_cluster_size=20,
        )
        assignments = payload["assignments"]
        self.assertIn("cluster_id", assignments.columns)
        self.assertIn("cluster_code", assignments.columns)
        self.assertIn("cluster_similarity", assignments.columns)
        self.assertGreater(len(payload["summary"]["clusters"]), 0)

    def test_cluster_outcome_stats_compute_correctly(self):
        strategy_artifact = self._build_insight_strategy_artifact("cluster_outcomes_fixture")
        payload = fit_market_situation_clusters(
            strategy_artifact=strategy_artifact,
            pca_components=3,
            max_clusters=4,
            min_cluster_size=20,
        )
        stats = compute_cluster_outcome_stats(payload["assignments"])
        self.assertGreater(len(stats), 0)
        self.assertIn("median_return", stats.columns)
        self.assertIn("win_rate", stats.columns)
        self.assertIn("yearly_median_return_std", stats.columns)

    def test_outcome_aggregation_works_for_historical_twins(self):
        strategy_artifact = self._build_insight_strategy_artifact("insight_strategy_fixture_outcomes")
        frame, meta = load_market_state_frame(
            strategy_artifact=strategy_artifact,
            feature_artifact=None,
            label_artifact=None,
            prediction_artifacts=(),
        )
        probe_date = str(frame[frame["symbol"] == "NVDA"]["date"].sort_values().iloc[170].date())
        embedding = compute_market_state_embedding(
            symbol="NVDA",
            date=probe_date,
            strategy_artifact=strategy_artifact,
            feature_artifact=None,
            label_artifact=None,
            prediction_artifacts=(),
        )
        index = build_similarity_index(frame, meta["embedding_columns"])
        matches = find_similar_market_states(
            embedding["embedding_vector"],
            index,
            k=8,
            query_date=embedding["date"],
            exclude_symbol="NVDA",
            exclude_date=embedding["date"],
        )
        matches = enrich_similarity_matches_with_outcomes(matches, frame, price_col=resolve_price_column(frame))
        summary = aggregate_outcome_distribution(matches)
        self.assertGreater(int(summary["sample_size"]), 0)
        self.assertEqual(int(summary["primary_horizon_days"]), 60)
        self.assertIsNotNone(summary["median_return"])
        self.assertEqual([row["horizon_days"] for row in summary["horizon_rows"]], [30, 60, 90, 180])

    def test_market_situation_similarity_returns_historical_twins(self):
        strategy_artifact = self._build_insight_strategy_artifact("market_similarity_fixture")
        market_artifact = self._build_market_situation_artifact(strategy_artifact, "market_similarity_fixture")
        bundle = load_market_situation_cluster_artifact(market_artifact)
        oracle_df, _ = build_oracle_state_dataset(strategy_artifact=strategy_artifact)
        target_row = oracle_df.iloc[180]
        cluster_embedding = compute_cluster_state_embedding(target_row, bundle.embedding_model)
        nearest = find_nearest_clusters(cluster_embedding, bundle, side=str(target_row.get("side") or "long"), top_n=2)
        self.assertGreaterEqual(len(nearest), 1)
        matches = find_similar_historical_states(
            str(nearest[0]["cluster_id"]),
            cluster_embedding,
            bundle,
            k=5,
            before_date=str(pd.Timestamp(target_row["date"]).date()),
            exclude_symbol=str(target_row["symbol"]),
            exclude_date=str(pd.Timestamp(target_row["date"]).date()),
        )
        self.assertEqual(len(matches), 5)
        self.assertTrue(all(str(row["date"]) < str(pd.Timestamp(target_row["date"]).date()) for row in matches))

    def test_opportunity_score_stays_bounded(self):
        strategy_artifact = self._build_insight_strategy_artifact("insight_strategy_fixture_score")
        frame, meta = load_market_state_frame(
            strategy_artifact=strategy_artifact,
            feature_artifact=None,
            label_artifact=None,
            prediction_artifacts=(),
        )
        embedding = compute_market_state_embedding(
            symbol="MSFT",
            strategy_artifact=strategy_artifact,
            feature_artifact=None,
            label_artifact=None,
            prediction_artifacts=(),
        )
        index = build_similarity_index(frame, meta["embedding_columns"])
        matches = find_similar_market_states(
            embedding["embedding_vector"],
            index,
            k=10,
            query_date=embedding["date"],
            exclude_symbol="MSFT",
            exclude_date=embedding["date"],
        )
        matches = enrich_similarity_matches_with_outcomes(matches, frame, price_col=resolve_price_column(frame))
        summary = aggregate_outcome_distribution(matches)
        scoring = compute_opportunity_summary(
            row=embedding["row"],
            state_frame=frame,
            outcome_summary=summary,
            similarity_rows=matches,
        )
        self.assertGreaterEqual(float(scoring["opportunity_score"]), 0.0)
        self.assertLessEqual(float(scoring["opportunity_score"]), 100.0)
        self.assertGreaterEqual(float(scoring["confidence_score"]), 0.0)
        self.assertLessEqual(float(scoring["confidence_score"]), 100.0)
        self.assertGreaterEqual(float(scoring["market_familiarity_score"]), 0.0)
        self.assertLessEqual(float(scoring["market_familiarity_score"]), 100.0)
        self.assertGreaterEqual(float(scoring["risk_score"]), 0.0)
        self.assertLessEqual(float(scoring["risk_score"]), 100.0)

    def test_market_state_api_returns_json(self):
        strategy_artifact = self._build_insight_strategy_artifact("insight_strategy_fixture_api")
        market_artifact = self._build_market_situation_artifact(strategy_artifact, "market_state_api_clusters")
        response = self.client.get(
            reverse("pipeline-market-state-api"),
            {
                "symbol": "AAPL",
                "strategy_artifact_id": strategy_artifact.id,
                "market_situation_artifact_id": market_artifact.id,
                "k": 5,
                "search_method": "hybrid",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["symbol"], "AAPL")
        self.assertIn("historical_twins", payload)
        self.assertEqual(len(payload["historical_twins"]), 5)
        self.assertIn("opportunity", payload)
        self.assertIn("current_cluster", payload)
        self.assertIn("same_symbol_twins", payload)
        self.assertIn("cross_symbol_twins", payload)
        self.assertIn("stock_insight", payload)
        self.assertIn("market_situation_explanation", payload)

    def test_market_state_api_supports_prompt_ready_reasoning(self):
        strategy_artifact = self._build_insight_strategy_artifact("insight_strategy_fixture_api_prompt")
        market_artifact = self._build_market_situation_artifact(strategy_artifact, "market_state_api_prompt_clusters")
        response = self.client.get(
            reverse("pipeline-market-state-api"),
            {
                "symbol": "AAPL",
                "strategy_artifact_id": strategy_artifact.id,
                "market_situation_artifact_id": market_artifact.id,
                "k": 5,
                "search_method": "hybrid",
                "reasoning_mode": "llm_prompt",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["reasoning_mode"], "llm_prompt")
        self.assertIn("llm_prompt", payload["stock_insight"])
        self.assertIn("Canonical Features", payload["stock_insight"]["llm_prompt"])

    def test_opportunities_page_renders_insight_dashboard(self):
        strategy_artifact = self._build_insight_strategy_artifact("insight_strategy_fixture_dashboard")
        market_artifact = self._build_market_situation_artifact(strategy_artifact, "insight_dashboard_clusters")
        response = self.client.get(
            reverse("pipeline-opportunities"),
            {
                "strategy_artifact_id": strategy_artifact.id,
                "market_situation_artifact_id": market_artifact.id,
                "limit": 5,
                "search_method": "hybrid",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Top Opportunities Today")
        self.assertContains(response, "Opportunity Score")
        self.assertContains(response, "AAPL")
        self.assertContains(response, "Market Situation Artifact")

    def test_stock_intelligence_page_renders_historical_twins(self):
        strategy_artifact = self._build_insight_strategy_artifact("insight_strategy_fixture_stock")
        market_artifact = self._build_market_situation_artifact(strategy_artifact, "stock_intelligence_clusters")
        response = self.client.get(
            reverse("pipeline-stock-intelligence-symbol", args=["NVDA"]),
            {
                "strategy_artifact_id": strategy_artifact.id,
                "market_situation_artifact_id": market_artifact.id,
                "k": 6,
                "search_method": "hybrid",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "NVDA Market Situation")
        self.assertContains(response, "Historical Twins")
        self.assertContains(response, "Outcome Distribution")
        self.assertContains(response, "Reasoning Layer")
        self.assertContains(response, "Caveats And Uncertainty")
        self.assertContains(response, "Current Situation Family")
        self.assertContains(response, "NVDA Historical Analogs")
        self.assertContains(response, "Cross-Symbol Analogs")

    def test_portfolio_analysis_page_renders_groups(self):
        strategy_artifact = self._build_insight_strategy_artifact("insight_strategy_fixture_portfolio")
        market_artifact = self._build_market_situation_artifact(strategy_artifact, "portfolio_clusters")
        response = self.client.get(
            reverse("pipeline-portfolio-analysis"),
            {
                "strategy_artifact_id": strategy_artifact.id,
                "market_situation_artifact_id": market_artifact.id,
                "symbols": "AAPL,MSFT,NVDA",
                "search_method": "hybrid",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Portfolio Situation Review")
        self.assertContains(response, "Portfolio Insight")
        self.assertContains(response, "Reasoning Layer")
        self.assertContains(response, "Strong Positions")
        self.assertContains(response, "Opportunity Breakdown")
        self.assertContains(response, "Portfolio Cluster Exposure")

    def test_market_situations_page_renders_saved_artifact(self):
        strategy_artifact = self._build_insight_strategy_artifact("market_situations_page_fixture")
        market_artifact = self._build_market_situation_artifact(strategy_artifact, "market_situations_page_fixture")
        response = self.client.get(reverse("pipeline-market-situations"), {"artifact_id": market_artifact.id})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Market Situation Taxonomy")
        self.assertContains(response, "Situation Families")
        self.assertContains(response, "Clusters")

    def test_run_market_situation_clustering_command_materializes_artifact(self):
        strategy_artifact = self._build_insight_strategy_artifact("market_situation_command_fixture")
        out = StringIO()
        with patch("analysis.situation_clustering.ARTIFACT_DIR", self.temp_path):
            call_command(
                "run_market_situation_clustering",
                strategy_artifact=strategy_artifact.id,
                output_basename="market_situation_command_fixture",
                stdout=out,
            )
        payload = json.loads(out.getvalue())
        artifact = Artifact.objects.get(pk=int(payload["artifact_id"]))
        self.assertEqual(artifact.artifact_type, "MARKET_SITUATION_CLUSTER")
        self.assertTrue(Path(artifact.uri).exists())

    def test_run_historical_market_search_command_outputs_same_and_cross_matches(self):
        strategy_artifact = self._build_insight_strategy_artifact("historical_market_search_fixture")
        market_artifact = self._build_market_situation_artifact(strategy_artifact, "historical_market_search_clusters")
        out = StringIO()
        call_command(
            "run_historical_market_search",
            symbol="NVDA",
            strategy_artifact=strategy_artifact.id,
            market_situation_artifact=market_artifact.id,
            search_method="hybrid",
            top_k=4,
            stdout=out,
        )
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["symbol"], "NVDA")
        self.assertEqual(payload["search"]["method"], "hybrid")
        self.assertIn("same_symbol_twins", payload)
        self.assertIn("cross_symbol_twins", payload)

    def test_run_market_insight_reasoning_command_outputs_stock_and_portfolio_payloads(self):
        strategy_artifact = self._build_insight_strategy_artifact("market_insight_reasoning_fixture")
        market_artifact = self._build_market_situation_artifact(strategy_artifact, "market_insight_reasoning_clusters")

        stock_out = StringIO()
        call_command(
            "run_market_insight_reasoning",
            symbol="NVDA",
            strategy_artifact=strategy_artifact.id,
            market_situation_artifact=market_artifact.id,
            search_method="hybrid",
            reasoning_mode="llm_prompt",
            stdout=stock_out,
        )
        stock_payload = json.loads(stock_out.getvalue())
        self.assertEqual(stock_payload["kind"], "stock_insight_reasoning")
        self.assertEqual(stock_payload["symbol"], "NVDA")
        self.assertEqual(stock_payload["reasoning_mode"], "llm_prompt")
        self.assertIn("stock_insight", stock_payload)
        self.assertIn("llm_prompt", stock_payload["stock_insight"])

        portfolio_out = StringIO()
        call_command(
            "run_market_insight_reasoning",
            symbols="AAPL,MSFT,NVDA",
            strategy_artifact=strategy_artifact.id,
            market_situation_artifact=market_artifact.id,
            search_method="hybrid",
            stdout=portfolio_out,
        )
        portfolio_payload = json.loads(portfolio_out.getvalue())
        self.assertEqual(portfolio_payload["kind"], "portfolio_insight_reasoning")
        self.assertEqual(portfolio_payload["symbols"], ["AAPL", "MSFT", "NVDA"])
        self.assertIn("portfolio_insight", portfolio_payload)


class UniverseScalingTests(Mag7FixtureMixin, TestCase):
    def setUp(self):
        super().setUp()
        self.create_screened_symbols()

    def test_resolve_symbol_universe_filters_us_exchanges_and_market_cap(self):
        symbols = resolve_symbol_universe(
            min_market_cap=100_000_000_000.0,
            country="US",
            exchanges=["NASDAQ", "NYSE"],
            exclude_pooled_vehicles=True,
        )
        self.assertEqual(symbols, ["AAPL", "MSFT", "NVDA", "ORCL", "CRM", "UBER"])

    def test_market_cap_tier_resolution_scales_symbol_counts(self):
        tier_1t = resolve_market_cap_tier_symbols(tier_key="1t", country="US", exchanges=["NASDAQ", "NYSE"])
        tier_100b = resolve_market_cap_tier_symbols(tier_key="100b", country="US", exchanges=["NASDAQ", "NYSE"])
        tier_10b = resolve_market_cap_tier_symbols(tier_key="10b", country="US", exchanges=["NASDAQ", "NYSE"])
        self.assertEqual(tier_1t, ["AAPL", "MSFT", "NVDA"])
        self.assertGreater(len(tier_100b), len(tier_1t))
        self.assertGreater(len(tier_10b), len(tier_100b))
        self.assertEqual(tier_10b[-1], "DUOL")

    def test_resolve_symbol_universe_excludes_payload_funds_when_requested(self):
        Symbol.objects.create(
            symbol="FUNDX",
            company_name="Broad Market Index Vehicle",
            exchange="NASDAQ",
            country="US",
            market_cap=600_000_000_000.0,
            payload={"isFund": True},
        )
        symbols = resolve_symbol_universe(
            min_market_cap=100_000_000_000.0,
            country="US",
            exchanges=["NASDAQ", "NYSE"],
            exclude_pooled_vehicles=True,
        )
        self.assertNotIn("FUNDX", symbols)

    def test_resolve_symbol_universe_excludes_requested_symbol_prefixes(self):
        Symbol.objects.create(
            symbol="TIER1234",
            company_name="tier1 synthetic 1234",
            exchange="NASDAQ",
            country="US",
            market_cap=700_000_000_000.0,
            payload={},
        )
        symbols = resolve_symbol_universe(
            min_market_cap=100_000_000_000.0,
            country="US",
            exchanges=["NASDAQ", "NYSE"],
            exclude_pooled_vehicles=True,
            exclude_symbol_prefixes=["TIER"],
        )
        self.assertNotIn("TIER1234", symbols)

    def test_universe_job_supports_screen_filters(self):
        run = PipelineRun.objects.create(
            name="us-cap-tier-universe",
            requested_job="universe",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
        )
        with patch("pipeline.services.ARTIFACT_DIR", self.temp_path):
            artifact = execute_pipeline_run(
                pipeline_run=run,
                target_job="universe",
                mode="strict",
                config={
                    "min_market_cap": 100_000_000_000.0,
                    "country": "US",
                    "exchanges": ["NASDAQ", "NYSE"],
                    "exclude_pooled_vehicles": True,
                    "limit": 10,
                },
            )
        payload = json.loads(Path(artifact.uri).read_text(encoding="utf-8"))
        self.assertEqual(payload["symbols"], ["AAPL", "MSFT", "NVDA", "ORCL", "CRM", "UBER"])
        self.assertEqual(payload["filters"]["country"], "US")
        self.assertEqual(payload["filters"]["exchanges"], ["NASDAQ", "NYSE"])

    def test_market_cap_tier_command_supports_dry_run(self):
        out = StringIO()
        call_command(
            "run_us_market_cap_tier_research",
            tiers="1t,100b,10b",
            dry_run=True,
            country="US",
            exchanges="NASDAQ,NYSE",
            stdout=out,
        )
        payload = json.loads(out.getvalue())
        self.assertEqual([row["tier"] for row in payload["tiers"]], ["1t", "100b", "10b"])
        counts = [int(row["symbol_count"]) for row in payload["tiers"]]
        self.assertEqual(counts, sorted(counts))

    def test_pipeline_lab_starts_screened_research_suite(self):
        class ImmediateThread:
            def __init__(self, target=None, daemon=None):
                self.target = target

            def start(self):
                if self.target is not None:
                    self.target()

        with patch("pipeline.views.run_optimal_trade_research_suite") as runner, patch(
            "pipeline.views.threading.Thread",
            ImmediateThread,
        ):
            response = self.client.post(
                reverse("pipeline-lab"),
                {
                    "lab_action": "run_research_suite",
                    "name": "us-tier-suite",
                    "universe_mode": "us_market_cap_screen",
                    "market_cap_tier": "100b",
                    "min_market_cap": "",
                    "country": "US",
                    "exchanges_csv": "NASDAQ,NYSE",
                    "max_symbols": "",
                    "profile_name": "small_universe_fast",
                    "test_start_year": 2024,
                    "test_end_year": 2024,
                    "min_profit_pct": 12.0,
                    "transaction_cost_bps": 10.0,
                    "resume_existing": "on",
                    "feature_artifact_id": 0,
                    "label_artifact_id": 0,
                },
            )
        self.assertEqual(response.status_code, 200)
        runner.assert_called_once()
        kwargs = runner.call_args.kwargs
        self.assertEqual(kwargs["symbols"], ["AAPL", "MSFT", "NVDA", "ORCL", "CRM", "UBER"])
        self.assertEqual(kwargs["profile_name"], "small_universe_fast")
