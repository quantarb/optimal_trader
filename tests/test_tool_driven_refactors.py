from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


fake_view_support = types.ModuleType("pipeline.view_support")


def _safe_float(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


fake_view_support._safe_float = _safe_float
fake_view_support._to_int = _to_int
sys.modules.setdefault("pipeline.view_support", fake_view_support)

fake_ml_execution = types.ModuleType("ml.execution")
fake_ml_execution._dedupe_label_frame = lambda frame: frame
fake_ml_execution.build_feature_frame_from_artifacts = lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError)
fake_ml_execution.load_artifact_csv_frame = lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError)
fake_ml_execution.infer_feature_family_columns = lambda *args, **kwargs: {}
fake_ml_execution.score_model_from_artifact_inputs = lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError)
fake_ml_execution.train_model_from_artifact_inputs = lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError)
sys.modules.setdefault("ml.execution", fake_ml_execution)


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")

import django

django.setup()

from analysis import diagnostics
from analysis import market_insight_schema
from pipeline import artifact_backtest_support
from pipeline import artifact_support
from tools.product_quality_analysis import cli as product_quality_cli
from utils import workflow
from workflows import strategy


class ToolDrivenRefactorTests(unittest.TestCase):
    def test_build_diagnostic_panel_merges_inputs_and_ranks_scores(self):
        classifier = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "AAPL", "prediction_score": "0.9", "label": "1", "trade_return": "0.12"},
                {"date": "2024-01-02", "symbol": "MSFT", "prediction_score": "0.2", "label": "0", "trade_return": "-0.03"},
            ]
        )
        regressor = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "AAPL", "prediction": "0.10"},
                {"date": "2024-01-02", "symbol": "MSFT", "prediction": "-0.01"},
            ]
        )
        autoencoder = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "AAPL", "prediction_score": "0.7", "prediction": "0.05"},
                {"date": "2024-01-02", "symbol": "MSFT", "prediction_score": "0.3", "prediction": "0.20"},
            ]
        )

        panel = diagnostics._build_diagnostic_panel(
            classifier=classifier,
            regressor=regressor,
            autoencoder=autoencoder,
        )

        self.assertEqual(list(panel["symbol"]), ["AAPL", "MSFT"])
        self.assertIn("combined_rank_mean", panel.columns)
        self.assertEqual(int(panel.iloc[0]["label"]), 1)
        self.assertGreater(float(panel.iloc[0]["combined_rank_mean"]), float(panel.iloc[1]["combined_rank_mean"]))

    def test_artifact_symbol_summary_reuses_research_query(self):
        rows = [
            {"symbol": "aapl", "date": "2024-01-01", "prediction_score": "1.0"},
            {"symbol": "AAPL", "date": "2024-01-02", "prediction_score": "3.0"},
            {"symbol": "msft", "date": "2024-01-03", "prediction_score": "2.0"},
        ]

        with patch("pipeline.artifact_support._research_query_for_symbol", return_value="?artifact=1") as mocked_query:
            summary = artifact_support._artifact_symbol_summary(rows, artifact=SimpleNamespace())

        self.assertEqual(mocked_query.call_count, 1)
        self.assertEqual(summary[0]["symbol"], "AAPL")
        self.assertAlmostEqual(float(summary[0]["avg_prediction_score"]), 2.0)
        self.assertEqual(summary[0]["research_query"], "?artifact=1")

    def test_collect_prediction_components_uses_suffix_rules(self):
        feature_df = pd.DataFrame(
            [
                {
                    "ret_1": 0.01,
                    "clf__prediction_score": 0.8,
                    "reg__prediction": 0.2,
                    "ae__prediction_score": 0.9,
                    "ae__prediction": 0.15,
                    "mtl__mtl_prob_buy": 0.4,
                    "mtl__mtl_trade_return": 0.6,
                    "mtl__mtl_cluster_confidence": 0.5,
                }
            ]
        )
        panel_meta = {
            "extra_panel_sources": [
                {"artifact_type": "CLASSIFIER_PREDICTIONS", "columns": ["clf__prediction_score"]},
                {"artifact_type": "REGRESSOR_PREDICTIONS", "columns": ["reg__prediction"]},
                {"artifact_type": "AUTOENCODER_SCORES", "columns": ["ae__prediction_score", "ae__prediction"]},
                {
                    "artifact_type": "MTL_PREDICTIONS",
                    "columns": ["mtl__mtl_prob_buy", "mtl__mtl_trade_return", "mtl__mtl_cluster_confidence"],
                },
            ]
        }

        enriched = strategy._collect_prediction_components(feature_df, panel_meta)

        self.assertAlmostEqual(float(enriched.iloc[0]["prob_buy"]), 0.6)
        self.assertAlmostEqual(float(enriched.iloc[0]["ranking"]), 0.4)
        self.assertAlmostEqual(float(enriched.iloc[0]["ae_familiarity"]), 0.7)
        self.assertAlmostEqual(float(enriched.iloc[0]["ae_reconstruction_error"]), 0.15)

    def test_selected_routes_uses_supplied_inventory_once(self):
        inventories = [SimpleNamespace(name="inventory")]
        route_a = SimpleNamespace(name="home")
        route_b = SimpleNamespace(name="detail")

        with patch("tools.product_quality_analysis.snapshot_support.discover_routes", return_value=[route_a, route_b]) as mocked_discover:
            selected = product_quality_cli._selected_routes(
                SimpleNamespace(),
                "detail",
                inventories=inventories,
            )

        self.assertEqual(mocked_discover.call_count, 1)
        self.assertIs(mocked_discover.call_args.args[1], inventories)
        self.assertEqual([route.name for route in selected], ["detail"])

    def test_market_insight_helpers_preserve_outcome_and_analog_shapes(self):
        summary = market_insight_schema._outcome_summary_from_payload(
            {
                "primary_horizon_days": 20,
                "median_return": "0.11",
                "horizon_rows": [{"horizon_days": 20, "sample_size": 3, "win_rate": "0.67"}],
            }
        )
        analogs = market_insight_schema._analogs_from_rows(
            [
                {
                    "symbol": "AAPL",
                    "date": "2024-01-01",
                    "similarity_score": "0.8",
                    "match_type": "same_symbol",
                    "explanations": [{"explanation": "Momentum aligned"}],
                }
            ]
        )

        self.assertEqual(summary.primary_horizon_days, 20)
        self.assertEqual(summary.horizon_rows[0].sample_size, 3)
        self.assertEqual(analogs[0].symbol, "AAPL")
        self.assertEqual(analogs[0].explanation_tags, ["Momentum aligned"])

    def test_build_equity_curve_context_preserves_series_and_summary(self):
        backtest_artifact = SimpleNamespace(
            content={},
            metadata={
                "equity_curve": [
                    {"date": "2024-01-02", "equity": 1.01, "net_daily_return": 0.01},
                    {"date": "2024-01-03", "equity": 0.9898, "net_daily_return": -0.02},
                ]
            },
        )
        preview_rows = [
            {"symbol": "AAPL", "date": "2024-01-02", "realized_return": "0.05"},
            {"symbol": "MSFT", "date": "2024-01-03", "realized_return": "-0.02"},
        ]
        content_payload = {
            "daily_rows": [
                {"date": "2024-01-02", "turnover": "0.2", "positions": "2", "net_daily_return": "0.01"},
                {"date": "2024-01-03", "turnover": "0.1", "positions": "1", "net_daily_return": "-0.02"},
            ],
            "max_drawdown": "-0.10",
        }

        def loader(_artifact, limit):
            self.assertGreater(limit, 0)
            return preview_rows, content_payload

        context = artifact_backtest_support.build_equity_curve_context(
            backtest_artifact,
            load_artifact_preview_rows=loader,
            normalized_date=lambda value: str(value or "")[:10],
            safe_float=_safe_float,
            to_int=_to_int,
        )

        self.assertEqual(context["equity_curve_count"], 2)
        self.assertEqual(len(context["monthly_return_rows"]), 1)
        self.assertEqual(context["contribution_rows"][0]["symbol"], "AAPL")
        self.assertIn("turnover_series_json", context)
        self.assertIn("report_summary", context)

    def test_universe_artifact_record_uses_file_symbols_and_metadata_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload_path = os.path.join(tmp_dir, "universe.json")
            with open(payload_path, "w", encoding="utf-8") as handle:
                json.dump({"name": "Large Cap Universe", "symbols": ["aapl", "MSFT", "aapl"]}, handle)

            artifact = SimpleNamespace(
                pk=17,
                uri=payload_path,
                content={"symbols": ["tsla"]},
                metadata={"name": "Metadata Name"},
                pipeline_run=SimpleNamespace(name="Run Name"),
            )

            record = workflow._build_universe_artifact_record(artifact)

        self.assertEqual(record.artifact_id, 17)
        self.assertEqual(record.name, "Metadata Name")
        self.assertEqual(record.symbols, ["AAPL", "MSFT"])


if __name__ == "__main__":
    unittest.main()
