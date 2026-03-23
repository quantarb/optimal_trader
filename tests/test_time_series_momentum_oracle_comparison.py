from __future__ import annotations

import os
import sys
import tempfile
import types
from io import StringIO
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import django
import numpy as np
import pandas as pd
from django.core.management import call_command
from django.test import SimpleTestCase

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")
django.setup()

fake_raw_stack = types.ModuleType("ml.raw_stack")
fake_raw_stack.train_ae = lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError)
sys.modules.setdefault("ml.raw_stack", fake_raw_stack)

from ml.model_runtime import score_artifact_rows
from pipeline.time_series_momentum_oracle_comparison import (
    select_symbols_with_stable_metadata_filter,
    write_time_series_momentum_oracle_comparison_report,
)
from workflows import strategy_signal_support


class _DummyProbabilityModel:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        positive = (pd.to_numeric(frame["feature_a"], errors="coerce").fillna(0.0) > 0).astype(float)
        return np.column_stack([1.0 - positive, positive])


class _DummyDirectionalClassifier:
    def __init__(self) -> None:
        self._used_features = ["feature_a", "feature_b"]
        self.model = _DummyProbabilityModel()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.where(pd.to_numeric(frame["feature_a"], errors="coerce").fillna(0.0) > 0.0, 1.0, -1.0)


class OracleComparisonUnitTests(TestCase):
    def test_score_artifact_rows_complete_case_drops_incomplete_feature_rows(self):
        feature_df = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "AAA", "feature_a": 1.0, "feature_b": np.nan},
                {"date": "2024-01-03", "symbol": "BBB", "feature_a": 2.0, "feature_b": 3.0},
            ]
        )
        model = _DummyDirectionalClassifier()

        any_coverage = score_artifact_rows(
            model_obj=model,
            feature_df=feature_df,
            feature_cols=["feature_a", "feature_b"],
            missing_feature_policy="any_coverage",
        )
        complete_case = score_artifact_rows(
            model_obj=model,
            feature_df=feature_df,
            feature_cols=["feature_a", "feature_b"],
            missing_feature_policy="complete_case",
        )

        self.assertEqual(any_coverage["symbol"].tolist(), ["AAA", "BBB"])
        self.assertEqual(complete_case["symbol"].tolist(), ["BBB"])

    def test_collect_prediction_components_carries_classifier_direction(self):
        feature_df = pd.DataFrame(
            [
                {
                    "ret_1": 0.01,
                    "clf__prediction_score": 0.25,
                    "clf__raw_prediction": -1.0,
                }
            ]
        )
        panel_meta = {
            "extra_panel_sources": [
                {
                    "artifact_type": "CLASSIFIER_PREDICTIONS",
                    "columns": ["clf__prediction_score", "clf__raw_prediction"],
                }
            ]
        }

        enriched = strategy_signal_support._collect_prediction_components(feature_df, panel_meta)

        self.assertAlmostEqual(float(enriched.iloc[0]["prob_buy"]), 0.25)
        self.assertAlmostEqual(float(enriched.iloc[0]["direction"]), -1.0)

    def test_stable_metadata_filter_excludes_market_cap_numeric_columns(self):
        metadata_rows = [
            {
                "symbol": "AAA",
                "sector": "Technology",
                "industry": "Software",
                "country": "US",
                "exchange": "NASDAQ",
                "avg_market_cap": 1_000_000_000_000.0,
            },
            {
                "symbol": "BBB",
                "sector": "Technology",
                "industry": "Semiconductors",
                "country": "US",
                "exchange": "NASDAQ",
                "avg_market_cap": 900_000_000_000.0,
            },
            {
                "symbol": "CCC",
                "sector": "Utilities",
                "industry": "Electric",
                "country": "US",
                "exchange": "NYSE",
                "avg_market_cap": 100_000_000_000.0,
            },
            {
                "symbol": "DDD",
                "sector": "Utilities",
                "industry": "Water",
                "country": "US",
                "exchange": "NYSE",
                "avg_market_cap": 90_000_000_000.0,
            },
        ]
        target_rows = [
            {"symbol": "AAA", "symbol_profitable": 1},
            {"symbol": "BBB", "symbol_profitable": 1},
            {"symbol": "CCC", "symbol_profitable": 0},
            {"symbol": "DDD", "symbol_profitable": 0},
        ]

        result = select_symbols_with_stable_metadata_filter(
            metadata_rows=metadata_rows,
            target_rows=target_rows,
            minimum_selected_symbols=1,
            max_depth=2,
            min_samples_leaf=1,
        )

        self.assertEqual(set(result["selected_symbols"]), {"AAA", "BBB"})
        self.assertTrue(result["feature_columns"])
        self.assertFalse(any("avg_market_cap" in str(column) for column in result["feature_columns"]))

    def test_report_writer_includes_runtime_and_tree_sections(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "tsmom_oracle_report.md"
            write_time_series_momentum_oracle_comparison_report(
                report_path=report_path,
                payload={
                    "backtest_start_date": "2020-01-01",
                    "backtest_end_date": "",
                    "pre_backtest_end_date": "2019-12-31",
                    "label_config": {"k_params": {"YE": [1, 2, 4, 8]}, "min_profit_pct": 10.0},
                    "feature_usage": {
                        "artifact_scope": "full_feature_artifact",
                        "missing_feature_policy": "complete_case",
                    },
                    "universe_rows": [
                        {"universe_key": "1t", "universe_label": "1T+ market cap", "symbol_count": 2},
                    ],
                    "aggregate_rows": [
                        {
                            "universe_key": "1t",
                            "universe_label": "1T+ market cap",
                            "strategy_name": "baseline",
                            "filter_name": "no_filter",
                            "variant_name": "baseline__no_filter",
                            "sharpe": 0.45,
                            "cumulative_return": 0.12,
                            "max_drawdown": -0.08,
                            "trades": 14,
                            "selected_symbol_count": 2,
                            "total_runtime_seconds": 5.0,
                        },
                    ],
                    "filter_diagnostic_rows": [
                        {
                            "universe_key": "1t",
                            "universe_label": "1T+ market cap",
                            "strategy_name": "baseline",
                            "selection_count": 1,
                            "universe_symbol_count": 2,
                            "positive_target_rate": 0.5,
                            "tree_depth": 1,
                            "top_features": [("sector_Technology", 1.0)],
                            "selected_sector_counts": {"Technology": 1},
                            "selected_industry_counts": {"Software": 1},
                            "selected_country_counts": {"US": 1},
                            "selected_exchange_counts": {"NASDAQ": 1},
                            "tree_rules": "|--- sector_Technology <= 0.50",
                        }
                    ],
                    "symbol_diagnostics_aggregate_rows": [
                        {
                            "universe_label": "1T+ market cap",
                            "strategy_name": "baseline",
                            "symbol": "AAA",
                            "sharpe": 0.6,
                            "avg_trade_return": 0.04,
                            "trade_count": 3,
                        }
                    ],
                    "runtime_rows": [
                        {
                            "universe_label": "1T+ market cap",
                            "strategy_name": "baseline",
                            "filter_name": "no_filter",
                            "dataset_build_seconds": 0.0,
                            "fit_seconds": 0.0,
                            "score_seconds": 0.0,
                            "backtest_seconds": 1.5,
                            "filter_training_time_sec": 0.2,
                            "total_runtime_seconds": 1.7,
                        }
                    ],
                    "summary_json_path": "data/pipeline_artifacts/summary.json",
                    "summary_csv_path": "data/pipeline_artifacts/summary.csv",
                    "filter_diagnostics_csv_path": "data/pipeline_artifacts/filter.csv",
                    "symbol_diagnostics_csv_path": "data/pipeline_artifacts/symbols.csv",
                    "symbol_diagnostics_aggregate_csv_path": "data/pipeline_artifacts/symbols_agg.csv",
                    "runtime_analysis_csv_path": "data/pipeline_artifacts/runtime.csv",
                },
            )

            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("## Runtime Comparison", report_text)
            self.assertIn("Walk-forward optimization: not used", report_text)
            self.assertIn("complete_case", report_text)
            self.assertIn("Decision Tree Summaries", report_text)


class OracleComparisonCommandTests(SimpleTestCase):
    def test_command_uses_requested_defaults(self):
        stdout = StringIO()
        with patch(
            "pipeline.management.commands.run_time_series_momentum_oracle_comparison.run_time_series_momentum_oracle_comparison_experiment",
            return_value={"summary_json_path": "data/pipeline_artifacts/test.json"},
        ) as mocked_runner, patch(
            "pipeline.management.commands.run_time_series_momentum_oracle_comparison.write_time_series_momentum_oracle_comparison_report"
        ) as mocked_writer:
            call_command("run_time_series_momentum_oracle_comparison", stdout=stdout)

        mocked_runner.assert_called_once()
        kwargs = mocked_runner.call_args.kwargs
        self.assertEqual(kwargs["tiers"], ["1t", "100b", "10b"])
        self.assertEqual(kwargs["backtest_start_date"], "2020-01-01")
        self.assertEqual(kwargs["label_ks"], [1, 2, 4, 8])
        self.assertEqual(float(kwargs["min_profit_pct"]), 10.0)
        self.assertIsNone(kwargs["max_symbols_per_tier"])
        mocked_writer.assert_called_once()
        self.assertIn("Research summary: data/pipeline_artifacts/test.json", stdout.getvalue())
