from __future__ import annotations

import unittest

import pandas as pd

from analysis.oracle_entry_exit_dataset import build_oracle_entry_exit_frames
from domain.backtests import StrategyBacktestSpec, StrategyDatasetSpec
from domain.features.technical import compute_features_worldclass
from domain.features.panel import representation_embedding_dataset_rows
from domain.features.specs import FeatureBuildSpec
from domain.trades.operations import apply_trade_deduplication
from domain.trades.panel import labels_panel_to_trades_df
from ml.frameworks.transformers.context_family_mtl import ContextFamilyMTLDataSpec
from ml.metrics import (
    build_action_classification_report_df,
    build_action_f1_comparison_df,
    build_context_numeric_result,
    build_flair_action_report,
    build_flair_regression_result,
    build_regression_task_report_df,
)


class ResearchCoreUnitTests(unittest.TestCase):
    def test_representation_embedding_rows_group_feature_families(self):
        frame = pd.DataFrame(
            [{"date": "2024-01-01", "symbol": "AAPL", "close": 100.0, "ret_1": 0.05, "km__ev_to_ebitda": 8.2}]
        )
        grouped = {"prices_div_adj": ["close", "ret_1"], "key_metrics": ["km__ev_to_ebitda"], "representation_embedding": []}
        rows = representation_embedding_dataset_rows(frame, grouped)
        self.assertEqual(rows[0]["families"]["price_technical"]["Close"], 100.0)
        self.assertEqual(rows[0]["families"]["valuation_quality"]["EV To EBITDA"], 8.2)

    def test_feature_build_spec_extracts_layered_config(self):
        spec = FeatureBuildSpec.from_mapping(
            {
                "include_price_technicals": True,
                "include_economic_indicators": False,
                "feature_start_date": "2024-01-01",
                "feature_end_date": "2024-03-31",
                "include_representation_embedding": True,
                "representation_embedding_store_dir": "/tmp/embeddings",
            }
        )
        self.assertTrue(spec.toggles.include_price_technicals)
        self.assertFalse(spec.toggles.include_economic_indicators)
        self.assertEqual(spec.start_date, "2024-01-01")
        self.assertEqual(spec.representation_embedding.store_dir, "/tmp/embeddings")

    def test_strategy_dataset_spec_extracts_prediction_inputs(self):
        spec = StrategyDatasetSpec.from_mapping(
            {
                "strategy_definition_id": "7",
                "prediction_artifact_ids": ["2", "2", "5", "bad"],
                "label_artifact_id": "9",
                "strategy_start_date": "2024-01-01",
                "strategy_end_date": "2024-03-31",
            }
        )
        self.assertEqual(spec.strategy_definition_id, 7)
        self.assertEqual(spec.prediction_artifact_ids, (2, 5))
        self.assertEqual(spec.label_artifact_id, 9)
        self.assertEqual(spec.start_date, "2024-01-01")
        self.assertEqual(spec.end_date, "2024-03-31")

    def test_strategy_backtest_spec_uses_transaction_cost_as_slippage_fallback(self):
        spec = StrategyBacktestSpec.from_mapping(
            {
                "transaction_cost_bps": "12.5",
                "execution_delay_days": "2",
                "turnover_half_l1": "false",
            }
        )
        self.assertEqual(spec.effective_slippage_bps(), 12.5)
        self.assertEqual(spec.execution_delay_days, 2)
        self.assertFalse(spec.turnover_half_l1)

    def test_trade_deduplication_prefers_best_return_for_same_entry(self):
        trade_rows = [
            {"symbol": "AAPL", "side": "long", "entry_date": "2024-01-01", "exit_date": "2024-01-03"},
            {"symbol": "AAPL", "side": "long", "entry_date": "2024-01-01", "exit_date": "2024-01-04"},
        ]
        completed = [
            {"symbol": "AAPL", "side": "long", "entry_date": "2024-01-01", "exit_date": "2024-01-03", "ret_dec": 0.05},
            {"symbol": "AAPL", "side": "long", "entry_date": "2024-01-01", "exit_date": "2024-01-04", "ret_dec": 0.15},
        ]
        _, kept = apply_trade_deduplication(trade_rows, completed, mode="entry_date")
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["ret_dec"], 0.15)

    def test_labels_panel_to_trades_df_pairs_rows(self):
        trades = labels_panel_to_trades_df(
            pd.DataFrame(
                [
                    {"date": "2024-01-01", "symbol": "AAPL", "side": "long", "horizon": "W", "trade_return": 0.12},
                    {"date": "2024-01-04", "symbol": "AAPL", "side": "long", "horizon": "W", "trade_return": 0.12},
                ]
            )
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["symbol"], "AAPL")

    def test_compute_features_worldclass_adds_21_day_return(self):
        dates = pd.bdate_range("2024-01-01", periods=30)
        prices = pd.DataFrame(
            {
                "open": pd.Series(range(100, 130), index=dates, dtype=float),
                "high": pd.Series(range(101, 131), index=dates, dtype=float),
                "low": pd.Series(range(99, 129), index=dates, dtype=float),
                "close": pd.Series(range(100, 130), index=dates, dtype=float),
                "volume": pd.Series([1_000_000] * len(dates), index=dates, dtype=float),
            },
            index=dates,
        )
        feature_df = compute_features_worldclass(prices)
        self.assertIn("Ret21d", feature_df.columns)
        expected = (129.0 / 108.0) - 1.0
        self.assertAlmostEqual(float(feature_df.iloc[-1]["Ret21d"]), expected, places=10)

    def test_build_oracle_entry_exit_frames_uses_action_sequence_labels(self):
        universe_df = pd.DataFrame(
            [
                {"symbol": "AAPL", "company_name": "Apple Inc.", "sector": "Tech", "industry": "Hardware"},
            ]
        )
        label_df = pd.DataFrame(
            [
                {
                    "trade_id": "trade-1",
                    "event": "entry",
                    "action_label": "buy",
                    "symbol": "AAPL",
                    "date": "2024-01-02",
                    "entry_date": "2024-01-02",
                    "exit_date": "2024-01-05",
                    "trade_return": 0.12,
                    "hold_days": 3,
                    "side": "long",
                },
                {
                    "trade_id": "trade-1",
                    "event": "exit",
                    "action_label": "sell",
                    "symbol": "AAPL",
                    "date": "2024-01-05",
                    "entry_date": "2024-01-02",
                    "exit_date": "2024-01-05",
                    "trade_return": 0.12,
                    "hold_days": 3,
                    "side": "long",
                },
            ]
        )
        completed_trades_df = pd.DataFrame()
        price_lookup_df = pd.DataFrame(
            [
                {"symbol": "AAPL", "date_text": "2024-01-02", "adj_open": 100.0, "adj_high": 101.0, "adj_low": 99.0, "adj_close": 100.5, "volume": 1_000_000},
                {"symbol": "AAPL", "date_text": "2024-01-05", "adj_open": 110.0, "adj_high": 111.0, "adj_low": 109.0, "adj_close": 110.5, "volume": 1_200_000},
            ]
        )

        trade_pair_df, state_df = build_oracle_entry_exit_frames(
            universe_df=universe_df,
            label_df=label_df,
            completed_trades_df=completed_trades_df,
            price_lookup_df=price_lookup_df,
        )

        self.assertEqual(len(trade_pair_df), 1)
        self.assertEqual(len(state_df), 2)
        self.assertEqual(trade_pair_df.iloc[0]["entry_action"], "buy")
        self.assertEqual(trade_pair_df.iloc[0]["exit_action"], "sell")
        self.assertIn("signed_trade_return_pct", trade_pair_df.columns)
        self.assertEqual(set(state_df["event_role"].tolist()), {"entry", "exit"})

    def test_flair_metric_helpers_match_macro_f1_report(self):
        y_true = ["buy", "buy", "short", "short"]
        y_pred = ["buy", "short", "short", "short"]
        flair_report = build_flair_action_report("entry", y_true, y_pred, ["buy", "short"])
        report_df = build_action_classification_report_df("dev", "entry", y_true, y_pred, ["buy", "short"])
        macro_from_df = float(report_df.loc[report_df["label"] == "macro avg", "f1-score"].iloc[0])
        self.assertAlmostEqual(float(flair_report["dict"]["macro avg"]["f1-score"]), macro_from_df)
        self.assertIn("micro avg", flair_report["dict"])

    def test_flair_regression_result_includes_expected_metrics(self):
        result = build_flair_regression_result("entry_return", [0.1, 0.5, 0.9], [0.2, 0.4, 0.8])
        self.assertEqual(result["task"], "entry_return")
        self.assertEqual(result["support"], 3)
        self.assertIn("AVG: mse:", result["detailed_result"])

    def test_build_action_f1_comparison_df_matches_report_macro_f1(self):
        metrics = {"entry_action_macro_f1": 0.6666666666666666}
        task_buffers = {
            "entry": {
                "action_true": ["buy", "buy", "short"],
                "action_pred": ["buy", "short", "short"],
            }
        }
        comparison_df = build_action_f1_comparison_df("dev", metrics, task_buffers, {"entry": ["buy", "short"]})
        self.assertEqual(len(comparison_df), 1)
        self.assertAlmostEqual(float(comparison_df.iloc[0]["difference"]), 0.0)

    def test_build_regression_task_report_df_includes_context_numeric_rows(self):
        report_df = build_regression_task_report_df(
            "dev",
            metrics={},
            task_buffers={},
            regression_specs=[("entry_return", [0.1, 0.5], [0.2, 0.4])],
            context_numeric_specs=[("transition", 2, 0.95, 0.31)],
        )
        self.assertEqual(set(report_df["task"].tolist()), {"entry_return", "transition"})
        transition_row = report_df.loc[report_df["task"] == "transition"].iloc[0]
        self.assertAlmostEqual(float(transition_row["context_cosine"]), 0.95)
        self.assertAlmostEqual(float(transition_row["numeric_recon_mae"]), 0.31)

    def test_build_context_numeric_result_formats_flair_like_summary(self):
        result = build_context_numeric_result("entry_reconstruction", 5, 0.99, 0.12)
        self.assertEqual(result["support"], 5)
        self.assertIn("AVG: context_cosine:", result["detailed_result"])

    def test_context_family_mtl_data_spec_builds_reverse_action_maps(self):
        spec = ContextFamilyMTLDataSpec(
            state_market_cols=["m1"],
            state_fundamental_cols=["f1"],
            state_macro_cols=["x1"],
            entry_market_cols=["em1"],
            entry_fundamental_cols=["ef1"],
            entry_macro_cols=["ex1"],
            exit_market_cols=["xm1"],
            exit_fundamental_cols=["xf1"],
            exit_macro_cols=["xx1"],
            state_market_stats={"mean": {"m1": 0.0}, "std": {"m1": 1.0}},
            state_fundamental_stats={"mean": {"f1": 0.0}, "std": {"f1": 1.0}},
            state_macro_stats={"mean": {"x1": 0.0}, "std": {"x1": 1.0}},
            exit_market_stats={"mean": {"xm1": 0.0}, "std": {"xm1": 1.0}},
            exit_fundamental_stats={"mean": {"xf1": 0.0}, "std": {"xf1": 1.0}},
            exit_macro_stats={"mean": {"xx1": 0.0}, "std": {"xx1": 1.0}},
            entry_action_to_id={"buy": 0, "short": 1},
            exit_action_to_id={"sell": 0, "cover": 1},
        )
        self.assertEqual(spec.id_to_entry_action[0], "buy")
        self.assertEqual(spec.id_to_exit_action[1], "cover")
