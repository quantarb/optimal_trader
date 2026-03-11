from __future__ import annotations

import unittest

import pandas as pd

from domain.backtests import StrategyBacktestSpec, StrategyDatasetSpec
from domain.features.panel import representation_embedding_dataset_rows
from domain.features.specs import FeatureBuildSpec
from domain.trades.operations import apply_trade_deduplication
from domain.trades.panel import labels_panel_to_trades_df


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
