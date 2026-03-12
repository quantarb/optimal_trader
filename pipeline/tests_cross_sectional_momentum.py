from __future__ import annotations

import pandas as pd
from django.test import SimpleTestCase

from pipeline.symbol_diagnostics import aggregate_symbol_diagnostic_rows
from pipeline.strategy_definitions import ResolvedStrategyDefinition, apply_strategy_definition
from pipeline.universe_selection import filter_symbols_by_price_history


class CrossSectionalMomentumCapabilityTests(SimpleTestCase):
    def test_apply_strategy_definition_supports_cross_sectional_quantile_sleeves(self):
        feature_df = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "A", "strategy_score": 0.40, "px__ret_63_d": 0.40},
                {"date": "2024-01-02", "symbol": "B", "strategy_score": 0.30, "px__ret_63_d": 0.30},
                {"date": "2024-01-02", "symbol": "C", "strategy_score": -0.10, "px__ret_63_d": -0.10},
                {"date": "2024-01-02", "symbol": "D", "strategy_score": -0.20, "px__ret_63_d": -0.20},
                {"date": "2024-02-01", "symbol": "A", "strategy_score": 0.50, "px__ret_63_d": 0.50},
                {"date": "2024-02-01", "symbol": "B", "strategy_score": -0.40, "px__ret_63_d": -0.40},
                {"date": "2024-02-01", "symbol": "C", "strategy_score": 0.20, "px__ret_63_d": 0.20},
                {"date": "2024-02-01", "symbol": "D", "strategy_score": -0.30, "px__ret_63_d": -0.30},
                {"date": "2024-03-01", "symbol": "A", "strategy_score": -0.50, "px__ret_63_d": -0.50},
                {"date": "2024-03-01", "symbol": "B", "strategy_score": 0.60, "px__ret_63_d": 0.60},
                {"date": "2024-03-01", "symbol": "C", "strategy_score": 0.30, "px__ret_63_d": 0.30},
                {"date": "2024-03-01", "symbol": "D", "strategy_score": -0.10, "px__ret_63_d": -0.10},
            ]
        )
        definition = ResolvedStrategyDefinition(
            definition_id=1,
            name="cross-sectional",
            slug="cross-sectional",
            strategy_type="notebook_topk_v1",
            config={
                "rebalance_freq": "M",
                "gross_exposure": 1.0,
                "selection_side": "long_short",
                "signal_combination": "direct",
                "portfolio_construction": "cross_sectional_quantiles",
                "cross_sectional_score_field": "px__ret_63_d",
                "cross_sectional_bucket_count": 2,
                "long_bucket": "top",
                "short_bucket": "bottom",
                "holding_period_rebalances": 2,
                "ranking_lag_days": 0,
            },
        )

        strategy_df, meta = apply_strategy_definition(feature_df, definition)
        feb_rows = strategy_df[strategy_df["date"] == "2024-02-01"].set_index("symbol")
        mar_rows = strategy_df[strategy_df["date"] == "2024-03-01"].set_index("symbol")

        self.assertEqual(meta["strategy_config"]["portfolio_construction"], "cross_sectional_quantiles")
        self.assertAlmostEqual(float(feb_rows.loc["A", "target_weight"]), 0.25, places=8)
        self.assertAlmostEqual(float(feb_rows.loc["B", "target_weight"]), 0.00, places=8)
        self.assertAlmostEqual(float(feb_rows.loc["C", "target_weight"]), 0.00, places=8)
        self.assertAlmostEqual(float(feb_rows.loc["D", "target_weight"]), -0.25, places=8)
        self.assertEqual(int(feb_rows.loc["A", "selected_on_rebalance"]), 1)
        self.assertEqual(int(feb_rows.loc["A", "cross_sectional_bucket"]), 2)
        self.assertEqual(int(feb_rows.loc["D", "cross_sectional_bucket"]), 1)

        self.assertAlmostEqual(float(mar_rows.loc["A", "target_weight"]), 0.00, places=8)
        self.assertAlmostEqual(float(mar_rows.loc["B", "target_weight"]), 0.00, places=8)
        self.assertAlmostEqual(float(mar_rows.loc["C", "target_weight"]), 0.25, places=8)
        self.assertAlmostEqual(float(mar_rows.loc["D", "target_weight"]), -0.25, places=8)

    def test_filter_symbols_by_price_history_respects_required_window(self):
        def fake_loader(symbols, *, start_date=None, end_date=None):
            del start_date, end_date
            full_index = pd.date_range("2023-01-02", periods=260, freq="B")
            short_index = pd.date_range("2024-01-02", periods=20, freq="B")
            return {
                "AAPL": pd.DataFrame({"close": range(len(full_index))}, index=full_index),
                "MSFT": pd.DataFrame({"close": range(len(short_index))}, index=short_index),
            }

        filtered = filter_symbols_by_price_history(
            ["AAPL", "MSFT"],
            required_start_date="2023-01-15",
            required_end_date="2023-12-15",
            min_history_days=200,
            history_loader=fake_loader,
        )

        self.assertEqual(filtered, ["AAPL"])

    def test_aggregate_symbol_diagnostics_supports_custom_group_keys(self):
        rows = aggregate_symbol_diagnostic_rows(
            [
                {
                    "strategy_name": "jt1993_j6_k6_lag5",
                    "fold_name": "wf_2024",
                    "symbol": "AAPL",
                    "sharpe": 1.2,
                    "avg_trade_return": 0.03,
                    "hit_rate": 0.6,
                    "max_drawdown": -0.1,
                    "trade_count": 4,
                    "turnover": 1.0,
                    "active_days": 10,
                    "selected_days": 10,
                    "avg_abs_weight": 0.1,
                    "cumulative_return": 0.12,
                    "final_equity": 1.12,
                }
            ],
            group_keys=("strategy_name", "symbol"),
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["strategy_name"], "jt1993_j6_k6_lag5")
