from __future__ import annotations

import pandas as pd
from django.test import SimpleTestCase

from pipeline.factor_analysis import compute_factor_correlation_rows, compute_strategy_factor_exposure_rows, summarize_return_frame
from pipeline.factor_signals import build_multi_factor_score_frame
from pipeline.strategy_definitions import ResolvedStrategyDefinition, apply_strategy_definition


class FamaFrenchFactorCapabilityTests(SimpleTestCase):
    def test_apply_strategy_definition_supports_long_short_factor_portfolios(self):
        feature_df = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "A", "market_cap_signal": 100.0, "strategy_score": 0.0},
                {"date": "2024-01-02", "symbol": "B", "market_cap_signal": 90.0, "strategy_score": 0.0},
                {"date": "2024-01-02", "symbol": "C", "market_cap_signal": 80.0, "strategy_score": 0.0},
                {"date": "2024-01-02", "symbol": "D", "market_cap_signal": 30.0, "strategy_score": 0.0},
                {"date": "2024-01-02", "symbol": "E", "market_cap_signal": 20.0, "strategy_score": 0.0},
                {"date": "2024-01-02", "symbol": "F", "market_cap_signal": 10.0, "strategy_score": 0.0},
            ]
        )
        definition = ResolvedStrategyDefinition(
            definition_id=1,
            name="factor",
            slug="factor",
            strategy_type="notebook_topk_v1",
            config={
                "rebalance_freq": "M",
                "gross_exposure": 1.0,
                "selection_side": "long_short",
                "signal_combination": "direct",
                "portfolio_construction": "long_short_factor",
                "factor_signal": "market_cap_signal",
                "long_quantile": 0.5,
                "short_quantile": 0.5,
                "higher_score_is_better": False,
            },
        )

        strategy_df, meta = apply_strategy_definition(feature_df, definition)
        rows = strategy_df.set_index("symbol")

        self.assertEqual(meta["strategy_config"]["portfolio_construction"], "long_short_factor")
        self.assertAlmostEqual(float(rows.loc["D", "target_weight"]), 1.0 / 6.0, places=8)
        self.assertAlmostEqual(float(rows.loc["E", "target_weight"]), 1.0 / 6.0, places=8)
        self.assertAlmostEqual(float(rows.loc["F", "target_weight"]), 1.0 / 6.0, places=8)
        self.assertAlmostEqual(float(rows.loc["A", "target_weight"]), -1.0 / 6.0, places=8)
        self.assertAlmostEqual(float(rows.loc["B", "target_weight"]), -1.0 / 6.0, places=8)
        self.assertAlmostEqual(float(rows.loc["C", "target_weight"]), -1.0 / 6.0, places=8)
        self.assertEqual(int(rows.loc["F", "cross_sectional_bucket"]), 2)
        self.assertEqual(int(rows.loc["A", "cross_sectional_bucket"]), 1)

    def test_build_multi_factor_score_frame_combines_cross_sectional_components(self):
        feature_df = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "AAA", "mom": 0.30, "value": 3.0, "size": 100.0},
                {"date": "2024-01-02", "symbol": "BBB", "mom": 0.10, "value": 2.0, "size": 50.0},
                {"date": "2024-01-02", "symbol": "CCC", "mom": -0.20, "value": 1.0, "size": 10.0},
            ]
        )

        scored_df, meta = build_multi_factor_score_frame(
            feature_df,
            factor_components=[
                {"name": "momentum", "expression": "mom", "weight": 1.0, "transform": "rank_pct", "higher_is_better": True},
                {"name": "value", "expression": "value", "weight": 1.0, "transform": "rank_pct", "higher_is_better": False},
                {"name": "size", "expression": "size", "weight": 1.0, "transform": "rank_pct", "higher_is_better": False},
            ],
            output_col="factor_model_score",
        )

        self.assertEqual(len(meta["component_columns"]), 3)
        ordered = scored_df.sort_values("factor_model_score", ascending=False)["symbol"].tolist()
        self.assertEqual(ordered[0], "CCC")
        self.assertEqual(ordered[-1], "AAA")

    def test_factor_analysis_computes_metrics_correlations_and_exposures(self):
        dates = pd.date_range("2024-01-02", periods=6, freq="B")
        mkt = pd.DataFrame({"date": dates, "net_daily_return": [0.01, 0.00, 0.02, -0.01, 0.01, 0.00], "turnover": 0.1})
        smb = pd.DataFrame({"date": dates, "net_daily_return": [0.00, 0.01, -0.01, 0.02, 0.01, -0.02], "turnover": 0.2})
        hml = pd.DataFrame({"date": dates, "net_daily_return": [0.02, -0.01, 0.01, 0.00, -0.01, 0.01], "turnover": 0.15})
        strategy = pd.DataFrame(
            {
                "date": dates,
                "net_daily_return": (
                    0.001
                    + (0.5 * mkt["net_daily_return"])
                    + (1.2 * smb["net_daily_return"])
                    - (0.3 * hml["net_daily_return"])
                ),
                "turnover": 0.3,
            }
        )

        summary = summarize_return_frame(mkt, series_name="MKT", series_kind="factor")
        correlation_rows = compute_factor_correlation_rows({"MKT": mkt, "SMB": smb, "HML": hml})
        exposure_rows = compute_strategy_factor_exposure_rows({"multi_factor_rank": strategy}, {"MKT": mkt, "SMB": smb, "HML": hml})

        self.assertEqual(summary["series_name"], "MKT")
        self.assertEqual(summary["days"], 6)
        self.assertTrue(any(row["left_factor"] == "MKT" and row["right_factor"] == "MKT" and float(row["correlation"]) == 1.0 for row in correlation_rows))
        self.assertEqual(len(exposure_rows), 1)
        self.assertAlmostEqual(float(exposure_rows[0]["alpha"]), 0.001, places=6)
        self.assertAlmostEqual(float(exposure_rows[0]["beta_mkt"]), 0.5, places=6)
        self.assertAlmostEqual(float(exposure_rows[0]["beta_smb"]), 1.2, places=6)
        self.assertAlmostEqual(float(exposure_rows[0]["beta_hml"]), -0.3, places=6)
        self.assertAlmostEqual(float(exposure_rows[0]["r_squared"]), 1.0, places=6)
