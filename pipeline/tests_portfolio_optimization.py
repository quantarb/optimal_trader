from __future__ import annotations

import numpy as np
import pandas as pd
from django.test import SimpleTestCase, TestCase

from backtest.backtest import ExecutionConfig, backtest_panel, build_panel_from_daily_by_symbol
from pipeline.models import Artifact, PipelineRun
from pipeline.portfolio_optimization import (
    PortfolioConstraintConfig,
    PortfolioOptimizationConfig,
    PortfolioOptimizationResult,
    PortfolioRiskModelConfig,
    RiskModelEstimate,
    optimize_mean_variance_portfolio,
)
from pipeline.portfolio_optimization_research import (
    _collect_weight_and_diagnostic_rows,
    write_portfolio_optimization_report,
)
from pipeline.strategy_definitions import ResolvedStrategyDefinition, apply_strategy_definition
from pipeline.test_support import Mag7FixtureMixin


class PortfolioOptimizationCapabilityTests(SimpleTestCase):
    def test_optimize_mean_variance_portfolio_respects_constraints(self):
        expected_returns = pd.Series({"A": 0.08, "B": 0.03, "C": -0.02, "D": -0.07}, dtype=float)
        covariance = np.array(
            [
                [0.040, 0.010, 0.002, 0.001],
                [0.010, 0.035, 0.001, 0.002],
                [0.002, 0.001, 0.030, 0.008],
                [0.001, 0.002, 0.008, 0.045],
            ],
            dtype=float,
        )
        risk_model = RiskModelEstimate(
            symbols=tuple(expected_returns.index),
            covariance=covariance,
            idiosyncratic_variance=np.diag(covariance).astype(float),
            factor_loadings=None,
            factor_names=(),
            model_type="sample_covariance",
            observations=63,
            shrinkage=0.15,
            variance_floor=1e-6,
            condition_number=10.0,
            min_eigenvalue=0.02,
            max_eigenvalue=0.20,
        )
        config = PortfolioOptimizationConfig(
            expected_return_input="predicted_return",
            alpha_scale=1.0,
            normalize_expected_returns=False,
            demean_expected_returns=False,
            risk_aversion=2.0,
            turnover_penalty=0.1,
            constraints=PortfolioConstraintConfig(
                gross_exposure_limit=1.0,
                net_exposure_target=0.0,
                max_name_weight=0.35,
                allow_short=True,
            ),
        )

        result = optimize_mean_variance_portfolio(
            expected_returns,
            risk_model=risk_model,
            previous_weights={"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0},
            config=config,
        )

        self.assertIsInstance(result, PortfolioOptimizationResult)
        self.assertLessEqual(float(result.gross_exposure), 1.0 + 1e-6)
        self.assertAlmostEqual(float(result.net_exposure), 0.0, places=6)
        self.assertLessEqual(float(result.max_abs_weight), 0.35 + 1e-6)
        self.assertGreater(float(result.weights["A"]), 0.0)
        self.assertLess(float(result.weights["D"]), 0.0)

    def test_optimize_mean_variance_portfolio_applies_turnover_cap(self):
        expected_returns = pd.Series({"A": -0.05, "B": -0.01, "C": 0.03, "D": 0.07}, dtype=float)
        covariance = np.eye(4, dtype=float) * 0.04
        risk_model = RiskModelEstimate(
            symbols=tuple(expected_returns.index),
            covariance=covariance,
            idiosyncratic_variance=np.diag(covariance).astype(float),
            factor_loadings=None,
            factor_names=(),
            model_type="sample_covariance",
            observations=20,
            shrinkage=0.1,
            variance_floor=1e-6,
            condition_number=1.0,
            min_eigenvalue=0.04,
            max_eigenvalue=0.04,
        )
        config = PortfolioOptimizationConfig(
            expected_return_input="predicted_return",
            alpha_scale=1.0,
            normalize_expected_returns=False,
            demean_expected_returns=False,
            risk_aversion=0.5,
            turnover_penalty=0.0,
            turnover_cap=0.05,
            constraints=PortfolioConstraintConfig(
                gross_exposure_limit=1.0,
                net_exposure_target=0.0,
                max_name_weight=0.30,
                allow_short=True,
            ),
        )

        result = optimize_mean_variance_portfolio(
            expected_returns,
            risk_model=risk_model,
            previous_weights={"A": 0.25, "B": 0.25, "C": -0.25, "D": -0.25},
            config=config,
        )

        self.assertLessEqual(float(result.turnover), 0.05 + 1e-6)

    def test_apply_strategy_definition_supports_optimized_mean_variance_construction(self):
        feature_df = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "A", "strategy_score": 0.8, "ret_1": 0.03, "close": 100.0, "volume": 1_000_000},
                {"date": "2024-01-02", "symbol": "B", "strategy_score": 0.4, "ret_1": 0.02, "close": 100.0, "volume": 1_000_000},
                {"date": "2024-01-02", "symbol": "C", "strategy_score": -0.3, "ret_1": -0.01, "close": 100.0, "volume": 1_000_000},
                {"date": "2024-01-02", "symbol": "D", "strategy_score": -0.7, "ret_1": -0.03, "close": 100.0, "volume": 1_000_000},
                {"date": "2024-01-03", "symbol": "A", "strategy_score": 0.7, "ret_1": 0.01, "close": 101.0, "volume": 1_000_000},
                {"date": "2024-01-03", "symbol": "B", "strategy_score": 0.3, "ret_1": 0.00, "close": 101.0, "volume": 1_000_000},
                {"date": "2024-01-03", "symbol": "C", "strategy_score": -0.2, "ret_1": -0.02, "close": 99.0, "volume": 1_000_000},
                {"date": "2024-01-03", "symbol": "D", "strategy_score": -0.6, "ret_1": -0.01, "close": 99.0, "volume": 1_000_000},
                {"date": "2024-01-04", "symbol": "A", "strategy_score": 0.9, "ret_1": 0.02, "close": 102.0, "volume": 1_000_000},
                {"date": "2024-01-04", "symbol": "B", "strategy_score": 0.2, "ret_1": 0.01, "close": 102.0, "volume": 1_000_000},
                {"date": "2024-01-04", "symbol": "C", "strategy_score": -0.1, "ret_1": -0.01, "close": 98.0, "volume": 1_000_000},
                {"date": "2024-01-04", "symbol": "D", "strategy_score": -0.8, "ret_1": -0.02, "close": 98.0, "volume": 1_000_000},
                {"date": "2024-02-01", "symbol": "A", "strategy_score": 0.6, "ret_1": 0.01, "close": 103.0, "volume": 1_000_000},
                {"date": "2024-02-01", "symbol": "B", "strategy_score": 0.1, "ret_1": 0.00, "close": 103.0, "volume": 1_000_000},
                {"date": "2024-02-01", "symbol": "C", "strategy_score": -0.2, "ret_1": -0.01, "close": 97.0, "volume": 1_000_000},
                {"date": "2024-02-01", "symbol": "D", "strategy_score": -0.9, "ret_1": -0.03, "close": 97.0, "volume": 1_000_000},
            ]
        )
        definition = ResolvedStrategyDefinition(
            definition_id=1,
            name="optimized",
            slug="optimized",
            strategy_type="notebook_topk_v1",
            config={
                "rebalance_freq": "M",
                "gross_exposure": 1.0,
                "selection_side": "long_short",
                "signal_combination": "direct",
                "portfolio_construction": "optimized_mean_variance",
                "cross_sectional_score_field": "strategy_score",
                "portfolio_optimization": {
                    "expected_return_input": "ranking_score",
                    "alpha_scale": 0.05,
                    "risk_aversion": 1.5,
                    "turnover_penalty": 0.1,
                    "constraints": {
                        "gross_exposure_limit": 1.0,
                        "net_exposure_target": 0.0,
                        "max_name_weight": 0.40,
                    },
                    "risk_model": {
                        "model_type": "sample_covariance",
                        "lookback_days": 3,
                        "min_observations": 2,
                        "shrinkage": 0.10,
                    },
                },
            },
        )

        strategy_df, meta = apply_strategy_definition(feature_df, definition)
        feb_rows = strategy_df[strategy_df["date"] == "2024-02-01"].set_index("symbol")

        self.assertEqual(meta["strategy_config"]["portfolio_construction"], "optimized_mean_variance")
        self.assertIn("portfolio_optimization", meta["strategy_config"])
        self.assertLessEqual(float(feb_rows["target_weight"].abs().sum()), 1.0 + 1e-6)
        self.assertAlmostEqual(float(feb_rows["target_weight"].sum()), 0.0, places=6)
        self.assertLessEqual(float(feb_rows["target_weight"].abs().max()), 0.40 + 1e-6)
        self.assertIn("expected_return_estimate", strategy_df.columns)
        self.assertIn("optimization_status", strategy_df.columns)
        self.assertTrue(str(feb_rows.iloc[0]["optimization_status"]))

    def test_optimized_strategy_backtest_runs_with_existing_workflow(self):
        feature_df = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "A", "strategy_score": 0.7, "ret_1": 0.03, "close": 100.0, "volume": 1_000_000},
                {"date": "2024-01-02", "symbol": "B", "strategy_score": -0.6, "ret_1": -0.02, "close": 100.0, "volume": 1_000_000},
                {"date": "2024-01-03", "symbol": "A", "strategy_score": 0.8, "ret_1": 0.02, "close": 101.0, "volume": 1_000_000},
                {"date": "2024-01-03", "symbol": "B", "strategy_score": -0.7, "ret_1": -0.01, "close": 99.0, "volume": 1_000_000},
                {"date": "2024-02-01", "symbol": "A", "strategy_score": 0.6, "ret_1": 0.01, "close": 102.0, "volume": 1_000_000},
                {"date": "2024-02-01", "symbol": "B", "strategy_score": -0.5, "ret_1": -0.03, "close": 98.0, "volume": 1_000_000},
            ]
        )
        definition = ResolvedStrategyDefinition(
            definition_id=1,
            name="optimized-backtest",
            slug="optimized-backtest",
            strategy_type="notebook_topk_v1",
            config={
                "rebalance_freq": "M",
                "gross_exposure": 1.0,
                "selection_side": "long_short",
                "signal_combination": "direct",
                "portfolio_construction": "optimized_mean_variance",
                "cross_sectional_score_field": "strategy_score",
                "portfolio_optimization": {
                    "expected_return_input": "ranking_score",
                    "alpha_scale": 0.05,
                    "risk_aversion": 1.0,
                    "constraints": {
                        "gross_exposure_limit": 1.0,
                        "net_exposure_target": 0.0,
                        "max_name_weight": 0.60,
                    },
                    "risk_model": {
                        "model_type": "sample_covariance",
                        "lookback_days": 2,
                        "min_observations": 2,
                        "shrinkage": 0.05,
                    },
                },
            },
        )
        strategy_df, _meta = apply_strategy_definition(feature_df, definition)
        panel_input: dict[str, pd.DataFrame] = {}
        for symbol, group in feature_df.groupby("symbol", sort=True):
            panel_input[str(symbol)] = group[["date", "close"]].copy()
        panel = build_panel_from_daily_by_symbol(panel_input, include_cols=["close"])
        weights = (
            strategy_df.assign(date=pd.to_datetime(strategy_df["date"], errors="coerce"))
            .pivot_table(index="date", columns="symbol", values="target_weight", aggfunc="last")
            .sort_index()
        )

        class StaticWeightStrategy:
            name = "optimized-static"

            def __init__(self, frame: pd.DataFrame):
                self._frame = frame

            def compute_weights(self, panel: pd.DataFrame) -> pd.DataFrame:
                dates = sorted(panel.index.get_level_values("date").unique().tolist())
                columns = sorted(panel.index.get_level_values("symbol").unique().tolist())
                return self._frame.reindex(index=dates, columns=columns).fillna(0.0)

        result = backtest_panel(
            panel,
            strategy=StaticWeightStrategy(weights),
            cfg=ExecutionConfig(
                price_col="close",
                fee_bps=0.0,
                slippage_bps=0.0,
                use_lagged_weights=True,
            ),
        )

        self.assertGreater(len(result.equity_curve), 0)
        self.assertFalse(result.returns.empty)
        self.assertGreaterEqual(float(result.turnover.sum()), 0.0)


class PortfolioOptimizationResearchTests(Mag7FixtureMixin, TestCase):
    def test_collect_weight_and_diagnostic_rows_uses_strategy_and_backtest_artifacts(self):
        run = PipelineRun.objects.create(
            name="portfolio-opt-test",
            requested_job="portfolio_opt",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        strategy_artifact = Artifact.objects.create(
            pipeline_run=run,
            artifact_type="STRATEGY_DATASET",
            key="portfolio_opt_strategy",
            uri=self.write_csv(
                "portfolio_opt_strategy",
                [
                    "date",
                    "symbol",
                    "target_weight",
                    "rebalance_date",
                    "optimization_status",
                    "optimization_success",
                    "optimization_objective",
                    "optimization_variance",
                    "optimization_expected_portfolio_return",
                    "optimization_turnover",
                    "optimization_gross_exposure",
                    "optimization_net_exposure",
                    "optimization_max_abs_weight",
                    "optimization_constraint_violation",
                    "optimization_iterations",
                    "risk_model_type",
                    "risk_model_observations",
                    "risk_model_condition_number",
                    "risk_model_min_eigenvalue",
                    "risk_model_max_eigenvalue",
                    "risk_model_shrinkage",
                    "risk_model_variance_floor",
                    "neutrality_exposure_summary",
                ],
                [
                    {
                        "date": "2024-01-02",
                        "symbol": "AAPL",
                        "target_weight": 0.40,
                        "rebalance_date": 1,
                        "optimization_status": "ok",
                        "optimization_success": 1,
                        "optimization_objective": -0.02,
                        "optimization_variance": 0.01,
                        "optimization_expected_portfolio_return": 0.03,
                        "optimization_turnover": 0.20,
                        "optimization_gross_exposure": 0.80,
                        "optimization_net_exposure": 0.0,
                        "optimization_max_abs_weight": 0.40,
                        "optimization_constraint_violation": 0.0,
                        "optimization_iterations": 12,
                        "risk_model_type": "sample_covariance",
                        "risk_model_observations": 63,
                        "risk_model_condition_number": 4.0,
                        "risk_model_min_eigenvalue": 0.01,
                        "risk_model_max_eigenvalue": 0.04,
                        "risk_model_shrinkage": 0.10,
                        "risk_model_variance_floor": 0.000001,
                        "neutrality_exposure_summary": "",
                    },
                    {
                        "date": "2024-01-02",
                        "symbol": "MSFT",
                        "target_weight": -0.40,
                        "rebalance_date": 1,
                        "optimization_status": "ok",
                        "optimization_success": 1,
                        "optimization_objective": -0.02,
                        "optimization_variance": 0.01,
                        "optimization_expected_portfolio_return": 0.03,
                        "optimization_turnover": 0.20,
                        "optimization_gross_exposure": 0.80,
                        "optimization_net_exposure": 0.0,
                        "optimization_max_abs_weight": 0.40,
                        "optimization_constraint_violation": 0.0,
                        "optimization_iterations": 12,
                        "risk_model_type": "sample_covariance",
                        "risk_model_observations": 63,
                        "risk_model_condition_number": 4.0,
                        "risk_model_min_eigenvalue": 0.01,
                        "risk_model_max_eigenvalue": 0.04,
                        "risk_model_shrinkage": 0.10,
                        "risk_model_variance_floor": 0.000001,
                        "neutrality_exposure_summary": "",
                    },
                ],
            ),
            content={"rows": 2},
            metadata={},
        )
        backtest_artifact = Artifact.objects.create(
            pipeline_run=run,
            artifact_type="BACKTEST_RESULT",
            key="portfolio_opt_backtest",
            uri=self.write_csv(
                "portfolio_opt_backtest",
                ["date", "symbol", "target_weight"],
                [{"date": "2024-01-02", "symbol": "AAPL", "target_weight": 0.40}],
            ),
            content={
                "daily_rows": [
                    {"date": "2024-01-02", "positions": 2, "gross_exposure": 0.80, "turnover": 0.20, "net_daily_return": 0.01, "equity": 1.01},
                    {"date": "2024-01-03", "positions": 2, "gross_exposure": 0.80, "turnover": 0.00, "net_daily_return": 0.02, "equity": 1.03},
                ]
            },
            metadata={},
        )

        optimized_weight_rows, covariance_rows, turnover_rows, exposure_rows = _collect_weight_and_diagnostic_rows(
            [
                {
                    "signal_source": "baseline_momentum",
                    "signal_label": "Baseline Momentum",
                    "construction": "optimized_mean_variance",
                    "construction_label": "Optimized Mean-Variance",
                    "fold_name": "wf_2024",
                    "strategy_artifact_id": int(strategy_artifact.id),
                    "backtest_artifact_id": int(backtest_artifact.id),
                }
            ]
        )

        self.assertEqual(len(optimized_weight_rows), 2)
        self.assertEqual(len(covariance_rows), 1)
        self.assertEqual(len(turnover_rows), 2)
        self.assertEqual(len(exposure_rows), 1)
        self.assertEqual(covariance_rows[0]["risk_model_type"], "sample_covariance")

    def test_write_portfolio_optimization_report_includes_required_sections(self):
        report_path = self.temp_path / "portfolio_optimization_research.md"
        write_portfolio_optimization_report(
            report_path=report_path,
            payload={
                "symbols": ["AAPL", "MSFT"],
                "optimizer_settings": {
                    "expected_return_input": "ranking_score",
                    "risk_model_type": "sample_covariance",
                    "risk_lookback_days": 63,
                    "net_exposure_target": 0.0,
                    "max_name_weight": 0.10,
                    "turnover_penalty": 0.1,
                    "turnover_cap": 0.2,
                },
                "aggregate_rows": [
                    {
                        "signal_source": "baseline_momentum",
                        "signal_label": "Baseline Momentum",
                        "construction": "equal_weight_quantiles",
                        "construction_label": "Equal-Weight Quantiles",
                        "sharpe": 0.40,
                        "total_return": 0.08,
                        "max_drawdown": -0.10,
                        "total_turnover": 5.0,
                        "trade_count": 20,
                        "positive_fold_rate": 0.50,
                    },
                    {
                        "signal_source": "baseline_momentum",
                        "signal_label": "Baseline Momentum",
                        "construction": "optimized_mean_variance",
                        "construction_label": "Optimized Mean-Variance",
                        "sharpe": 0.60,
                        "total_return": 0.12,
                        "max_drawdown": -0.08,
                        "total_turnover": 4.0,
                        "trade_count": 18,
                        "positive_fold_rate": 0.75,
                    },
                ],
                "comparison_rows": [
                    {
                        "signal_source": "baseline_momentum",
                        "signal_label": "Baseline Momentum",
                        "sharpe_delta": 0.20,
                        "total_return_delta": 0.04,
                        "drawdown_delta": 0.02,
                        "turnover_delta": -1.0,
                    }
                ],
                "summary_json_path": "data/pipeline_artifacts/portfolio_metrics.json",
                "summary_csv_path": "data/pipeline_artifacts/portfolio_metrics.csv",
                "optimized_weights_csv_path": "data/pipeline_artifacts/optimized_weights.csv",
                "covariance_diagnostics_csv_path": "data/pipeline_artifacts/covariance_diagnostics.csv",
                "turnover_diagnostics_csv_path": "data/pipeline_artifacts/turnover_diagnostics.csv",
                "exposure_diagnostics_csv_path": "data/pipeline_artifacts/exposure_diagnostics.csv",
            },
        )

        report_text = report_path.read_text(encoding="utf-8")
        self.assertIn("## 1. Experiment setup", report_text)
        self.assertIn("## 3. Performance comparison", report_text)
        self.assertIn("## 5. Key observations", report_text)
        self.assertIn("Optimized Mean-Variance", report_text)
