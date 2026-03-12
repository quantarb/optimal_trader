from django.test import TestCase

from fmp.models import Symbol
from domain.backtests import StrategyBacktestSpec
from workflows.strategy import run_strategy_backtest
from .models import Artifact, PipelineRun
from .symbol_diagnostics import compute_symbol_strategy_diagnostics
from .symbol_filters import (
    build_symbol_feature_summary,
    build_symbol_metadata_filter_summary,
    select_symbols_with_learned_filter,
    select_symbols_with_metadata_filter,
    select_top_symbols_from_diagnostics,
)
from .test_support import Mag7FixtureMixin
from .time_series_momentum_market_cap_policy_comparison import write_market_cap_policy_comparison_report
from .time_series_momentum_policy_comparison import write_policy_comparison_report


class PolicyComparisonCapabilityTests(Mag7FixtureMixin, TestCase):
    def test_run_strategy_backtest_respects_allowed_symbols(self):
        strategy_run = PipelineRun.objects.create(
            name="allowed-symbols-strategy",
            requested_job="build_strategy_dataset",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        strategy_artifact = Artifact.objects.create(
            pipeline_run=strategy_run,
            artifact_type="STRATEGY_DATASET",
            key="allowed_symbols_strategy",
            uri=self.write_csv(
                "allowed_symbols_strategy",
                ["date", "symbol", "strategy_signal", "target_weight", "strategy_score", "ret_1", "close"],
                [
                    {"date": "2024-01-02", "symbol": "AAPL", "strategy_signal": 1, "target_weight": 0.5, "strategy_score": 0.6, "ret_1": 0.02, "close": 100.0},
                    {"date": "2024-01-02", "symbol": "MSFT", "strategy_signal": -1, "target_weight": -0.5, "strategy_score": -0.4, "ret_1": -0.01, "close": 100.0},
                    {"date": "2024-01-03", "symbol": "AAPL", "strategy_signal": 1, "target_weight": 0.5, "strategy_score": 0.6, "ret_1": 0.01, "close": 101.0},
                    {"date": "2024-01-03", "symbol": "MSFT", "strategy_signal": -1, "target_weight": -0.5, "strategy_score": -0.4, "ret_1": -0.02, "close": 99.0},
                ],
            ),
            content={},
            metadata={},
        )

        result = run_strategy_backtest(
            spec=StrategyBacktestSpec.from_mapping(
                {
                    "backtest_start_date": "2024-01-02",
                    "backtest_end_date": "2024-01-03",
                    "execution_delay_days": 0,
                    "allowed_symbols": ["AAPL"],
                }
            ),
            strategy_dataset_artifact=strategy_artifact,
        )

        self.assertEqual(sorted(result.trade_frame["symbol"].unique().tolist()), ["AAPL"])
        self.assertEqual(len(result.daily_rows), 2)

    def test_symbol_diagnostics_and_filters_rank_profitable_symbols(self):
        backtest_run = PipelineRun.objects.create(
            name="symbol-diagnostics-backtest",
            requested_job="backtest_strategy",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        backtest_artifact = Artifact.objects.create(
            pipeline_run=backtest_run,
            artifact_type="BACKTEST_RESULT",
            key="symbol_diagnostics_backtest",
            uri=self.write_csv(
                "symbol_diagnostics_backtest",
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
                    {"date": "2024-01-02", "symbol": "AAPL", "strategy_signal": 1, "strategy_score": 0.7, "target_weight": 0.5, "effective_weight": 0.5, "asset_return": 0.03, "gross_exposure": 0.5, "realized_return": 0.015, "turnover": 0.5, "turnover_cost": 0.0},
                    {"date": "2024-01-02", "symbol": "MSFT", "strategy_signal": -1, "strategy_score": -0.6, "target_weight": -0.5, "effective_weight": -0.5, "asset_return": 0.02, "gross_exposure": 0.5, "realized_return": -0.01, "turnover": 0.5, "turnover_cost": 0.0},
                    {"date": "2024-01-03", "symbol": "AAPL", "strategy_signal": 1, "strategy_score": 0.7, "target_weight": 0.5, "effective_weight": 0.5, "asset_return": 0.02, "gross_exposure": 0.5, "realized_return": 0.01, "turnover": 0.0, "turnover_cost": 0.0},
                    {"date": "2024-01-03", "symbol": "MSFT", "strategy_signal": -1, "strategy_score": -0.6, "target_weight": -0.5, "effective_weight": -0.5, "asset_return": -0.03, "gross_exposure": 0.5, "realized_return": 0.015, "turnover": 0.0, "turnover_cost": 0.0},
                    {"date": "2024-01-04", "symbol": "AAPL", "strategy_signal": 0, "strategy_score": 0.0, "target_weight": 0.0, "effective_weight": 0.0, "asset_return": 0.0, "gross_exposure": 0.0, "realized_return": 0.0, "turnover": 0.5, "turnover_cost": 0.0},
                    {"date": "2024-01-04", "symbol": "MSFT", "strategy_signal": 0, "strategy_score": 0.0, "target_weight": 0.0, "effective_weight": 0.0, "asset_return": 0.0, "gross_exposure": 0.0, "realized_return": 0.0, "turnover": 0.5, "turnover_cost": 0.0},
                ],
            ),
            content={"daily_rows": [{"date": "2024-01-02"}, {"date": "2024-01-03"}, {"date": "2024-01-04"}]},
            metadata={"backtest_config": {"fee_bps": 0.0, "slippage_bps": 0.0, "turnover_half_l1": True}},
        )
        diagnostics = compute_symbol_strategy_diagnostics(backtest_artifact)
        diag_by_symbol = {row["symbol"]: row for row in diagnostics}
        self.assertGreater(diag_by_symbol["AAPL"]["avg_trade_return"], diag_by_symbol["MSFT"]["avg_trade_return"])

        simple_filter = select_top_symbols_from_diagnostics(
            diagnostics,
            selection_fraction=0.5,
            minimum=1,
        )
        self.assertEqual(simple_filter["selected_symbols"], ["AAPL"])

        feature_run = PipelineRun.objects.create(
            name="symbol-feature-summary",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="symbol_feature_summary",
            uri=self.write_csv(
                "symbol_feature_summary",
                ["date", "symbol", "ret_1", "px__ret_252_d", "evt__revision"],
                [
                    {"date": "2024-01-01", "symbol": "AAPL", "ret_1": 0.02, "px__ret_252_d": 0.25, "evt__revision": 1.0},
                    {"date": "2024-01-02", "symbol": "AAPL", "ret_1": 0.03, "px__ret_252_d": 0.27, "evt__revision": 1.0},
                    {"date": "2024-01-01", "symbol": "MSFT", "ret_1": -0.01, "px__ret_252_d": -0.10, "evt__revision": 0.0},
                    {"date": "2024-01-02", "symbol": "MSFT", "ret_1": -0.02, "px__ret_252_d": -0.12, "evt__revision": 0.0},
                ],
            ),
            content={"rows": 4},
            metadata={},
        )
        feature_summary = build_symbol_feature_summary(feature_artifact, end_date="2024-01-02")
        learned_filter = select_symbols_with_learned_filter(
            feature_summary_rows=feature_summary,
            diagnostic_rows=diagnostics,
            selection_fraction=0.5,
            minimum=1,
            model_kind="decision_tree_regressor",
        )
        self.assertEqual(learned_filter["selected_symbols"], ["AAPL"])

    def test_policy_comparison_report_includes_required_sections(self):
        report_path = self.temp_path / "policy_comparison_report.md"
        write_policy_comparison_report(
            report_path=report_path,
            payload={
                "symbols": ["AAPL", "MSFT"],
                "aggregate_rows": [
                    {
                        "variant_name": "baseline__no_filter",
                        "strategy_name": "baseline",
                        "filter_name": "no_filter",
                        "sharpe": 0.3,
                        "total_return": 0.08,
                        "max_drawdown": -0.12,
                        "total_turnover": 5.0,
                        "trade_count": 20,
                        "mean_selected_symbol_count": 2.0,
                        "positive_fold_rate": 0.5,
                    },
                    {
                        "variant_name": "baseline__simple_filter",
                        "strategy_name": "baseline",
                        "filter_name": "simple_filter",
                        "sharpe": 0.5,
                        "total_return": 0.10,
                        "max_drawdown": -0.09,
                        "total_turnover": 4.0,
                        "trade_count": 18,
                        "mean_selected_symbol_count": 1.0,
                        "positive_fold_rate": 0.75,
                    },
                    {
                        "variant_name": "baseline__learned_filter",
                        "strategy_name": "baseline",
                        "filter_name": "learned_filter",
                        "sharpe": 0.4,
                        "total_return": 0.09,
                        "max_drawdown": -0.10,
                        "total_turnover": 4.5,
                        "trade_count": 19,
                        "mean_selected_symbol_count": 1.0,
                        "positive_fold_rate": 0.5,
                    },
                    {
                        "variant_name": "model__no_filter",
                        "strategy_name": "model",
                        "filter_name": "no_filter",
                        "sharpe": 0.2,
                        "total_return": 0.05,
                        "max_drawdown": -0.15,
                        "total_turnover": 6.0,
                        "trade_count": 22,
                        "mean_selected_symbol_count": 2.0,
                        "positive_fold_rate": 0.5,
                    },
                    {
                        "variant_name": "model__simple_filter",
                        "strategy_name": "model",
                        "filter_name": "simple_filter",
                        "sharpe": 0.25,
                        "total_return": 0.06,
                        "max_drawdown": -0.14,
                        "total_turnover": 5.5,
                        "trade_count": 21,
                        "mean_selected_symbol_count": 1.0,
                        "positive_fold_rate": 0.5,
                    },
                    {
                        "variant_name": "model__learned_filter",
                        "strategy_name": "model",
                        "filter_name": "learned_filter",
                        "sharpe": 0.35,
                        "total_return": 0.07,
                        "max_drawdown": -0.11,
                        "total_turnover": 5.2,
                        "trade_count": 20,
                        "mean_selected_symbol_count": 1.0,
                        "positive_fold_rate": 0.75,
                    },
                ],
                "symbol_diagnostics_aggregate_rows": [
                    {"strategy_name": "baseline", "filter_name": "no_filter", "symbol": "AAPL", "sharpe": 0.8},
                    {"strategy_name": "model", "filter_name": "no_filter", "symbol": "MSFT", "sharpe": 0.6},
                ],
                "selection_rows": [
                    {"strategy_name": "baseline", "filter_name": "simple_filter", "selected_symbols": ["AAPL"]},
                    {"strategy_name": "model", "filter_name": "learned_filter", "selected_symbols": ["MSFT"]},
                ],
                "summary_json_path": "data/pipeline_artifacts/test.json",
                "summary_csv_path": "data/pipeline_artifacts/test.csv",
                "symbol_diagnostics_test_csv_path": "data/pipeline_artifacts/test_test.csv",
                "symbol_diagnostics_aggregate_csv_path": "data/pipeline_artifacts/test_agg.csv",
            },
        )

        report_text = report_path.read_text(encoding="utf-8")
        self.assertIn("## 2. Walk-forward comparison", report_text)
        self.assertIn("baseline__simple_filter", report_text)
        self.assertIn("## 5. Symbol-level performance analysis", report_text)

    def test_metadata_filter_uses_symbol_metadata_only(self):
        for symbol, sector, industry, exchange, market_cap in (
            ("AAA1", "Technology", "Software", "NASDAQ", 200.0),
            ("BBB1", "Technology", "Semiconductors", "NASDAQ", 180.0),
            ("CCC1", "Utilities", "Electric", "NYSE", 90.0),
            ("DDD1", "Utilities", "Water", "NYSE", 85.0),
        ):
            Symbol.objects.update_or_create(
                symbol=symbol,
                defaults={
                    "company_name": f"{symbol} Corp",
                    "sector": sector,
                    "industry": industry,
                    "exchange": exchange,
                    "country": "US",
                    "market_cap": market_cap,
                },
            )

        feature_run = PipelineRun.objects.create(
            name="metadata-filter-features",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
        )
        feature_artifact = Artifact.objects.create(
            pipeline_run=feature_run,
            artifact_type="FEATURES",
            key="metadata_filter_features",
            uri=self.write_csv(
                "metadata_filter_features",
                ["date", "symbol", "km__marketcap", "px__ret_252_d", "px__ret_21_d"],
                [
                    {"date": "2024-01-01", "symbol": "AAA1", "km__marketcap": 210.0, "px__ret_252_d": 0.15, "px__ret_21_d": 0.02},
                    {"date": "2024-01-01", "symbol": "BBB1", "km__marketcap": 190.0, "px__ret_252_d": 0.14, "px__ret_21_d": 0.01},
                    {"date": "2024-01-01", "symbol": "CCC1", "km__marketcap": 88.0, "px__ret_252_d": -0.05, "px__ret_21_d": 0.01},
                    {"date": "2024-01-01", "symbol": "DDD1", "km__marketcap": 86.0, "px__ret_252_d": -0.04, "px__ret_21_d": 0.01},
                ],
            ),
            content={"rows": 4},
            metadata={},
        )
        metadata_rows = build_symbol_metadata_filter_summary(feature_artifact, end_date="2024-01-01")
        filter_result = select_symbols_with_metadata_filter(
            metadata_rows=metadata_rows,
            target_rows=[
                {"symbol": "AAA1", "symbol_profitable": 1},
                {"symbol": "BBB1", "symbol_profitable": 1},
                {"symbol": "CCC1", "symbol_profitable": 0},
                {"symbol": "DDD1", "symbol_profitable": 0},
            ],
            target_col="symbol_profitable",
            minimum_selected_symbols=1,
            max_depth=2,
            min_samples_leaf=1,
        )

        self.assertEqual(filter_result["selection_count"], 2)
        self.assertEqual(set(filter_result["selected_symbols"]), {"AAA1", "BBB1"})
        self.assertGreaterEqual(int(filter_result["tree_depth"]), 1)
        self.assertIn("sector", "\n".join(filter_result.get("feature_columns") or []))

    def test_market_cap_policy_comparison_report_includes_runtime_and_conclusions(self):
        report_path = self.temp_path / "market_cap_policy_comparison_report.md"
        write_market_cap_policy_comparison_report(
            report_path=report_path,
            payload={
                "universe_rows": [
                    {
                        "universe_key": "1t",
                        "universe_label": "1T+ market cap",
                        "symbol_count": 2,
                        "status": "succeeded",
                    }
                ],
                "aggregate_rows": [
                    {
                        "universe_key": "1t",
                        "universe_label": "1T+ market cap",
                        "policy_name": "baseline",
                        "strategy_name": "baseline",
                        "filter_name": "no_filter",
                        "variant_name": "baseline__no_filter",
                        "sharpe": 0.25,
                        "total_return": 0.10,
                        "max_drawdown": -0.12,
                        "total_turnover": 5.0,
                        "trade_count": 12,
                        "positive_fold_rate": 0.5,
                        "total_runtime_sec": 2.0,
                    },
                    {
                        "universe_key": "1t",
                        "universe_label": "1T+ market cap",
                        "policy_name": "model",
                        "strategy_name": "model",
                        "filter_name": "no_filter",
                        "variant_name": "model__no_filter",
                        "sharpe": 0.45,
                        "total_return": 0.18,
                        "max_drawdown": -0.10,
                        "total_turnover": 4.0,
                        "trade_count": 10,
                        "positive_fold_rate": 1.0,
                        "total_runtime_sec": 4.0,
                    },
                    {
                        "universe_key": "1t",
                        "universe_label": "1T+ market cap",
                        "policy_name": "model",
                        "strategy_name": "model",
                        "filter_name": "profitable_filter",
                        "variant_name": "model__profitable_filter",
                        "sharpe": 0.35,
                        "total_return": 0.12,
                        "max_drawdown": -0.08,
                        "total_turnover": 3.0,
                        "trade_count": 8,
                        "positive_fold_rate": 0.5,
                        "total_runtime_sec": 5.0,
                    },
                ],
                "symbol_diagnostics_aggregate_rows": [
                    {"universe_label": "1T+ market cap", "strategy_name": "model", "symbol": "AAA1", "sharpe": 0.8, "avg_trade_return": 0.05, "trade_count": 3}
                ],
                "filter_diagnostic_rows": [
                    {
                        "universe_key": "1t",
                        "universe_label": "1T+ market cap",
                        "strategy_name": "model",
                        "filter_name": "profitable_filter",
                        "fold_name": "wf_2024",
                        "selection_count": 1,
                        "tree_depth": 1,
                        "top_features": [("sector_Technology", 1.0)],
                        "selected_sector_counts": {"Technology": 1},
                        "selected_industry_counts": {"Software": 1},
                    }
                ],
                "runtime_rows": [
                    {
                        "universe_label": "1T+ market cap",
                        "policy_name": "baseline",
                        "filter_name": "no_filter",
                        "total_runtime_sec": 2.0,
                        "model_training_time_sec": 0.0,
                        "filter_training_time_sec": 0.0,
                        "backtest_time_sec": 1.0,
                    },
                    {
                        "universe_label": "1T+ market cap",
                        "policy_name": "model",
                        "filter_name": "no_filter",
                        "total_runtime_sec": 4.0,
                        "model_training_time_sec": 2.5,
                        "filter_training_time_sec": 0.0,
                        "backtest_time_sec": 0.8,
                    },
                ],
                "summary_json_path": "data/pipeline_artifacts/test.json",
                "summary_csv_path": "data/pipeline_artifacts/test.csv",
                "fold_results_csv_path": "data/pipeline_artifacts/test_folds.csv",
                "symbol_diagnostics_test_csv_path": "data/pipeline_artifacts/test_symbols.csv",
                "symbol_diagnostics_aggregate_csv_path": "data/pipeline_artifacts/test_symbols_agg.csv",
                "filter_diagnostics_csv_path": "data/pipeline_artifacts/test_filters.csv",
                "runtime_analysis_csv_path": "data/pipeline_artifacts/test_runtime.csv",
            },
        )

        report_text = report_path.read_text(encoding="utf-8")
        self.assertIn("## Runtime Comparison", report_text)
        self.assertIn("Does added complexity justify runtime cost?", report_text)
        self.assertIn("Runtime analysis CSV", report_text)
