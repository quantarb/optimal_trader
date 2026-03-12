from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from pipeline.portfolio_optimization_research import (
    run_portfolio_optimization_research,
    write_portfolio_optimization_report,
)


class Command(BaseCommand):
    help = "Run reusable portfolio-optimization research on top of the existing signal and backtest stack."

    def add_arguments(self, parser):
        parser.add_argument("--symbols", default="")
        parser.add_argument("--symbol-limit", type=int, default=20)
        parser.add_argument("--candidate-limit", type=int, default=60)
        parser.add_argument("--min-market-cap", type=float, default=25_000_000_000.0)
        parser.add_argument("--test-start-year", type=int, default=2022)
        parser.add_argument("--test-end-year", type=int, default=2025)
        parser.add_argument("--bucket-count", type=int, default=10)
        parser.add_argument("--prediction-artifact-ids", default="")
        parser.add_argument("--skip-characteristics-factor", action="store_true")
        parser.add_argument("--expected-return-input", default="ranking_score")
        parser.add_argument("--risk-model-type", default="sample_covariance")
        parser.add_argument("--risk-lookback-days", type=int, default=63)
        parser.add_argument("--risk-shrinkage", type=float, default=0.15)
        parser.add_argument("--risk-factor-count", type=int, default=3)
        parser.add_argument("--risk-aversion", type=float, default=5.0)
        parser.add_argument("--turnover-penalty", type=float, default=0.0)
        parser.add_argument("--turnover-cap", type=float, default=None)
        parser.add_argument("--max-name-weight", type=float, default=0.10)
        parser.add_argument("--net-exposure-target", type=float, default=0.0)
        parser.add_argument("--sector-neutral", action="store_true")
        parser.add_argument("--alpha-quantile", type=float, default=0.2)
        parser.add_argument("--alpha-scale", type=float, default=0.05)
        parser.add_argument("--n-factors", type=int, default=3)
        parser.add_argument("--exposure-lookback-days", type=int, default=63)
        parser.add_argument("--minimum-exposure-observations", type=int, default=30)
        parser.add_argument("--random-state", type=int, default=1337)
        parser.add_argument("--fee-bps", type=float, default=2.0)
        parser.add_argument("--slippage-bps", type=float, default=8.0)
        parser.add_argument("--short-borrow-bps-annual", type=float, default=25.0)
        parser.add_argument("--execution-delay-days", type=int, default=1)
        parser.add_argument("--output-basename", default="portfolio_optimization_research")
        parser.add_argument("--resume", action="store_true")

    def handle(self, *args, **options):
        start_year = int(options["test_start_year"])
        end_year = int(options["test_end_year"])
        if start_year > end_year:
            raise CommandError("test-start-year must be <= test-end-year.")

        requested_symbols = [
            str(symbol).strip().upper()
            for symbol in str(options["symbols"] or "").split(",")
            if str(symbol).strip()
        ]
        prediction_artifact_ids = [
            int(token)
            for token in str(options["prediction_artifact_ids"] or "").split(",")
            if str(token).strip()
        ]
        payload = run_portfolio_optimization_research(
            requested_symbols=requested_symbols or None,
            symbol_limit=int(options["symbol_limit"]),
            candidate_limit=int(options["candidate_limit"]),
            min_market_cap=float(options["min_market_cap"]),
            test_start_year=start_year,
            test_end_year=end_year,
            bucket_count=int(options["bucket_count"]),
            fee_bps=float(options["fee_bps"]),
            slippage_bps=float(options["slippage_bps"]),
            short_borrow_bps_annual=float(options["short_borrow_bps_annual"]),
            execution_delay_days=int(options["execution_delay_days"]),
            output_basename=str(options["output_basename"]),
            resume_existing=bool(options["resume"]),
            include_characteristics_factor=not bool(options["skip_characteristics_factor"]),
            prediction_artifact_ids=prediction_artifact_ids,
            expected_return_input=str(options["expected_return_input"]),
            risk_model_type=str(options["risk_model_type"]),
            risk_lookback_days=int(options["risk_lookback_days"]),
            risk_shrinkage=float(options["risk_shrinkage"]),
            risk_factor_count=int(options["risk_factor_count"]),
            risk_aversion=float(options["risk_aversion"]),
            turnover_penalty=float(options["turnover_penalty"]),
            turnover_cap=options["turnover_cap"],
            max_name_weight=float(options["max_name_weight"]),
            net_exposure_target=float(options["net_exposure_target"]),
            sector_neutral=bool(options["sector_neutral"]),
            alpha_quantile=float(options["alpha_quantile"]),
            alpha_scale=float(options["alpha_scale"]),
            n_factors=int(options["n_factors"]),
            exposure_lookback_days=int(options["exposure_lookback_days"]),
            minimum_exposure_observations=int(options["minimum_exposure_observations"]),
            random_state=int(options["random_state"]),
        )
        report_path = Path("docs") / "research" / "portfolio_optimization_research.md"
        write_portfolio_optimization_report(report_path=report_path, payload=payload)
        self.stdout.write(self.style.SUCCESS(f"Research summary: {payload['summary_json_path']}"))
        self.stdout.write(self.style.SUCCESS(f"Research report: {report_path}"))
