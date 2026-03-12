from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from pipeline.characteristics_factor_model import (
    run_characteristics_factor_research,
    write_characteristics_factor_report,
)


class Command(BaseCommand):
    help = "Run the characteristics-factor cross-sectional ranking research workflow."

    def add_arguments(self, parser):
        parser.add_argument("--symbols", default="")
        parser.add_argument("--symbol-limit", type=int, default=20)
        parser.add_argument("--candidate-limit", type=int, default=60)
        parser.add_argument("--min-market-cap", type=float, default=25_000_000_000.0)
        parser.add_argument("--test-start-year", type=int, default=2022)
        parser.add_argument("--test-end-year", type=int, default=2025)
        parser.add_argument("--lookback-days", type=int, default=252)
        parser.add_argument("--forward-horizon-days", type=int, default=21)
        parser.add_argument("--start-offset-days", type=int, default=1)
        parser.add_argument("--bucket-count", type=int, default=10)
        parser.add_argument("--n-factors", type=int, default=3)
        parser.add_argument("--exposure-lookback-days", type=int, default=63)
        parser.add_argument("--minimum-exposure-observations", type=int, default=30)
        parser.add_argument("--evaluate-feature-subsets", action="store_true")
        parser.add_argument("--fee-bps", type=float, default=2.0)
        parser.add_argument("--slippage-bps", type=float, default=8.0)
        parser.add_argument("--short-borrow-bps-annual", type=float, default=25.0)
        parser.add_argument("--execution-delay-days", type=int, default=1)
        parser.add_argument("--output-basename", default="characteristics_factor_research")
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
        payload = run_characteristics_factor_research(
            requested_symbols=requested_symbols or None,
            symbol_limit=int(options["symbol_limit"]),
            candidate_limit=int(options["candidate_limit"]),
            min_market_cap=float(options["min_market_cap"]),
            test_start_year=start_year,
            test_end_year=end_year,
            lookback_days=int(options["lookback_days"]),
            forward_horizon_days=int(options["forward_horizon_days"]),
            start_offset_days=int(options["start_offset_days"]),
            bucket_count=int(options["bucket_count"]),
            n_factors=int(options["n_factors"]),
            exposure_lookback_days=int(options["exposure_lookback_days"]),
            minimum_exposure_observations=int(options["minimum_exposure_observations"]),
            evaluate_feature_subsets=bool(options["evaluate_feature_subsets"]),
            fee_bps=float(options["fee_bps"]),
            slippage_bps=float(options["slippage_bps"]),
            short_borrow_bps_annual=float(options["short_borrow_bps_annual"]),
            execution_delay_days=int(options["execution_delay_days"]),
            output_basename=str(options["output_basename"]),
            resume_existing=bool(options["resume"]),
        )
        report_path = Path("docs") / "research" / "characteristics_factor_research.md"
        write_characteristics_factor_report(report_path=report_path, payload=payload)
        self.stdout.write(self.style.SUCCESS(f"Research summary: {payload['summary_json_path']}"))
        self.stdout.write(self.style.SUCCESS(f"Research report: {report_path}"))
