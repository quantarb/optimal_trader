from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError

from pipeline.time_series_momentum_research import (
    DEFAULT_TSMOM_PROXY_SYMBOLS,
    run_time_series_momentum_research,
)


class Command(BaseCommand):
    help = "Run the Time Series Momentum paper replication on the existing platform."

    def add_arguments(self, parser):
        parser.add_argument("--symbols", default=",".join(DEFAULT_TSMOM_PROXY_SYMBOLS))
        parser.add_argument("--test-start-year", type=int, default=2011)
        parser.add_argument("--test-end-year", type=int, default=2025)
        parser.add_argument("--fee-bps", type=float, default=2.0)
        parser.add_argument("--slippage-bps", type=float, default=8.0)
        parser.add_argument("--short-borrow-bps-annual", type=float, default=25.0)
        parser.add_argument("--execution-delay-days", type=int, default=1)
        parser.add_argument("--output-basename", default="time_series_momentum_research")
        parser.add_argument("--resume", action="store_true")

    def handle(self, *args, **options):
        requested_symbols = [
            str(token).strip().upper()
            for token in str(options["symbols"] or "").split(",")
            if str(token).strip()
        ]
        if not requested_symbols:
            raise CommandError("At least one symbol is required.")
        try:
            output = run_time_series_momentum_research(
                symbols=requested_symbols,
                test_start_year=int(options["test_start_year"]),
                test_end_year=int(options["test_end_year"]),
                fee_bps=float(options["fee_bps"]),
                slippage_bps=float(options["slippage_bps"]),
                short_borrow_bps_annual=float(options["short_borrow_bps_annual"]),
                execution_delay_days=int(options["execution_delay_days"]),
                output_basename=str(options["output_basename"]).strip(),
                resume_existing=bool(options["resume"]),
            )
        except ValueError as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(json.dumps(output, indent=2, sort_keys=True))
