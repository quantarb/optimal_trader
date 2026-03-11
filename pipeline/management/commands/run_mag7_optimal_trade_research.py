from __future__ import annotations

import json

from django.core.management.base import BaseCommand

from pipeline.management.commands.run_mag7_backtest import MAG7_SYMBOLS
from pipeline.research_suite import research_profile_names, run_optimal_trade_research_suite


class Command(BaseCommand):
    help = "Run the opinionated optimal-trade-first MAG7 research suite."

    def add_arguments(self, parser):
        parser.add_argument("--test-start-year", type=int, default=2023)
        parser.add_argument("--test-end-year", type=int, default=2025)
        parser.add_argument("--min-profit-pct", type=float, default=12.0)
        parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
        parser.add_argument("--profile", default="broad_universe_long_history", choices=research_profile_names())
        parser.add_argument("--resume", action="store_true", help="Reuse completed suite and fold artifacts when present.")
        parser.add_argument("--output-basename", default="mag7_optimal_trade_research")

    def handle(self, *args, **options):
        folds = []
        for year in range(int(options["test_start_year"]), int(options["test_end_year"]) + 1):
            folds.append(
                {
                    "name": f"wf_{year}",
                    "train_end_date": f"{year - 1}-12-31",
                    "backtest_start_date": f"{year}-01-01",
                    "backtest_end_date": f"{year}-12-31",
                }
            )
        payload = run_optimal_trade_research_suite(
            symbols=MAG7_SYMBOLS,
            folds=folds,
            min_profit_pct=float(options["min_profit_pct"]),
            transaction_cost_bps=float(options["transaction_cost_bps"]),
            profile_name=str(options["profile"]).strip(),
            output_basename=str(options["output_basename"]).strip(),
            resume_existing=bool(options["resume"]),
        )
        self.stdout.write(self.style.SUCCESS(json.dumps(payload, indent=2, sort_keys=True)))
