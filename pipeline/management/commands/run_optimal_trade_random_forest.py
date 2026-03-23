from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError

from pipeline.optimal_trade_random_forest import (
    DEFAULT_OPTIMAL_TRADE_ETF_SYMBOLS,
    run_optimal_trade_random_forest_training,
)


class Command(BaseCommand):
    help = "Train a pre-2020 random-forest optimal-trade classifier for the ETF proxy universe."

    def add_arguments(self, parser):
        parser.add_argument("--symbols", default=",".join(DEFAULT_OPTIMAL_TRADE_ETF_SYMBOLS))
        parser.add_argument("--feature-start-date", default="1900-01-01")
        parser.add_argument("--train-end-date", default="2019-12-31")
        parser.add_argument("--ye-list", default="1,2,4,8")
        parser.add_argument("--min-profit-pct", type=float, default=5.0)
        parser.add_argument("--min-feature-coverage-pct", type=float, default=10.0)
        parser.add_argument("--output-basename", default="optimal_trade_rf_etf_pre2020_min5")
        parser.add_argument("--model-name-prefix", default="optimal_trade_rf_etf_pre2020_min5")
        parser.add_argument("--n-estimators", type=int, default=400)
        parser.add_argument("--random-state", type=int, default=1337)
        parser.add_argument("--max-depth", type=int, default=0)
        parser.add_argument("--min-samples-leaf", type=int, default=1)
        parser.add_argument("--min-samples-split", type=int, default=2)
        parser.add_argument(
            "--skip-download-missing-prices",
            action="store_true",
            help="Do not try to fetch missing prices from FMP before training.",
        )

    def handle(self, *args, **options):
        requested_symbols = [
            str(token).strip().upper()
            for token in str(options["symbols"] or "").split(",")
            if str(token).strip()
        ]
        if not requested_symbols:
            raise CommandError("At least one symbol is required.")

        try:
            payload = run_optimal_trade_random_forest_training(
                symbols=requested_symbols,
                feature_start_date=str(options["feature_start_date"]).strip(),
                train_end_date=str(options["train_end_date"]).strip(),
                ye_list=str(options["ye_list"]).strip(),
                min_profit_pct=float(options["min_profit_pct"]),
                min_feature_coverage_pct=float(options["min_feature_coverage_pct"]),
                output_basename=str(options["output_basename"]).strip(),
                model_name_prefix=str(options["model_name_prefix"]).strip(),
                n_estimators=int(options["n_estimators"]),
                random_state=int(options["random_state"]),
                download_missing_prices=not bool(options["skip_download_missing_prices"]),
                max_depth=None if int(options["max_depth"] or 0) <= 0 else int(options["max_depth"]),
                min_samples_leaf=int(options["min_samples_leaf"]),
                min_samples_split=int(options["min_samples_split"]),
            )
        except ValueError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(json.dumps(payload, indent=2, sort_keys=True))
