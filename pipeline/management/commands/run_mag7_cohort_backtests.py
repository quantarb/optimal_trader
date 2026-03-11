from __future__ import annotations

import json

from django.core.management.base import BaseCommand

from pipeline.cohort_runner import run_model_cohort_backtests
from pipeline.management.commands.run_mag7_backtest import MAG7_SYMBOLS


class Command(BaseCommand):
    help = "Run grouped MAG7 model cohort backtests and persist a comparison summary."

    def add_arguments(self, parser):
        parser.add_argument("--fit-job", default="fit_regressor", choices=["fit_classifier", "fit_regressor", "fit_autoencoder"])
        parser.add_argument("--train-end-date", default="2019-12-31")
        parser.add_argument("--backtest-start-date", default="2020-01-01")
        parser.add_argument("--backtest-end-date", default="")
        parser.add_argument("--min-profit-pct", type=float, default=10.0)
        parser.add_argument("--transaction-cost-bps", type=float, default=10.0)

    def handle(self, *args, **options):
        payload = run_model_cohort_backtests(
            symbols=MAG7_SYMBOLS,
            fit_job=str(options["fit_job"]).strip(),
            base_model_config={
                "model_name": "mag7_cohort_model",
                "split_ratio": 1.0,
                "min_profit_pct": float(options["min_profit_pct"]),
                "feature_family_mode": "grouped_family",
                "feature_family_groups": [
                    ["prices_div_adj"],
                    ["income_statement", "income_statement_growth"],
                    ["analyst_estimates"],
                ],
                "label_horizon_mode": "grouped_k",
                "label_k_groups": [[1, 2], [4, 8]],
            },
            train_end_date=str(options["train_end_date"]).strip(),
            backtest_start_date=str(options["backtest_start_date"]).strip(),
            backtest_end_date=str(options["backtest_end_date"]).strip(),
            strategy_definition_slug="mag7-cohort-backtest-cli",
            strategy_definition_name="MAG7 Cohort Backtest Strategy",
            strategy_config={
                "gate_quantile": 0.5,
                "top_k": 20,
                "rebalance_freq": "W",
                "gross_exposure": 0.8,
                "selection_side": "long_only",
            },
            transaction_cost_bps=float(options["transaction_cost_bps"]),
            output_basename="mag7_cohort_backtest_summary",
        )
        self.stdout.write(self.style.SUCCESS(json.dumps(payload, indent=2, sort_keys=True)))
