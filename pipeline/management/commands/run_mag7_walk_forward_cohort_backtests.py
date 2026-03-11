from __future__ import annotations

import json

from django.core.management.base import BaseCommand

from pipeline.cohort_runner import run_walk_forward_model_cohort_backtests


MAG7_SYMBOLS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]


class Command(BaseCommand):
    help = "Run MAG7 walk-forward cohort backtests and persist aggregate fold summaries."

    def add_arguments(self, parser):
        parser.add_argument("--train-start-year", type=int, default=2020)
        parser.add_argument("--test-start-year", type=int, default=2023)
        parser.add_argument("--test-end-year", type=int, default=2025)
        parser.add_argument("--top-k", type=int, default=3)
        parser.add_argument("--gate-quantile", type=float, default=0.5)
        parser.add_argument("--rebalance-freq", default="W")
        parser.add_argument("--gross-exposure", type=float, default=0.8)
        parser.add_argument("--transaction-cost-bps", type=float, default=10.0)

    def handle(self, *args, **options):
        test_start_year = int(options["test_start_year"])
        test_end_year = int(options["test_end_year"])
        folds = []
        for year in range(test_start_year, test_end_year + 1):
            folds.append(
                {
                    "name": f"wf_{year}",
                    "train_end_date": f"{year - 1}-12-31",
                    "backtest_start_date": f"{year}-01-01",
                    "backtest_end_date": f"{year}-12-31",
                }
            )

        payload = run_walk_forward_model_cohort_backtests(
            symbols=MAG7_SYMBOLS,
            fit_job="fit_regressor",
            base_model_config={
                "model_name": "mag7_walk_forward_model",
                "split_ratio": 1.0,
                "feature_family_mode": "grouped_family",
                "feature_family_groups": [["prices_div_adj"], ["income_statement", "income_statement_growth"]],
                "label_horizon_mode": "grouped_k",
                "label_k_groups": [[1], [2]],
            },
            folds=folds,
            strategy_definition_slug="mag7-walk-forward-cohort-cli",
            strategy_definition_name="MAG7 Walk Forward Cohort Strategy",
            strategy_config={
                "gate_quantile": float(options["gate_quantile"]),
                "top_k": int(options["top_k"]),
                "rebalance_freq": str(options["rebalance_freq"]).strip().upper(),
                "gross_exposure": float(options["gross_exposure"]),
                "selection_side": "long_only",
            },
            transaction_cost_bps=float(options["transaction_cost_bps"]),
            output_basename="mag7_walk_forward_cohort_summary",
        )
        self.stdout.write(self.style.SUCCESS(json.dumps(payload, indent=2, sort_keys=True)))
