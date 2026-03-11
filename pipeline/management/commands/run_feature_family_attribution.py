from __future__ import annotations

import json

from django.core.management.base import BaseCommand

from analysis.feature_attribution import run_feature_family_attribution_suite
from pipeline.management.commands.run_mag7_backtest import MAG7_SYMBOLS
from pipeline.research_suite import research_profile_names, resolve_research_profile


class Command(BaseCommand):
    help = "Run a walk-forward feature-family attribution suite and score bundles on oracle coverage plus tradability."

    def add_arguments(self, parser):
        parser.add_argument("--symbols", default=",".join(MAG7_SYMBOLS))
        parser.add_argument("--fit-job", default="fit_mtl", choices=["fit_classifier", "fit_regressor", "fit_mtl"])
        parser.add_argument("--profile", default="broad_universe_long_history", choices=research_profile_names())
        parser.add_argument("--test-start-year", type=int, default=2024)
        parser.add_argument("--test-end-year", type=int, default=2025)
        parser.add_argument("--min-profit-pct", type=float, default=10.0)
        parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
        parser.add_argument("--selection-quantile", type=float, default=0.8)
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--output-basename", default="feature_family_attribution")

    def handle(self, *args, **options):
        symbols = [str(token).strip().upper() for token in str(options["symbols"] or "").split(",") if str(token).strip()]
        profile = resolve_research_profile(str(options["profile"]).strip())
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

        fit_job = str(options["fit_job"]).strip()
        model_prefix = {
            "fit_classifier": "oracle_attr_clf",
            "fit_regressor": "oracle_attr_reg",
            "fit_mtl": "oracle_attr_mtl",
        }[fit_job]
        base_model_config = {
            "model_name": model_prefix,
            "split_ratio": 1.0,
            "min_profit_pct": float(options["min_profit_pct"]),
            "label_horizon_mode": "grouped_k",
            "label_k_groups": [[1, 2, 4, 8]],
            "min_abs_trade_return_pct": 8.0,
            "max_hold_days": 90,
            "sample_weight_mode": "trade_return_abs",
        }
        payload = run_feature_family_attribution_suite(
            symbols=symbols,
            folds=folds,
            fit_job=fit_job,
            base_model_config=base_model_config,
            feature_family_groups=[list(group) for group in list(profile.get("feature_family_groups") or [])],
            feature_config=dict(profile.get("feature_config") or {}),
            transaction_cost_bps=float(options["transaction_cost_bps"]),
            backtest_config=dict(profile.get("backtest_config") or {}),
            selection_quantile=float(options["selection_quantile"]),
            output_basename=str(options["output_basename"]).strip(),
            resume_existing=bool(options["resume"]),
        )
        self.stdout.write(self.style.SUCCESS(json.dumps(payload, indent=2, sort_keys=True)))
