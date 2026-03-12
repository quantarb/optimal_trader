from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from pipeline.time_series_momentum_market_cap_policy_comparison import (
    DEFAULT_US_EXCHANGES,
    run_market_cap_policy_comparison_experiment,
    write_market_cap_policy_comparison_report,
)
from pipeline.universe_selection import MARKET_CAP_TIERS
from pipeline.time_series_momentum_policy_comparison import build_yearly_folds


class Command(BaseCommand):
    help = "Run the TSMOM baseline-vs-ML market-cap tier policy comparison experiment."

    def add_arguments(self, parser):
        parser.add_argument("--tiers", default="1t,100b,10b", help="Comma-separated market-cap tiers: 1t,100b,10b")
        parser.add_argument("--test-start-year", type=int, default=2018)
        parser.add_argument("--test-end-year", type=int, default=2025)
        parser.add_argument("--fee-bps", type=float, default=2.0)
        parser.add_argument("--slippage-bps", type=float, default=8.0)
        parser.add_argument("--short-borrow-bps-annual", type=float, default=25.0)
        parser.add_argument("--execution-delay-days", type=int, default=1)
        parser.add_argument("--country", default="US")
        parser.add_argument("--exchanges", default=",".join(DEFAULT_US_EXCHANGES))
        parser.add_argument("--max-symbols-per-tier", type=int, default=0, help="Optional symbol cap per tier. Zero means full tier universe.")
        parser.add_argument("--minimum-filter-symbols", type=int, default=5)
        parser.add_argument("--filter-max-depth", type=int, default=3)
        parser.add_argument("--filter-min-samples-leaf", type=int, default=3)
        parser.add_argument("--output-basename", default="time_series_momentum_market_cap_policy_comparison")
        parser.add_argument("--resume", action="store_true")

    def handle(self, *args, **options):
        start_year = int(options["test_start_year"])
        end_year = int(options["test_end_year"])
        if start_year > end_year:
            raise CommandError("test-start-year must be <= test-end-year.")

        tiers = [str(token).strip().lower() for token in str(options["tiers"] or "").split(",") if str(token).strip()]
        if not tiers:
            raise CommandError("At least one market-cap tier is required.")
        invalid_tiers = [tier for tier in tiers if tier not in MARKET_CAP_TIERS]
        if invalid_tiers:
            raise CommandError(f"Unknown tier(s): {', '.join(invalid_tiers)}")

        exchanges = [str(token).strip().upper() for token in str(options["exchanges"] or "").split(",") if str(token).strip()]
        if not exchanges:
            raise CommandError("At least one exchange is required.")

        payload = run_market_cap_policy_comparison_experiment(
            tiers=tiers,
            folds=build_yearly_folds(start_year, end_year),
            backtest_config={
                "fee_bps": float(options["fee_bps"]),
                "slippage_bps": float(options["slippage_bps"]),
                "short_borrow_bps_annual": float(options["short_borrow_bps_annual"]),
                "execution_delay_days": int(options["execution_delay_days"]),
                "turnover_half_l1": True,
                "use_lagged_weights": True,
                "min_price": 5.0,
                "min_dollar_volume": 5_000_000.0,
            },
            country=str(options["country"] or "US").strip(),
            exchanges=exchanges,
            max_symbols_per_tier=int(options["max_symbols_per_tier"] or 0) or None,
            minimum_filter_symbols=int(options["minimum_filter_symbols"]),
            filter_max_depth=int(options["filter_max_depth"]),
            filter_min_samples_leaf=int(options["filter_min_samples_leaf"]),
            output_basename=str(options["output_basename"]).strip(),
            resume_existing=bool(options["resume"]),
        )

        report_path = Path("docs") / "research" / "time_series_momentum_market_cap_policy_comparison_report.md"
        write_market_cap_policy_comparison_report(
            report_path=report_path,
            payload=payload,
        )
        payload = dict(payload)
        payload["report_path"] = str(report_path)
        Path(str(payload["summary_json_path"])).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        output = {
            "mode": "time_series_momentum_market_cap_policy_comparison",
            "tiers": tiers,
            "summary_json_path": str(payload.get("summary_json_path") or ""),
            "summary_csv_path": str(payload.get("summary_csv_path") or ""),
            "fold_results_csv_path": str(payload.get("fold_results_csv_path") or ""),
            "symbol_diagnostics_test_csv_path": str(payload.get("symbol_diagnostics_test_csv_path") or ""),
            "symbol_diagnostics_aggregate_csv_path": str(payload.get("symbol_diagnostics_aggregate_csv_path") or ""),
            "filter_diagnostics_csv_path": str(payload.get("filter_diagnostics_csv_path") or ""),
            "runtime_analysis_csv_path": str(payload.get("runtime_analysis_csv_path") or ""),
            "runtime_stage_csv_path": str(payload.get("runtime_stage_csv_path") or ""),
            "report_path": str(report_path),
            "aggregate_rows": list(payload.get("aggregate_rows") or []),
            "universe_rows": list(payload.get("universe_rows") or []),
        }
        self.stdout.write(json.dumps(output, indent=2, sort_keys=True))
