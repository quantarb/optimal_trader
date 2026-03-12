from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from pipeline.time_series_momentum_policy_comparison import (
    DEFAULT_UNIVERSE_EXCLUDED_PREFIXES,
    build_yearly_folds,
    resolve_large_cap_symbols,
    run_policy_comparison_experiment,
    write_policy_comparison_report,
)


class Command(BaseCommand):
    help = "Compare the baseline TSMOM signal policy against an oracle-trade model policy with symbol filtering."

    def add_arguments(self, parser):
        parser.add_argument("--test-start-year", type=int, default=2018)
        parser.add_argument("--test-end-year", type=int, default=2025)
        parser.add_argument("--universe-limit", type=int, default=24)
        parser.add_argument("--min-market-cap", type=float, default=10_000_000_000.0)
        parser.add_argument("--fee-bps", type=float, default=2.0)
        parser.add_argument("--slippage-bps", type=float, default=8.0)
        parser.add_argument("--short-borrow-bps-annual", type=float, default=25.0)
        parser.add_argument("--execution-delay-days", type=int, default=1)
        parser.add_argument("--selection-fraction", type=float, default=0.5)
        parser.add_argument("--minimum-filter-symbols", type=int, default=8)
        parser.add_argument("--exclude-symbol-prefixes", default=",".join(DEFAULT_UNIVERSE_EXCLUDED_PREFIXES))
        parser.add_argument("--output-basename", default="time_series_momentum_policy_comparison")
        parser.add_argument("--resume", action="store_true")

    def handle(self, *args, **options):
        start_year = int(options["test_start_year"])
        end_year = int(options["test_end_year"])
        if start_year > end_year:
            raise CommandError("test-start-year must be <= test-end-year.")

        exclude_symbol_prefixes = [
            str(token).strip().upper()
            for token in str(options["exclude_symbol_prefixes"] or "").split(",")
            if str(token).strip()
        ]
        symbols = resolve_large_cap_symbols(
            limit=int(options["universe_limit"]),
            min_market_cap=float(options["min_market_cap"]),
            exclude_symbol_prefixes=exclude_symbol_prefixes or DEFAULT_UNIVERSE_EXCLUDED_PREFIXES,
        )
        if not symbols:
            raise CommandError("No symbols were resolved for the experiment universe.")

        payload = run_policy_comparison_experiment(
            symbols=symbols,
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
            selection_fraction=float(options["selection_fraction"]),
            minimum_filter_symbols=int(options["minimum_filter_symbols"]),
            output_basename=str(options["output_basename"]).strip(),
            resume_existing=bool(options["resume"]),
        )

        report_path = Path("docs") / "research" / "time_series_momentum_policy_comparison_report.md"
        write_policy_comparison_report(
            report_path=report_path,
            payload=payload,
        )
        payload = dict(payload)
        payload["report_path"] = str(report_path)
        Path(str(payload["summary_json_path"])).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        output = {
            "mode": "time_series_momentum_policy_comparison",
            "symbols": symbols,
            "summary_json_path": str(payload.get("summary_json_path") or ""),
            "summary_csv_path": str(payload.get("summary_csv_path") or ""),
            "symbol_diagnostics_test_csv_path": str(payload.get("symbol_diagnostics_test_csv_path") or ""),
            "symbol_diagnostics_aggregate_csv_path": str(payload.get("symbol_diagnostics_aggregate_csv_path") or ""),
            "report_path": str(report_path),
            "aggregate_rows": list(payload.get("aggregate_rows") or []),
        }
        self.stdout.write(json.dumps(output, indent=2, sort_keys=True))
