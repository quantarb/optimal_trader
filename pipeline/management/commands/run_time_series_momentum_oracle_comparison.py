from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from pipeline.time_series_momentum_oracle_comparison import (
    DEFAULT_LABEL_KS,
    DEFAULT_US_EXCHANGES,
    run_time_series_momentum_oracle_comparison_experiment,
    write_time_series_momentum_oracle_comparison_report,
)
from pipeline.universe_selection import MARKET_CAP_TIERS


class Command(BaseCommand):
    help = "Run the TSMOM vs oracle-label ML comparison across predefined market-cap universes."

    def add_arguments(self, parser):
        parser.add_argument("--tiers", default="1t,100b,10b")
        parser.add_argument("--backtest-start-date", default="2020-01-01")
        parser.add_argument("--backtest-end-date", default="")
        parser.add_argument("--fee-bps", type=float, default=2.0)
        parser.add_argument("--slippage-bps", type=float, default=8.0)
        parser.add_argument("--short-borrow-bps-annual", type=float, default=25.0)
        parser.add_argument("--execution-delay-days", type=int, default=1)
        parser.add_argument("--country", default="US")
        parser.add_argument("--exchanges", default=",".join(DEFAULT_US_EXCHANGES))
        parser.add_argument("--max-symbols-per-tier", type=int, default=0)
        parser.add_argument("--minimum-filter-symbols", type=int, default=5)
        parser.add_argument("--filter-max-depth", type=int, default=3)
        parser.add_argument("--filter-min-samples-leaf", type=int, default=3)
        parser.add_argument("--label-ks", default=",".join(str(value) for value in DEFAULT_LABEL_KS))
        parser.add_argument("--min-profit-pct", type=float, default=10.0)
        parser.add_argument("--output-basename", default="time_series_momentum_oracle_comparison")
        parser.add_argument("--report-path", default="docs/research/time_series_momentum_oracle_comparison_report.md")
        parser.add_argument("--resume", action="store_true")

    def handle(self, *args, **options):
        tiers = [
            str(token).strip().lower()
            for token in str(options["tiers"] or "").split(",")
            if str(token).strip()
        ]
        if not tiers:
            raise CommandError("At least one tier is required.")
        invalid_tiers = [tier for tier in tiers if tier not in MARKET_CAP_TIERS]
        if invalid_tiers:
            raise CommandError(f"Unsupported tiers: {', '.join(invalid_tiers)}")

        exchanges = [
            str(token).strip().upper()
            for token in str(options["exchanges"] or "").split(",")
            if str(token).strip()
        ]
        if not exchanges:
            raise CommandError("At least one exchange is required.")

        label_ks: list[int] = []
        for token in str(options["label_ks"] or "").split(","):
            try:
                parsed = int(str(token).strip())
            except Exception:
                continue
            if parsed > 0 and parsed not in label_ks:
                label_ks.append(parsed)
        if not label_ks:
            raise CommandError("At least one positive oracle label horizon is required.")

        max_symbols_per_tier = int(options["max_symbols_per_tier"])
        payload = run_time_series_momentum_oracle_comparison_experiment(
            tiers=tiers,
            backtest_start_date=str(options["backtest_start_date"] or ""),
            backtest_end_date=str(options["backtest_end_date"] or ""),
            fee_bps=float(options["fee_bps"]),
            slippage_bps=float(options["slippage_bps"]),
            short_borrow_bps_annual=float(options["short_borrow_bps_annual"]),
            execution_delay_days=int(options["execution_delay_days"]),
            country=str(options["country"] or "US").strip().upper() or "US",
            exchanges=exchanges,
            max_symbols_per_tier=max_symbols_per_tier if max_symbols_per_tier > 0 else None,
            minimum_filter_symbols=int(options["minimum_filter_symbols"]),
            filter_max_depth=int(options["filter_max_depth"]),
            filter_min_samples_leaf=int(options["filter_min_samples_leaf"]),
            label_ks=label_ks,
            min_profit_pct=float(options["min_profit_pct"]),
            output_basename=str(options["output_basename"] or "time_series_momentum_oracle_comparison"),
            resume_existing=bool(options["resume"]),
        )
        report_path = Path(str(options["report_path"] or "docs/research/time_series_momentum_oracle_comparison_report.md"))
        write_time_series_momentum_oracle_comparison_report(report_path=report_path, payload=payload)
        self.stdout.write(self.style.SUCCESS(f"Research summary: {payload['summary_json_path']}"))
        self.stdout.write(self.style.SUCCESS(f"Research report: {report_path}"))
