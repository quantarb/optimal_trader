from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from pipeline.research_suite import research_profile_names, run_optimal_trade_research_suite
from pipeline.universe_selection import DEFAULT_US_EXCHANGES, MARKET_CAP_TIERS, resolve_market_cap_tier_symbols


class Command(BaseCommand):
    help = "Run the optimal-trade research suite across US market-cap tiers."

    def add_arguments(self, parser):
        parser.add_argument("--tiers", default="1t,100b,10b", help="Comma-separated tier keys: 1t,100b,10b")
        parser.add_argument("--test-start-year", type=int, default=2023)
        parser.add_argument("--test-end-year", type=int, default=2025)
        parser.add_argument("--min-profit-pct", type=float, default=12.0)
        parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
        parser.add_argument("--profile", default="broad_universe_long_history", choices=research_profile_names())
        parser.add_argument("--resume", action="store_true", help="Reuse completed suite and fold artifacts when present.")
        parser.add_argument("--max-symbols", type=int, default=0, help="Optional cap per tier. Zero means no cap.")
        parser.add_argument("--country", default="US")
        parser.add_argument("--exchanges", default=",".join(DEFAULT_US_EXCHANGES))
        parser.add_argument("--output-prefix", default="us_market_cap_tier_research")
        parser.add_argument("--dry-run", action="store_true", help="Resolve symbol universes only and skip research execution.")

    def handle(self, *args, **options):
        raw_tiers = [str(token).strip().lower() for token in str(options["tiers"] or "").split(",") if str(token).strip()]
        if not raw_tiers:
            raise CommandError("At least one tier key is required.")
        invalid = [tier for tier in raw_tiers if tier not in MARKET_CAP_TIERS]
        if invalid:
            raise CommandError(f"Unknown tier(s): {', '.join(invalid)}")

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

        exchanges = [str(token).strip().upper() for token in str(options["exchanges"] or "").split(",") if str(token).strip()]
        max_symbols = int(options["max_symbols"] or 0)
        payload_rows: list[dict[str, object]] = []
        for tier in raw_tiers:
            symbols = resolve_market_cap_tier_symbols(
                tier_key=tier,
                country=str(options["country"] or "US").strip(),
                exchanges=exchanges,
                limit=max_symbols or None,
                exclude_pooled_vehicles=True,
            )
            row: dict[str, object] = {
                "tier": tier,
                "min_market_cap": float(MARKET_CAP_TIERS[tier]),
                "country": str(options["country"] or "US").strip(),
                "exchanges": exchanges,
                "symbol_count": len(symbols),
                "symbols_preview": symbols[:20],
            }
            if options["dry_run"]:
                payload_rows.append(row)
                continue
            if not symbols:
                row["status"] = "skipped"
                row["error"] = "no_symbols"
                payload_rows.append(row)
                continue
            output_basename = f"{str(options['output_prefix']).strip()}__{tier}"
            result = run_optimal_trade_research_suite(
                symbols=symbols,
                folds=folds,
                min_profit_pct=float(options["min_profit_pct"]),
                transaction_cost_bps=float(options["transaction_cost_bps"]),
                profile_name=str(options["profile"]).strip(),
                output_basename=output_basename,
                resume_existing=bool(options["resume"]),
            )
            row.update(
                {
                    "status": "succeeded",
                    "output_basename": output_basename,
                    "summary_json_path": str(result.get("summary_json_path") or ""),
                    "summary_csv_path": str(result.get("summary_csv_path") or ""),
                    "report_summary": dict(result.get("report_summary") or {}),
                }
            )
            payload_rows.append(row)

        combined = {
            "profile_name": str(options["profile"]).strip(),
            "tiers": payload_rows,
            "folds": folds,
            "country": str(options["country"] or "US").strip(),
            "exchanges": exchanges,
            "max_symbols": max_symbols,
            "dry_run": bool(options["dry_run"]),
        }
        output_path = Path("data") / "pipeline_artifacts" / f"{str(options['output_prefix']).strip()}__summary.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8")
        combined["summary_path"] = str(output_path)
        self.stdout.write(json.dumps(combined, indent=2, sort_keys=True))
