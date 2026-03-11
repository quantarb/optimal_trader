from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from analysis.insights import build_portfolio_analysis, build_stock_intelligence


class Command(BaseCommand):
    help = "Export deterministic or prompt-ready market insight reasoning for a stock or portfolio."

    def add_arguments(self, parser):
        parser.add_argument("--symbol", default="")
        parser.add_argument("--symbols", default="")
        parser.add_argument("--date", default="")
        parser.add_argument("--strategy-artifact", type=int, default=0)
        parser.add_argument("--features", type=int, default=0)
        parser.add_argument("--labels", type=int, default=0)
        parser.add_argument("--market-situation-artifact", type=int, default=0)
        parser.add_argument("--prediction-artifacts", default="")
        parser.add_argument("--search-method", default="hybrid", choices=["numeric", "text_embedding", "hybrid"])
        parser.add_argument("--top-k", type=int, default=10)
        parser.add_argument("--reasoning-mode", default="deterministic")
        parser.add_argument("--output-json", default="")

    def handle(self, *args, **options):
        symbol = str(options.get("symbol") or "").strip().upper()
        symbols = [
            value.strip().upper()
            for value in str(options.get("symbols") or "").split(",")
            if value.strip()
        ]
        if bool(symbol) == bool(symbols):
            raise CommandError("Provide either --symbol for stock insight or --symbols for portfolio insight.")
        prediction_ids = [
            int(value)
            for value in str(options.get("prediction_artifacts") or "").split(",")
            if str(value).strip().isdigit() and int(value) > 0
        ]
        reasoning_mode = str(options.get("reasoning_mode") or "deterministic").strip() or "deterministic"

        if symbol:
            try:
                payload = build_stock_intelligence(
                    symbol=symbol,
                    date=str(options.get("date") or "").strip() or None,
                    strategy_artifact_id=int(options.get("strategy_artifact") or 0),
                    feature_artifact_id=int(options.get("features") or 0),
                    label_artifact_id=int(options.get("labels") or 0),
                    prediction_artifact_ids=prediction_ids,
                    market_situation_artifact_id=int(options.get("market_situation_artifact") or 0),
                    twin_count=max(int(options.get("top_k") or 10), 1),
                    search_method=str(options.get("search_method") or "hybrid"),
                    reasoning_mode=reasoning_mode,
                )
            except Exception as exc:
                raise CommandError(str(exc)) from exc
            rendered_payload = {
                "kind": "stock_insight_reasoning",
                "symbol": payload.get("symbol"),
                "date": payload.get("date"),
                "reasoning_mode": payload.get("reasoning_mode"),
                "stock_insight": payload.get("stock_insight") or {},
                "market_situation_explanation": payload.get("market_situation_explanation") or {},
                "reasoning_input": payload.get("reasoning_input") or {},
                "opportunity": payload.get("opportunity") or {},
                "outcome_summary": payload.get("outcome_summary") or {},
                "artifacts": payload.get("artifacts") or {},
            }
        else:
            try:
                payload = build_portfolio_analysis(
                    symbols=symbols,
                    strategy_artifact_id=int(options.get("strategy_artifact") or 0),
                    feature_artifact_id=int(options.get("features") or 0),
                    label_artifact_id=int(options.get("labels") or 0),
                    prediction_artifact_ids=prediction_ids,
                    market_situation_artifact_id=int(options.get("market_situation_artifact") or 0),
                    search_method=str(options.get("search_method") or "hybrid"),
                    reasoning_mode=reasoning_mode,
                )
            except Exception as exc:
                raise CommandError(str(exc)) from exc
            rendered_payload = {
                "kind": "portfolio_insight_reasoning",
                "symbols": payload.get("symbols") or [],
                "as_of_date": payload.get("as_of_date") or "",
                "reasoning_mode": payload.get("reasoning_mode"),
                "portfolio_insight": payload.get("portfolio_insight") or {},
                "portfolio_insight_input": payload.get("portfolio_insight_input") or {},
                "portfolio_score": payload.get("portfolio_score"),
                "regime_similarity_score": payload.get("regime_similarity_score"),
                "risk_concentration_score": payload.get("risk_concentration_score"),
                "artifacts": payload.get("artifacts") or {},
            }

        rendered = json.dumps(rendered_payload, indent=2, sort_keys=True, default=str)
        output_path = str(options.get("output_json") or "").strip()
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered, encoding="utf-8")
        self.stdout.write(rendered)
