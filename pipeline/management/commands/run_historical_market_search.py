from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from analysis.insights import build_stock_intelligence


class Command(BaseCommand):
    help = "Run historical market situation search for a symbol/date across numeric, embedding, or hybrid retrieval."

    def add_arguments(self, parser):
        parser.add_argument("--symbol", required=True)
        parser.add_argument("--date", default="")
        parser.add_argument("--strategy-artifact", type=int, default=0)
        parser.add_argument("--features", type=int, default=0)
        parser.add_argument("--labels", type=int, default=0)
        parser.add_argument("--market-situation-artifact", type=int, default=0)
        parser.add_argument("--prediction-artifacts", default="")
        parser.add_argument("--search-method", default="hybrid", choices=["numeric", "text_embedding", "hybrid"])
        parser.add_argument("--top-k", type=int, default=10)
        parser.add_argument("--output-json", default="")

    def handle(self, *args, **options):
        prediction_ids = [
            int(value)
            for value in str(options.get("prediction_artifacts") or "").split(",")
            if str(value).strip().isdigit() and int(value) > 0
        ]
        try:
            payload = build_stock_intelligence(
                symbol=str(options["symbol"]).strip().upper(),
                date=str(options.get("date") or "").strip() or None,
                strategy_artifact_id=int(options.get("strategy_artifact") or 0),
                feature_artifact_id=int(options.get("features") or 0),
                label_artifact_id=int(options.get("labels") or 0),
                prediction_artifact_ids=prediction_ids,
                market_situation_artifact_id=int(options.get("market_situation_artifact") or 0),
                twin_count=max(int(options.get("top_k") or 10), 1),
                search_method=str(options.get("search_method") or "hybrid"),
            )
        except Exception as exc:
            raise CommandError(str(exc)) from exc

        output_path = str(options.get("output_json") or "").strip()
        rendered = json.dumps(payload, indent=2, sort_keys=True, default=str)
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered, encoding="utf-8")
        self.stdout.write(rendered)
