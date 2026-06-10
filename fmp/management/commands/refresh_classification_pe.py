from __future__ import annotations

from datetime import date, timedelta

from django.core.management.base import BaseCommand, CommandError

from fmp.classification_pe import refresh_classification_pe
from fmp.models import Symbol


class Command(BaseCommand):
    help = "Refresh FMP historical sector and industry P/E series for stored symbol classifications."

    def add_arguments(self, parser):
        parser.add_argument("--symbols", default="", help="Optional comma-separated symbol filter.")
        parser.add_argument("--from", dest="from_date", default="")
        parser.add_argument("--to", dest="to_date", default="")

    def handle(self, *args, **options):
        try:
            end_date = date.fromisoformat(options["to_date"]) if options.get("to_date") else date.today()
            start_date = date.fromisoformat(options["from_date"]) if options.get("from_date") else end_date - timedelta(days=3650)
        except ValueError as exc:
            raise CommandError("Dates must use YYYY-MM-DD format.") from exc
        queryset = Symbol.objects.exclude(exchange="")
        requested = [item.strip().upper() for item in str(options.get("symbols") or "").split(",") if item.strip()]
        if requested:
            queryset = queryset.filter(symbol__in=requested)
        try:
            results = refresh_classification_pe(queryset.iterator(), start_date=start_date, end_date=end_date)
        except Exception as exc:
            raise CommandError(str(exc)) from exc
        observations = sum(int(item["observations_saved"]) for item in results)
        self.stdout.write(self.style.SUCCESS(f"Refreshed {len(results)} P/E series and saved {observations} observations."))
