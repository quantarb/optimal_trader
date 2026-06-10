from __future__ import annotations

from django.core.management.base import BaseCommand

from fmp.symbol_metadata import refresh_symbol_metadata_from_fmp


class Command(BaseCommand):
    help = "Fetch complete FMP profiles for symbols whose required metadata is incomplete."

    def add_arguments(self, parser):
        parser.add_argument("symbols", nargs="*", help="Optional symbols to validate; defaults to every stored symbol.")
        parser.add_argument("--force", action="store_true", help="Ignore the profile refresh cooldown.")
        parser.add_argument("--max-symbols", type=int, default=None)

    def handle(self, *args, **options):
        result = refresh_symbol_metadata_from_fmp(
            symbols=options.get("symbols") or None,
            force=bool(options.get("force")),
            max_symbols=options.get("max_symbols"),
            progress_logger=self.stdout.write,
        )
        if result.empty:
            self.stdout.write(self.style.SUCCESS("No symbols selected for metadata refresh."))
            return
        counts = result["status"].value_counts().to_dict()
        summary = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        self.stdout.write(self.style.SUCCESS(f"Symbol metadata refresh complete: {summary}"))
