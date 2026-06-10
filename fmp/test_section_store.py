from __future__ import annotations

from django.test import TestCase

from fmp.models import Symbol, SymbolSectionHistorical
from fmp.section_store import save_historical_section, sync_symbol_historical_ranges


class SectionStoreTests(TestCase):
    def setUp(self):
        self.symbol = Symbol.objects.create(symbol="TEST")

    def test_daily_series_updates_same_date_instead_of_duplicating(self):
        save_historical_section(
            self.symbol,
            "prices_div_adj",
            [{"date": "2026-01-02", "adjClose": 100.0}],
            dedupe_by_date=True,
        )
        save_historical_section(
            self.symbol,
            "prices_div_adj",
            [{"date": "2026-01-02", "adjClose": 101.0}],
            dedupe_by_date=True,
        )

        rows = SymbolSectionHistorical.objects.filter(symbol=self.symbol, section_key="prices_div_adj")
        self.assertEqual(rows.count(), 1)
        self.assertEqual(rows.get().payload["adjClose"], 101.0)

    def test_event_series_preserves_distinct_same_date_records(self):
        save_historical_section(
            self.symbol,
            "insider_trading",
            [
                {"date": "2026-01-02", "transaction": "buy"},
                {"date": "2026-01-02", "transaction": "sell"},
            ],
            dedupe_by_date=False,
        )

        self.assertEqual(
            SymbolSectionHistorical.objects.filter(symbol=self.symbol, section_key="insider_trading").count(),
            2,
        )

    def test_sync_updates_cached_range_metadata(self):
        save_historical_section(
            self.symbol,
            "prices_div_adj",
            [
                {"date": "2026-01-02", "adjClose": 100.0},
                {"date": "2026-01-05", "adjClose": 102.0},
            ],
            dedupe_by_date=True,
        )
        sync_symbol_historical_ranges(self.symbol, ["prices_div_adj"])
        self.symbol.refresh_from_db()

        self.assertEqual(
            self.symbol.historical_date_ranges["prices_div_adj"],
            {"min_date": "2026-01-02", "max_date": "2026-01-05", "count": 2},
        )
