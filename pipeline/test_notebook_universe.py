from __future__ import annotations

from unittest.mock import patch

from django.test import TestCase

from fmp.models import Symbol, SymbolSectionSnapshot
from pipeline.notebook_universe import normalize_symbols, resolve_notebook_universe


class FakeProfileClient:
    def get_json(self, path, *, params=None):
        symbol = str((params or {}).get("symbol") or "")
        return [{
            "symbol": symbol,
            "companyName": f"{symbol} Company",
            "exchangeShortName": "NASDAQ",
            "country": "US",
            "sector": "Technology",
            "industry": "Software",
            "ipoDate": "2020-01-02",
        }]


class NotebookUniverseTests(TestCase):
    def test_normalize_symbols_accepts_lists_and_comma_separated_text(self):
        self.assertEqual(normalize_symbols(" aapl,MSFT\naapl "), ("AAPL", "MSFT"))
        self.assertEqual(normalize_symbols([" tsla ", "TSLA", "nvda"]), ("TSLA", "NVDA"))

    def test_explicit_symbols_create_records_and_repair_metadata_first(self):
        resolved = resolve_notebook_universe(
            {"symbols": "aapl, msft", "source": "local"},
            metadata_client=FakeProfileClient(),
        )

        self.assertEqual(resolved.symbols, ("AAPL", "MSFT"))
        self.assertEqual(resolved.source, "explicit symbols")
        self.assertEqual(Symbol.objects.filter(symbol__in=resolved.symbols, sector="Technology").count(), 2)

    def test_local_universe_validates_complete_metadata_without_profile_calls(self):
        complete = Symbol.objects.create(
            symbol="DONE",
            company_name="Done Company",
            exchange="NYSE",
            country="US",
            sector="Industrials",
            industry="Machinery",
            market_cap=20_000_000_000,
            payload={"ipoDate": "2001-01-01"},
        )
        SymbolSectionSnapshot.objects.create(symbol=complete, section_key="profile", payload={"symbol": "DONE"})

        resolved = resolve_notebook_universe(
            {
                "source": "local",
                "symbols": [],
                "country": "US",
                "exchanges": ["NYSE"],
                "min_market_cap": 10_000_000_000,
            },
            metadata_client=FakeProfileClient(),
        )

        self.assertEqual(resolved.symbols, ("DONE",))
        self.assertEqual(resolved.source, "local DB")

    @patch("data.universe_fmp.screen_companies_fmp")
    def test_screener_records_are_persisted_then_validated(self, mocked_screen):
        mocked_screen.return_value = (
            ("SCRN",),
            [{
                "symbol": "SCRN",
                "companyName": "Screened Company",
                "exchangeShortName": "NASDAQ",
                "country": "US",
                "sector": "Technology",
                "industry": "Software",
                "marketCap": 50_000_000_000,
                "ipoDate": "2018-05-01",
            }],
        )

        resolved = resolve_notebook_universe(
            {"source": "screener", "symbols": [], "exchanges": ["NASDAQ"]},
            api_key="test-key",
            metadata_client=FakeProfileClient(),
        )

        self.assertEqual(resolved.symbols, ("SCRN",))
        self.assertEqual(Symbol.objects.get(symbol="SCRN").company_name, "SCRN Company")
        self.assertTrue(
            SymbolSectionSnapshot.objects.filter(symbol__symbol="SCRN", section_key="profile").exists()
        )

    @patch("data.universe_fmp.screen_companies_fmp")
    def test_sparse_screener_record_preserves_existing_profile_metadata(self, mocked_screen):
        complete = Symbol.objects.create(
            symbol="KEEP",
            company_name="Keep Company",
            exchange="NASDAQ",
            country="US",
            sector="Technology",
            industry="Software",
            payload={"ipoDate": "2010-01-01", "companyName": "Keep Company"},
        )
        SymbolSectionSnapshot.objects.create(symbol=complete, section_key="profile", payload={"symbol": "KEEP"})
        mocked_screen.return_value = (("KEEP",), [{"symbol": "KEEP", "marketCap": 99_000_000_000}])

        resolve_notebook_universe(
            {"source": "screener", "symbols": []},
            api_key="test-key",
            metadata_client=FakeProfileClient(),
        )

        symbol = Symbol.objects.get(symbol="KEEP")
        self.assertEqual(symbol.company_name, "Keep Company")
        self.assertEqual(symbol.payload["ipoDate"], "2010-01-01")

    @patch("data.universe_fmp.screen_companies_fmp")
    def test_screener_records_normalize_non_json_values(self, mocked_screen):
        mocked_screen.return_value = (
            ("SAFE",),
            [{
                "symbol": "SAFE",
                "companyName": "Safe Company",
                "exchangeShortName": "NASDAQ",
                "country": "US",
                "sector": "Technology",
                "industry": "Software",
                "marketCap": float("nan"),
                "ipoDate": "2019-01-02",
                "nested": {"invalid": float("nan")},
            }],
        )

        resolve_notebook_universe(
            {"source": "screener", "symbols": []},
            api_key="test-key",
            metadata_client=FakeProfileClient(),
        )

        symbol = Symbol.objects.get(symbol="SAFE")
        self.assertIsNone(symbol.market_cap)
        self.assertIsNone(symbol.payload["nested"]["invalid"])
