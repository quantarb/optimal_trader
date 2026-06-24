from __future__ import annotations

from unittest.mock import patch

from django.test import TestCase

from fmp.models import Symbol
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

    def test_explicit_symbols_create_records_without_profile_refresh(self):
        resolved = resolve_notebook_universe(
            {"symbols": "aapl, msft", "source": "local"},
            metadata_client=FakeProfileClient(),
        )

        self.assertEqual(resolved.symbols, ("AAPL", "MSFT"))
        self.assertEqual(resolved.source, "explicit symbols")
        self.assertEqual(Symbol.objects.filter(symbol__in=resolved.symbols).count(), 2)
        self.assertEqual(Symbol.objects.filter(symbol__in=resolved.symbols, sector="Technology").count(), 0)

    @patch("data.warehouse_universe.use_warehouse_screener", return_value=False)
    @patch("pipeline.universe_selection.resolve_symbol_universe", return_value=("DONE",))
    def test_local_universe_validates_complete_metadata_without_profile_calls(self, _mock_resolve, _mock_wh):
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

    @patch("data.warehouse_universe.use_warehouse_screener", return_value=False)
    @patch("data.universe_fmp.screen_companies_fmp")
    def test_screener_records_are_persisted_without_profile_refresh(self, mocked_screen, _mock_wh):
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
        self.assertEqual(Symbol.objects.get(symbol="SCRN").company_name, "Screened Company")
        self.assertEqual(Symbol.objects.get(symbol="SCRN").payload["ipoDate"], "2018-05-01")

    @patch("data.warehouse_universe.use_warehouse_screener", return_value=False)
    @patch("data.universe_fmp.screen_companies_fmp")
    def test_sparse_screener_record_preserves_existing_profile_metadata(self, mocked_screen, _mock_wh):
        complete = Symbol.objects.create(
            symbol="KEEP",
            company_name="Keep Company",
            exchange="NASDAQ",
            country="US",
            sector="Technology",
            industry="Software",
            payload={"ipoDate": "2010-01-01", "companyName": "Keep Company"},
        )
        mocked_screen.return_value = (("KEEP",), [{"symbol": "KEEP", "marketCap": 99_000_000_000}])

        resolve_notebook_universe(
            {"source": "screener", "symbols": []},
            api_key="test-key",
            metadata_client=FakeProfileClient(),
        )

        symbol = Symbol.objects.get(symbol="KEEP")
        self.assertEqual(symbol.company_name, "Keep Company")
        self.assertEqual(symbol.payload["ipoDate"], "2010-01-01")

    @patch("data.warehouse_universe.use_warehouse_screener", return_value=False)
    @patch("data.universe_fmp.screen_companies_fmp")
    def test_screener_records_normalize_non_json_values(self, mocked_screen, _mock_wh):
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

    @patch("fmp.symbol_metadata.symbols_missing_optional_metadata", return_value=())
    @patch("data.warehouse_universe.use_warehouse_screener", return_value=True)
    @patch("data.warehouse_universe.sync_warehouse_catalog_profiles_to_django_symbols")
    @patch("data.warehouse_universe.screen_universe_from_warehouse")
    def test_warehouse_screener_path_stores_catalog_then_syncs_metadata(
        self,
        mocked_screen,
        mocked_sync,
        _mock_use_wh,
        _mock_skip_optional,
    ):
        mocked_screen.return_value = (("WH",), "fmp")

        def _sync_catalog(symbols, **kwargs):
            symbol_obj, _created = Symbol.objects.get_or_create(symbol=symbols[0])
            symbol_obj.company_name = "Warehouse Co"
            symbol_obj.exchange = "NASDAQ"
            symbol_obj.country = "US"
            symbol_obj.sector = "Technology"
            symbol_obj.industry = "Software"
            symbol_obj.payload = {"ipoDate": "2015-01-01"}
            symbol_obj.save()

        mocked_sync.side_effect = _sync_catalog

        resolved = resolve_notebook_universe(
            {"source": "screener", "symbols": [], "exchanges": ["NASDAQ"]},
            api_key="test-key",
        )

        self.assertEqual(resolved.symbols, ("WH",))
        self.assertTrue(resolved.source.startswith("warehouse screener"))
        mocked_screen.assert_called_once()
        mocked_sync.assert_called_once()
