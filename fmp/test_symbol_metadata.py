from __future__ import annotations

from datetime import datetime, timezone as datetime_timezone

from django.test import TestCase

from fmp.models import Symbol, SymbolSectionSnapshot, SymbolSectionState
from fmp.symbol_dates import symbol_listing_date
from fmp.symbol_metadata import (
    apply_profile_metadata,
    refresh_symbol_metadata_from_fmp,
    sync_symbol_metadata_from_fmp,
    symbol_metadata_missing,
)


class FakeProfileClient:
    def __init__(self, payloads):
        self.payloads = dict(payloads)
        self.calls: list[str] = []

    def get_json(self, path, *, params=None):
        self.assert_path = path
        symbol = str((params or {}).get("symbol") or "")
        self.calls.append(symbol)
        return self.payloads.get(symbol, [])


class SymbolMetadataRepairTests(TestCase):
    def test_apply_profile_metadata_populates_symbol_and_listing_date(self):
        symbol = Symbol.objects.create(symbol="NEW")

        updated = apply_profile_metadata(
            symbol,
            {
                "symbol": "NEW",
                "companyName": "New Company",
                "exchangeShortName": "NASDAQ",
                "country": "US",
                "sector": "Technology",
                "industry": "Software",
                "ipoDate": "2024-03-15",
                "marketCap": 123456,
            },
        )
        symbol.refresh_from_db()

        self.assertIn("payload", updated)
        self.assertEqual(symbol.company_name, "New Company")
        self.assertEqual(symbol.exchange, "NASDAQ")
        self.assertEqual(symbol.sector, "Technology")
        self.assertEqual(symbol_listing_date(symbol).isoformat(), "2024-03-15")
        self.assertEqual(symbol_metadata_missing(symbol), ["profile_snapshot"])

    def test_repair_targets_only_incomplete_symbols_and_saves_profile_state(self):
        incomplete = Symbol.objects.create(symbol="MISS")
        complete = Symbol.objects.create(
            symbol="DONE",
            company_name="Done Company",
            exchange="NYSE",
            country="US",
            sector="Industrials",
            industry="Machinery",
            payload={"ipoDate": "2000-01-01"},
        )
        SymbolSectionSnapshot.objects.create(symbol=complete, section_key="profile", payload={"symbol": "DONE"})
        client = FakeProfileClient(
            {
                "MISS": [
                    {
                        "companyName": "Missing Company",
                        "exchangeShortName": "NASDAQ",
                        "country": "US",
                        "sector": "Technology",
                        "industry": "Software",
                        "ipoDate": "2022-06-01",
                    }
                ]
            }
        )

        result = refresh_symbol_metadata_from_fmp(client=client)
        incomplete.refresh_from_db()

        self.assertEqual(client.calls, ["MISS"])
        statuses = result.set_index("symbol")["status"].to_dict()
        self.assertEqual(statuses, {"DONE": "skipped_complete", "MISS": "updated"})
        self.assertEqual(incomplete.company_name, "Missing Company")
        self.assertTrue(SymbolSectionSnapshot.objects.filter(symbol=incomplete, section_key="profile").exists())
        self.assertTrue(SymbolSectionState.objects.filter(symbol=incomplete, section_key="profile").exists())

    def test_recent_profile_attempt_observes_cooldown(self):
        symbol = Symbol.objects.create(symbol="EMPTY")
        SymbolSectionState.objects.create(
            symbol=symbol,
            section_key="profile",
            kind="snapshot",
            last_fetched_at=datetime.now(tz=datetime_timezone.utc),
        )
        client = FakeProfileClient({"EMPTY": []})

        result = refresh_symbol_metadata_from_fmp(symbols=["EMPTY"], client=client)

        self.assertEqual(client.calls, [])
        self.assertEqual(result.iloc[0]["status"], "skipped_recent")

    def test_ensure_metadata_raises_when_profile_cannot_supply_requirements(self):
        Symbol.objects.create(symbol="EMPTY")
        client = FakeProfileClient({"EMPTY": []})

        with self.assertRaisesRegex(RuntimeError, "EMPTY"):
            sync_symbol_metadata_from_fmp(symbols=["EMPTY"], client=client, force=True)

    def test_complete_symbol_is_not_downloaded_even_when_force_is_set(self):
        complete = Symbol.objects.create(
            symbol="DONE",
            company_name="Done Company",
            exchange="NYSE",
            country="US",
            sector="Industrials",
            industry="Machinery",
            payload={"ipoDate": "2000-01-01"},
        )
        SymbolSectionSnapshot.objects.create(symbol=complete, section_key="profile", payload={"symbol": "DONE"})
        client = FakeProfileClient({"DONE": [{"companyName": "Replacement"}]})

        result = refresh_symbol_metadata_from_fmp(symbols=["DONE"], client=client, force=True)

        self.assertEqual(client.calls, [])
        self.assertEqual(result.iloc[0]["status"], "skipped_complete")

    def test_profile_persistence_keeps_the_complete_response(self):
        symbol = Symbol.objects.create(symbol="FULL")
        record = {
            "symbol": "FULL",
            "companyName": "Full Company",
            "exchangeShortName": "NASDAQ",
            "country": "US",
            "sector": "Technology",
            "industry": "Software",
            "ipoDate": "2021-02-03",
            "description": "Complete company description",
            "ceo": "Example CEO",
            "defaultImage": True,
            "optionalNullField": None,
        }

        refresh_symbol_metadata_from_fmp(
            symbols=["FULL"],
            client=FakeProfileClient({"FULL": [record]}),
        )
        symbol.refresh_from_db()
        snapshot = SymbolSectionSnapshot.objects.get(symbol=symbol, section_key="profile")

        self.assertEqual(snapshot.payload, [record])
        self.assertEqual(symbol.payload["description"], record["description"])
        self.assertEqual(symbol.payload["ceo"], record["ceo"])
        self.assertIn("optionalNullField", symbol.payload)
        self.assertIsNone(symbol.payload["optionalNullField"])

    def test_profile_persistence_normalizes_non_json_values(self):
        symbol = Symbol.objects.create(symbol="SAFE")
        record = {
            "symbol": "SAFE",
            "companyName": "Safe Company",
            "exchangeShortName": "NASDAQ",
            "country": "US",
            "sector": "Technology",
            "industry": "Software",
            "ipoDate": "2020-01-02",
            "marketCap": float("nan"),
            "nested": {"invalid": float("nan")},
        }

        refresh_symbol_metadata_from_fmp(
            symbols=["SAFE"],
            client=FakeProfileClient({"SAFE": [record]}),
        )
        symbol.refresh_from_db()
        snapshot = SymbolSectionSnapshot.objects.get(symbol=symbol, section_key="profile")

        self.assertIsNone(symbol.market_cap)
        self.assertIsNone(symbol.payload["nested"]["invalid"])
        self.assertIsNone(snapshot.payload[0]["nested"]["invalid"])
