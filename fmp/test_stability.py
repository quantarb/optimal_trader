from __future__ import annotations

from datetime import date, datetime, timedelta, timezone as datetime_timezone

from django.test import TestCase

from fmp.endpoints.base import EndpointDefinition
from fmp.models import Symbol, SymbolSectionState
from fmp.section_store import save_historical_section
from fmp.stability import assess_historical_section_stability


class HistoricalSectionStabilityTests(TestCase):
    def setUp(self):
        self.symbol = Symbol.objects.create(symbol="TEST")
        self.now = datetime(2026, 6, 8, 12, tzinfo=datetime_timezone.utc)

    def endpoint(self, **overrides):
        values = {
            "key": "prices",
            "title": "Prices",
            "kind": "historical",
            "threshold_days": 1,
            "min_history_years": 1,
            "max_rows": 10,
            "candidates": [("/prices", {"symbol": "TEST"})],
            "dedupe_by_date": True,
        }
        values.update(overrides)
        return EndpointDefinition(**values)

    def mark_fetched(self, section_key: str):
        SymbolSectionState.objects.create(
            symbol=self.symbol,
            section_key=section_key,
            kind="historical",
            last_fetched_at=self.now,
        )

    def test_dense_daily_data_is_stable(self):
        records = []
        current = date(2025, 6, 9)
        while current <= date(2026, 6, 8):
            if current.weekday() < 5:
                records.append({"date": current.isoformat(), "close": 100.0})
            current += timedelta(days=1)
        save_historical_section(self.symbol, "prices", records, dedupe_by_date=True)
        self.mark_fetched("prices")

        result = assess_historical_section_stability(
            self.symbol,
            self.endpoint(),
            target_start=date(2006, 6, 8),
            target_end=date(2026, 6, 8),
            now=self.now,
        )

        self.assertTrue(result.stable)
        self.assertEqual(result.reason, "stable_historical_section")
        self.assertGreaterEqual(result.density_ratio, 0.99)

    def test_sparse_daily_data_is_not_stable_even_with_wide_range(self):
        save_historical_section(
            self.symbol,
            "prices",
            [
                {"date": "2025-06-09", "close": 90.0},
                {"date": "2026-06-08", "close": 100.0},
            ],
            dedupe_by_date=True,
        )
        self.mark_fetched("prices")

        result = assess_historical_section_stability(
            self.symbol,
            self.endpoint(),
            target_start=date(2006, 6, 8),
            target_end=date(2026, 6, 8),
            now=self.now,
        )

        self.assertFalse(result.stable)
        self.assertEqual(result.reason, "sparse_observation_density")

    def test_recently_confirmed_empty_event_section_is_stable(self):
        endpoint = self.endpoint(
            key="splits",
            candidates=[("/splits", {"symbol": "TEST"})],
            dedupe_by_date=False,
            min_history_years=15,
            threshold_days=30,
        )
        self.mark_fetched("splits")

        result = assess_historical_section_stability(
            self.symbol,
            endpoint,
            target_start=date(2006, 6, 8),
            target_end=date(2026, 6, 8),
            now=self.now,
        )

        self.assertTrue(result.stable)
        self.assertEqual(result.reason, "stable_event_section")
