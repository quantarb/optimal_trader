from __future__ import annotations

from datetime import date

from django.test import SimpleTestCase

from fmp.endpoints.base import EndpointDefinition
from fmp.records import dedupe_historical_records
from fmp.transport import fetch_first_success, fetch_historical_records, run_with_retries, with_date_window


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get_json(self, path, *, params=None):
        params = dict(params or {})
        self.calls.append((path, params))
        response = self.responses(path, params) if callable(self.responses) else self.responses[(path, params.get("page"))]
        if isinstance(response, Exception):
            raise response
        return response


class FMPTransportTests(SimpleTestCase):
    def test_explicit_page_policy_fetches_until_short_page(self):
        client = FakeClient(
            lambda _path, params: (
                [{"id": 1}, {"id": 2}]
                if params["page"] == 0
                else ([{"id": 3}] if params["page"] == 1 else [])
            )
        )
        endpoint = EndpointDefinition(
            key="events",
            title="Events",
            kind="historical",
            threshold_days=1,
            max_rows=10,
            candidates=[("/stable/events", {"symbol": "AAPL"})],
            pagination="page",
            page_size=2,
        )

        records = fetch_historical_records(client, endpoint)

        self.assertEqual([item["id"] for item in records], [1, 2, 3])
        self.assertEqual([params["page"] for _, params in client.calls], [0, 1, 2])

    def test_candidate_fallback_uses_second_endpoint(self):
        client = FakeClient(
            lambda path, _params: RuntimeError("unsupported") if path == "/first" else [{"ok": True}]
        )
        result = fetch_first_success(client, [("/first", {}), ("/second", {})])
        self.assertEqual(result, [{"ok": True}])

    def test_date_window_and_chunking_are_endpoint_policy(self):
        client = FakeClient(lambda _path, params: [{"date": params["from"]}])
        endpoint = EndpointDefinition(
            key="prices",
            title="Prices",
            kind="historical",
            threshold_days=1,
            max_rows=10,
            candidates=[("/prices", {"symbol": "AAPL"})],
            supports_date_window=True,
            chunk_years=1,
        )
        endpoint = EndpointDefinition(
            **{
                **endpoint.__dict__,
                "candidates": with_date_window(
                    endpoint.candidates,
                    from_date=date(2020, 1, 1),
                    to_date=date(2022, 1, 2),
                    endpoint=endpoint,
                ),
            }
        )

        records = fetch_historical_records(client, endpoint)

        self.assertEqual(len(records), 3)
        self.assertEqual(client.calls[0][1]["from"], "2020-01-01")
        self.assertEqual(client.calls[-1][1]["to"], "2022-01-02")

    def test_identity_dedupe_preserves_distinct_same_day_events(self):
        records = [
            {"date": "2026-01-02", "transaction": "buy", "shares": 10},
            {"date": "2026-01-02", "transaction": "sell", "shares": 5},
            {"date": "2026-01-02", "transaction": "buy", "shares": 10},
        ]
        identity_deduped = dedupe_historical_records(records, by_date=False)
        daily_deduped = dedupe_historical_records(records, by_date=True)

        self.assertEqual(len(identity_deduped), 2)
        self.assertEqual(len(daily_deduped), 1)

    def test_retry_helper_reports_retries_and_backoff(self):
        calls = []
        sleeps = []

        def fetch():
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("temporary")
            return "ok"

        result, retries = run_with_retries(fetch, max_attempts=3, base_delay_s=0.5, sleep_fn=sleeps.append)
        self.assertEqual(result, "ok")
        self.assertEqual(retries, 2)
        self.assertEqual(sleeps, [0.5, 1.0])

    def test_non_paginated_endpoint_preserves_intentional_limit(self):
        client = FakeClient(lambda _path, params: [{"limit": params["limit"]}])
        endpoint = EndpointDefinition(
            key="news",
            title="News",
            kind="historical",
            threshold_days=1,
            max_rows=50,
            candidates=[("/stable/news/stock", {"symbols": "AAPL", "limit": 50})],
        )

        records = fetch_historical_records(client, endpoint)

        self.assertEqual(records, [{"limit": 50}])
