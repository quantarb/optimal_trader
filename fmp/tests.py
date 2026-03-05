from django.test import SimpleTestCase

from fmp.endpoints import get_symbol_endpoint_definitions
from fmp.endpoints.helpers import GRANULARITY_PREFERENCE, preferred_period
from fmp.models import Symbol


class EndpointRegistryTests(SimpleTestCase):
    def test_symbol_endpoint_registry_has_unique_keys(self):
        symbol_obj = Symbol(symbol="AAPL")

        endpoints = get_symbol_endpoint_definitions(symbol_obj)
        keys = [endpoint.key for endpoint in endpoints]

        self.assertEqual(len(keys), len(set(keys)))
        self.assertIn("prices_div_adj", keys)
        self.assertIn("profile", keys)
        self.assertIn("peer_symbols", keys)
        self.assertIn("income_statement_growth", keys)
        self.assertIn("balance_sheet_growth", keys)
        self.assertIn("cash_flow_growth", keys)

    def test_endpoint_definitions_keep_raw_candidate_shapes(self):
        symbol_obj = Symbol(symbol="MSFT")

        endpoints = get_symbol_endpoint_definitions(symbol_obj)
        profile = next(endpoint for endpoint in endpoints if endpoint.key == "profile")

        self.assertEqual(profile.candidates, [("/stable/profile", {"symbol": "MSFT"})])

    def test_period_endpoints_use_lowest_granularity_policy(self):
        symbol_obj = Symbol(symbol="NVDA")

        endpoints = get_symbol_endpoint_definitions(symbol_obj)
        expected_periods = {
            "key_metrics": preferred_period("quarter", "annual"),
            "ratios": preferred_period("quarter", "annual"),
            "analyst_estimates": preferred_period("annual"),
            "income_statement": preferred_period("quarter", "annual"),
            "income_statement_growth": preferred_period("quarter", "annual"),
            "balance_sheet": preferred_period("quarter", "annual"),
            "balance_sheet_growth": preferred_period("quarter", "annual"),
            "cash_flow": preferred_period("quarter", "annual"),
            "cash_flow_growth": preferred_period("quarter", "annual"),
            "financial_growth": preferred_period("quarter", "annual"),
        }

        for endpoint in endpoints:
            expected = expected_periods.get(endpoint.key)
            for _path, params in endpoint.candidates:
                if expected is None:
                    self.assertNotIn("period", params)
                    self.assertEqual(endpoint.supported_periods, ())
                elif "period" in params:
                    self.assertEqual(params["period"], expected)
                    self.assertEqual(params["period"], preferred_period(*endpoint.supported_periods))

    def test_preferred_period_uses_finest_supported_granularity(self):
        self.assertEqual(preferred_period("annual", "quarter"), "quarter")
        self.assertEqual(preferred_period("day", "quarter"), "day")
        self.assertEqual(GRANULARITY_PREFERENCE[0], "day")
