from __future__ import annotations

from django.test import SimpleTestCase

from domain.features.panel import needed_sparse_sections
from domain.features.specs import FeatureToggleSpec
from workflows.fmp_feature_families import _fmp_endpoint_builders


class NoTtmFinancialStatementFeatureTests(SimpleTestCase):
    def test_legacy_ttm_toggle_is_ignored(self):
        toggles = FeatureToggleSpec.from_mapping({"include_ttm_financial_statements": True})
        sections = needed_sparse_sections(toggles)
        self.assertNotIn("income_statement_ttm", sections)
        self.assertNotIn("cash_flow_ttm", sections)
        self.assertNotIn("balance_sheet_ttm", sections)

    def test_fmp_endpoint_builders_do_not_expose_ttm_families(self):
        builders = _fmp_endpoint_builders(filing_lag_days=45)
        self.assertNotIn("income_statement_ttm", builders)
        self.assertNotIn("cash_flow_ttm", builders)
        self.assertNotIn("balance_sheet_ttm", builders)
