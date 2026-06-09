from __future__ import annotations

from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase

from domain.features.panel import needed_sparse_sections
from domain.features.specs import BuiltFeatureSet, FeatureBuildSpec, FeatureToggleSpec
from features.feature_builders import build_ttm_financial_statement_features
from fmp.models import Symbol
from workflows.feature_runtime import build_symbol_feature_result
from workflows.fmp_feature_families import _fmp_endpoint_builders


class TtmFinancialStatementFeatureTests(SimpleTestCase):
    def setUp(self):
        dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
        self.target_index = pd.MultiIndex.from_arrays(
            [dates, ["AAPL", "AAPL"]],
            names=["date", "symbol"],
        )
        self.prices = pd.DataFrame({"close": [100.0, 110.0]}, index=dates)
        self.symbol = Symbol(symbol="AAPL")

    def _built(self, column: str, values: list[float]) -> BuiltFeatureSet:
        return BuiltFeatureSet(
            df=pd.DataFrame({column: values}, index=self.target_index),
            feature_cols=[column],
        )

    def test_toggle_primes_all_ttm_sections(self):
        sections = needed_sparse_sections(
            FeatureToggleSpec(
                include_price_technicals=False,
                include_time_calendar_features=False,
                include_fundamental_change=False,
                include_statement_quality=False,
                include_ttm_financial_statements=True,
                include_event_features=False,
                include_ownership_features=False,
                include_economic_indicators=False,
                include_treasury_rates=False,
            )
        )
        self.assertEqual(
            sections,
            [
                "key_metrics_ttm",
                "ratios_ttm",
                "income_statement_ttm",
                "cash_flow_ttm",
                "balance_sheet_ttm",
            ],
        )

    def test_options_backtest_builder_exposes_separate_ttm_model_families(self):
        builders = _fmp_endpoint_builders(filing_lag_days=45)
        self.assertTrue(
            {
                "key_metrics_ttm",
                "ratios_ttm",
                "income_statement_ttm",
                "cash_flow_ttm",
                "balance_sheet_ttm",
            }.issubset(builders)
        )

    def test_builder_merges_five_families_and_reuses_ttm_market_cap(self):
        captured_market_caps: list[pd.Series | None] = []

        def fake_cash_flow(*args, market_cap=None, **kwargs):
            captured_market_caps.append(market_cap)
            return self._built("cf_ttm__freecashflow", [20.0, 20.0])

        def fake_balance_sheet(*args, market_cap=None, **kwargs):
            captured_market_caps.append(market_cap)
            return self._built("bs_ttm__totalassets", [500.0, 500.0])

        with (
            patch(
                "features.feature_builders.build_key_metrics_ttm_features",
                return_value=self._built("km_ttm__marketcap", [1000.0, 1100.0]),
            ),
            patch(
                "features.feature_builders.build_ratios_ttm_features",
                return_value=self._built("rt_ttm__currentratio", [1.2, 1.2]),
            ),
            patch(
                "features.feature_builders.build_income_statement_ttm_features",
                return_value=self._built("is_ttm__weightedaverageshsoutdil", [10.0, 10.0]),
            ),
            patch("features.feature_builders.build_cash_flow_ttm_features", side_effect=fake_cash_flow),
            patch("features.feature_builders.build_balance_sheet_ttm_features", side_effect=fake_balance_sheet),
        ):
            built = build_ttm_financial_statement_features(
                self.symbol,
                self.target_index,
                df_prices=self.prices,
            )

        self.assertEqual(
            built.feature_cols,
            [
                "km_ttm__marketcap",
                "rt_ttm__currentratio",
                "is_ttm__weightedaverageshsoutdil",
                "cf_ttm__freecashflow",
                "bs_ttm__totalassets",
            ],
        )
        self.assertEqual(len(captured_market_caps), 2)
        for market_cap in captured_market_caps:
            self.assertIsNotNone(market_cap)
            self.assertEqual(list(market_cap), [1000.0, 1100.0])

    def test_symbol_workflow_exposes_ttm_family_metadata(self):
        toggles = FeatureToggleSpec(
            include_price_technicals=False,
            include_time_calendar_features=False,
            include_fundamental_change=False,
            include_statement_quality=False,
            include_ttm_financial_statements=True,
            include_event_features=False,
            include_ownership_features=False,
            include_economic_indicators=False,
            include_treasury_rates=False,
        )
        built = BuiltFeatureSet(
            df=pd.DataFrame(
                {
                    "km_ttm__marketcap": [1000.0, 1100.0],
                    "rt_ttm__currentratio": [1.2, 1.2],
                    "is_ttm__revenue": [400.0, 400.0],
                    "cf_ttm__freecashflow": [20.0, 20.0],
                    "bs_ttm__totalassets": [500.0, 500.0],
                },
                index=self.target_index,
            ),
            feature_cols=[
                "km_ttm__marketcap",
                "rt_ttm__currentratio",
                "is_ttm__revenue",
                "cf_ttm__freecashflow",
                "bs_ttm__totalassets",
            ],
        )
        with patch("workflows.feature_runtime.build_ttm_financial_statement_features", return_value=built):
            result = build_symbol_feature_result(
                symbol="AAPL",
                symbol_obj=self.symbol,
                df_prices=self.prices,
                build_spec=FeatureBuildSpec(toggles=toggles),
                economic_df=pd.DataFrame(),
                treasury_df=pd.DataFrame(),
                representation_meta={"columns": []},
            )

        self.assertEqual(result.grouped_feature_columns["key_metrics_ttm"], ["km_ttm__marketcap"])
        self.assertEqual(result.grouped_feature_columns["ratios_ttm"], ["rt_ttm__currentratio"])
        self.assertEqual(result.grouped_feature_columns["income_statement_ttm"], ["is_ttm__revenue"])
        self.assertEqual(result.grouped_feature_columns["cash_flow_ttm"], ["cf_ttm__freecashflow"])
        self.assertEqual(result.grouped_feature_columns["balance_sheet_ttm"], ["bs_ttm__totalassets"])
