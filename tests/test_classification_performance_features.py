from __future__ import annotations

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")

import django
import numpy as np
import pandas as pd
from django.test import TestCase

django.setup()

from domain.models.feature_families import infer_feature_family_columns
from domain.features.specs import FeatureBuildSpec, FeatureToggleSpec
from features.classification_performance_features import (
    build_industry_performance_features,
    build_sector_performance_features,
)
from fmp.classification_performance import (
    ClassificationPerformanceKey,
    INDUSTRY_PERFORMANCE_CATEGORY,
    SECTOR_PERFORMANCE_CATEGORY,
    classification_performance_series_code,
    normalize_classification_performance_frame,
    save_classification_performance_frame,
)
from fmp.models import MacroObservation, Symbol
from workflows.feature_runtime import build_symbol_feature_result


class ClassificationPerformanceFeatureTests(TestCase):
    def setUp(self):
        self.symbol = Symbol.objects.create(
            symbol="TEST",
            exchange="NASDAQ",
            sector="Technology",
            industry="Software - Infrastructure",
        )
        self.dates = pd.bdate_range("2026-01-02", periods=25)
        self.target_index = pd.MultiIndex.from_arrays(
            [self.dates, ["TEST"] * len(self.dates)], names=["date", "symbol"]
        )
        self.prices = pd.DataFrame(
            {"close": 100.0 * np.cumprod(np.repeat(1.02, len(self.dates)))},
            index=self.dates,
        )

    def _save(self, category: str, classification: str, *, missing_index: int | None = None):
        rows = []
        for index, day in enumerate(self.dates):
            if index == missing_index:
                continue
            rows.append(
                {
                    "date": day.date().isoformat(),
                    "averageChange": 1.0,
                    "exchange": "NASDAQ",
                    "sector" if category == SECTOR_PERFORMANCE_CATEGORY else "industry": classification,
                }
            )
        return save_classification_performance_frame(
            ClassificationPerformanceKey(category, classification, "NASDAQ"),
            pd.DataFrame(rows),
        )

    def test_normalizes_fmp_percentage_points_to_decimal_returns(self):
        normalized = normalize_classification_performance_frame(
            pd.DataFrame([{"date": "2026-06-08", "averageChange": -0.6643558425512563}])
        )
        self.assertAlmostEqual(float(normalized.iloc[0]["return_decimal"]), -0.006643558425512563)

    def test_sector_features_compound_returns_and_do_not_forward_fill_missing_days(self):
        result = self._save(SECTOR_PERFORMANCE_CATEGORY, "Technology", missing_index=10)
        self.assertEqual(result["observations_saved"], 24)
        built = build_sector_performance_features(self.symbol, self.target_index, df_prices=self.prices)

        self.assertTrue(pd.isna(built.df.iloc[10]["sector_perf__return_1d"]))
        self.assertAlmostEqual(float(built.df.iloc[4]["sector_perf__return_5d"]), (1.01**5) - 1.0)
        self.assertAlmostEqual(float(built.df.iloc[4]["sector_perf__stock_excess_1d"]), 0.01)
        self.assertTrue(pd.isna(built.df.iloc[20]["sector_perf__return_21d"]))

    def test_industry_features_use_industry_exchange_series(self):
        self._save(INDUSTRY_PERFORMANCE_CATEGORY, "Software - Infrastructure")
        built = build_industry_performance_features(self.symbol, self.target_index, df_prices=self.prices)
        code = classification_performance_series_code(
            INDUSTRY_PERFORMANCE_CATEGORY, "Software - Infrastructure", "NASDAQ"
        )

        self.assertEqual(MacroObservation.objects.filter(series__code=code).count(), 25)
        self.assertAlmostEqual(float(built.df.iloc[-1]["industry_perf__return_21d"]), (1.01**21) - 1.0)
        grouped = infer_feature_family_columns(built.feature_cols)
        self.assertEqual(grouped["industry_performance"], built.feature_cols)

    def test_feature_panel_toggles_add_both_classification_families(self):
        self._save(SECTOR_PERFORMANCE_CATEGORY, "Technology")
        self._save(INDUSTRY_PERFORMANCE_CATEGORY, "Software - Infrastructure")
        result = build_symbol_feature_result(
            symbol="TEST",
            symbol_obj=self.symbol,
            df_prices=self.prices,
            build_spec=FeatureBuildSpec(
                toggles=FeatureToggleSpec(
                    include_price_technicals=False,
                    include_time_calendar_features=False,
                    include_fundamental_change=False,
                    include_statement_quality=False,
                    include_event_features=False,
                    include_ownership_features=False,
                    include_economic_indicators=False,
                    include_treasury_rates=False,
                    include_sector_performance=True,
                    include_industry_performance=True,
                )
            ),
            economic_df=pd.DataFrame(),
            treasury_df=pd.DataFrame(),
            representation_meta={},
        )

        self.assertEqual(len(result.grouped_feature_columns["sector_performance"]), 6)
        self.assertEqual(len(result.grouped_feature_columns["industry_performance"]), 6)
        self.assertIn("sector_perf__stock_excess_21d", result.symbol_frame.columns)
        self.assertIn("industry_perf__stock_excess_21d", result.symbol_frame.columns)
