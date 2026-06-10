from __future__ import annotations

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")

import django
import pandas as pd
from django.test import TestCase

django.setup()

from domain.features.specs import FeatureBuildSpec, FeatureToggleSpec
from domain.models.feature_families import infer_feature_family_columns
from features.classification_pe_features import build_industry_pe_features, build_sector_pe_features
from fmp.classification_pe import (
    ClassificationPEKey,
    INDUSTRY_PE_CATEGORY,
    SECTOR_PE_CATEGORY,
    classification_pe_series_code,
    normalize_classification_pe_frame,
    save_classification_pe_frame,
)
from fmp.models import MacroObservation, Symbol
from workflows.feature_runtime import build_symbol_feature_result


class ClassificationPEFeatureTests(TestCase):
    def setUp(self):
        self.symbol = Symbol.objects.create(
            symbol="TEST",
            exchange="NASDAQ",
            sector="Technology",
            industry="Software - Infrastructure",
        )
        self.dates = pd.bdate_range("2025-01-02", periods=270)
        self.index = pd.MultiIndex.from_arrays(
            [self.dates, ["TEST"] * len(self.dates)], names=["date", "symbol"]
        )
        self.prices = pd.DataFrame({"close": range(100, 370)}, index=self.dates, dtype=float)

    def _save(self, category: str, classification: str, *, missing_index: int | None = None):
        rows = []
        for position, day in enumerate(self.dates):
            if position == missing_index:
                continue
            rows.append({"date": day.date().isoformat(), "pe": 20.0 + position * 0.1})
        return save_classification_pe_frame(
            ClassificationPEKey(category, classification, "NASDAQ"), pd.DataFrame(rows)
        )

    def test_normalizes_pe_and_discards_invalid_rows(self):
        normalized = normalize_classification_pe_frame(
            pd.DataFrame(
                [
                    {"date": "2026-06-08", "pe": 44.25},
                    {"date": "bad", "pe": 50.0},
                    {"date": "2026-06-09", "pe": "bad"},
                ]
            )
        )
        self.assertEqual(len(normalized), 1)
        self.assertAlmostEqual(float(normalized.iloc[0]["pe"]), 44.25)

    def test_sector_pe_features_preserve_missing_dates_and_build_history_state(self):
        self._save(SECTOR_PE_CATEGORY, "Technology", missing_index=100)
        built = build_sector_pe_features(self.symbol, self.index)

        self.assertTrue(pd.isna(built.df.iloc[100]["sector_pe__level"]))
        self.assertAlmostEqual(float(built.df.iloc[1]["sector_pe__change_1d"]), 20.1 / 20.0 - 1.0)
        self.assertTrue(pd.isna(built.df.iloc[100]["sector_pe__zscore_63d"]))
        self.assertTrue(pd.notna(built.df.iloc[-1]["sector_pe__zscore_252d"]))

    def test_industry_pe_family_storage_and_inference(self):
        self._save(INDUSTRY_PE_CATEGORY, "Software - Infrastructure")
        built = build_industry_pe_features(self.symbol, self.index)
        code = classification_pe_series_code(INDUSTRY_PE_CATEGORY, "Software - Infrastructure", "NASDAQ")

        self.assertEqual(MacroObservation.objects.filter(series__code=code).count(), 270)
        self.assertEqual(infer_feature_family_columns(built.feature_cols)["industry_pe"], built.feature_cols)
        self.assertEqual(len(built.feature_cols), 6)

    def test_feature_panel_adds_sector_and_industry_pe_families(self):
        self._save(SECTOR_PE_CATEGORY, "Technology")
        self._save(INDUSTRY_PE_CATEGORY, "Software - Infrastructure")
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
                    include_sector_pe=True,
                    include_industry_pe=True,
                )
            ),
            economic_df=pd.DataFrame(),
            treasury_df=pd.DataFrame(),
            representation_meta={},
        )
        self.assertEqual(len(result.grouped_feature_columns["sector_pe"]), 6)
        self.assertEqual(len(result.grouped_feature_columns["industry_pe"]), 6)
