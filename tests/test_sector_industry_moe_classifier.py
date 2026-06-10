from __future__ import annotations

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")

import django
import pandas as pd
from django.test import TestCase

django.setup()

from fmp.models import Symbol, SymbolSectionSnapshot
from ml.artifact_datasets import attach_symbol_classifications
from ml.base import FitSpec
from ml.frameworks.sklearn import SklearnRoutedMoERFClassifier
from ml.model_runtime import fit_model_for_algorithm
from pipeline.forms import FitModelPipelineForm


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["T1", "T2", "T3", "T4", "U1", "U2", "U3", "U4"],
            "sector": ["Technology"] * 4 + ["Utilities"] * 4,
            "industry": ["Software"] * 4 + ["Electric Utilities"] * 4,
            "feature_a": [0.0, 0.1, 0.9, 1.0, 1.0, 0.9, 0.1, 0.0],
            "feature_b": [1.0, 0.9, 0.1, 0.0, 0.0, 0.1, 0.9, 1.0],
            "target": [0, 0, 1, 1, 0, 0, 1, 1],
            "sample_weight": [1.0] * 8,
        }
    )


class RoutedMoERFClassifierTests(TestCase):
    def test_sector_experts_route_rows_and_fallback_for_unseen_sector(self):
        frame = _training_frame()
        model = SklearnRoutedMoERFClassifier(
            route_col="sector",
            min_expert_rows=4,
            n_estimators=8,
            max_depth=3,
            random_state=7,
        )
        model.fit(
            frame,
            FitSpec(
                feature_cols=["feature_a", "feature_b"],
                target_col="target",
                weight_col="sample_weight",
                split_ratio=1.0,
            ),
            verbose=False,
        )

        self.assertEqual(set(model._experts), {"Technology", "Utilities"})
        scored = model.predict_frame(
            pd.DataFrame(
                {
                    "sector": ["Technology", "Utilities", "Healthcare"],
                    "feature_a": [0.9, 0.9, 0.9],
                    "feature_b": [0.1, 0.1, 0.1],
                }
            )
        )
        self.assertEqual(scored["moe_expert"].tolist(), ["Technology", "Utilities", "__global__"])
        self.assertEqual(scored["moe_used_fallback"].tolist(), [False, False, True])
        self.assertTrue(set(scored["prediction"]).issubset({0, 1}))
        self.assertTrue(scored["clf__prob_1"].between(0.0, 1.0).all())

    def test_runtime_builds_industry_routed_moe(self):
        model = fit_model_for_algorithm(
            algorithm="industry_moe_random_forest_classifier",
            train_df=_training_frame(),
            feature_cols=["feature_a", "feature_b"],
            model_params={"min_expert_rows": 4, "n_estimators": 6, "max_depth": 2},
            target_col="target",
            split_ratio=1.0,
        )
        self.assertEqual(model.route_col, "industry")
        self.assertEqual(set(model._experts), {"Software", "Electric Utilities"})

    def test_symbol_classifications_are_attached_from_fmp_profile_fields(self):
        Symbol.objects.create(symbol="AAPL", sector="Technology", industry="Consumer Electronics")
        Symbol.objects.create(symbol="NEE", sector="Utilities", industry="Regulated Electric")
        frame = attach_symbol_classifications(
            pd.DataFrame(
                {
                    "date": ["2025-01-01", "2025-01-01", "2025-01-01"],
                    "symbol": ["AAPL", "NEE", "MISSING"],
                    "close": [100.0, 80.0, 20.0],
                }
            )
        )
        self.assertEqual(frame["sector"].tolist(), ["Technology", "Utilities", "Unknown"])
        self.assertEqual(
            frame["industry"].tolist(),
            ["Consumer Electronics", "Regulated Electric", "Unknown"],
        )

    def test_profile_endpoint_snapshot_fills_blank_symbol_classifications(self):
        symbol = Symbol.objects.create(symbol="SOFT")
        SymbolSectionSnapshot.objects.create(
            symbol=symbol,
            section_key="profile",
            payload=[{"symbol": "SOFT", "sector": "Technology", "industry": "Software - Application"}],
        )
        frame = attach_symbol_classifications(
            pd.DataFrame({"date": ["2025-01-01"], "symbol": ["SOFT"], "close": [50.0]})
        )
        self.assertEqual(frame.loc[0, "sector"], "Technology")
        self.assertEqual(frame.loc[0, "industry"], "Software - Application")

    def test_pipeline_form_exposes_sector_and_industry_moe_algorithms(self):
        choices = dict(FitModelPipelineForm().fields["algorithm"].choices)
        self.assertIn("sector_moe_random_forest_classifier", choices)
        self.assertIn("industry_moe_random_forest_classifier", choices)
