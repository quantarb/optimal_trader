from __future__ import annotations

import unittest

import pandas as pd

from ml.base import FitSpec
from ml.frameworks.sklearn import SklearnMoERFClassifier


class SklearnMoERFClassifierTests(unittest.TestCase):
    def test_trains_family_specific_trees_and_scores_available_families(self) -> None:
        frame = pd.DataFrame(
            {
                "tech_a": [0.1, 0.2, 0.9, 1.0, 0.3, 0.8],
                "tech_b": [1.0, 0.9, 0.1, 0.0, 0.8, 0.2],
                "fund_a": [10.0, 11.0, 30.0, 32.0, None, None],
                "fund_b": [4.0, 5.0, 20.0, 22.0, None, None],
                "target": [0, 0, 1, 1, 0, 1],
                "sample_weight": [1, 1, 1, 1, 1, 1],
            }
        )
        model = SklearnMoERFClassifier(
            {
                "technical": ["tech_a", "tech_b"],
                "fundamental": ["fund_a", "fund_b"],
            },
            n_estimators=6,
            max_depth=3,
            min_samples_leaf=1,
            random_state=7,
        )

        model.fit(
            frame,
            FitSpec(
                feature_cols=["tech_a", "tech_b", "fund_a", "fund_b"],
                target_col="target",
                weight_col="sample_weight",
                split_ratio=1.0,
            ),
            verbose=False,
        )

        self.assertEqual(sum(len(trees) for trees in model._trees_by_family.values()), 6)
        self.assertEqual(len(model.model), 2)
        self.assertEqual(set(model._trees_by_family), {"technical", "fundamental"})
        scored = model.predict_moe_frame(frame)
        self.assertIn("technical__prob_buy", scored.columns)
        self.assertIn("fundamental__prob_buy", scored.columns)
        self.assertIn("clf__prob_1", scored.columns)
        self.assertTrue(scored.loc[4:, "fundamental__prob_buy"].isna().all())
        self.assertTrue(scored["clf__prob_1"].between(0.0, 1.0).all())


if __name__ == "__main__":
    unittest.main()
