from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from django.test import SimpleTestCase, TestCase

from pipeline.characteristics_factor_model import (
    LatentFactorSpec,
    build_characteristic_factor_targets,
    build_characteristic_rank_panel,
    fit_characteristic_factor_ranker,
    score_characteristic_factor_ranker,
)
from pipeline.prediction_artifacts import save_prediction_frame_artifact
from pipeline.ranking_diagnostics import assign_cross_sectional_buckets, build_signal_ranking_panel
from pipeline.strategy_definitions import ResolvedStrategyDefinition, apply_strategy_definition


class CharacteristicsFactorResearchTests(SimpleTestCase):
    def test_build_characteristic_rank_panel_reuses_existing_feature_families(self):
        feature_df = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "AAA", "px__ret_21_d": 0.05, "px__ret_252_d": 0.20, "km__marketcap": 100.0, "text_field": "x"},
                {"date": "2024-01-02", "symbol": "BBB", "px__ret_21_d": -0.01, "px__ret_252_d": 0.10, "km__marketcap": 80.0, "text_field": "y"},
                {"date": "2024-01-03", "symbol": "AAA", "px__ret_21_d": 0.06, "px__ret_252_d": 0.25, "km__marketcap": 101.0, "text_field": "x"},
                {"date": "2024-01-03", "symbol": "BBB", "px__ret_21_d": -0.02, "px__ret_252_d": 0.11, "km__marketcap": 79.0, "text_field": "y"},
            ]
        )
        label_df = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "AAA", "future_rank_pct": 1.0, "trade_return": 0.03},
                {"date": "2024-01-02", "symbol": "BBB", "future_rank_pct": 0.0, "trade_return": -0.01},
                {"date": "2024-01-03", "symbol": "AAA", "future_rank_pct": 1.0, "trade_return": 0.04},
                {"date": "2024-01-03", "symbol": "BBB", "future_rank_pct": 0.0, "trade_return": -0.02},
            ]
        )

        panel, feature_cols, meta = build_characteristic_rank_panel(
            feature_df,
            label_df,
            feature_families=["prices_div_adj"],
        )

        self.assertEqual(int(meta["rows"]), 4)
        self.assertIn("px__ret_21_d", feature_cols)
        self.assertIn("px__ret_252_d", feature_cols)
        self.assertNotIn("km__marketcap", feature_cols)
        self.assertNotIn("text_field", feature_cols)

    def test_build_characteristic_factor_targets_produces_rebalance_exposures(self):
        dates = pd.bdate_range("2024-01-02", periods=30)
        factor_one = np.linspace(-0.01, 0.015, len(dates))
        factor_two = np.sin(np.arange(len(dates)) / 4.0) * 0.006
        betas = {
            "AAA": (1.0, 0.2),
            "BBB": (0.2, 1.0),
            "CCC": (-0.7, 0.4),
        }
        rows: list[dict[str, object]] = []
        for symbol, (beta_one, beta_two) in betas.items():
            close = 100.0
            for idx, date_value in enumerate(dates):
                daily_ret = (beta_one * factor_one[idx]) + (beta_two * factor_two[idx])
                close *= 1.0 + daily_ret
                rows.append(
                    {
                        "date": date_value.strftime("%Y-%m-%d"),
                        "symbol": symbol,
                        "ret_1": daily_ret,
                        "px__adj_close": close,
                        "px__ret_21_d": daily_ret * 5.0,
                    }
                )
        feature_df = pd.DataFrame(rows)
        rebalance_dates = [dates[14], dates[19], dates[24], dates[28]]
        label_rows: list[dict[str, object]] = []
        for date_value in rebalance_dates:
            for rank_index, symbol in enumerate(["AAA", "BBB", "CCC"]):
                label_rows.append(
                    {
                        "date": date_value.strftime("%Y-%m-%d"),
                        "symbol": symbol,
                        "future_rank_pct": max(0.0, 1.0 - (rank_index * 0.5)),
                        "trade_return": 0.03 - (rank_index * 0.02),
                    }
                )
        label_df = pd.DataFrame(label_rows)

        factor_return_df, exposure_df, factor_cols, meta = build_characteristic_factor_targets(
            feature_df,
            label_df,
            train_end_date=dates[19].strftime("%Y-%m-%d"),
            score_end_date=dates[28].strftime("%Y-%m-%d"),
            spec=LatentFactorSpec(n_factors=2, exposure_lookback_days=10, minimum_exposure_observations=6),
        )

        self.assertEqual(int((meta["basis_meta"] or {}).get("n_factors") or 0), 2)
        self.assertEqual(len(factor_cols), 2)
        self.assertFalse(factor_return_df.empty)
        self.assertFalse(exposure_df.empty)
        self.assertTrue(all(column.endswith("_beta") for column in factor_cols))

    def test_fit_and_score_characteristic_factor_ranker_outputs_ordered_scores(self):
        train_panel = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "AAA", "char_1": 1.0, "char_2": 0.1, "latent_factor_1_beta": 1.0, "latent_factor_2_beta": 0.1, "future_rank_pct": 0.95},
                {"date": "2024-01-02", "symbol": "BBB", "char_1": 0.6, "char_2": 0.4, "latent_factor_1_beta": 0.6, "latent_factor_2_beta": 0.4, "future_rank_pct": 0.75},
                {"date": "2024-01-02", "symbol": "CCC", "char_1": 0.2, "char_2": 0.9, "latent_factor_1_beta": 0.2, "latent_factor_2_beta": 0.9, "future_rank_pct": 0.35},
                {"date": "2024-01-03", "symbol": "AAA", "char_1": 1.1, "char_2": 0.1, "latent_factor_1_beta": 1.1, "latent_factor_2_beta": 0.1, "future_rank_pct": 0.98},
                {"date": "2024-01-03", "symbol": "BBB", "char_1": 0.5, "char_2": 0.4, "latent_factor_1_beta": 0.5, "latent_factor_2_beta": 0.4, "future_rank_pct": 0.70},
                {"date": "2024-01-03", "symbol": "CCC", "char_1": 0.1, "char_2": 1.0, "latent_factor_1_beta": 0.1, "latent_factor_2_beta": 1.0, "future_rank_pct": 0.30},
            ]
        )
        score_panel = pd.DataFrame(
            [
                {"date": "2024-01-04", "symbol": "AAA", "char_1": 1.05, "char_2": 0.1},
                {"date": "2024-01-04", "symbol": "BBB", "char_1": 0.55, "char_2": 0.45},
                {"date": "2024-01-04", "symbol": "CCC", "char_1": 0.15, "char_2": 0.95},
            ]
        )

        state = fit_characteristic_factor_ranker(
            train_panel,
            feature_cols=["char_1", "char_2"],
            factor_cols=["latent_factor_1_beta", "latent_factor_2_beta"],
            factor_premia={"alpha": 0.0, "latent_factor_1_beta": 1.0, "latent_factor_2_beta": -0.5},
            random_state=17,
            n_estimators=80,
            max_depth=4,
            min_samples_leaf=1,
        )
        scored = score_characteristic_factor_ranker(state, score_panel)

        ordered = scored.sort_values("prediction_score", ascending=False)["symbol"].tolist()
        self.assertEqual(ordered[0], "AAA")
        self.assertEqual(ordered[-1], "CCC")
        self.assertGreater(float(state["train_score_rank_corr"]), 0.5)

    def test_characteristic_factor_scores_work_with_cross_sectional_quantiles(self):
        feature_df = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "AAA", "strategy_score": 0.9},
                {"date": "2024-01-02", "symbol": "BBB", "strategy_score": 0.5},
                {"date": "2024-01-02", "symbol": "CCC", "strategy_score": -0.2},
                {"date": "2024-01-03", "symbol": "AAA", "strategy_score": 0.8},
                {"date": "2024-01-03", "symbol": "BBB", "strategy_score": 0.4},
                {"date": "2024-01-03", "symbol": "CCC", "strategy_score": -0.3},
            ]
        )
        definition = ResolvedStrategyDefinition(
            definition_id=1,
            name="characteristics-factor",
            slug="characteristics-factor",
            strategy_type="notebook_topk_v1",
            config={
                "rebalance_freq": "M",
                "gross_exposure": 1.0,
                "selection_side": "long_short",
                "signal_combination": "direct",
                "action_source_field": "ranking",
                "portfolio_construction": "cross_sectional_quantiles",
                "cross_sectional_score_field": "strategy_score",
                "cross_sectional_bucket_count": 3,
                "long_bucket": "top",
                "short_bucket": "bottom",
                "holding_period_rebalances": 1,
                "ranking_lag_days": 0,
                "higher_score_is_better": True,
            },
        )

        strategy_df, meta = apply_strategy_definition(feature_df, definition)
        rows = strategy_df[strategy_df["date"] == "2024-01-02"].set_index("symbol")

        self.assertEqual(meta["strategy_config"]["portfolio_construction"], "cross_sectional_quantiles")
        self.assertGreater(float(rows.loc["AAA", "target_weight"]), 0.0)
        self.assertLess(float(rows.loc["CCC", "target_weight"]), 0.0)
        self.assertEqual(int(rows.loc["AAA", "cross_sectional_bucket"]), 3)
        self.assertEqual(int(rows.loc["CCC", "cross_sectional_bucket"]), 1)


class CharacteristicsFactorArtifactTests(TestCase):
    def test_prediction_artifact_can_feed_ranking_diagnostics(self):
        prediction_df = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "AAA", "prediction": 0.8, "prediction_score": 0.8},
                {"date": "2024-01-02", "symbol": "BBB", "prediction": 0.2, "prediction_score": 0.2},
                {"date": "2024-01-02", "symbol": "CCC", "prediction": -0.1, "prediction_score": -0.1},
            ]
        )
        label_df = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "AAA", "future_rank_pct": 1.0, "trade_return": 0.03},
                {"date": "2024-01-02", "symbol": "BBB", "future_rank_pct": 0.5, "trade_return": 0.01},
                {"date": "2024-01-02", "symbol": "CCC", "future_rank_pct": 0.0, "trade_return": -0.02},
            ]
        )

        artifact = save_prediction_frame_artifact(
            prediction_df,
            run_name="Characteristics Factor Test Predictions",
            requested_job="score_characteristic_factor_ranker",
        )
        panel = build_signal_ranking_panel(
            artifact,
            label_df,
            variant_name="characteristics_factor_rf_all_features",
            fold_name="wf_2024",
            variant_kind="model",
            variant_label="Characteristics Factor All Features",
            feature_scope="all_features",
            symbol_metadata_lookup={},
        )
        bucketed = assign_cross_sectional_buckets(panel, bucket_count=3)

        self.assertTrue(Path(str(artifact.uri)).exists())
        self.assertEqual(int(bucketed["bucket"].max()), 3)
        top_symbol = bucketed.sort_values("bucket", ascending=False)["symbol"].iloc[0]
        self.assertEqual(top_symbol, "AAA")
