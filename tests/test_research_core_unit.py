from __future__ import annotations

import unittest

import pandas as pd

from analysis.oracle_entry_exit_dataset import build_oracle_entry_exit_frames
from domain.backtests import StrategyBacktestSpec, StrategyDatasetSpec
from domain.features.technical import compute_features_worldclass
from domain.features.panel import representation_embedding_dataset_rows
from domain.features.specs import FeatureBuildSpec
from domain.trades.operations import apply_trade_deduplication
from domain.trades.panel import labels_panel_to_trades_df
from ml.frameworks.transformers.context_family_mtl import ContextFamilyMTLDataSpec
from ml.metrics import (
    build_action_classification_report_df,
    build_action_f1_comparison_df,
    build_context_numeric_result,
    build_flair_action_report,
    build_flair_regression_result,
    build_regression_task_report_df,
)
from ml.rl.oracle_imitation import (
    build_expert_position_panel,
    build_transition_sample_weights,
    build_transition_training_mask,
    build_long_only_expert_action_panel,
    build_long_only_expert_position_panel,
    rollout_position_policy,
    rollout_long_only_position_policy,
    rollout_position_aware_long_only_policy,
    train_position_cloning_rf,
    train_position_cloning_hgb,
    train_long_only_behavior_cloning_rf,
    train_long_only_position_cloning_rf,
    train_position_aware_long_only_behavior_cloning_rf,
)


class ResearchCoreUnitTests(unittest.TestCase):
    def test_representation_embedding_rows_group_feature_families(self):
        frame = pd.DataFrame(
            [{"date": "2024-01-01", "symbol": "AAPL", "close": 100.0, "ret_1": 0.05, "km__ev_to_ebitda": 8.2}]
        )
        grouped = {"prices_div_adj": ["close", "ret_1"], "key_metrics": ["km__ev_to_ebitda"], "representation_embedding": []}
        rows = representation_embedding_dataset_rows(frame, grouped)
        self.assertEqual(rows[0]["families"]["price_technical"]["Close"], 100.0)
        self.assertEqual(rows[0]["families"]["valuation_quality"]["EV To EBITDA"], 8.2)

    def test_feature_build_spec_extracts_layered_config(self):
        spec = FeatureBuildSpec.from_mapping(
            {
                "include_price_technicals": True,
                "include_economic_indicators": False,
                "feature_start_date": "2024-01-01",
                "feature_end_date": "2024-03-31",
                "include_representation_embedding": True,
                "representation_embedding_store_dir": "/tmp/embeddings",
            }
        )
        self.assertTrue(spec.toggles.include_price_technicals)
        self.assertFalse(spec.toggles.include_economic_indicators)
        self.assertEqual(spec.start_date, "2024-01-01")
        self.assertEqual(spec.representation_embedding.store_dir, "/tmp/embeddings")

    def test_strategy_dataset_spec_extracts_prediction_inputs(self):
        spec = StrategyDatasetSpec.from_mapping(
            {
                "strategy_definition_id": "7",
                "prediction_artifact_ids": ["2", "2", "5", "bad"],
                "label_artifact_id": "9",
                "strategy_start_date": "2024-01-01",
                "strategy_end_date": "2024-03-31",
            }
        )
        self.assertEqual(spec.strategy_definition_id, 7)
        self.assertEqual(spec.prediction_artifact_ids, (2, 5))
        self.assertEqual(spec.label_artifact_id, 9)
        self.assertEqual(spec.start_date, "2024-01-01")
        self.assertEqual(spec.end_date, "2024-03-31")

    def test_strategy_backtest_spec_uses_transaction_cost_as_slippage_fallback(self):
        spec = StrategyBacktestSpec.from_mapping(
            {
                "transaction_cost_bps": "12.5",
                "execution_delay_days": "2",
                "turnover_half_l1": "false",
            }
        )
        self.assertEqual(spec.effective_slippage_bps(), 12.5)
        self.assertEqual(spec.execution_delay_days, 2)
        self.assertFalse(spec.turnover_half_l1)

    def test_trade_deduplication_prefers_best_return_for_same_entry(self):
        trade_rows = [
            {"symbol": "AAPL", "side": "long", "entry_date": "2024-01-01", "exit_date": "2024-01-03"},
            {"symbol": "AAPL", "side": "long", "entry_date": "2024-01-01", "exit_date": "2024-01-04"},
        ]
        completed = [
            {"symbol": "AAPL", "side": "long", "entry_date": "2024-01-01", "exit_date": "2024-01-03", "ret_dec": 0.05},
            {"symbol": "AAPL", "side": "long", "entry_date": "2024-01-01", "exit_date": "2024-01-04", "ret_dec": 0.15},
        ]
        _, kept = apply_trade_deduplication(trade_rows, completed, mode="entry_date")
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["ret_dec"], 0.15)

    def test_labels_panel_to_trades_df_pairs_rows(self):
        trades = labels_panel_to_trades_df(
            pd.DataFrame(
                [
                    {"date": "2024-01-01", "symbol": "AAPL", "side": "long", "horizon": "W", "trade_return": 0.12},
                    {"date": "2024-01-04", "symbol": "AAPL", "side": "long", "horizon": "W", "trade_return": 0.12},
                ]
            )
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["symbol"], "AAPL")

    def test_compute_features_worldclass_adds_21_day_return(self):
        dates = pd.bdate_range("2024-01-01", periods=30)
        prices = pd.DataFrame(
            {
                "open": pd.Series(range(100, 130), index=dates, dtype=float),
                "high": pd.Series(range(101, 131), index=dates, dtype=float),
                "low": pd.Series(range(99, 129), index=dates, dtype=float),
                "close": pd.Series(range(100, 130), index=dates, dtype=float),
                "volume": pd.Series([1_000_000] * len(dates), index=dates, dtype=float),
            },
            index=dates,
        )
        feature_df = compute_features_worldclass(prices)
        self.assertIn("Ret21d", feature_df.columns)
        expected = (129.0 / 108.0) - 1.0
        self.assertAlmostEqual(float(feature_df.iloc[-1]["Ret21d"]), expected, places=10)

    def test_build_oracle_entry_exit_frames_uses_action_sequence_labels(self):
        universe_df = pd.DataFrame(
            [
                {"symbol": "AAPL", "company_name": "Apple Inc.", "sector": "Tech", "industry": "Hardware"},
            ]
        )
        label_df = pd.DataFrame(
            [
                {
                    "trade_id": "trade-1",
                    "event": "entry",
                    "action_label": "buy",
                    "symbol": "AAPL",
                    "date": "2024-01-02",
                    "entry_date": "2024-01-02",
                    "exit_date": "2024-01-05",
                    "trade_return": 0.12,
                    "hold_days": 3,
                    "side": "long",
                },
                {
                    "trade_id": "trade-1",
                    "event": "exit",
                    "action_label": "sell",
                    "symbol": "AAPL",
                    "date": "2024-01-05",
                    "entry_date": "2024-01-02",
                    "exit_date": "2024-01-05",
                    "trade_return": 0.12,
                    "hold_days": 3,
                    "side": "long",
                },
            ]
        )
        completed_trades_df = pd.DataFrame()
        price_lookup_df = pd.DataFrame(
            [
                {"symbol": "AAPL", "date_text": "2024-01-02", "adj_open": 100.0, "adj_high": 101.0, "adj_low": 99.0, "adj_close": 100.5, "volume": 1_000_000},
                {"symbol": "AAPL", "date_text": "2024-01-05", "adj_open": 110.0, "adj_high": 111.0, "adj_low": 109.0, "adj_close": 110.5, "volume": 1_200_000},
            ]
        )

        trade_pair_df, state_df = build_oracle_entry_exit_frames(
            universe_df=universe_df,
            label_df=label_df,
            completed_trades_df=completed_trades_df,
            price_lookup_df=price_lookup_df,
        )

        self.assertEqual(len(trade_pair_df), 1)
        self.assertEqual(len(state_df), 2)
        self.assertEqual(trade_pair_df.iloc[0]["entry_action"], "buy")
        self.assertEqual(trade_pair_df.iloc[0]["exit_action"], "sell")
        self.assertIn("signed_trade_return_pct", trade_pair_df.columns)
        self.assertEqual(set(state_df["event_role"].tolist()), {"entry", "exit"})

    def test_flair_metric_helpers_match_macro_f1_report(self):
        y_true = ["buy", "buy", "short", "short"]
        y_pred = ["buy", "short", "short", "short"]
        flair_report = build_flair_action_report("entry", y_true, y_pred, ["buy", "short"])
        report_df = build_action_classification_report_df("dev", "entry", y_true, y_pred, ["buy", "short"])
        macro_from_df = float(report_df.loc[report_df["label"] == "macro avg", "f1-score"].iloc[0])
        self.assertAlmostEqual(float(flair_report["dict"]["macro avg"]["f1-score"]), macro_from_df)
        self.assertIn("micro avg", flair_report["dict"])

    def test_flair_regression_result_includes_expected_metrics(self):
        result = build_flair_regression_result("entry_return", [0.1, 0.5, 0.9], [0.2, 0.4, 0.8])
        self.assertEqual(result["task"], "entry_return")
        self.assertEqual(result["support"], 3)
        self.assertIn("AVG: mse:", result["detailed_result"])

    def test_build_action_f1_comparison_df_matches_report_macro_f1(self):
        metrics = {"entry_action_macro_f1": 0.6666666666666666}
        task_buffers = {
            "entry": {
                "action_true": ["buy", "buy", "short"],
                "action_pred": ["buy", "short", "short"],
            }
        }
        comparison_df = build_action_f1_comparison_df("dev", metrics, task_buffers, {"entry": ["buy", "short"]})
        self.assertEqual(len(comparison_df), 1)
        self.assertAlmostEqual(float(comparison_df.iloc[0]["difference"]), 0.0)

    def test_build_regression_task_report_df_includes_context_numeric_rows(self):
        report_df = build_regression_task_report_df(
            "dev",
            metrics={},
            task_buffers={},
            regression_specs=[("entry_return", [0.1, 0.5], [0.2, 0.4])],
            context_numeric_specs=[("transition", 2, 0.95, 0.31)],
        )
        self.assertEqual(set(report_df["task"].tolist()), {"entry_return", "transition"})
        transition_row = report_df.loc[report_df["task"] == "transition"].iloc[0]
        self.assertAlmostEqual(float(transition_row["context_cosine"]), 0.95)
        self.assertAlmostEqual(float(transition_row["numeric_recon_mae"]), 0.31)

    def test_build_context_numeric_result_formats_flair_like_summary(self):
        result = build_context_numeric_result("entry_reconstruction", 5, 0.99, 0.12)
        self.assertEqual(result["support"], 5)
        self.assertIn("AVG: context_cosine:", result["detailed_result"])

    def test_context_family_mtl_data_spec_builds_reverse_action_maps(self):
        spec = ContextFamilyMTLDataSpec(
            state_market_cols=["m1"],
            state_fundamental_cols=["f1"],
            state_macro_cols=["x1"],
            entry_market_cols=["em1"],
            entry_fundamental_cols=["ef1"],
            entry_macro_cols=["ex1"],
            exit_market_cols=["xm1"],
            exit_fundamental_cols=["xf1"],
            exit_macro_cols=["xx1"],
            state_market_stats={"mean": {"m1": 0.0}, "std": {"m1": 1.0}},
            state_fundamental_stats={"mean": {"f1": 0.0}, "std": {"f1": 1.0}},
            state_macro_stats={"mean": {"x1": 0.0}, "std": {"x1": 1.0}},
            exit_market_stats={"mean": {"xm1": 0.0}, "std": {"xm1": 1.0}},
            exit_fundamental_stats={"mean": {"xf1": 0.0}, "std": {"xf1": 1.0}},
            exit_macro_stats={"mean": {"xx1": 0.0}, "std": {"xx1": 1.0}},
            entry_action_to_id={"buy": 0, "short": 1},
            exit_action_to_id={"sell": 0, "cover": 1},
        )
        self.assertEqual(spec.id_to_entry_action[0], "buy")
        self.assertEqual(spec.id_to_exit_action[1], "cover")

    def test_build_long_only_expert_action_panel_marks_buy_sell_and_positions(self):
        daily_state_df = pd.DataFrame(
            [
                {"symbol": "AAPL", "date_text": "2024-01-01", "feature_a": 0.0},
                {"symbol": "AAPL", "date_text": "2024-01-02", "feature_a": 1.0},
                {"symbol": "AAPL", "date_text": "2024-01-03", "feature_a": 2.0},
                {"symbol": "AAPL", "date_text": "2024-01-04", "feature_a": 3.0},
                {"symbol": "AAPL", "date_text": "2024-01-05", "feature_a": 4.0},
            ]
        )
        trade_pair_df = pd.DataFrame(
            [
                {
                    "trade_id": "t1",
                    "symbol": "AAPL",
                    "side": "long",
                    "entry_date_text": "2024-01-02",
                    "exit_date_text": "2024-01-04",
                }
            ]
        )

        panel_df = build_long_only_expert_action_panel(daily_state_df, trade_pair_df)

        self.assertEqual(panel_df["expert_action"].tolist(), ["hold", "buy", "hold", "sell", "hold"])
        self.assertEqual(panel_df["expert_position_before"].tolist(), [0, 0, 1, 1, 0])
        self.assertEqual(panel_df["expert_position_after"].tolist(), [0, 1, 1, 0, 0])

    def test_build_long_only_expert_position_panel_labels_holding_window_long(self):
        daily_state_df = pd.DataFrame(
            [
                {"symbol": "AAPL", "date_text": "2024-01-01", "feature_a": 0.0},
                {"symbol": "AAPL", "date_text": "2024-01-02", "feature_a": 1.0},
                {"symbol": "AAPL", "date_text": "2024-01-03", "feature_a": 2.0},
                {"symbol": "AAPL", "date_text": "2024-01-04", "feature_a": 3.0},
                {"symbol": "AAPL", "date_text": "2024-01-05", "feature_a": 4.0},
            ]
        )
        trade_pair_df = pd.DataFrame(
            [
                {
                    "trade_id": "t1",
                    "symbol": "AAPL",
                    "side": "long",
                    "entry_date_text": "2024-01-02",
                    "exit_date_text": "2024-01-04",
                }
            ]
        )

        panel_df = build_long_only_expert_position_panel(daily_state_df, trade_pair_df)

        self.assertEqual(panel_df["expert_position_label"].tolist(), ["flat", "long", "long", "flat", "flat"])
        self.assertEqual(panel_df["entry_signal"].tolist(), [0, 1, 0, 0, 0])
        self.assertEqual(panel_df["exit_signal"].tolist(), [0, 0, 0, 1, 0])

    def test_build_expert_position_panel_labels_long_flat_short_sequence(self):
        daily_state_df = pd.DataFrame(
            [
                {"symbol": "AAPL", "date_text": "2024-01-01", "feature_a": 0.0},
                {"symbol": "AAPL", "date_text": "2024-01-02", "feature_a": 1.0},
                {"symbol": "AAPL", "date_text": "2024-01-03", "feature_a": 2.0},
                {"symbol": "AAPL", "date_text": "2024-01-04", "feature_a": 3.0},
                {"symbol": "AAPL", "date_text": "2024-01-05", "feature_a": 4.0},
                {"symbol": "AAPL", "date_text": "2024-01-06", "feature_a": 5.0},
                {"symbol": "AAPL", "date_text": "2024-01-07", "feature_a": 6.0},
            ]
        )
        trade_pair_df = pd.DataFrame(
            [
                {
                    "trade_id": "t1",
                    "symbol": "AAPL",
                    "side": "long",
                    "entry_date_text": "2024-01-02",
                    "exit_date_text": "2024-01-04",
                },
                {
                    "trade_id": "t2",
                    "symbol": "AAPL",
                    "side": "short",
                    "entry_date_text": "2024-01-06",
                    "exit_date_text": "2024-01-08",
                },
            ]
        )

        panel_df = build_expert_position_panel(daily_state_df, trade_pair_df)

        self.assertEqual(
            panel_df["expert_position_label"].tolist(),
            ["flat", "long", "long", "flat", "flat", "short", "short"],
        )
        self.assertEqual(panel_df["expert_action"].tolist(), ["hold", "buy", "hold", "sell", "hold", "short", "hold"])

    def test_build_transition_sample_weights_prioritizes_transitions_and_nearby_rows(self):
        panel_df = pd.DataFrame(
            [
                {"symbol": "AAPL", "date_text": "2024-01-01", "expert_action": "hold"},
                {"symbol": "AAPL", "date_text": "2024-01-02", "expert_action": "buy"},
                {"symbol": "AAPL", "date_text": "2024-01-03", "expert_action": "hold"},
                {"symbol": "AAPL", "date_text": "2024-01-04", "expert_action": "hold"},
                {"symbol": "AAPL", "date_text": "2024-01-05", "expert_action": "sell"},
                {"symbol": "AAPL", "date_text": "2024-01-06", "expert_action": "hold"},
            ]
        )

        weighted_df = build_transition_sample_weights(
            panel_df,
            near_window=1,
            transition_weight=1.0,
            near_weight=0.5,
            interior_weight=0.1,
        )

        self.assertEqual(weighted_df["transition_distance_steps"].tolist(), [1, 0, 1, 1, 0, 1])
        self.assertEqual(weighted_df["sample_weight"].tolist(), [0.5, 1.0, 0.5, 0.5, 1.0, 0.5])

    def test_build_transition_training_mask_keeps_all_transitions(self):
        panel_df = pd.DataFrame(
            [
                {"symbol": "AAPL", "date_text": "2024-01-01", "expert_action": "hold"},
                {"symbol": "AAPL", "date_text": "2024-01-02", "expert_action": "buy"},
                {"symbol": "AAPL", "date_text": "2024-01-03", "expert_action": "hold"},
                {"symbol": "AAPL", "date_text": "2024-01-04", "expert_action": "sell"},
                {"symbol": "AAPL", "date_text": "2024-01-05", "expert_action": "hold"},
            ]
        )

        masked_df = build_transition_training_mask(
            panel_df,
            near_window=1,
            transition_keep_prob=1.0,
            near_keep_prob=0.0,
            interior_keep_prob=0.0,
            random_state=7,
        )

        self.assertEqual(masked_df["keep_for_training"].tolist(), [False, True, False, True, False])

    def test_train_long_only_behavior_cloning_rf_returns_oos_reports(self):
        panel_rows = []
        for idx in range(6):
            panel_rows.append(
                {
                    "symbol": "AAPL",
                    "date_text": f"2019-01-0{idx + 1}",
                    "date": pd.Timestamp(f"2019-01-0{idx + 1}"),
                    "feature_signal": 0.0 if idx < 2 else (1.0 if idx < 4 else 2.0),
                    "expert_action": ["hold", "hold", "buy", "buy", "sell", "sell"][idx],
                    "expert_action_id": [0, 0, 1, 1, 2, 2][idx],
                }
            )
        for idx in range(6):
            panel_rows.append(
                {
                    "symbol": "AAPL",
                    "date_text": f"2020-01-0{idx + 1}",
                    "date": pd.Timestamp(f"2020-01-0{idx + 1}"),
                    "feature_signal": 0.0 if idx < 2 else (1.0 if idx < 4 else 2.0),
                    "expert_action": ["hold", "hold", "buy", "buy", "sell", "sell"][idx],
                    "expert_action_id": [0, 0, 1, 1, 2, 2][idx],
                }
            )

        result = train_long_only_behavior_cloning_rf(
            pd.DataFrame(panel_rows),
            ["feature_signal"],
            train_cutoff="2020-01-01",
            rf_kwargs={"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 1, "random_state": 7, "n_jobs": 1},
        )

        self.assertEqual(set(result.summary_df["split"].tolist()), {"train", "out_of_sample"})
        self.assertIn("prob_buy", result.scored_oos_df.columns)
        self.assertIn("prob_sell", result.scored_oos_df.columns)
        self.assertIn("macro avg", set(result.report_df["label"].tolist()))

    def test_train_long_only_position_cloning_rf_returns_position_reports(self):
        panel_rows = []
        for idx in range(6):
            panel_rows.append(
                {
                    "symbol": "AAPL",
                    "date_text": f"2019-01-0{idx + 1}",
                    "date": pd.Timestamp(f"2019-01-0{idx + 1}"),
                    "feature_signal": 0.0 if idx < 2 else (1.0 if idx < 4 else 2.0),
                    "expert_position_label": ["flat", "flat", "long", "long", "flat", "flat"][idx],
                    "expert_position_id": [0, 0, 1, 1, 0, 0][idx],
                }
            )
        for idx in range(6):
            panel_rows.append(
                {
                    "symbol": "AAPL",
                    "date_text": f"2020-01-0{idx + 1}",
                    "date": pd.Timestamp(f"2020-01-0{idx + 1}"),
                    "feature_signal": 0.0 if idx < 2 else (1.0 if idx < 4 else 2.0),
                    "expert_position_label": ["flat", "flat", "long", "long", "flat", "flat"][idx],
                    "expert_position_id": [0, 0, 1, 1, 0, 0][idx],
                }
            )

        result = train_long_only_position_cloning_rf(
            pd.DataFrame(panel_rows),
            ["feature_signal"],
            train_cutoff="2020-01-01",
            rf_kwargs={"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 1, "random_state": 7, "n_jobs": 1},
        )

        self.assertEqual(set(result.summary_df["split"].tolist()), {"train", "out_of_sample"})
        self.assertIn("prob_long", result.scored_oos_df.columns)
        self.assertIn("macro avg", set(result.report_df["label"].tolist()))

    def test_train_position_cloning_rf_returns_three_state_reports(self):
        panel_rows = []
        labels = ["flat", "long", "long", "flat", "short", "short", "flat"]
        ids = [0, 1, 1, 0, 2, 2, 0]
        for idx in range(len(labels)):
            panel_rows.append(
                {
                    "symbol": "AAPL",
                    "date_text": f"2019-01-0{idx + 1}",
                    "date": pd.Timestamp(f"2019-01-0{idx + 1}"),
                    "feature_signal": float(idx),
                    "expert_position_label": labels[idx],
                    "expert_position_id": ids[idx],
                }
            )
        for idx in range(len(labels)):
            panel_rows.append(
                {
                    "symbol": "AAPL",
                    "date_text": f"2020-01-0{idx + 1}",
                    "date": pd.Timestamp(f"2020-01-0{idx + 1}"),
                    "feature_signal": float(idx),
                    "expert_position_label": labels[idx],
                    "expert_position_id": ids[idx],
                }
            )

        result = train_position_cloning_rf(
            pd.DataFrame(panel_rows),
            ["feature_signal"],
            train_cutoff="2020-01-01",
            rf_kwargs={"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 1, "random_state": 7, "n_jobs": 1},
        )

        self.assertEqual(set(result.summary_df["split"].tolist()), {"train", "out_of_sample"})
        self.assertIn("prob_short", result.scored_oos_df.columns)
        self.assertIn("short", set(result.report_df["label"].tolist()))

    def test_train_position_cloning_hgb_returns_three_state_reports(self):
        panel_rows = []
        labels = ["flat", "long", "long", "flat", "short", "short", "flat"] * 3
        ids = [0, 1, 1, 0, 2, 2, 0] * 3
        train_dates = pd.date_range("2019-01-01", periods=len(labels), freq="D")
        oos_dates = pd.date_range("2020-01-01", periods=len(labels), freq="D")
        for dt, label, label_id, feature_signal in zip(train_dates, labels, ids, range(len(labels))):
            panel_rows.append(
                {
                    "symbol": "AAPL",
                    "date_text": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "date": pd.Timestamp(dt),
                    "feature_signal": float(feature_signal % 7),
                    "expert_position_label": label,
                    "expert_position_id": label_id,
                }
            )
        for dt, label, label_id, feature_signal in zip(oos_dates, labels, ids, range(len(labels))):
            panel_rows.append(
                {
                    "symbol": "AAPL",
                    "date_text": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "date": pd.Timestamp(dt),
                    "feature_signal": float(feature_signal % 7),
                    "expert_position_label": label,
                    "expert_position_id": label_id,
                }
            )

        result = train_position_cloning_hgb(
            pd.DataFrame(panel_rows),
            ["feature_signal"],
            train_cutoff="2020-01-01",
            hgb_kwargs={"max_iter": 40, "max_depth": 3, "min_samples_leaf": 1, "random_state": 7, "early_stopping": False},
        )

        self.assertEqual(set(result.summary_df["split"].tolist()), {"train", "out_of_sample"})
        self.assertIn("prob_short", result.scored_oos_df.columns)
        self.assertIn("short", set(result.report_df["label"].tolist()))

    def test_train_position_aware_long_only_behavior_cloning_rf_returns_policy_state_metrics(self):
        panel_rows = []
        train_dates = pd.date_range("2019-01-01", periods=8, freq="D")
        oos_dates = pd.date_range("2020-01-01", periods=8, freq="D")
        train_specs = [
            (train_dates[0], 0.0, "hold", 0),
            (train_dates[1], 1.0, "buy", 0),
            (train_dates[2], 2.0, "hold", 1),
            (train_dates[3], 3.0, "sell", 1),
            (train_dates[4], 0.0, "hold", 0),
            (train_dates[5], 1.0, "buy", 0),
            (train_dates[6], 2.0, "hold", 1),
            (train_dates[7], 3.0, "sell", 1),
        ]
        oos_specs = [
            (oos_dates[0], 0.0, "hold", 0),
            (oos_dates[1], 1.0, "buy", 0),
            (oos_dates[2], 2.0, "hold", 1),
            (oos_dates[3], 3.0, "sell", 1),
            (oos_dates[4], 0.0, "hold", 0),
            (oos_dates[5], 1.0, "buy", 0),
            (oos_dates[6], 2.0, "hold", 1),
            (oos_dates[7], 3.0, "sell", 1),
        ]
        for dt, feature_signal, action, position_before in train_specs + oos_specs:
            panel_rows.append(
                {
                    "symbol": "AAPL",
                    "date": pd.Timestamp(dt),
                    "date_text": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "feature_signal": feature_signal,
                    "expert_action": action,
                    "expert_action_id": {"hold": 0, "buy": 1, "sell": 2}[action],
                    "expert_position_before": position_before,
                }
            )

        result = train_position_aware_long_only_behavior_cloning_rf(
            pd.DataFrame(panel_rows),
            ["feature_signal"],
            train_cutoff="2020-01-01",
            rf_kwargs={"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 1, "random_state": 7, "n_jobs": 1},
        )

        self.assertEqual(set(result.summary_df["policy_state"].tolist()), {"flat", "long"})
        self.assertIn("prob_buy", result.scored_oos_df.columns)
        self.assertIn("prob_sell", result.scored_oos_df.columns)
        self.assertEqual(set(result.report_df["policy_state"].tolist()), {"flat", "long"})

    def test_rollout_position_aware_long_only_policy_uses_simulated_position(self):
        train_rows = []
        oos_rows = []
        train_dates = pd.date_range("2019-01-01", periods=4, freq="D")
        oos_dates = pd.date_range("2020-01-01", periods=4, freq="D")
        specs = [
            (0.0, "hold", 0),
            (1.0, "buy", 0),
            (2.0, "hold", 1),
            (3.0, "sell", 1),
        ]
        for dt, (feature_signal, action, position_before) in zip(train_dates, specs):
            train_rows.append(
                {
                    "symbol": "AAPL",
                    "date": pd.Timestamp(dt),
                    "date_text": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "feature_signal": feature_signal,
                    "expert_action": action,
                    "expert_action_id": {"hold": 0, "buy": 1, "sell": 2}[action],
                    "expert_position_before": position_before,
                }
            )
        for dt, (feature_signal, action, position_before) in zip(oos_dates, specs):
            oos_rows.append(
                {
                    "symbol": "AAPL",
                    "date": pd.Timestamp(dt),
                    "date_text": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "feature_signal": feature_signal,
                    "expert_action": action,
                    "expert_action_id": {"hold": 0, "buy": 1, "sell": 2}[action],
                    "expert_position_before": position_before,
                }
            )

        result = train_position_aware_long_only_behavior_cloning_rf(
            pd.DataFrame(train_rows + oos_rows),
            ["feature_signal"],
            train_cutoff="2020-01-01",
            rf_kwargs={"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 1, "random_state": 7, "n_jobs": 1},
        )
        rollout_df = rollout_position_aware_long_only_policy(
            pd.DataFrame(oos_rows),
            ["feature_signal"],
            flat_model=result.flat_model,
            long_model=result.long_model,
        )

        self.assertEqual(rollout_df["sim_position_before"].tolist(), [0, 0, 1, 1])
        self.assertEqual(rollout_df["pred_action"].tolist(), ["hold", "buy", "hold", "sell"])

    def test_rollout_long_only_position_policy_turns_position_changes_into_actions(self):
        panel_rows = []
        dates = pd.date_range("2020-01-01", periods=4, freq="D")
        specs = [
            (0.0, "flat", 0),
            (1.0, "long", 1),
            (2.0, "long", 1),
            (3.0, "flat", 0),
        ]
        for dt, (feature_signal, position_label, position_id) in zip(dates, specs):
            panel_rows.append(
                {
                    "symbol": "AAPL",
                    "date": pd.Timestamp(dt),
                    "date_text": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "adj_close": 100.0 + feature_signal,
                    "feature_signal": feature_signal,
                    "expert_position_label": position_label,
                    "expert_position_id": position_id,
                }
            )
        train_rows = []
        for dt, (feature_signal, position_label, position_id) in zip(pd.date_range("2019-01-01", periods=4, freq="D"), specs):
            train_rows.append(
                {
                    "symbol": "AAPL",
                    "date": pd.Timestamp(dt),
                    "date_text": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "feature_signal": feature_signal,
                    "expert_position_label": position_label,
                    "expert_position_id": position_id,
                }
            )
        result = train_long_only_position_cloning_rf(
            pd.DataFrame(train_rows + panel_rows),
            ["feature_signal"],
            train_cutoff="2020-01-01",
            rf_kwargs={"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 1, "random_state": 7, "n_jobs": 1},
        )
        rollout_df = rollout_long_only_position_policy(
            pd.DataFrame(panel_rows),
            ["feature_signal"],
            model=result.model,
        )

        self.assertEqual(rollout_df["pred_position_label"].tolist(), ["flat", "long", "long", "flat"])
        self.assertEqual(rollout_df["pred_action"].tolist(), ["hold", "buy", "hold", "sell"])

    def test_rollout_position_policy_turns_three_state_changes_into_actions(self):
        panel_rows = []
        dates = pd.date_range("2020-01-01", periods=7, freq="D")
        specs = [
            (0.0, "flat", 0),
            (1.0, "long", 1),
            (2.0, "long", 1),
            (3.0, "flat", 0),
            (4.0, "short", 2),
            (5.0, "short", 2),
            (6.0, "flat", 0),
        ]
        for dt, (feature_signal, position_label, position_id) in zip(dates, specs):
            panel_rows.append(
                {
                    "symbol": "AAPL",
                    "date": pd.Timestamp(dt),
                    "date_text": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "feature_signal": feature_signal,
                    "expert_position_label": position_label,
                    "expert_position_id": position_id,
                }
            )
        train_rows = []
        for dt, (feature_signal, position_label, position_id) in zip(pd.date_range("2019-01-01", periods=7, freq="D"), specs):
            train_rows.append(
                {
                    "symbol": "AAPL",
                    "date": pd.Timestamp(dt),
                    "date_text": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "feature_signal": feature_signal,
                    "expert_position_label": position_label,
                    "expert_position_id": position_id,
                }
            )
        result = train_position_cloning_rf(
            pd.DataFrame(train_rows + panel_rows),
            ["feature_signal"],
            train_cutoff="2020-01-01",
            rf_kwargs={"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 1, "random_state": 7, "n_jobs": 1},
        )
        rollout_df = rollout_position_policy(
            pd.DataFrame(panel_rows),
            ["feature_signal"],
            model=result.model,
        )

        self.assertEqual(
            rollout_df["pred_position_label"].tolist(),
            ["flat", "long", "long", "flat", "short", "short", "flat"],
        )
        self.assertEqual(
            rollout_df["pred_action"].tolist(),
            ["hold", "buy", "hold", "sell", "short", "hold", "cover"],
        )

    def test_rollout_position_policy_flips_directly_between_long_and_short(self):
        panel_rows = []
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        specs = [
            (0.0, "flat", 0),
            (1.0, "long", 1),
            (2.0, "short", 2),
            (3.0, "long", 1),
            (4.0, "flat", 0),
        ]
        for dt, (feature_signal, position_label, position_id) in zip(dates, specs):
            panel_rows.append(
                {
                    "symbol": "AAPL",
                    "date": pd.Timestamp(dt),
                    "date_text": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "feature_signal": feature_signal,
                    "expert_position_label": position_label,
                    "expert_position_id": position_id,
                }
            )
        train_rows = []
        for dt, (feature_signal, position_label, position_id) in zip(pd.date_range("2019-01-01", periods=5, freq="D"), specs):
            train_rows.append(
                {
                    "symbol": "AAPL",
                    "date": pd.Timestamp(dt),
                    "date_text": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "feature_signal": feature_signal,
                    "expert_position_label": position_label,
                    "expert_position_id": position_id,
                }
            )
        result = train_position_cloning_rf(
            pd.DataFrame(train_rows + panel_rows),
            ["feature_signal"],
            train_cutoff="2020-01-01",
            rf_kwargs={"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 1, "random_state": 7, "n_jobs": 1},
        )
        rollout_df = rollout_position_policy(
            pd.DataFrame(panel_rows),
            ["feature_signal"],
            model=result.model,
        )

        self.assertEqual(
            rollout_df["pred_position_label"].tolist(),
            ["flat", "long", "short", "long", "flat"],
        )
        self.assertEqual(
            rollout_df["pred_action"].tolist(),
            ["hold", "buy", "short", "buy", "sell"],
        )
        self.assertEqual(
            rollout_df["sim_position_label_after"].tolist(),
            ["flat", "long", "short", "long", "flat"],
        )
