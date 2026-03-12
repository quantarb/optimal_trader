from __future__ import annotations

import pandas as pd
from django.test import SimpleTestCase

from pipeline.cross_sectional_rank_labels import (
    CrossSectionalRankLabelSpec,
    build_cross_sectional_rank_label_frame,
)
from pipeline.ranking_diagnostics import (
    aggregate_bucket_overlap_rows,
    aggregate_ranking_summary_rows,
    assign_cross_sectional_buckets,
    build_signal_ranking_panel,
    compute_bucket_overlap_rows,
    compute_bucket_return_rows,
    compute_ranking_summary_rows,
    compute_top_bucket_stability_rows,
)


class OracleRankingResearchTests(SimpleTestCase):
    def test_cross_sectional_rank_labels_compute_forward_percentiles(self):
        feature_df = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "close": 100.0},
                {"date": "2024-01-02", "symbol": "AAA", "close": 110.0},
                {"date": "2024-01-03", "symbol": "AAA", "close": 105.0},
                {"date": "2024-01-01", "symbol": "BBB", "close": 100.0},
                {"date": "2024-01-02", "symbol": "BBB", "close": 90.0},
                {"date": "2024-01-03", "symbol": "BBB", "close": 108.0},
            ]
        )
        label_df, meta = build_cross_sectional_rank_label_frame(
            feature_df,
            spec=CrossSectionalRankLabelSpec(
                horizon_days=1,
                rebalance_freq="D",
                start_offset_days=0,
                minimum_cross_section=2,
                target_col="future_rank_pct",
                forward_return_col="trade_return",
                price_col_candidates=("close",),
            ),
        )

        self.assertEqual(int(meta["rows"]), 4)
        day1 = label_df[label_df["date"] == "2024-01-01"].set_index("symbol")
        self.assertAlmostEqual(float(day1.loc["AAA", "trade_return"]), 0.10, places=6)
        self.assertAlmostEqual(float(day1.loc["BBB", "trade_return"]), -0.10, places=6)
        self.assertAlmostEqual(float(day1.loc["AAA", "future_rank_pct"]), 1.0, places=6)
        self.assertAlmostEqual(float(day1.loc["BBB", "future_rank_pct"]), 0.0, places=6)

        day2 = label_df[label_df["date"] == "2024-01-02"].set_index("symbol")
        self.assertAlmostEqual(float(day2.loc["AAA", "future_rank_pct"]), 0.0, places=6)
        self.assertAlmostEqual(float(day2.loc["BBB", "future_rank_pct"]), 1.0, places=6)

    def test_ranking_diagnostics_compute_ic_bucket_returns_and_overlap(self):
        label_df = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "future_rank_pct": 1.0, "trade_return": 0.04},
                {"date": "2024-01-01", "symbol": "BBB", "future_rank_pct": 0.666667, "trade_return": 0.02},
                {"date": "2024-01-01", "symbol": "CCC", "future_rank_pct": 0.333333, "trade_return": -0.01},
                {"date": "2024-01-01", "symbol": "DDD", "future_rank_pct": 0.0, "trade_return": -0.03},
                {"date": "2024-01-02", "symbol": "AAA", "future_rank_pct": 1.0, "trade_return": 0.03},
                {"date": "2024-01-02", "symbol": "BBB", "future_rank_pct": 0.666667, "trade_return": 0.01},
                {"date": "2024-01-02", "symbol": "CCC", "future_rank_pct": 0.333333, "trade_return": -0.02},
                {"date": "2024-01-02", "symbol": "DDD", "future_rank_pct": 0.0, "trade_return": -0.04},
            ]
        )
        baseline_scores = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "score": 0.9},
                {"date": "2024-01-01", "symbol": "BBB", "score": 0.7},
                {"date": "2024-01-01", "symbol": "CCC", "score": 0.3},
                {"date": "2024-01-01", "symbol": "DDD", "score": 0.1},
                {"date": "2024-01-02", "symbol": "AAA", "score": 0.8},
                {"date": "2024-01-02", "symbol": "BBB", "score": 0.6},
                {"date": "2024-01-02", "symbol": "CCC", "score": 0.2},
                {"date": "2024-01-02", "symbol": "DDD", "score": 0.0},
            ]
        )
        model_scores = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "score": 0.9},
                {"date": "2024-01-01", "symbol": "BBB", "score": 0.2},
                {"date": "2024-01-01", "symbol": "CCC", "score": 0.8},
                {"date": "2024-01-01", "symbol": "DDD", "score": 0.1},
                {"date": "2024-01-02", "symbol": "AAA", "score": 0.7},
                {"date": "2024-01-02", "symbol": "BBB", "score": 0.3},
                {"date": "2024-01-02", "symbol": "CCC", "score": 0.6},
                {"date": "2024-01-02", "symbol": "DDD", "score": 0.0},
            ]
        )

        baseline_panel = build_signal_ranking_panel(
            baseline_scores,
            label_df,
            variant_name="baseline_momentum",
            fold_name="wf_2024",
            variant_kind="baseline",
            variant_label="Baseline Momentum",
            feature_scope="prices_only",
            symbol_metadata_lookup={},
        )
        model_panel = build_signal_ranking_panel(
            model_scores,
            label_df,
            variant_name="oracle_rank_rf_all_features",
            fold_name="wf_2024",
            variant_kind="model",
            variant_label="ML All Features",
            feature_scope="all_features",
            symbol_metadata_lookup={},
        )

        baseline_bucketed = assign_cross_sectional_buckets(baseline_panel, bucket_count=2)
        model_bucketed = assign_cross_sectional_buckets(model_panel, bucket_count=2)
        ranking_rows = compute_ranking_summary_rows(baseline_bucketed, bucket_count=2)
        ranking_aggregate = aggregate_ranking_summary_rows(ranking_rows)
        bucket_rows = compute_bucket_return_rows(baseline_bucketed)
        overlap_rows = compute_bucket_overlap_rows(
            baseline_bucketed,
            model_bucketed,
            bucket_count=2,
            left_variant_name="baseline_momentum",
            right_variant_name="oracle_rank_rf_all_features",
        )
        overlap_aggregate = aggregate_bucket_overlap_rows(overlap_rows)

        self.assertEqual(len(ranking_rows), 1)
        self.assertAlmostEqual(float(ranking_rows[0]["mean_spearman_ic"]), 1.0, places=6)
        self.assertGreater(float(ranking_rows[0]["mean_long_short_spread"]), 0.0)
        self.assertEqual(ranking_aggregate[0]["variant_name"], "baseline_momentum")
        top_bucket = next(row for row in bucket_rows if int(row["bucket"]) == 2)
        self.assertGreater(float(top_bucket["avg_forward_return"]), 0.0)
        self.assertEqual(len(overlap_rows), 2)
        self.assertAlmostEqual(float(overlap_aggregate[0]["jaccard"]), 1.0 / 3.0, places=6)

    def test_top_bucket_stability_rows_measure_fold_overlap(self):
        bucketed_df = pd.DataFrame(
            [
                {"variant_name": "oracle_rank_rf_all_features", "fold_name": "wf_2024", "date": "2024-01-01", "symbol": "AAA", "bucket": 2, "sector": "Tech", "instrument_type": "stock"},
                {"variant_name": "oracle_rank_rf_all_features", "fold_name": "wf_2024", "date": "2024-01-01", "symbol": "BBB", "bucket": 2, "sector": "Tech", "instrument_type": "stock"},
                {"variant_name": "oracle_rank_rf_all_features", "fold_name": "wf_2025", "date": "2025-01-01", "symbol": "AAA", "bucket": 2, "sector": "Tech", "instrument_type": "stock"},
                {"variant_name": "oracle_rank_rf_all_features", "fold_name": "wf_2025", "date": "2025-01-01", "symbol": "CCC", "bucket": 2, "sector": "Health", "instrument_type": "stock"},
            ]
        )

        stability_rows, symbol_rows = compute_top_bucket_stability_rows(bucketed_df, bucket_count=2)

        self.assertEqual(len(stability_rows), 1)
        self.assertEqual(stability_rows[0]["variant_name"], "oracle_rank_rf_all_features")
        self.assertAlmostEqual(float(stability_rows[0]["mean_pairwise_jaccard"]), 1.0 / 3.0, places=6)
        aaa_row = next(row for row in symbol_rows if row["symbol"] == "AAA")
        self.assertEqual(int(aaa_row["folds_selected"]), 2)
