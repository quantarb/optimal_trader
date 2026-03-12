from __future__ import annotations

import pandas as pd
from django.test import SimpleTestCase, TestCase
from django.utils import timezone

from fmp.models import Symbol
from pipeline.cross_sectional_rank_labels import (
    CrossSectionalRankLabelSpec,
    build_cross_sectional_rank_label_frame,
)
from pipeline.models import Artifact, PipelineRun
from pipeline.oracle_residual_label_experiment import (
    aggregate_prediction_diagnostic_rows,
    compute_prediction_diagnostic_rows,
)
from pipeline.prediction_artifacts import save_prediction_frame_artifact
from pipeline.service_runtime import artifact_payload_hash, write_frame_artifact


class OracleResidualLabelFrameTests(TestCase):
    def setUp(self):
        super().setUp()
        Symbol.objects.bulk_create(
            [
                Symbol(symbol="AAA", company_name="AAA Corp", sector="Technology", payload={}),
                Symbol(symbol="BBB", company_name="BBB Corp", sector="Financial Services", payload={}),
                Symbol(symbol="CCC", company_name="CCC ETF", sector="Utilities", payload={"isEtf": True}),
                Symbol(symbol="DDD", company_name="DDD Corp", sector="Technology", payload={}),
            ]
        )

    def test_residual_label_variant_adds_factor_expected_and_residual_targets(self):
        feature_df = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "close": 100.0, "km__marketcap": 300.0, "px__ret_252d": 0.30, "px__ret_21d": 0.05, "px__vol_20": 0.12},
                {"date": "2024-01-02", "symbol": "AAA", "close": 106.0, "km__marketcap": 302.0, "px__ret_252d": 0.32, "px__ret_21d": 0.06, "px__vol_20": 0.11},
                {"date": "2024-01-03", "symbol": "AAA", "close": 108.0, "km__marketcap": 303.0, "px__ret_252d": 0.31, "px__ret_21d": 0.04, "px__vol_20": 0.10},
                {"date": "2024-01-01", "symbol": "BBB", "close": 100.0, "km__marketcap": 180.0, "px__ret_252d": 0.12, "px__ret_21d": 0.02, "px__vol_20": 0.18},
                {"date": "2024-01-02", "symbol": "BBB", "close": 101.0, "km__marketcap": 181.0, "px__ret_252d": 0.13, "px__ret_21d": 0.03, "px__vol_20": 0.17},
                {"date": "2024-01-03", "symbol": "BBB", "close": 100.5, "km__marketcap": 182.0, "px__ret_252d": 0.14, "px__ret_21d": 0.02, "px__vol_20": 0.19},
                {"date": "2024-01-01", "symbol": "CCC", "close": 100.0, "km__marketcap": 90.0, "px__ret_252d": 0.08, "px__ret_21d": 0.01, "px__vol_20": 0.09},
                {"date": "2024-01-02", "symbol": "CCC", "close": 98.0, "km__marketcap": 91.0, "px__ret_252d": 0.07, "px__ret_21d": -0.01, "px__vol_20": 0.08},
                {"date": "2024-01-03", "symbol": "CCC", "close": 97.5, "km__marketcap": 91.5, "px__ret_252d": 0.06, "px__ret_21d": -0.02, "px__vol_20": 0.08},
                {"date": "2024-01-01", "symbol": "DDD", "close": 100.0, "km__marketcap": 220.0, "px__ret_252d": 0.20, "px__ret_21d": 0.04, "px__vol_20": 0.15},
                {"date": "2024-01-02", "symbol": "DDD", "close": 97.0, "km__marketcap": 219.0, "px__ret_252d": 0.18, "px__ret_21d": 0.01, "px__vol_20": 0.16},
                {"date": "2024-01-03", "symbol": "DDD", "close": 99.0, "km__marketcap": 221.0, "px__ret_252d": 0.19, "px__ret_21d": 0.02, "px__vol_20": 0.15},
            ]
        )

        label_df, meta = build_cross_sectional_rank_label_frame(
            feature_df,
            spec=CrossSectionalRankLabelSpec(
                horizon_days=1,
                rebalance_freq="D",
                start_offset_days=0,
                minimum_cross_section=4,
                label_variant="residual",
                target_col="future_rank_pct",
                forward_return_col="trade_return",
                residualize_targets=True,
                residual_target_col="residual_rank_pct",
                residual_return_col="residual_return",
                fitted_return_col="factor_expected_return",
                price_col_candidates=("close",),
            ),
        )

        self.assertTrue(bool(meta["residualization_enabled"]))
        self.assertEqual(meta["active_target_col"], "residual_rank_pct")
        self.assertEqual(meta["size_col"], "km__marketcap")
        self.assertEqual(meta["volatility_col"], "px__vol_20")
        self.assertIn("factor_expected_return", label_df.columns)
        self.assertIn("residual_return", label_df.columns)
        self.assertIn("residual_rank_pct", label_df.columns)
        self.assertTrue(label_df["residual_return"].notna().all())
        self.assertTrue(label_df["residual_rank_pct"].between(0.0, 1.0).all())
        self.assertEqual(set(label_df["label"].unique().tolist()), {0, 1})


class OracleResidualLabelGuardTests(SimpleTestCase):
    def test_residual_label_variant_requires_residualization(self):
        feature_df = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "close": 100.0},
                {"date": "2024-01-02", "symbol": "AAA", "close": 101.0},
                {"date": "2024-01-01", "symbol": "BBB", "close": 100.0},
                {"date": "2024-01-02", "symbol": "BBB", "close": 99.0},
            ]
        )

        with self.assertRaisesMessage(ValueError, "label_variant='residual' requires residualize_targets=True"):
            build_cross_sectional_rank_label_frame(
                feature_df,
                spec=CrossSectionalRankLabelSpec(
                    horizon_days=1,
                    rebalance_freq="D",
                    start_offset_days=0,
                    minimum_cross_section=2,
                    label_variant="residual",
                    residualize_targets=False,
                    price_col_candidates=("close",),
                ),
            )


class OracleResidualDiagnosticsTests(TestCase):
    def _build_label_artifact(self, frame: pd.DataFrame) -> Artifact:
        stored = write_frame_artifact(
            "test_oracle_residual_labels",
            frame=frame,
            fieldnames=list(frame.columns),
        )
        now = timezone.now()
        run = PipelineRun.objects.create(
            name="test-oracle-residual-labels",
            requested_job="test_oracle_residual_labels",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
            started_at=now,
            finished_at=now,
        )
        content = {"rows": int(len(frame))}
        return Artifact.objects.create(
            pipeline_run=run,
            artifact_type="LABELS",
            key="test_oracle_residual_labels",
            uri=str(stored.uri),
            content=content,
            metadata=stored.storage_metadata(),
            payload_hash=artifact_payload_hash(content, str(stored.uri)),
        )

    def test_prediction_diagnostics_compare_raw_and_residual_ic(self):
        label_df = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "AAA", "trade_return": 0.05, "residual_return": 0.01, "future_rank_pct": 1.0},
                {"date": "2024-01-02", "symbol": "BBB", "trade_return": 0.02, "residual_return": 0.03, "future_rank_pct": 0.666667},
                {"date": "2024-01-02", "symbol": "CCC", "trade_return": -0.01, "residual_return": -0.02, "future_rank_pct": 0.333333},
                {"date": "2024-01-02", "symbol": "DDD", "trade_return": -0.03, "residual_return": -0.01, "future_rank_pct": 0.0},
            ]
        )
        prediction_df = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "AAA", "prediction": 0.9, "prediction_score": 0.9},
                {"date": "2024-01-02", "symbol": "BBB", "prediction": 0.6, "prediction_score": 0.6},
                {"date": "2024-01-02", "symbol": "CCC", "prediction": 0.2, "prediction_score": 0.2},
                {"date": "2024-01-02", "symbol": "DDD", "prediction": 0.1, "prediction_score": 0.1},
            ]
        )

        label_artifact = self._build_label_artifact(label_df)
        prediction_artifact = save_prediction_frame_artifact(
            prediction_df,
            run_name="Oracle Residual Diagnostic Predictions",
            requested_job="score_oracle_residual_test",
        )
        diagnostic_row = compute_prediction_diagnostic_rows(
            prediction_artifact,
            label_artifact,
            variant_name="raw_label_model",
            variant_kind="model",
            variant_label="Raw Label Model",
            label_type="raw",
            fold_name="wf_2024",
            target_col="future_rank_pct",
        )

        self.assertEqual(int(diagnostic_row["scored_rows"]), 4)
        self.assertAlmostEqual(float(diagnostic_row["mean_forward_return_ic"]), 1.0, places=6)
        self.assertGreater(float(diagnostic_row["mean_residual_return_ic"]), 0.0)

        aggregate_rows = aggregate_prediction_diagnostic_rows([diagnostic_row])
        self.assertEqual(len(aggregate_rows), 1)
        self.assertEqual(aggregate_rows[0]["variant_name"], "raw_label_model")
        self.assertEqual(int(aggregate_rows[0]["fold_count"]), 1)
