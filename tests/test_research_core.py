from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from django.test import TestCase

from domain.features.panel import representation_embedding_dataset_rows
from domain.features.specs import FeatureBuildSpec
from domain.labels.specs import LabelBuildSpec
from domain.trades.operations import apply_trade_deduplication
from domain.trades.panel import labels_panel_to_trades_df
from fmp.models import Symbol
from pipeline.models import Artifact, PipelineRun
from pipeline.service_runtime import write_frame_artifact
from workflows.features import build_feature_panel_frame_for_symbols
from workflows.labels import build_oracle_labels
from workflows.modeling import (
    build_model_scoring_spec,
    build_model_training_spec,
    score_model_workflow,
    train_model_workflow,
)


class DomainFeatureTests(unittest.TestCase):
    def test_representation_embedding_rows_group_feature_families(self):
        frame = pd.DataFrame(
            [{"date": "2024-01-01", "symbol": "AAPL", "close": 100.0, "ret_1": 0.05, "km__ev_to_ebitda": 8.2}]
        )
        grouped = {"prices_div_adj": ["close", "ret_1"], "key_metrics": ["km__ev_to_ebitda"], "representation_embedding": []}
        rows = representation_embedding_dataset_rows(frame, grouped)
        self.assertEqual(rows[0]["families"]["price_technical"]["Close"], 100.0)
        self.assertEqual(rows[0]["families"]["valuation_quality"]["EV To EBITDA"], 8.2)


class DomainTradeTests(unittest.TestCase):
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


class ResearchWorkflowTests(TestCase):
    def _price_frame(self, periods: int = 20) -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-02", periods=periods)
        close = pd.Series(
            [100, 112, 96, 118, 92, 121, 89, 124, 94, 128, 90, 130, 95, 133, 91, 135, 97, 138, 93, 140],
            index=dates,
        )
        frame = pd.DataFrame(index=dates)
        frame["open"] = close - 1.0
        frame["high"] = close + 2.0
        frame["low"] = close - 2.0
        frame["close"] = close
        frame["volume"] = 1_000_000
        frame["adj_open"] = frame["open"]
        frame["adj_high"] = frame["high"]
        frame["adj_low"] = frame["low"]
        frame["adj_close"] = frame["close"]
        frame.index.name = "date"
        return frame

    def _artifact_from_frame(self, *, run: PipelineRun, artifact_type: str, key: str, frame: pd.DataFrame) -> Artifact:
        stored = write_frame_artifact(key, frame=frame, fieldnames=list(frame.columns))
        return Artifact.objects.create(
            pipeline_run=run,
            artifact_type=artifact_type,
            key=key,
            uri=stored.uri,
            content={"rows": int(len(frame))},
            metadata=stored.storage_metadata(),
        )

    def _feature_spec(self) -> FeatureBuildSpec:
        return FeatureBuildSpec.from_mapping(
            {
                "include_price_technicals": True,
                "include_fundamental_change": False,
                "include_statement_quality": False,
                "include_event_features": False,
                "include_ownership_features": False,
                "include_economic_indicators": False,
                "include_treasury_rates": False,
            }
        )

    def _label_spec(self) -> LabelBuildSpec:
        return LabelBuildSpec(
            k_params={"W": [1]},
            min_profit_pct=0.0,
            buy_execution="adj_high",
            sell_execution="adj_low",
            short_execution="adj_low",
            cover_execution="adj_high",
            trade_dedup_mode="entry_date",
        )

    def test_model_training_and_scoring_from_artifacts(self):
        Symbol.objects.create(symbol="AAPL", company_name="Apple Inc.")
        feature_run = PipelineRun.objects.create(name="features", requested_job="features")
        label_run = PipelineRun.objects.create(name="labels", requested_job="labels")
        feature_frame = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAPL", "close": 100.0, "ret_1": 0.10},
                {"date": "2024-01-02", "symbol": "AAPL", "close": 101.0, "ret_1": -0.20},
                {"date": "2024-01-03", "symbol": "AAPL", "close": 102.0, "ret_1": 0.05},
                {"date": "2024-01-04", "symbol": "AAPL", "close": 101.5, "ret_1": -0.03},
            ]
        )
        label_frame = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAPL", "label": 1, "market_position": 1, "trade_return": 0.10, "hold_days": 1, "side": "long", "freq": "W", "k": 1},
                {"date": "2024-01-02", "symbol": "AAPL", "label": 0, "market_position": 0, "trade_return": -0.20, "hold_days": 1, "side": "short", "freq": "W", "k": 1},
                {"date": "2024-01-03", "symbol": "AAPL", "label": 1, "market_position": 1, "trade_return": 0.05, "hold_days": 1, "side": "long", "freq": "W", "k": 1},
                {"date": "2024-01-04", "symbol": "AAPL", "label": 0, "market_position": 0, "trade_return": -0.03, "hold_days": 1, "side": "short", "freq": "W", "k": 1},
            ]
        )
        feature_artifact = self._artifact_from_frame(run=feature_run, artifact_type="FEATURES", key="core_features", frame=feature_frame)
        label_artifact = self._artifact_from_frame(run=label_run, artifact_type="LABELS", key="core_labels", frame=label_frame)
        saved_model = train_model_workflow(
            spec=build_model_training_spec({"params": {"n_estimators": 5}, "split_ratio": 0.5}),
            feature_artifact=feature_artifact,
            label_artifact=label_artifact,
        )
        prediction_df, score_meta = score_model_workflow(
            spec=build_model_scoring_spec({}, saved_model_id=int(saved_model.id)),
            feature_artifact=feature_artifact,
        )
        self.assertEqual(len(prediction_df), 4)
        self.assertIn("prediction_score", prediction_df.columns)
        self.assertGreaterEqual(int(score_meta["rows_scored"]), 4)

    def test_end_to_end_smoke_feature_label_train_score(self):
        Symbol.objects.create(symbol="AAPL", company_name="Apple Inc.")
        price_frames = {"AAPL": self._price_frame()}
        with patch("workflows.features.load_adjusted_price_frames", return_value=price_frames):
            feature_frame, _, feature_meta = build_feature_panel_frame_for_symbols(symbols=["AAPL"], spec=self._feature_spec())
        label_result = build_oracle_labels(["AAPL"], spec=self._label_spec(), price_frames=price_frames)
        self.assertGreater(len(feature_frame), 0)
        self.assertGreater(len(label_result.label_rows), 0)
        self.assertIn("prices_div_adj", feature_meta["feature_family_columns"])

        feature_run = PipelineRun.objects.create(name="smoke-features", requested_job="features")
        label_run = PipelineRun.objects.create(name="smoke-labels", requested_job="labels")
        feature_artifact = self._artifact_from_frame(run=feature_run, artifact_type="FEATURES", key="smoke_features", frame=feature_frame)
        label_artifact = self._artifact_from_frame(run=label_run, artifact_type="LABELS", key="smoke_labels", frame=pd.DataFrame(label_result.label_rows))
        saved_model = train_model_workflow(
            spec=build_model_training_spec(
                {
                    "algorithm": "random_forest_regressor",
                    "task_type": "regression",
                    "target_col": "trade_return",
                    "params": {"n_estimators": 10},
                    "split_ratio": 0.5,
                }
            ),
            feature_artifact=feature_artifact,
            label_artifact=label_artifact,
        )
        prediction_df, _ = score_model_workflow(
            spec=build_model_scoring_spec({}, saved_model_id=int(saved_model.id)),
            feature_artifact=feature_artifact,
        )
        self.assertGreater(len(prediction_df), 0)
        self.assertTrue(Path(saved_model.metadata["predictions_uri"]).exists())
