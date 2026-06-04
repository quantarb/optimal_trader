from __future__ import annotations

import os

import django
import numpy as np
import pandas as pd
from django.apps import apps

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")
if not apps.ready:
    django.setup()

from domain.labels.directional import add_binary_classification_labels
from domain.models.specs import FitSpec
from ml.frameworks.sklearn import SklearnRFClassifier


def test_optimal_directional_labels_use_target():
    events = pd.DataFrame(
        {
            "event": ["entry", "entry", "exit", "exit"],
            "side": ["long", "short", "long", "short"],
            "horizon": ["W_k1", "W_k1", "W_k1", "W_k1"],
            "trade_return": [0.12, 0.08, 0.12, 0.08],
        },
        index=pd.date_range("2024-01-01", periods=4),
    )

    labels = add_binary_classification_labels(events, use_sample_weight=True)

    assert "target" in labels.columns
    assert "side" in labels.columns
    assert set(labels["side"]) == {"long", "short"}
    assert "sample_weight" in labels.columns


def test_target_classifier_treats_one_as_positive_class():
    df = pd.DataFrame(
        {
            "feature": np.arange(8, dtype=float),
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    clf = SklearnRFClassifier(random_state=1, n_estimators=20, max_depth=3)

    clf.fit(
        df,
        FitSpec(feature_cols=["feature"], target_col="target", split_ratio=1.0),
        verbose=False,
    )

    positive_idx = clf.positive_class_index()
    class_name = clf._class_mapping[clf._classes[positive_idx]]
    assert class_name == "1"


def test_target_classifier_uses_external_validation_frame():
    train_df = pd.DataFrame(
        {
            "feature": np.arange(8, dtype=float),
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    validation_df = pd.DataFrame(
        {
            "feature": np.arange(8, 12, dtype=float),
            "target": [0, 0, 1, 1],
        }
    )
    clf = SklearnRFClassifier(random_state=1, n_estimators=20, max_depth=3)

    clf.fit(
        train_df,
        FitSpec(feature_cols=["feature"], target_col="target", split_ratio=1.0),
        verbose=False,
        validation_df=validation_df,
    )

    assert clf._train_stats["eval_mode"] == "external_validation"
    assert clf._train_stats["n_test"] == len(validation_df)
