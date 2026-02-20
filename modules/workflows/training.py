from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from modules.models.base import FitSpec
from modules.models.sklearn.classifier import SklearnRFClassifier
from modules.models.sklearn.regressor import SklearnRFRegressor
from modules.models.torch.autoencoder.config import AutoEncoderConfig
from modules.models.torch.autoencoder.adapter import TorchAutoEncoder


def train_rf_models(train_df: pd.DataFrame, feature_list: Sequence[str]):
    """Train RF classifier/regressor used by the raw stack notebook."""
    spec_clf = FitSpec(
        feature_cols=list(feature_list) + ["market_position"],
        target_col="label",
        weight_col="sample_weight",
        split_ratio=0.8,
    )
    clf = SklearnRFClassifier(
        random_state=1337,
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(train_df, spec_clf, verbose=True)

    spec_reg = FitSpec(
        feature_cols=list(feature_list),
        target_col="rank_y",
        weight_col="sample_weight",
        split_ratio=0.8,
    )
    reg = SklearnRFRegressor(
        test_size=0.2,
        random_state=1337,
        n_estimators=200,
        max_depth=12,
        max_features="sqrt",
        n_jobs=-1,
    )
    reg.fit(train_df, spec_reg, verbose=True)
    return clf, reg


def train_ae(train_df: pd.DataFrame, feature_list: Sequence[str]):
    """Train numeric-only AE used by the raw stack notebook."""
    numeric_cols = [c for c in list(feature_list) if pd.api.types.is_numeric_dtype(train_df[c])]

    spec_ae = FitSpec(
        feature_cols=numeric_cols,
        target_col="label",  # unused by AE fit
        weight_col=None,
        split_ratio=0.8,
    )

    cfg = AutoEncoderConfig(
        n_layers=3,
        min_layer_dim=2,
        denoise_std=0.03,
        latent_ref_max_points=50000,
    )

    ae = TorchAutoEncoder(cfg=cfg)
    ae.fit(
        train_df,
        spec_ae,
        numeric_cols=numeric_cols,
        categorical_cols=[],
        verbose=True,
    )
    return ae, numeric_cols


def save_raw_stack_artifacts(
    *,
    clf_raw: Any,
    reg_raw: Any,
    ae_raw: Any,
    raw_feature_list: Sequence[str],
    ae_raw_numeric_cols: Sequence[str],
    artifact_dir: str = "./artifacts/raw_stack",
    flavor_space: Any | None = None,
) -> Path:
    """Persist raw-stack artifacts for separate inference notebook/workflow."""
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "clf_raw.pkl", "wb") as f:
        pickle.dump(clf_raw, f)
    with open(out_dir / "reg_raw.pkl", "wb") as f:
        pickle.dump(reg_raw, f)
    with open(out_dir / "ae_raw.pkl", "wb") as f:
        pickle.dump(ae_raw, f)

    if flavor_space is not None:
        with open(out_dir / "flavor_space_raw.pkl", "wb") as f:
            pickle.dump(flavor_space, f)

    meta = {
        "artifact_version": 1,
        "stack": "raw",
        "feature_list": list(raw_feature_list),
        "ae_numeric_cols": list(ae_raw_numeric_cols),
        "has_flavor_space": bool(flavor_space is not None),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return out_dir
