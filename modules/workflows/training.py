from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from modules.models.base import FitSpec
from modules.models.sklearn.classifier import SklearnRFClassifier
from modules.models.sklearn.regressor import SklearnRFRegressor
from modules.models.torch.autoencoder.config import AutoEncoderConfig
from modules.models.torch.autoencoder.adapter import TorchAutoEncoder


@dataclass(frozen=True)
class RawRFModels:
    """Bundle for raw-stack RF models with backward-compatible unpacking."""

    clf: Any
    ranking_reg: Any
    trade_return_reg: Any | None = None
    duration_reg: Any | None = None

    def __iter__(self):
        # Backward compatibility for: clf_raw, reg_raw = train_rf_models(...)
        yield self.clf
        yield self.ranking_reg


def train_rf_models(
    train_df: pd.DataFrame,
    feature_list: Sequence[str],
    *,
    split_ratio: float = 0.8,
    classifier_target_col: str = "label",
    ranking_target_col: str = "rank_y",
    classifier_market_position_col: str = "market_position",
    train_trade_return_model: bool = True,
    trade_return_target_col: str = "trade_return",
    train_duration_model: bool = True,
    duration_target_col: str = "trade_duration_days",
) -> RawRFModels:
    """Train RF classifier + ranking regressor, with optional return/duration regressors."""
    if classifier_target_col not in train_df.columns:
        raise KeyError(
            f"Missing classifier target column '{classifier_target_col}'. "
            f"Available targets include: {[c for c in ['target', 'label', 'rank_y', 'trade_return', 'trade_duration_days'] if c in train_df.columns]}"
        )
    if ranking_target_col not in train_df.columns:
        raise KeyError(f"Missing ranking target column '{ranking_target_col}'.")

    clf_feature_cols = list(feature_list)
    if classifier_market_position_col:
        if classifier_market_position_col not in train_df.columns:
            raise KeyError(f"Missing classifier market-position column '{classifier_market_position_col}'.")
        clf_feature_cols = clf_feature_cols + [classifier_market_position_col]

    spec_clf = FitSpec(
        feature_cols=clf_feature_cols,
        target_col=classifier_target_col,
        weight_col="sample_weight",
        split_ratio=float(split_ratio),
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
        target_col=ranking_target_col,
        weight_col="sample_weight",
        split_ratio=float(split_ratio),
    )
    reg = SklearnRFRegressor(
        test_size=max(0.0, 1.0 - float(split_ratio)),
        random_state=1337,
        n_estimators=200,
        max_depth=12,
        max_features="sqrt",
        n_jobs=-1,
    )
    reg.fit(train_df, spec_reg, verbose=True)

    trade_return_reg = None
    if train_trade_return_model and trade_return_target_col in train_df.columns:
        return_df = train_df.copy()
        return_df[trade_return_target_col] = pd.to_numeric(return_df[trade_return_target_col], errors="coerce")
        return_df = return_df.dropna(subset=[trade_return_target_col])
        if not return_df.empty:
            spec_ret = FitSpec(
                feature_cols=list(feature_list),
                target_col=trade_return_target_col,
                weight_col="sample_weight",
                split_ratio=float(split_ratio),
            )
            trade_return_reg = SklearnRFRegressor(
                test_size=max(0.0, 1.0 - float(split_ratio)),
                random_state=1337,
                n_estimators=200,
                max_depth=12,
                max_features="sqrt",
                n_jobs=-1,
            )
            trade_return_reg.fit(return_df, spec_ret, verbose=True)

    duration_reg = None
    if train_duration_model and duration_target_col in train_df.columns:
        duration_df = train_df.copy()
        duration_df[duration_target_col] = pd.to_numeric(duration_df[duration_target_col], errors="coerce")
        duration_df = duration_df.dropna(subset=[duration_target_col])
        if not duration_df.empty:
            spec_dur = FitSpec(
                feature_cols=list(feature_list),
                target_col=duration_target_col,
                weight_col="sample_weight",
                split_ratio=float(split_ratio),
            )
            duration_reg = SklearnRFRegressor(
                test_size=max(0.0, 1.0 - float(split_ratio)),
                random_state=1337,
                n_estimators=200,
                max_depth=12,
                max_features="sqrt",
                n_jobs=-1,
            )
            duration_reg.fit(duration_df, spec_dur, verbose=True)

    return RawRFModels(
        clf=clf,
        ranking_reg=reg,
        trade_return_reg=trade_return_reg,
        duration_reg=duration_reg,
    )


def train_ae(train_df: pd.DataFrame, feature_list: Sequence[str], *, verbose: bool = True):
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
        verbose=verbose,
    )
    return ae, numeric_cols


def save_raw_stack_artifacts(
    *,
    clf_raw: Any,
    reg_trade_return_raw: Any | None = None,
    reg_raw: Any | None = None,
    reg_duration_raw: Any | None = None,
    ae_raw: Any,
    raw_feature_list: Sequence[str],
    ae_raw_numeric_cols: Sequence[str],
    artifact_dir: str = "./artifacts/raw_stack",
    flavor_space: Any | None = None,
) -> Path:
    """Persist raw-stack artifacts for separate inference notebook/workflow."""
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_reg = reg_trade_return_raw if reg_trade_return_raw is not None else reg_raw
    if primary_reg is None:
        raise ValueError("save_raw_stack_artifacts requires a trade-return regressor.")

    with open(out_dir / "clf_raw.pkl", "wb") as f:
        pickle.dump(clf_raw, f)
    with open(out_dir / "reg_raw.pkl", "wb") as f:
        pickle.dump(primary_reg, f)
    with open(out_dir / "reg_trade_return_raw.pkl", "wb") as f:
        pickle.dump(primary_reg, f)
    if reg_duration_raw is not None:
        with open(out_dir / "reg_duration_raw.pkl", "wb") as f:
            pickle.dump(reg_duration_raw, f)
    with open(out_dir / "ae_raw.pkl", "wb") as f:
        pickle.dump(ae_raw, f)

    meta = {
        "artifact_version": 1,
        "stack": "raw",
        "feature_list": list(raw_feature_list),
        "ae_numeric_cols": list(ae_raw_numeric_cols),
        "has_trade_return_regressor": bool(primary_reg is not None),
        "has_duration_regressor": bool(reg_duration_raw is not None),
        "has_flavor_space": False,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return out_dir


def save_raw_stack_artifacts_to_db(
    *,
    clf_raw: Any,
    reg_trade_return_raw: Any | None = None,
    reg_raw: Any | None = None,
    reg_duration_raw: Any | None = None,
    ae_raw: Any,
    raw_feature_list: Sequence[str],
    ae_raw_numeric_cols: Sequence[str],
    model_prefix: str = "raw_stack",
) -> dict[str, Any]:
    """Persist raw-stack artifacts into the Django ModelArtifact table."""
    from ml.store import save_model_artifact

    primary_reg = reg_trade_return_raw if reg_trade_return_raw is not None else reg_raw
    if primary_reg is None:
        raise ValueError("save_raw_stack_artifacts_to_db requires a trade-return regressor.")

    shared_metadata = {
        "stack": "raw",
        "feature_list": list(raw_feature_list),
        "ae_numeric_cols": list(ae_raw_numeric_cols),
    }

    saved = {
        "classifier": save_model_artifact(
            name=f"{model_prefix}_classifier",
            model_obj=clf_raw,
            framework="sklearn",
            task_type="classification",
            target_col="label",
            feature_cols=raw_feature_list,
            metadata=shared_metadata,
        ),
        "trade_return_regressor": save_model_artifact(
            name=f"{model_prefix}_trade_return_regressor",
            model_obj=primary_reg,
            framework="sklearn",
            task_type="regression",
            target_col="trade_return",
            feature_cols=raw_feature_list,
            metadata=shared_metadata,
        ),
        "autoencoder": save_model_artifact(
            name=f"{model_prefix}_autoencoder",
            model_obj=ae_raw,
            framework="torch",
            task_type="embedding",
            target_col="",
            feature_cols=ae_raw_numeric_cols,
            metadata=shared_metadata,
        ),
    }

    if reg_duration_raw is not None:
        saved["duration_regressor"] = save_model_artifact(
            name=f"{model_prefix}_duration_regressor",
            model_obj=reg_duration_raw,
            framework="sklearn",
            task_type="regression",
            target_col="trade_duration_days",
            feature_cols=raw_feature_list,
            metadata=shared_metadata,
        )

    return saved
