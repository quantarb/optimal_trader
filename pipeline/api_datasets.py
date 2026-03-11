from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from data import (
    Entry2ExitTextConfig,
    MLDatasetConfig,
    build_dataset,
    prepare_entry2exit_dataset as _prepare_entry2exit_dataset_unified,
    prepare_ml_dataset as _prepare_ml_dataset_unified,
)
from pipeline.contracts_runtime import DatasetArtifacts
from pipeline.window import TimeWindow
from utils.panel import ensure_panel_index


def build_dataset_artifacts(
    *,
    ctx,
    symbols: Sequence[str],
    train_window: TimeWindow,
    infer_window: TimeWindow,
    k_params: Dict[str, int],
    execution_params: Dict[str, Any],
    weighting: Dict[str, Any],
    add_rank_labels: bool = True,
    add_rank_tasks_to_mtl: bool = True,
    debug_data_quality: bool = False,
    data_quality_overrides: Optional[Dict[str, Any]] = None,
    skip_on_error: bool = True,
    verbose_data: bool = True,
) -> DatasetArtifacts:
    out = build_dataset(
        ctx=ctx,
        symbols=list(symbols),
        train_start=str(train_window.start.date()),
        train_end=str(train_window.end.date()),
        infer_start=str(infer_window.start.date()),
        infer_end=str(infer_window.end.date()),
        k_params=dict(k_params),
        execution_params=dict(execution_params),
        weighting=dict(weighting),
        add_rank_labels=bool(add_rank_labels),
        add_rank_tasks_to_mtl=bool(add_rank_tasks_to_mtl),
        debug_data_quality=bool(debug_data_quality),
        data_quality_overrides=dict(data_quality_overrides) if data_quality_overrides else None,
        skip_on_error=bool(skip_on_error),
        verbose_data=bool(verbose_data),
    )
    return DatasetArtifacts(
        daily_by_symbol=out["daily_by_symbol"],
        training_df=out["training_df"],
        inference_panel=ensure_panel_index(out["inference_panel"]),
        feature_cols=out["feature_cols"],
        meta=out.get("meta", {}) or {},
    )


def prepare_ml_dataset(
    *,
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    target_cols: Union[str, List[str]] = "target",
    weight_col: Optional[str] = "sample_weight",
    drop_nan_features: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    return _prepare_ml_dataset_unified(
        features_df=features_df,
        labels_df=labels_df,
        target_cols=target_cols,
        weight_col=weight_col,
        config=MLDatasetConfig(drop_nan_features=bool(drop_nan_features)),
        verbose=verbose,
    )


def prepare_entry2exit_dataset(
    *,
    features_df: pd.DataFrame,
    labels_df: Optional[pd.DataFrame] = None,
    trades_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[Sequence[str]] = None,
    numeric_precision: int = 2,
    scientific_for_large_numbers: bool = True,
    scientific_threshold: float = 1_000_000.0,
    dedupe_source_duplicate_features: bool = True,
    compact_feature_names: bool = False,
    drop_missing_entry_rows: bool = True,
) -> pd.DataFrame:
    return _prepare_entry2exit_dataset_unified(
        features_df=features_df,
        labels_df=labels_df,
        trades_df=trades_df,
        feature_cols=feature_cols,
        config=Entry2ExitTextConfig(
            numeric_precision=int(numeric_precision),
            scientific_for_large_numbers=bool(scientific_for_large_numbers),
            scientific_threshold=float(scientific_threshold),
            dedupe_source_duplicate_features=bool(dedupe_source_duplicate_features),
            compact_feature_names=bool(compact_feature_names),
            drop_missing_entry_rows=bool(drop_missing_entry_rows),
        ),
    )


__all__ = [
    "build_dataset_artifacts",
    "prepare_entry2exit_dataset",
    "prepare_ml_dataset",
]
