from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import pandas as pd

from labels.trades import labels_panel_to_trades_df
from ml.frameworks.transformers.seq2seq import prepare_entry2exit_dataset as _prepare_entry2exit_dataset


@dataclass(frozen=True)
class MLDatasetConfig:
    drop_nan_features: bool = True


@dataclass(frozen=True)
class Entry2ExitTextConfig:
    numeric_precision: int = 2
    scientific_for_large_numbers: bool = True
    scientific_threshold: float = 1_000_000.0
    dedupe_source_duplicate_features: bool = True
    compact_feature_names: bool = False
    drop_missing_entry_rows: bool = True
    target_fields: Tuple[str, ...] = ("entry_px", "exit_px", "trade_return", "trade_duration_days")


def prepare_ml_dataset(
    *,
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    target_cols: Union[str, List[str]] = "target",
    weight_col: Optional[str] = "sample_weight",
    config: Optional[MLDatasetConfig] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Join dense features with sparse labels into a train-ready tabular dataset.
    """
    cfg = config or MLDatasetConfig()

    if verbose:
        print("--- Preparing ML Training Dataset ---")

    dataset = features_df.join(labels_df, how="inner")
    targets = [target_cols] if isinstance(target_cols, str) else list(target_cols)
    all_features = [c for c in features_df.columns if c in dataset.columns]
    valid_features = [c for c in all_features if dataset[c].notna().any()]

    if cfg.drop_nan_features and len(valid_features) > 0:
        initial_rows = len(dataset)
        dataset = dataset.dropna(subset=valid_features)
        if verbose:
            dropped = initial_rows - len(dataset)
            print(f"  - Dropped {dropped:,} rows due to NaN features.")

    if verbose:
        print(f"  - Final Training Rows: {len(dataset):,}")
        print(f"  - Active Features:     {len(valid_features)}")
        print(f"  - Targets:             {targets}")
        if weight_col:
            print(f"  - Sample Weight Col:   {weight_col}")

    return dataset, valid_features, targets


def prepare_entry2exit_dataset(
    *,
    features_df: pd.DataFrame,
    labels_df: Optional[pd.DataFrame] = None,
    trades_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[Sequence[str]] = None,
    config: Optional[Entry2ExitTextConfig] = None,
) -> pd.DataFrame:
    """
    Build canonical Entry->Exit text pairs from features + optimal trades.
    """
    if (labels_df is None) == (trades_df is None):
        raise ValueError("Provide exactly one of labels_df or trades_df.")

    cfg = config or Entry2ExitTextConfig()
    resolved_trades = trades_df if trades_df is not None else labels_panel_to_trades_df(labels_df)

    return _prepare_entry2exit_dataset(
        final_df=features_df,
        trades_df=resolved_trades,
        feature_cols=feature_cols,
        numeric_precision=int(cfg.numeric_precision),
        scientific_for_large_numbers=bool(cfg.scientific_for_large_numbers),
        scientific_threshold=float(cfg.scientific_threshold),
        dedupe_source_duplicate_features=bool(cfg.dedupe_source_duplicate_features),
        compact_feature_names=bool(cfg.compact_feature_names),
        target_fields=tuple(cfg.target_fields),
        drop_missing_entry_rows=bool(cfg.drop_missing_entry_rows),
    )
