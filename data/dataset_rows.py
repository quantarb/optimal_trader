from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from utils.normalize import normalize_cols
from data.schema import require_columns


@dataclass(frozen=True)
class EventTrainingDataset:
    training_df: pd.DataFrame


def build_event_training_dataset(
    *,
    df_features: pd.DataFrame,
    labels: pd.DataFrame,
    symbol: str,
) -> EventTrainingDataset:
    """Join event-level labels onto daily features to form training rows.

    Assumptions:
      - df_features is daily (DatetimeIndex) and includes feature columns.
      - labels is indexed by the same event date index (DatetimeIndex),
        containing at least: target, side, horizon and optionally trade_return/sample_weight.
      - Both frames are already normalized to lowercase columns upstream.
    """
    if df_features is None or len(df_features) == 0 or labels is None or len(labels) == 0:
        return EventTrainingDataset(training_df=pd.DataFrame())

    f = normalize_cols(df_features).copy()
    y = normalize_cols(labels).copy()

    if not isinstance(f.index, pd.DatetimeIndex):
        raise ValueError("df_features must be indexed by DatetimeIndex")
    if not isinstance(y.index, pd.DatetimeIndex):
        raise ValueError("labels must be indexed by DatetimeIndex")

    require_columns(y, ["target", "side", "horizon"], ctx=f"{symbol}:build_event_training_dataset")

    # Avoid dup column collisions: keep only label cols from y
    label_cols = [c for c in ["target", "side", "horizon", "trade_return", "sample_weight"] if c in y.columns]
    y = y[label_cols]

    joined = f.join(y, how="inner")
    joined["symbol"] = symbol

    # drop any rows with missing target
    joined = joined[joined["target"].notna()].copy()
    return EventTrainingDataset(training_df=joined)
