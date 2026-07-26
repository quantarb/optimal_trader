"""Point-in-time alignment helpers for issuer, event, macro, and instruments."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def assert_point_in_time(
    rows: pd.DataFrame,
    prediction_column: str = "prediction_date",
    available_column: str = "available_date",
) -> None:
    """Fail if any row is unavailable at its prediction timestamp."""
    missing = {prediction_column, available_column}.difference(rows.columns)
    if missing:
        raise KeyError(f"point-in-time rows missing columns: {sorted(missing)}")
    prediction = pd.to_datetime(rows[prediction_column], errors="raise")
    available = pd.to_datetime(rows[available_column], errors="raise")
    bad = available > prediction
    if bad.any():
        sample = rows.loc[bad, [prediction_column, available_column]].head(5).to_dict("records")
        raise ValueError(f"point-in-time leakage detected in {int(bad.sum())} rows: {sample}")


def align_available_rows(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    entity_columns: Sequence[str],
    prediction_column: str = "prediction_date",
    available_column: str = "available_date",
    value_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """As-of align observations to predictions without using future data."""
    required_prediction = {*entity_columns, prediction_column}
    required_observation = {*entity_columns, available_column}
    if missing := required_prediction.difference(predictions.columns):
        raise KeyError(f"predictions missing columns: {sorted(missing)}")
    if missing := required_observation.difference(observations.columns):
        raise KeyError(f"observations missing columns: {sorted(missing)}")
    left = predictions.copy()
    right = observations.copy()
    left[prediction_column] = pd.to_datetime(left[prediction_column], errors="raise")
    right[available_column] = pd.to_datetime(right[available_column], errors="raise")
    right = right.sort_values([*entity_columns, available_column])
    assert_point_in_time(
        pd.DataFrame({prediction_column: left[prediction_column], available_column: left[prediction_column]})
    )
    columns = list(value_columns or [column for column in right.columns if column not in {*entity_columns, available_column}])
    right = right[[*entity_columns, available_column, *columns]]
    chunks = []
    for keys, group in left.groupby(list(entity_columns), sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        candidates = right
        for column, value in zip(entity_columns, keys):
            candidates = candidates.loc[candidates[column].eq(value)]
        merged = pd.merge_asof(
            group.sort_values(prediction_column),
            candidates.sort_values(available_column),
            left_on=prediction_column,
            right_on=available_column,
            direction="backward",
            allow_exact_matches=True,
        )
        chunks.append(merged)
    return pd.concat(chunks, ignore_index=True) if chunks else left.copy()


def build_asof_memory(
    observations: pd.DataFrame,
    prediction_dates: Sequence[pd.Timestamp],
    *,
    entity_column: str = "issuer",
    available_column: str = "available_date",
    max_rows: int,
) -> list[pd.DataFrame]:
    """Build one strictly as-of, most-recent memory window per prediction date."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if entity_column not in observations or available_column not in observations:
        raise KeyError("observations require entity and available-date columns")
    rows = observations.copy()
    rows[available_column] = pd.to_datetime(rows[available_column], errors="raise")
    result = []
    for prediction_date in pd.to_datetime(list(prediction_dates), errors="raise"):
        eligible = rows.loc[rows[available_column].le(prediction_date)].sort_values(available_column)
        result.append(eligible.tail(max_rows).reset_index(drop=True))
    return result

