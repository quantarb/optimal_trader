"""Strict dataset contracts for multi-rate windows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .alignment import align_available_rows, assert_point_in_time


@dataclass(frozen=True)
class MultiRateWindow:
    prediction_date: pd.Timestamp
    asset_class: str
    daily: pd.DataFrame
    annual: pd.DataFrame
    quarterly: pd.DataFrame


def build_multirate_window(
    *,
    prediction_date: pd.Timestamp,
    asset_class: str,
    daily: pd.DataFrame,
    annual: pd.DataFrame,
    quarterly: pd.DataFrame,
    daily_entity_columns: tuple[str, ...] = ("issuer", "instrument_id"),
    issuer_column: str = "issuer",
    available_column: str = "available_date",
    daily_date_column: str = "date",
    max_daily_rows: int = 256,
    max_annual_rows: int = 5,
    max_quarterly_rows: int = 12,
) -> MultiRateWindow:
    """Construct one as-of window and reject fiscal-date-only inputs.

    Annual and quarterly frames must carry ``available_date``. The function
    deliberately does not infer availability from fiscal period end dates.
    """
    prediction_date = pd.Timestamp(prediction_date)
    # Existing quant-warehouse feature panels use ``date`` as the preserved
    # observation/as-of date. Raw warehouse frames may expose ``filing_date``
    # or ``accepted_date``; callers can normalize those before this boundary.
    for name, frame in (("annual", annual), ("quarterly", quarterly), ("daily", daily)):
        if available_column not in frame.columns:
            if "date" in frame.columns:
                frame[available_column] = pd.to_datetime(frame["date"], errors="raise")
            else:
                raise ValueError(f"{name} data must contain {available_column} or the warehouse observation date 'date'")
        if name != "daily" and issuer_column not in frame.columns:
            raise KeyError(f"{name} data missing {issuer_column}")
    annual = annual.copy()
    quarterly = quarterly.copy()
    annual[available_column] = pd.to_datetime(annual[available_column], errors="raise")
    quarterly[available_column] = pd.to_datetime(quarterly[available_column], errors="raise")
    daily = daily.copy()
    daily[daily_date_column] = pd.to_datetime(daily[daily_date_column], errors="raise")
    daily = daily.loc[daily[daily_date_column].le(prediction_date)].sort_values(daily_date_column).tail(max_daily_rows)
    daily["prediction_date"] = prediction_date
    assert_point_in_time(pd.DataFrame({"prediction_date": [prediction_date] * len(daily), "available_date": daily[available_column]}))
    issuer = daily[issuer_column].iloc[0] if len(daily) else None
    annual = annual.loc[annual[issuer_column].eq(issuer)] if issuer is not None else annual.iloc[0:0]
    quarterly = quarterly.loc[quarterly[issuer_column].eq(issuer)] if issuer is not None else quarterly.iloc[0:0]
    for frame in (annual, quarterly):
        assert_point_in_time(pd.DataFrame({"prediction_date": [prediction_date] * len(frame), "available_date": frame[available_column]}))
    annual = annual.loc[annual[available_column].le(prediction_date)].sort_values(available_column).tail(max_annual_rows).reset_index(drop=True)
    quarterly = quarterly.loc[quarterly[available_column].le(prediction_date)].sort_values(available_column).tail(max_quarterly_rows).reset_index(drop=True)
    return MultiRateWindow(prediction_date, asset_class, daily.reset_index(drop=True), annual, quarterly)
