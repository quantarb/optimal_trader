from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

RebalanceFreq = Literal["D", "W", "M"]  # daily, weekly, monthly


def apply_rebalance_schedule(
    weights_daily: pd.DataFrame,
    freq: RebalanceFreq = "D",
    anchor: Literal["period_start", "period_end"] = "period_start",
) -> pd.DataFrame:
    """Carry-forward weights between rebalance dates.

    weights_daily:
      index: DatetimeIndex trading days
      columns: symbols
    """
    if weights_daily.empty:
        return weights_daily

    w = weights_daily.copy()
    w.index = pd.DatetimeIndex(w.index).sort_values()

    if freq == "D":
        return w

    idx = w.index

    if freq == "W":
        period = idx.to_period("W")
    elif freq == "M":
        period = idx.to_period("M")
    else:
        raise ValueError("freq must be 'D', 'W', or 'M'")

    if anchor == "period_start":
        rebalance_dates = idx.to_series().groupby(period).min().values
    elif anchor == "period_end":
        rebalance_dates = idx.to_series().groupby(period).max().values
    else:
        raise ValueError("anchor must be 'period_start' or 'period_end'")

    rebalance_dates = pd.DatetimeIndex(rebalance_dates)
    mask = idx.isin(rebalance_dates)

    out = w.copy()
    out.loc[~mask, :] = np.nan
    out = out.ffill().fillna(0.0)
    return out
