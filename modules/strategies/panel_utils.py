from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from modules.utils.panel import panel_dates_symbols


def dates_symbols(panel: pd.DataFrame) -> tuple[pd.DatetimeIndex, List[str]]:
    return panel_dates_symbols(panel)


def col_to_matrix(panel: pd.DataFrame, col: str, dates: pd.DatetimeIndex, symbols: List[str]) -> np.ndarray:
    if col not in panel.columns:
        raise ValueError(f"panel missing required column '{col}'")
    m = panel[col].unstack("symbol").reindex(index=dates, columns=symbols)
    arr = m.to_numpy(dtype=np.float32, copy=False)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def empty_weights(dates: pd.DatetimeIndex, symbols: List[str]) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=dates, columns=symbols)
