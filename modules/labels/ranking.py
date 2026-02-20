from __future__ import annotations

import numpy as np
import pandas as pd

from modules.utils.normalize import normalize_cols
from modules.schema import require_columns


def add_rank_regression_labels(training_df: pd.DataFrame) -> pd.DataFrame:
    """Adds global rank labels using raw trade_return.

    rank_y = percentile_rank(trade_return) over all rows in [0, 1]
    """
    if training_df is None or len(training_df) == 0:
        return training_df.copy()

    df = normalize_cols(training_df).copy()
    require_columns(df, ["trade_return"], ctx="add_rank_regression_labels")

    ret = pd.to_numeric(df["trade_return"], errors="coerce")

    df["rank_y"] = np.nan
    # Keep side_profit for compatibility with existing notebooks/reports.
    if "target" in df.columns:
        tgt = pd.to_numeric(df["target"], errors="coerce")
        df["side_profit"] = ret.where(tgt == 1, -ret).astype(float)
    else:
        df["side_profit"] = ret.astype(float)

    valid = ret.notna()
    if not bool(valid.any()):
        return df

    work = pd.DataFrame(
        {
            "__ret": ret.astype(float),
            "__valid": valid.astype(bool),
        },
        index=df.index,
    )
    work_valid = work.loc[work["__valid"]]
    ranked = work_valid["__ret"].rank(method="average", pct=True).astype(float)
    rank_y = np.full(len(df), np.nan, dtype=float)
    valid_pos = np.flatnonzero(valid.to_numpy())
    rank_y[valid_pos] = ranked.to_numpy(dtype=float)
    df["rank_y"] = rank_y
    return df
