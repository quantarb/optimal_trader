from __future__ import annotations

import numpy as np
import pandas as pd

from utils.normalize import normalize_cols
from data.schema import require_columns


def add_rank_regression_labels(training_df: pd.DataFrame) -> pd.DataFrame:
    """Adds global rank labels using raw trade_return.

    rank_y = percentile_rank(trade_return) over all rows in [0, 1]
    """
    if training_df is None or len(training_df) == 0:
        return training_df.copy()

    df = normalize_cols(training_df).copy()
    require_columns(df, ["trade_return"], ctx="add_rank_regression_labels")

    ret = pd.to_numeric(df["trade_return"], errors="coerce")

    # Keep side_profit for compatibility with existing notebooks/reports.
    if "target" in df.columns:
        tgt = pd.to_numeric(df["target"], errors="coerce")
        df["side_profit"] = ret.where(tgt == 1, -ret).astype(float)
    else:
        df["side_profit"] = ret.astype(float)

    # Direct pandas rank — handles NaN naturally (they stay NaN in the output)
    df["rank_y"] = ret.rank(method="average", pct=True)
    return df
