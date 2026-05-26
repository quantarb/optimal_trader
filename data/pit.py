from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd


def asof_join_pit(
    *,
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str = "date",
    by: Optional[Sequence[str]] = ("symbol",),
    direction: str = "backward",
    tolerance: Optional[pd.Timedelta] = None,
    allow_exact_matches: bool = True,
) -> pd.DataFrame:
    """
    Point-in-time safe as-of join.

    Semantics:
      - for each row in left, use the most recent right row at or before `on`
        (or strict-before when allow_exact_matches=False)
      - optional grouping keys in `by` (for example symbol)
      - preserves left row cardinality
    """
    if left is None or right is None:
        raise ValueError("left and right must be non-null dataframes.")
    if left.empty:
        return left.copy()
    if right.empty:
        return left.copy()

    by_cols: list[str] = [c for c in list(by or []) if c]

    left_df = left.copy()
    right_df = right.copy()

    if on not in left_df.columns or on not in right_df.columns:
        raise ValueError(f"Both dataframes must include '{on}' column.")

    left_df[on] = pd.to_datetime(left_df[on], errors="coerce")
    right_df[on] = pd.to_datetime(right_df[on], errors="coerce")
    left_df = left_df.dropna(subset=[on])
    right_df = right_df.dropna(subset=[on])

    for col in by_cols:
        if col not in left_df.columns or col not in right_df.columns:
            raise ValueError(f"Grouping column '{col}' must exist in both dataframes.")
        left_df[col] = left_df[col].astype(str)
        right_df[col] = right_df[col].astype(str)

    sort_cols = [on] + by_cols
    left_df = left_df.sort_values(sort_cols).reset_index(drop=True)
    right_df = right_df.sort_values(sort_cols).reset_index(drop=True)

    if by_cols:
        merged = pd.merge_asof(
            left_df,
            right_df,
            on=on,
            by=by_cols,
            direction=direction,
            tolerance=tolerance,
            allow_exact_matches=allow_exact_matches,
        )
    else:
        merged = pd.merge_asof(
            left_df,
            right_df,
            on=on,
            direction=direction,
            tolerance=tolerance,
            allow_exact_matches=allow_exact_matches,
        )

    return merged


def broadcast_asof_to_target_index(
    *,
    sparse_df: pd.DataFrame,
    target_index: pd.Index,
    on: str = "date",
    by: Optional[Sequence[str]] = ("symbol",),
) -> pd.DataFrame:
    """
    Broadcast sparse time-series rows onto a target index via PIT-safe as-of join.
    """
    if sparse_df is None or sparse_df.empty:
        return pd.DataFrame(index=target_index)

    if isinstance(sparse_df.index, pd.MultiIndex):
        sparse = sparse_df.copy().reset_index()
    else:
        sparse = sparse_df.copy()
        if on not in sparse.columns and isinstance(sparse.index, pd.DatetimeIndex):
            sparse = sparse.reset_index().rename(columns={sparse.index.name or "index": on})
    dense = pd.DataFrame(index=target_index).reset_index()
    dense_index_cols = list(dense.columns)

    required = {on}
    if not required.issubset(sparse.columns):
        raise ValueError(f"sparse_df must have '{on}' as a column or index level.")
    if not required.issubset(dense.columns):
        raise ValueError(f"target_index must include '{on}'.")

    by_cols = [c for c in list(by or []) if c]
    if by_cols:
        for c in by_cols:
            if c not in sparse.columns:
                raise ValueError(f"sparse_df must include '{c}' for grouped as-of join.")
            if c not in dense.columns:
                raise ValueError(f"target_index must include '{c}' for grouped as-of join.")

    merged = asof_join_pit(
        left=dense,
        right=sparse,
        on=on,
        by=by_cols if by_cols else None,
        direction="backward",
    )

    if by_cols:
        out = merged.set_index([on] + by_cols).sort_index()
        if len(by_cols) == 1 and isinstance(target_index, pd.MultiIndex):
            out.index = out.index.set_names([on, by_cols[0]])
        return out
    if isinstance(target_index, pd.MultiIndex) and len(dense_index_cols) > 1:
        return merged.set_index(dense_index_cols).sort_index()
    return merged.set_index(on).sort_index()
