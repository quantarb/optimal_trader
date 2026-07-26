"""Point-in-time-safe cross-sectional rank-label utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import torch


def percentile_rank_within_group(
    values: torch.Tensor,
    group_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    descending: bool = True,
    min_group_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic percentile ranks and a validity mask.

    Ties receive their average rank. Invalid rows never participate in a group.
    A singleton group is masked instead of receiving a fabricated percentile.
    """
    if values.ndim != 1 or group_ids.ndim != 1 or valid_mask.ndim != 1:
        raise ValueError("values, group_ids, and valid_mask must be one-dimensional")
    if not (values.numel() == group_ids.numel() == valid_mask.numel()):
        raise ValueError("ranking inputs must have equal length")
    ranks = torch.zeros_like(values, dtype=torch.float32)
    output_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    valid_indices = torch.nonzero(valid_mask.bool(), as_tuple=False).flatten().tolist()
    groups: dict[object, list[int]] = {}
    for index in valid_indices:
        key = group_ids[index].item()
        groups.setdefault(key, []).append(index)
    for indices in groups.values():
        if len(indices) < min_group_size:
            continue
        ordered = sorted(
            indices,
            key=lambda index: (float(values[index].detach().cpu()), -index if descending else index),
            reverse=descending,
        )
        # Average positions for ties, then scale to [0, 1].
        position = 0
        while position < len(ordered):
            end = position + 1
            value = float(values[ordered[position]].detach().cpu())
            while end < len(ordered) and float(values[ordered[end]].detach().cpu()) == value:
                end += 1
            average_position = (position + end - 1) / 2
            percentile = (
                (len(ordered) - 1 - average_position) / (len(ordered) - 1)
                if descending
                else average_position / (len(ordered) - 1)
            )
            for ordered_index in ordered[position:end]:
                ranks[ordered_index] = percentile
                output_mask[ordered_index] = True
            position = end
    return ranks, output_mask


def _pandas_percentile_rank(
    frame: pd.DataFrame,
    value_column: str,
    group_columns: Sequence[str],
    output_column: str,
    valid_column: str,
    descending: bool,
    min_group_size: int,
) -> None:
    values = pd.to_numeric(frame[value_column], errors="coerce")
    eligible = values.notna()
    sizes = frame.loc[eligible].groupby(list(group_columns), dropna=False)[value_column].transform("size")
    eligible &= sizes.reindex(frame.index, fill_value=0).ge(min_group_size)
    rank = pd.Series(np.nan, index=frame.index, dtype="float32")
    if eligible.any():
        method = "average"
        grouped = values[eligible].groupby(
            [frame.loc[eligible, column] for column in group_columns], dropna=False
        )
        raw_rank = grouped.rank(method=method, ascending=descending, pct=False)
        group_size = grouped.transform("size")
        normalized = (raw_rank - 1.0) / (group_size - 1.0).clip(lower=1.0)
        rank.loc[eligible] = normalized.astype("float32")
    frame[output_column] = rank
    frame[valid_column] = rank.notna()


def add_hits_rank_targets(
    frame: pd.DataFrame,
    hit_columns: Iterable[str],
    *,
    date_column: str = "date",
    issuer_column: str = "issuer",
    asset_class_column: str = "asset_class",
    min_issuer_group_size: int = 2,
    min_asset_class_group_size: int = 5,
    descending: bool = True,
) -> pd.DataFrame:
    """Add issuer and asset-class percentile targets for every HITS column.

    The input frame must already contain only rows eligible on each prediction
    date. No future rows are consulted; labels are generated from the supplied
    realized HITS values only.
    """
    required = {date_column, issuer_column, asset_class_column, *hit_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"rank-label frame missing columns: {sorted(missing)}")
    out = frame.copy()
    out[date_column] = pd.to_datetime(out[date_column], errors="raise").dt.normalize()
    for hit_column in hit_columns:
        issuer_name = f"same_issuer_{hit_column}_rank"
        issuer_valid = f"{issuer_name}_valid"
        asset_name = f"same_asset_class_{hit_column}_rank"
        asset_valid = f"{asset_name}_valid"
        _pandas_percentile_rank(
            out,
            hit_column,
            [date_column, issuer_column],
            issuer_name,
            issuer_valid,
            descending,
            min_issuer_group_size,
        )
        _pandas_percentile_rank(
            out,
            hit_column,
            [date_column, asset_class_column],
            asset_name,
            asset_valid,
            descending,
            min_asset_class_group_size,
        )
    return out
