"""Programmatic heads derived from the resolved task inventory."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from torch import nn


def build_task_heads(
    tasks: Iterable[Mapping[str, object]],
    d_model: int,
    *,
    hidden: int | None = None,
) -> nn.ModuleDict:
    """Create heads from task metadata; no fixed HITS-name list is required."""
    hidden = hidden or d_model
    heads = nn.ModuleDict()
    for task in tasks:
        name = str(task["task_name"])
        if name in heads:
            continue
        task_type = str(task.get("task_type", "regression"))
        output_dim = max(1, int(task.get("output_dim", 1)))
        heads[name] = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
        )
    return heads


def expand_hits_task_inventory(tasks: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Add raw, same-issuer, and same-asset-class tasks for active HITS targets."""
    expanded: list[dict[str, object]] = []
    for task in tasks:
        row = dict(task)
        expanded.append(row)
        if str(row.get("task_type")) != "regression":
            continue
        name = str(row["task_name"])
        if not any(token in name.lower() for token in ("hub", "authority", "pagerank", "speed")):
            continue
        for prefix, level, weight in (
            ("same_issuer_", "same_issuer", 0.5),
            ("same_asset_class_", "same_asset_class", 0.75),
        ):
            derived = dict(row)
            derived.update(task_name=f"{prefix}{name}_rank", task_level=level, weight=weight, loss="percentile_huber_plus_pairwise")
            expanded.append(derived)
    return expanded

