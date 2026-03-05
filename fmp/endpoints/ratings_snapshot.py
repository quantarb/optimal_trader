from __future__ import annotations

from .base import EndpointDefinition


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="ratings_snapshot",
        title="Ratings Snapshot",
        kind="snapshot",
        threshold_days=7,
        max_rows=1,
        candidates=[("/stable/ratings-snapshot", {"symbol": symbol_obj.symbol})],
    )
