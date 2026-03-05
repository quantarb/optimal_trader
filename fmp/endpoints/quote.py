from __future__ import annotations

from .base import EndpointDefinition


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="quote",
        title="Quote Snapshot",
        kind="snapshot",
        threshold_days=1,
        max_rows=1,
        candidates=[("/stable/quote", {"symbol": symbol_obj.symbol})],
    )
