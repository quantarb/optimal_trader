from __future__ import annotations

from .base import EndpointDefinition


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="grades",
        title="Grades",
        kind="snapshot",
        threshold_days=7,
        max_rows=50,
        candidates=[("/stable/grades", {"symbol": symbol_obj.symbol})],
    )
