from __future__ import annotations

from .base import EndpointDefinition


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="splits",
        title="Splits",
        kind="historical",
        threshold_days=30,
        min_history_years=15,
        max_rows=50,
        candidates=[("/stable/splits", {"symbol": symbol_obj.symbol})],
    )
