from __future__ import annotations

from .base import EndpointDefinition


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="revenue_geographic_segmentation",
        title="Revenue Geographic Segmentation",
        kind="historical",
        threshold_days=90,
        min_history_years=10,
        max_rows=30,
        candidates=[("/stable/revenue-geographic-segmentation", {"symbol": symbol_obj.symbol})],
    )
