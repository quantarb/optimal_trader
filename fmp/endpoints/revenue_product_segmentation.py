from __future__ import annotations

from .base import EndpointDefinition


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="revenue_product_segmentation",
        title="Revenue Product Segmentation",
        kind="historical",
        threshold_days=90,
        min_history_years=10,
        max_rows=30,
        candidates=[("/stable/revenue-product-segmentation", {"symbol": symbol_obj.symbol})],
    )
