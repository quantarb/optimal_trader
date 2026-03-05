from __future__ import annotations

from .base import EndpointDefinition


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="etf_holders",
        title="ETF Holders",
        kind="snapshot",
        threshold_days=30,
        max_rows=100,
        candidates=[
            ("/stable/etf/asset-exposure", {"symbol": symbol_obj.symbol}),
            ("/stable/etf-holder", {"symbol": symbol_obj.symbol}),
        ],
    )
