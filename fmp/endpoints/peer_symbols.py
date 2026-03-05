from __future__ import annotations

from .base import EndpointDefinition


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="peer_symbols",
        title="Peer Symbols",
        kind="snapshot",
        threshold_days=30,
        max_rows=100,
        candidates=[("/stable/stock-peers", {"symbol": symbol_obj.symbol})],
    )
