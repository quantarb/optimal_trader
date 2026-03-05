from __future__ import annotations

from .base import EndpointDefinition


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="key_executives",
        title="Key Executives",
        kind="snapshot",
        threshold_days=90,
        max_rows=50,
        candidates=[("/stable/key-executives", {"symbol": symbol_obj.symbol})],
    )
