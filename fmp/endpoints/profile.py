from __future__ import annotations

from .base import EndpointDefinition


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="profile",
        title="Company Profile",
        kind="snapshot",
        threshold_days=30,
        max_rows=1,
        candidates=[("/stable/profile", {"symbol": symbol_obj.symbol})],
    )
