from __future__ import annotations

from .base import EndpointDefinition


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="earnings",
        title="Earnings",
        kind="historical",
        threshold_days=30,
        min_history_years=15,
        max_rows=24,
        candidates=[("/stable/earnings", {"symbol": symbol_obj.symbol})],
        stability_mode="periodic",
    )
