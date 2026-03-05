from __future__ import annotations

from .base import EndpointDefinition


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="dividends",
        title="Dividends",
        kind="historical",
        threshold_days=7,
        min_history_years=15,
        max_rows=50,
        candidates=[("/stable/dividends", {"symbol": symbol_obj.symbol})],
    )
