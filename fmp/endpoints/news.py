from __future__ import annotations

from .base import EndpointDefinition


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="news",
        title="News",
        kind="historical",
        threshold_days=1,
        min_history_years=0,
        max_rows=50,
        candidates=[("/stable/news/stock", {"symbols": symbol_obj.symbol, "limit": 50})],
    )
