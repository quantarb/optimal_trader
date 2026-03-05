from __future__ import annotations

from .base import EndpointDefinition
from .helpers import recent_year_quarters


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="institutional_holders",
        title="Institutional Holders",
        kind="snapshot",
        threshold_days=30,
        max_rows=100,
        candidates=[
            (
                "/stable/institutional-ownership/symbol-positions-summary",
                {"symbol": symbol_obj.symbol, "year": year, "quarter": quarter},
            )
            for year, quarter in recent_year_quarters(8)
        ],
    )
