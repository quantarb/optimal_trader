from __future__ import annotations

from .base import EndpointDefinition
from .helpers import recent_year_quarters


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="positions_summary",
        title="Positions Summary",
        kind="historical",
        threshold_days=30,
        max_rows=100,
        candidates=[
            (
                "/stable/institutional-ownership/symbol-positions-summary",
                {"symbol": symbol_obj.symbol, "year": year, "quarter": quarter},
            )
            for year, quarter in recent_year_quarters(8)
        ],
        dedupe_by_date=True,
        minimum_observations=1,
    )
