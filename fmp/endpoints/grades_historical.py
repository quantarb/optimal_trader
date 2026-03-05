from __future__ import annotations

from .base import EndpointDefinition


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="grades_historical",
        title="Grades Historical",
        kind="historical",
        threshold_days=30,
        min_history_years=5,
        max_rows=100,
        candidates=[("/stable/grades-historical", {"symbol": symbol_obj.symbol})],
    )
