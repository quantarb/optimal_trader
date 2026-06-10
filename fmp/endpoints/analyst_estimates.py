from __future__ import annotations

from .base import EndpointDefinition
from .helpers import paginated_params, preferred_period


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="analyst_estimates",
        title="Analyst Estimates",
        kind="historical",
        threshold_days=30,
        supported_periods=("annual",),
        min_history_years=5,
        max_rows=10,
        candidates=[
            (
                "/stable/analyst-estimates",
                {
                    "symbol": symbol_obj.symbol,
                    "period": preferred_period("annual"),
                    **paginated_params(page=0),
                },
            )
        ],
        pagination="page",
    )
