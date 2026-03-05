from __future__ import annotations

from .base import EndpointDefinition
from .helpers import period_limit_params


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="ratios",
        title="Ratios",
        kind="historical",
        threshold_days=7,
        supported_periods=("quarter", "annual"),
        min_history_years=10,
        max_rows=12,
        candidates=[("/stable/ratios", {"symbol": symbol_obj.symbol, **period_limit_params(12, "quarter", "annual")})],
    )
