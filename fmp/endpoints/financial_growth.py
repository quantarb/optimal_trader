from __future__ import annotations

from .base import EndpointDefinition
from .helpers import period_limit_params


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="financial_growth",
        title="Financial Statement Growth",
        kind="historical",
        threshold_days=30,
        supported_periods=("quarter", "annual"),
        min_history_years=10,
        max_rows=10,
        candidates=[("/stable/financial-growth", {"symbol": symbol_obj.symbol, **period_limit_params(10, "quarter", "annual")})],
    )
