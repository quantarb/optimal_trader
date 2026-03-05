from __future__ import annotations

from .base import EndpointDefinition
from .helpers import period_limit_params


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="income_statement",
        title="Income Statement",
        kind="historical",
        threshold_days=30,
        supported_periods=("quarter", "annual"),
        min_history_years=10,
        max_rows=10,
        candidates=[("/stable/income-statement", {"symbol": symbol_obj.symbol, **period_limit_params(10, "quarter", "annual")})],
    )
