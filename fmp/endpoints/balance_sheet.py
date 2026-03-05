from __future__ import annotations

from .base import EndpointDefinition
from .helpers import period_limit_params


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="balance_sheet",
        title="Balance Sheet",
        kind="historical",
        threshold_days=30,
        supported_periods=("quarter", "annual"),
        min_history_years=10,
        max_rows=10,
        candidates=[("/stable/balance-sheet-statement", {"symbol": symbol_obj.symbol, **period_limit_params(10, "quarter", "annual")})],
    )
