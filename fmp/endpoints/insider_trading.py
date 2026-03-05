from __future__ import annotations

from .base import EndpointDefinition
from .helpers import paginated_params


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="insider_trading",
        title="Insider Trading",
        kind="historical",
        threshold_days=7,
        min_history_years=1,
        max_rows=100,
        candidates=[("/stable/insider-trading/search", {"symbol": symbol_obj.symbol, **paginated_params(page=0)})],
    )
