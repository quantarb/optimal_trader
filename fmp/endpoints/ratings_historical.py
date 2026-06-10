from __future__ import annotations

from .base import EndpointDefinition
from .helpers import paginated_params


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="ratings_historical",
        title="Ratings Historical",
        kind="historical",
        threshold_days=30,
        min_history_years=5,
        max_rows=50,
        candidates=[("/stable/ratings-historical", {"symbol": symbol_obj.symbol, **paginated_params(page=0)})],
        pagination="page",
    )
