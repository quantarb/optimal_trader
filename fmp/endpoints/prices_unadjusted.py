from __future__ import annotations

from .base import EndpointDefinition
from .helpers import full_history_range_params


def build(symbol_obj) -> EndpointDefinition:
    params = {
        "symbol": symbol_obj.symbol,
        **full_history_range_params(),
        "__chunk_years": 10,
    }
    return EndpointDefinition(
        key="prices_unadjusted",
        title="Recent Prices (Unadjusted)",
        kind="historical",
        threshold_days=1,
        min_history_years=10,
        max_rows=60,
        candidates=[
            ("/stable/historical-price-eod", dict(params)),
            ("/stable/historical-price-eod/full", dict(params)),
        ],
    )
