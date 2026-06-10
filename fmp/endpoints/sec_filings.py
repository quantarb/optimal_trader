from __future__ import annotations

from .base import EndpointDefinition
from .helpers import today


def build(symbol_obj) -> EndpointDefinition:
    end_date = today()
    return EndpointDefinition(
        key="sec_filings",
        title="SEC Filings",
        kind="historical",
        threshold_days=7,
        min_history_years=5,
        max_rows=50,
        candidates=[
            (
                "/stable/sec-filings-search/symbol",
                {
                    "symbol": symbol_obj.symbol,
                    "from": end_date.replace(year=end_date.year - 5).isoformat(),
                    "to": end_date.isoformat(),
                    "page": 0,
                    "limit": 50,
                },
            )
        ],
        pagination="page",
        page_size=50,
        supports_date_window=True,
    )
