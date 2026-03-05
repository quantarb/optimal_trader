from __future__ import annotations

from .base import EndpointDefinition


def build(symbol_obj) -> EndpointDefinition:
    return EndpointDefinition(
        key="mutual_fund_holders",
        title="Mutual Fund Holders",
        kind="snapshot",
        threshold_days=30,
        max_rows=100,
        candidates=[("/stable/mutual-fund-holder", {"symbol": symbol_obj.symbol})],
    )
