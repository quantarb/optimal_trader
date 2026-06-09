from __future__ import annotations

from .base import build_symbol_endpoint

build = build_symbol_endpoint(
    key="cash_flow_ttm",
    title="Cash Flow TTM",
    kind="historical",
    threshold_days=7,
    max_rows=1,
    candidate_path="/stable/cash-flow-statement-ttm",
)
