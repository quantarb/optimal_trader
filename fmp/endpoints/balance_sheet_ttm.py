from __future__ import annotations

from .base import build_symbol_endpoint
from .helpers import limit_params

build = build_symbol_endpoint(
    key="balance_sheet_ttm",
    title="Balance Sheet TTM",
    kind="historical",
    threshold_days=30,
    supported_periods=("quarter", "annual"),
    min_history_years=10,
    max_rows=10,
    candidate_path="/stable/balance-sheet-statement-ttm",
    extra_params_builder=lambda _symbol_obj: limit_params(),
)
