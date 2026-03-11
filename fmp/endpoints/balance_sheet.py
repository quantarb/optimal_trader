from __future__ import annotations

from .base import build_symbol_endpoint
from .helpers import period_limit_params

build = build_symbol_endpoint(
    key="balance_sheet",
    title="Balance Sheet",
    kind="historical",
    threshold_days=30,
    supported_periods=("quarter", "annual"),
    min_history_years=10,
    max_rows=10,
    candidate_path="/stable/balance-sheet-statement",
    extra_params_builder=lambda _symbol_obj: period_limit_params(10, "quarter", "annual"),
)
