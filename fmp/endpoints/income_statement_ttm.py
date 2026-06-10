from __future__ import annotations

from .base import build_symbol_endpoint
from .helpers import limit_params

build = build_symbol_endpoint(
    key="income_statement_ttm",
    title="Income Statement TTM",
    kind="historical",
    threshold_days=30,
    supported_periods=("quarter", "annual"),
    min_history_years=10,
    max_rows=10,
    candidate_path="/stable/income-statement-ttm",
    extra_params_builder=lambda _symbol_obj: limit_params(),
)
