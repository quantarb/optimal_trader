from __future__ import annotations

from .base import build_symbol_endpoint
from .helpers import period_limit_params

build = build_symbol_endpoint(
    key="income_statement_growth",
    title="Income Statement Growth",
    kind="historical",
    threshold_days=30,
    supported_periods=("quarter", "annual"),
    min_history_years=10,
    max_rows=10,
    candidate_path="/stable/income-statement-growth",
    extra_params_builder=lambda _symbol_obj: period_limit_params(10, "quarter", "annual"),
)
