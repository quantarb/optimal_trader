from __future__ import annotations

from .base import build_symbol_endpoint

build = build_symbol_endpoint(
    key="income_statement_ttm",
    title="Income Statement TTM",
    kind="historical",
    threshold_days=7,
    max_rows=1,
    candidate_path="/stable/income-statement-ttm",
)
