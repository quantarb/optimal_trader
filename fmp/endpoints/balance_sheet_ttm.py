from __future__ import annotations

from .base import build_symbol_endpoint

build = build_symbol_endpoint(
    key="balance_sheet_ttm",
    title="Balance Sheet TTM",
    kind="historical",
    threshold_days=7,
    max_rows=1,
    candidate_path="/stable/balance-sheet-statement-ttm",
)
