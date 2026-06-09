from __future__ import annotations

from .base import build_symbol_endpoint

build = build_symbol_endpoint(
    key="ratios_ttm",
    title="Ratios TTM",
    kind="historical",
    threshold_days=7,
    max_rows=1,
    candidate_path="/stable/ratios-ttm",
)
