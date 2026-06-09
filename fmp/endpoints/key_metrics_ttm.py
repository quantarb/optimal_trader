from __future__ import annotations

from .base import build_symbol_endpoint

build = build_symbol_endpoint(
    key="key_metrics_ttm",
    title="Key Metrics TTM",
    kind="historical",
    threshold_days=7,
    max_rows=1,
    candidate_path="/stable/key-metrics-ttm",
)
