from __future__ import annotations

from .base import build_symbol_endpoint
from .helpers import period_limit_params

build = build_symbol_endpoint(
    key="key_metrics",
    title="Key Metrics",
    kind="historical",
    threshold_days=7,
    supported_periods=("quarter", "annual"),
    min_history_years=10,
    max_rows=12,
    candidate_path="/stable/key-metrics",
    extra_params_builder=lambda _symbol_obj: period_limit_params(12, "quarter", "annual"),
)
