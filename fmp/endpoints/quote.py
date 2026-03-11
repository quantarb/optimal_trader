from __future__ import annotations

from .base import build_symbol_endpoint

build = build_symbol_endpoint(
    key="quote",
    title="Quote Snapshot",
    kind="snapshot",
    threshold_days=1,
    max_rows=1,
    candidate_path="/stable/quote",
)
