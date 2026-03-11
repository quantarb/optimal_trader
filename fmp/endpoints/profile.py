from __future__ import annotations

from .base import build_symbol_endpoint

build = build_symbol_endpoint(
    key="profile",
    title="Company Profile",
    kind="snapshot",
    threshold_days=30,
    max_rows=1,
    candidate_path="/stable/profile",
)
