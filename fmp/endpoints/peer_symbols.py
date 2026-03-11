from __future__ import annotations

from .base import build_symbol_endpoint

build = build_symbol_endpoint(
    key="peer_symbols",
    title="Peer Symbols",
    kind="snapshot",
    threshold_days=30,
    max_rows=100,
    candidate_path="/stable/stock-peers",
)
