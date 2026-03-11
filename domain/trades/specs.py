from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TradeGenerationResult:
    """Oracle trade rows plus normalized completed trade payloads."""

    trade_rows: list[dict[str, Any]] = field(default_factory=list)
    completed_trades: list[dict[str, Any]] = field(default_factory=list)

