from __future__ import annotations

from dataclasses import dataclass
from typing import Any


Candidate = tuple[str, dict[str, Any]]


@dataclass(frozen=True)
class EndpointDefinition:
    key: str
    title: str
    kind: str
    threshold_days: int
    max_rows: int
    candidates: list[Candidate]
    supported_periods: tuple[str, ...] = ()
    min_history_years: int | None = None
    filter_symbol: bool = False
