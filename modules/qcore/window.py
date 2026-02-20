from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class TimeWindow:
    """Closed-open time window [start, end)."""

    start: pd.Timestamp
    end: pd.Timestamp

    @staticmethod
    def from_ymd(start: str, end: str) -> "TimeWindow":
        return TimeWindow(pd.Timestamp(start), pd.Timestamp(end))

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"Invalid window: end <= start ({self.start} .. {self.end})")
