from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd


@dataclass(frozen=True)
class BacktestSpec:
    """Minimal backtest workflow configuration."""

    name: str = ""
    config: dict[str, Any] = field(default_factory=dict)


class BacktestRunner(Protocol):
    """Backtest workflow interface."""

    def run(self, frame: pd.DataFrame, *, spec: BacktestSpec) -> pd.DataFrame:
        ...
