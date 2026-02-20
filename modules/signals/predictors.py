from __future__ import annotations

from typing import Iterable, Optional, Protocol

import pandas as pd

from modules.utils.panel import ensure_panel_index


class SignalProducer(Protocol):
    """A callable that adds one or more signal columns to a panel."""

    def __call__(self, panel: pd.DataFrame) -> pd.DataFrame:
        ...


def run_signal_producers(panel: pd.DataFrame, producers: Optional[Iterable[SignalProducer]] = None) -> pd.DataFrame:
    """Apply signal producers to the panel.

    In the refactor, this is usually a no-op because signals are already
    materialized as prediction columns by the training pipeline.
    """
    out = ensure_panel_index(panel)
    if not producers:
        return out
    for p in producers:
        out = p(out)
        out = ensure_panel_index(out)
    return out
