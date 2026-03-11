from __future__ import annotations

from typing import Protocol

import pandas as pd

from domain.labels.specs import LabelBuildSpec


class LabelBuilder(Protocol):
    """Pure label-builder contract."""

    def build(self, events: pd.DataFrame, *, spec: LabelBuildSpec) -> pd.DataFrame:
        ...

