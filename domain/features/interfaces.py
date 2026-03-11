from __future__ import annotations

from typing import Protocol

import pandas as pd

from domain.features.specs import BuiltFeatureSet, FeatureBuildSpec


class FeatureBuilder(Protocol):
    """Pure feature-family builder contract."""

    def build(self, *, symbol: str, target_index: pd.MultiIndex) -> BuiltFeatureSet:
        ...


class FeaturePanelBuilder(Protocol):
    """Panel builder contract for multi-symbol feature generation."""

    def build(self, *, symbols: list[str], spec: FeatureBuildSpec) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
        ...

