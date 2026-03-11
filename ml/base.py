from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceSpec:
    """Optional sequence/text settings for architecture-agnostic training specs."""

    input_col: str = "input_text"
    target_col: str = "target_text"
    sequence_id_col: str | None = None
    time_col: str | None = None
    padding: str = "max_length"
    max_source_length: int = 512
    max_target_length: int = 512


@dataclass(frozen=True)
class FitSpec:
    """Training spec shared across model adapters."""

    feature_cols: Sequence[str]
    target_col: str | None = None
    weight_col: str | None = None
    split_ratio: float = 0.8
    signal: str | None = None
    model_tag: str | None = None
    task_type: str = "tabular"
    sequence: SequenceSpec | None = None

    def is_sequence_task(self) -> bool:
        return self.sequence is not None or self.task_type in {"seq2seq", "sequence"}


class Model(Protocol):
    """Framework-agnostic model contract."""

    def fit(self, df_train: pd.DataFrame, spec: FitSpec) -> "Model":
        ...

    def predict(self, df: pd.DataFrame, *, feature_cols: Sequence[str]) -> np.ndarray:
        ...

    def metrics_report(self) -> dict[str, Any]:
        ...

    def summarize(self) -> None:
        ...


def print_model_section(title: str) -> None:
    """Utility for consistent log formatting across all model types."""

    print("\n" + "=" * 60)
    print(f"  DIAGNOSTIC: {title.upper()}")
    print("=" * 60)


def metrics_with_feature_importance(
    metrics: dict[str, Any],
    feature_importance: dict[str, float],
    *,
    top_n: int = 30,
) -> dict[str, Any]:
    out = dict(metrics)
    if feature_importance:
        top = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)[: int(top_n)]
        out["feature_importance_top"] = top
    return out


def copy_feature_importance(feature_importance: dict[str, float]) -> dict[str, float]:
    return dict(feature_importance)
