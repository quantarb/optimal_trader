# modules/models/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class SequenceSpec:
    """Optional sequence/text settings for architecture-agnostic training specs."""

    input_col: str = "input_text"
    target_col: str = "target_text"
    sequence_id_col: Optional[str] = None
    time_col: Optional[str] = None
    padding: str = "max_length"
    max_source_length: int = 512
    max_target_length: int = 512


@dataclass(frozen=True)
class FitSpec:
    """Training spec shared across model adapters."""
    feature_cols: Sequence[str]
    target_col: Optional[str] = None
    weight_col: Optional[str] = None

    # NEW: Control the random train/test split ratio
    split_ratio: float = 0.8  # Default to 80% train, 20% test

    signal: Optional[str] = None
    model_tag: Optional[str] = None
    task_type: str = "tabular"
    sequence: Optional[SequenceSpec] = None

    def is_sequence_task(self) -> bool:
        return self.sequence is not None or self.task_type in {"seq2seq", "sequence"}

class Model(Protocol):
    """Framework-agnostic model contract."""

    def fit(self, df_train: pd.DataFrame, spec: FitSpec) -> "Model":
        ...

    def predict(self, df: pd.DataFrame, *, feature_cols: Sequence[str]) -> np.ndarray:
        ...

    def metrics_report(self) -> dict:
        ...

    def summarize(self) -> None:
        """Each model implements its own diagnostic print for the LLM."""
        ...

def print_model_section(title: str):
    """Utility for consistent log formatting across all model types."""
    print("\n" + "="*60)
    print(f"  DIAGNOSTIC: {title.upper()}")
    print("="*60)


def metrics_with_feature_importance(
    metrics: Dict[str, Any],
    feature_importance: Dict[str, float],
    *,
    top_n: int = 30,
) -> Dict[str, Any]:
    out = dict(metrics)
    if feature_importance:
        top = sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)[: int(top_n)]
        out["feature_importance_top"] = top
    return out


def copy_feature_importance(feature_importance: Dict[str, float]) -> Dict[str, float]:
    return dict(feature_importance)
