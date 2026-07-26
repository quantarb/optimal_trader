"""Resolve the current experiment's task inventory without duplicating names."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def resolve_current_tasks(gnn_script: Path | None = None) -> list[dict[str, object]]:
    path = gnn_script or Path(__file__).resolve().parents[1] / "run_feature_family_gnn_smoke.py"
    spec = importlib.util.spec_from_file_location("current_gnn_registry", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load task registry: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows: list[dict[str, object]] = []
    graph_names = (
        "long_hub", "long_authority", "long_pagerank",
        "short_hub", "short_authority", "short_pagerank",
        *getattr(module, "SPEED_TARGET_COLS", ()),
    )
    for name in graph_names:
        rows.append({"task_name": name, "task_type": "regression", "task_level": "instrument", "target_column": name, "loss": "smooth_l1", "orientation": "long" if "long" in name else "short", "weight": 1.0, "applicable_asset_classes": "non_option"})
    for name in getattr(module, "ALL_EVENT_TARGETS", {}):
        rows.append({"task_name": name, "task_type": "event", "task_level": "instrument", "target_column": name, "loss": "prototype_or_bce", "orientation": "positive", "weight": 0.25, "applicable_asset_classes": "issuer_linked"})
    for name in getattr(module, "AUX_TARGET_COLS", ()):
        rows.append({"task_name": name, "task_type": "categorical", "task_level": "issuer_or_context", "target_column": name, "loss": "cross_entropy", "orientation": "n/a", "weight": 0.1, "applicable_asset_classes": "issuer_linked"})
    return rows


def task_inventory_frame(gnn_script: Path | None = None):
    import pandas as pd

    return pd.DataFrame(resolve_current_tasks(gnn_script))

