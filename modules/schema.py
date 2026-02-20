# ============================================================
# modules/schema.py
# Central schema contracts + validation + alignment helpers.
# ============================================================
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd


# --------------------------
# Canonical column sets
# --------------------------
# Daily features (post-normalization). Used for backtesting.
DAILY_REQUIRED: List[str] = ["close", "symbol"]

# Events produced by modules.events.generate_optimal_events
# (trade metadata lives HERE, not in labels)
EVENTS_REQUIRED: List[str] = [
    "event",          # "entry" / "exit"
    "side",           # "long" / "short"
    "horizon",        # "W"/"ME"/"QE"/"YE" (or others)
    "trade_id",       # stable id to link entry/exit
    "entry_px",
    "exit_px",
    "trade_return",   # net return (same on entry+exit rows)
]

EVENTS_OPTIONAL_DEFAULTS: Dict[str, Any] = {
    "trade_duration_days": np.nan,
    "trade_return_gross": np.nan,
}

# Labels produced by modules.labeler.label_events_directional
# (supervision contract)
LABELS_REQUIRED: List[str] = [
    "target",   # 0/1
    "side",
    "horizon",
]

LABELS_OPTIONAL_DEFAULTS: Dict[str, Any] = {
    "trade_return": np.nan,
    "sample_weight": 1.0,
}

# Optional backtest daily columns that reporting might summarize.
BACKTEST_DAILY_OPTIONAL_DEFAULTS: Dict[str, Any] = {
    "turnover": np.nan,
    "num_positions": np.nan,
}


# --------------------------
# Exceptions
# --------------------------
class SchemaError(ValueError):
    pass


# --------------------------
# Core helpers
# --------------------------
def require_columns(df: pd.DataFrame, cols: Iterable[str], *, ctx: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        prefix = f"[{ctx}] " if ctx else ""
        available = list(df.columns)
        raise SchemaError(
            f"{prefix}Missing required columns: {missing}. "
            f"Available (first 80): {available[:80]}"
        )


def ensure_columns(df: pd.DataFrame, defaults: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for c, v in defaults.items():
        if c not in out.columns:
            out[c] = v
    return out


def align_to_columns(df: pd.DataFrame, cols: List[str], *, fill_value: Any = np.nan) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = fill_value
    return out


def align_dict_to_columns(
    dfs: Dict[str, pd.DataFrame], cols: List[str], *, fill_value: Any = np.nan
) -> Dict[str, pd.DataFrame]:
    return {k: align_to_columns(v, cols, fill_value=fill_value) for k, v in dfs.items()}


def validate_feature_cols(feature_cols: List[str]) -> None:
    # Guardrails: label columns must never be treated as features
    bad = {"sample_weight", "trade_return", "target", "side", "horizon"} & set(feature_cols)
    if bad:
        raise SchemaError(f"These should NOT be in feature_cols: {sorted(bad)}")


def validate_daily_df(df_daily: pd.DataFrame, *, ctx: str = "daily") -> None:
    require_columns(df_daily, DAILY_REQUIRED, ctx=ctx)


def validate_events_df(events: pd.DataFrame, *, ctx: str = "events") -> pd.DataFrame:
    require_columns(events, EVENTS_REQUIRED, ctx=ctx)
    return ensure_columns(events, EVENTS_OPTIONAL_DEFAULTS)


def validate_labels_df(labels: pd.DataFrame, *, ctx: str = "labels") -> pd.DataFrame:
    """
    Labels are supervision rows (post-labeler).
    Trade metadata belongs to EVENTS.
    """
    require_columns(labels, LABELS_REQUIRED, ctx=ctx)

    out = ensure_columns(labels, LABELS_OPTIONAL_DEFAULTS)

    # Light sanity checks (keep strict enough to catch bugs, not strict on dtype)
    t = pd.to_numeric(out["target"], errors="coerce")
    if t.isna().any():
        raise SchemaError(f"[{ctx}] target contains non-numeric values")

    bad_side = set(out["side"].dropna().unique()) - {"long", "short"}
    if bad_side:
        raise SchemaError(f"[{ctx}] unexpected side values: {sorted(bad_side)}")

    return out
