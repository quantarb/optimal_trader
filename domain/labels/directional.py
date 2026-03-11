from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from data.schema import require_columns
from utils.normalize import normalize_cols


def add_binary_classification_labels(
    events: pd.DataFrame,
    *,
    use_sample_weight: bool = True,
    r_clip: float = 0.10,
    alpha: float = 4.0,
    horizon_balance: bool = True,
    horizon_balance_mode: str = "mass",
    entry_only_weighting: bool = True,
    horizon_factor_cap: Optional[float] = 3.0,
) -> pd.DataFrame:
    """Convert per-event rows into binary long-vs-short labels."""

    if events is None or len(events) == 0:
        return pd.DataFrame()

    ev = normalize_cols(events)
    require_columns(ev, ["event", "side", "horizon"], ctx="add_binary_classification_labels")
    base_cols = ["event", "side", "horizon"]
    extra_cols = ["trade_return"] if "trade_return" in ev.columns else []
    out = ev[base_cols + extra_cols].copy()

    def _to_target(row) -> int:
        side = row["side"]
        event = row["event"]
        if side == "long":
            return 1 if event == "entry" else 0
        if side == "short":
            return 0 if event == "entry" else 1
        raise ValueError(f"Unexpected side={side!r} event={event!r}")

    out["target"] = out.apply(_to_target, axis=1)
    if use_sample_weight and "trade_return" in out.columns:
        returns = pd.to_numeric(out["trade_return"], errors="coerce").fillna(0.0).to_numpy()
        clipped = np.clip(returns, 0.0, float(r_clip))
        denom = float(r_clip) if float(r_clip) > 0 else 1.0
        out["sample_weight"] = (1.0 + float(alpha) * (clipped / denom)).astype(float)
        is_entry = out["event"] == "entry"
        if entry_only_weighting:
            out.loc[~is_entry, "sample_weight"] = 1.0
        if horizon_balance:
            if horizon_balance_mode not in {"mass", "count"}:
                raise ValueError("horizon_balance_mode must be 'mass' or 'count'")
            entry_df = out.loc[is_entry].copy()
            if not entry_df.empty:
                if horizon_balance_mode == "count":
                    denom_series = entry_df.groupby(["side", "horizon"]).size().astype(float)
                else:
                    denom_series = entry_df.groupby(["side", "horizon"])["sample_weight"].sum().astype(float)
                inv = 1.0 / denom_series
                inv = inv / inv.mean()
                inv = inv.clip(lower=1.0)
                if horizon_factor_cap is not None:
                    inv = inv.clip(upper=float(horizon_factor_cap))
                factors = entry_df.set_index(["side", "horizon"]).index.map(inv).astype(float)
                out.loc[is_entry, "sample_weight"] *= factors.to_numpy()

    out = out.sort_index()
    keep = ["target", "side", "horizon"]
    if "trade_return" in out.columns:
        keep.append("trade_return")
    if "sample_weight" in out.columns:
        keep.append("sample_weight")
    return out[keep]


def add_action_labels(events: pd.DataFrame) -> pd.DataFrame:
    """Convert per-event rows into explicit trading action labels."""

    if events is None or len(events) == 0:
        return pd.DataFrame()

    ev = normalize_cols(events)
    require_columns(ev, ["event", "side", "horizon"], ctx="add_action_labels")
    base_cols = ["event", "side", "horizon"]
    extra_cols = ["trade_return"] if "trade_return" in ev.columns else []
    out = ev[base_cols + extra_cols].copy()

    def _to_label(row) -> str:
        side = row["side"]
        event = row["event"]
        if side == "long":
            if event == "entry":
                return "buy"
            if event == "exit":
                return "sell"
        if side == "short":
            if event == "entry":
                return "short"
            if event == "exit":
                return "cover"
        raise ValueError(f"Unexpected side={side!r} event={event!r}")

    label_to_position = {"buy": 0, "short": 0, "sell": 1, "cover": -1}
    out["label"] = out.apply(_to_label, axis=1)
    out["market_position"] = out["label"].map(label_to_position).astype(int)
    out = out.sort_index()
    keep = ["label", "market_position", "side", "horizon"]
    if "trade_return" in out.columns:
        keep.append("trade_return")
    return out[keep]

