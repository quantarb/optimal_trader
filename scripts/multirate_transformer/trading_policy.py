"""Legacy-compatible trading score policy for the multi-rate model."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def _date_percentile(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = out.groupby("date")[column].rank(pct=True, method="average")
    return out


def build_legacy_compatible_scores(
    predictions: pd.DataFrame,
    *,
    strategy: str = "return",
) -> pd.DataFrame:
    """Map new-model task outputs to the existing optimal-trader score schema.

    This intentionally follows the established strategy rather than using
    auxiliary event/context/rank heads as implicit trading rules:

    - HITS hub is the entry score;
    - HITS authority is the exit score;
    - all four components are percentile-ranked within each date;
    - direction agreement is computed from long versus short hub scores.
    """
    strategy = str(strategy).strip().lower()
    if strategy not in {"return", "speed"}:
        raise ValueError("strategy must be 'return' or 'speed'")
    required = {"symbol", "date"}
    prefix = "speed_" if strategy == "speed" else ""
    required.update({f"{prefix}long_hub", f"{prefix}long_authority", f"{prefix}short_hub", f"{prefix}short_authority"})
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"predictions missing trading heads: {sorted(missing)}")
    out = predictions.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.normalize()
    source_columns = [f"{prefix}{name}" for name in ("long_hub", "long_authority", "short_hub", "short_authority")]
    out = _date_percentile(out, source_columns)
    out["long_score"] = out[f"{prefix}long_hub"]
    out["long_exit_score"] = out[f"{prefix}long_authority"]
    out["short_score"] = out[f"{prefix}short_hub"]
    out["short_exit_score"] = out[f"{prefix}short_authority"]
    out["long_agree_count"] = (out["long_score"] >= out["short_score"]).astype(int)
    out["short_agree_count"] = (out["short_score"] > out["long_score"]).astype(int)
    out["model_count"] = 1
    return out


def task_output_frame(
    symbol: Sequence[str],
    date: Sequence[object],
    outputs: dict[str, object],
) -> pd.DataFrame:
    """Flatten one new-model output batch for score-policy consumption."""
    frame = pd.DataFrame({"symbol": list(symbol), "date": list(date)})
    for name, value in outputs.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = value
        if getattr(array, "ndim", 0) == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if getattr(array, "ndim", 0) != 2:
            raise ValueError(f"task output {name!r} must have shape [batch, sequence] or [batch, sequence, 1]")
        if array.shape[0] != len(frame) or array.shape[1] != 1:
            raise ValueError("task_output_frame expects one token per supplied row")
        frame[name] = array[:, 0]
    return frame

