from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import pandas as pd


def _safe_float(value: Any, default: float) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _normalize_component_name(raw_name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(raw_name or "").strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "component"


def evaluate_signal_expression(
    frame: pd.DataFrame,
    *,
    expression: str = "",
    strict: bool = False,
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    resolved = str(expression or "").strip()
    if not resolved:
        return pd.Series(0.0, index=frame.index, dtype=float)
    if resolved in frame.columns:
        return pd.to_numeric(frame[resolved], errors="coerce")
    try:
        return pd.to_numeric(frame.eval(resolved, engine="python"), errors="coerce")
    except Exception:
        if strict:
            raise ValueError(f"Could not evaluate signal expression: {resolved!r}")
        return pd.Series(index=frame.index, dtype=float)


def _rank_percentile(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    out = pd.Series(0.5, index=series.index, dtype=float)
    if valid.empty:
        return out
    ranking_source = valid if higher_is_better else (-1.0 * valid)
    if len(valid) == 1:
        out.loc[valid.index] = 0.5
        return out
    rank_positions = ranking_source.rank(method="average", ascending=True)
    out.loc[rank_positions.index] = (rank_positions - 1.0) / float(len(rank_positions) - 1)
    return out


def _zscore(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    out = pd.Series(0.0, index=series.index, dtype=float)
    if valid.empty:
        return out
    adjusted = valid if higher_is_better else (-1.0 * valid)
    mean = float(adjusted.mean())
    std = float(adjusted.std(ddof=0))
    if std <= 1e-12:
        out.loc[adjusted.index] = 0.0
        return out
    out.loc[adjusted.index] = (adjusted - mean) / std
    return out


def build_multi_factor_score_frame(
    frame: pd.DataFrame,
    *,
    factor_components: Sequence[Mapping[str, Any]],
    output_col: str = "multi_factor_score",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        return frame.copy(), {"component_columns": [], "output_col": output_col, "components": []}
    components = [dict(component) for component in list(factor_components or []) if dict(component)]
    if not components:
        raise ValueError("factor_components must contain at least one component definition.")

    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().all():
        raise ValueError("factor_components require a usable date column for cross-sectional normalization.")

    component_columns: list[str] = []
    component_meta: list[dict[str, Any]] = []
    weighted_total = pd.Series(0.0, index=out.index, dtype=float)
    total_abs_weight = 0.0

    for index, component in enumerate(components, start=1):
        raw_name = str(component.get("name") or component.get("field") or component.get("expression") or f"component_{index}")
        component_name = _normalize_component_name(raw_name)
        component_col = f"factor_component__{component_name}"
        expression = str(component.get("expression") or component.get("field") or "").strip()
        higher_is_better = bool(component.get("higher_is_better", True))
        transform = str(component.get("transform") or "rank_pct").strip().lower() or "rank_pct"
        weight = _safe_float(component.get("weight"), 1.0)

        raw_scores = evaluate_signal_expression(out, expression=expression, strict=True)
        transformed = pd.Series(index=out.index, dtype=float)
        for _date_value, group in out.groupby("date", sort=False):
            group_raw = raw_scores.loc[group.index]
            if transform == "zscore":
                transformed.loc[group.index] = _zscore(group_raw, higher_is_better=higher_is_better)
            else:
                transformed.loc[group.index] = _rank_percentile(group_raw, higher_is_better=higher_is_better)

        out[component_col] = pd.to_numeric(transformed, errors="coerce").fillna(0.0)
        component_columns.append(component_col)
        component_meta.append(
            {
                "name": component_name,
                "raw_name": raw_name,
                "expression": expression,
                "weight": float(weight),
                "transform": transform,
                "higher_is_better": bool(higher_is_better),
                "output_column": component_col,
            }
        )
        weighted_total = weighted_total + (out[component_col] * float(weight))
        total_abs_weight += abs(float(weight))

    denominator = total_abs_weight if total_abs_weight > 1e-12 else 1.0
    out[output_col] = (weighted_total / denominator).astype(float)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out, {
        "component_columns": component_columns,
        "output_col": output_col,
        "components": component_meta,
    }


__all__ = [
    "build_multi_factor_score_frame",
    "evaluate_signal_expression",
]
