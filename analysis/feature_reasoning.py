from __future__ import annotations

from typing import Any, Mapping, Sequence

from .market_insight_schema import CanonicalFeatureValue


def _direction_from_feature(feature: CanonicalFeatureValue) -> str:
    percentile = feature.percentile
    if percentile is not None:
        if percentile >= 0.9:
            return "very high"
        if percentile >= 0.75:
            return "above typical"
        if percentile <= 0.1:
            return "very low"
        if percentile <= 0.25:
            return "below typical"
    raw = feature.raw_value
    try:
        numeric_value = float(raw)
    except Exception:
        numeric_value = None
    if numeric_value is None:
        return "notable"
    if numeric_value > 0:
        return "positive"
    if numeric_value < 0:
        return "negative"
    return "flat"


def summarize_feature_family(feature_family_name: str, features: Sequence[CanonicalFeatureValue], limit: int = 3) -> dict[str, Any]:
    rows = list(features or [])
    if not rows:
        return {"family": str(feature_family_name), "summary_lines": [], "evidence": {"feature_count": 0}}
    ranked = sorted(
        rows,
        key=lambda item: (
            abs(float(item.zscore or 0.0)) if item.zscore is not None else abs((float(item.percentile or 0.5) - 0.5) * 2.0),
            item.display_name,
        ),
        reverse=True,
    )
    selected = ranked[: max(int(limit), 1)]
    summary_lines = [
        f"{feature.display_name} is {_direction_from_feature(feature)} at {feature.rendered_value}."
        for feature in selected
    ]
    return {
        "family": str(feature_family_name),
        "summary_lines": summary_lines,
        "evidence": {
            "feature_count": len(rows),
            "top_features": [feature.to_dict() for feature in selected],
        },
    }


def summarize_feature_extremes(features: Sequence[CanonicalFeatureValue], limit: int = 5) -> dict[str, Any]:
    rows = list(features or [])
    ranked = sorted(
        rows,
        key=lambda item: (
            abs(float(item.zscore or 0.0)) if item.zscore is not None else abs((float(item.percentile or 0.5) - 0.5) * 2.0),
            item.display_name,
        ),
        reverse=True,
    )[: max(int(limit), 1)]
    return {
        "summary_lines": [f"{item.display_name} stands out at {item.rendered_value}." for item in ranked],
        "evidence": {"top_features": [item.to_dict() for item in ranked]},
    }


def summarize_feature_changes(
    current_features: Sequence[CanonicalFeatureValue],
    baseline_features: Mapping[str, CanonicalFeatureValue] | Sequence[CanonicalFeatureValue] | None = None,
    limit: int = 4,
) -> dict[str, Any]:
    baseline_map: dict[str, CanonicalFeatureValue] = {}
    if isinstance(baseline_features, Mapping):
        baseline_map = {str(key): value for key, value in dict(baseline_features or {}).items()}
    else:
        baseline_map = {item.internal_name: item for item in list(baseline_features or [])}
    rows: list[tuple[float, str, dict[str, Any]]] = []
    for current in list(current_features or []):
        baseline = baseline_map.get(current.internal_name)
        if baseline is None:
            continue
        try:
            current_value = float(current.raw_value)
            baseline_value = float(baseline.raw_value)
        except Exception:
            continue
        delta = current_value - baseline_value
        if delta == 0.0:
            continue
        rows.append(
            (
                abs(delta),
                f"{current.display_name} changed from {baseline.rendered_value} to {current.rendered_value}.",
                {
                    "feature": current.to_dict(),
                    "baseline_feature": baseline.to_dict(),
                    "delta": delta,
                },
            )
        )
    rows.sort(key=lambda item: item[0], reverse=True)
    selected = rows[: max(int(limit), 1)]
    return {
        "summary_lines": [item[1] for item in selected],
        "evidence": {"changes": [item[2] for item in selected]},
    }
