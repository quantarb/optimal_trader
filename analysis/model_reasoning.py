from __future__ import annotations

from typing import Any, Sequence

from .market_insight_schema import FamiliaritySignals, ModelScore


def summarize_model_scores(model_scores: Sequence[ModelScore]) -> dict[str, Any]:
    rows = [item for item in list(model_scores or []) if item.value is not None]
    ranked = sorted(
        rows,
        key=lambda item: abs(float(item.percentile or 0.5) - 0.5) if item.percentile is not None else abs(float(item.value or 0.0)),
        reverse=True,
    )
    top_scores = ranked[:3]
    summary_lines = [f"{item.display_name} is at {item.rendered_value}." for item in top_scores]
    return {
        "summary_lines": summary_lines,
        "evidence": {"top_scores": [item.to_dict() for item in top_scores]},
    }


def detect_signal_conflicts(
    model_scores: Sequence[ModelScore],
    analog_summary: dict[str, Any],
    familiarity_signals: FamiliaritySignals,
) -> dict[str, Any]:
    score_map = {item.name: item for item in list(model_scores or [])}
    conflicts: list[dict[str, Any]] = []
    ranking = score_map.get("ranking") or score_map.get("combined_score") or score_map.get("opportunity_score")
    analog_positive_rate = analog_summary.get("positive_rate")
    if ranking and ranking.value is not None and analog_positive_rate is not None:
        if float(ranking.value) >= 70.0 and float(analog_positive_rate) < 0.5:
            conflicts.append(
                {
                    "kind": "ranking_vs_analogs",
                    "message": "Model ranking is strong, but historical analogs are not consistently favorable.",
                }
            )
    if familiarity_signals.market_familiarity_score is not None and analog_positive_rate is not None:
        if float(analog_positive_rate) >= 0.6 and float(familiarity_signals.market_familiarity_score) < 45.0:
            conflicts.append(
                {
                    "kind": "favorable_but_unfamiliar",
                    "message": "Historical analogs look favorable, but this state is less familiar than typical opportunity states.",
                }
            )
    if familiarity_signals.confidence_score is not None and familiarity_signals.market_familiarity_score is not None:
        if float(familiarity_signals.confidence_score) >= 70.0 and float(familiarity_signals.market_familiarity_score) < 40.0:
            conflicts.append(
                {
                    "kind": "confidence_vs_familiarity",
                    "message": "Confidence is high, but familiarity remains low, so the evidence is not uniformly aligned.",
                }
            )
    return {
        "summary_lines": [item["message"] for item in conflicts],
        "evidence": {"conflicts": conflicts},
    }


def summarize_signal_alignment(
    model_scores: Sequence[ModelScore],
    analog_summary: dict[str, Any],
    familiarity_signals: FamiliaritySignals,
) -> dict[str, Any]:
    conflicts = detect_signal_conflicts(model_scores, analog_summary, familiarity_signals)
    score_map = {item.name: item for item in list(model_scores or [])}
    alignment_lines: list[str] = []
    opportunity = score_map.get("opportunity_score")
    confidence = score_map.get("confidence_score")
    analog_positive_rate = analog_summary.get("positive_rate")
    if opportunity and opportunity.value is not None and analog_positive_rate is not None:
        if float(opportunity.value) >= 65.0 and float(analog_positive_rate) >= 0.6:
            alignment_lines.append("The model stack appears directionally aligned with the historical analog evidence.")
        elif float(opportunity.value) < 45.0 and float(analog_positive_rate) < 0.5:
            alignment_lines.append("Model scores and historical analogs are both cautious.")
        else:
            alignment_lines.append("Model scores and historical analogs are only partially aligned.")
    if confidence and confidence.value is not None:
        alignment_lines.append(f"Confidence is {confidence.rendered_value} and familiarity is {familiarity_signals.market_familiarity_label.lower() or 'unspecified'}.")
    return {
        "summary_lines": alignment_lines + list(conflicts.get("summary_lines") or []),
        "evidence": {
            "model_scores": [item.to_dict() for item in list(model_scores or [])],
            "conflicts": list((conflicts.get("evidence") or {}).get("conflicts") or []),
            "analog_positive_rate": analog_positive_rate,
        },
    }
