from __future__ import annotations

from typing import Any

from .market_insight_schema import FamiliaritySignals


def summarize_familiarity(familiarity_signals: FamiliaritySignals) -> dict[str, Any]:
    score = float(familiarity_signals.market_familiarity_score or 0.0)
    if score >= 75.0:
        summary = "This market state appears familiar based on the platform's novelty and analog evidence."
    elif score >= 50.0:
        summary = "This market state appears moderately familiar, though not at the highest familiarity levels."
    else:
        summary = "This market state appears less familiar than typical historical opportunity states."
    return {
        "summary_lines": [summary],
        "evidence": {
            "market_familiarity_score": familiarity_signals.market_familiarity_score,
            "market_familiarity_label": familiarity_signals.market_familiarity_label,
            "ae_familiarity_raw": familiarity_signals.ae_familiarity_raw,
        },
    }


def summarize_novelty_risk(familiarity_signals: FamiliaritySignals, analog_density: float | None = None) -> dict[str, Any]:
    score = float(familiarity_signals.market_familiarity_score or 0.0)
    density = float(analog_density if analog_density is not None else (familiarity_signals.analog_density or 0.0))
    lines: list[str] = []
    if score < 45.0:
        lines.append("Novelty risk is elevated because the current state is less familiar than typical historical opportunity states.")
    if density < 4.0:
        lines.append("The analog set is relatively thin, which makes the historical comparison less robust.")
    if not lines:
        lines.append("Novelty risk appears contained because the state is reasonably familiar and supported by a usable analog set.")
    return {
        "summary_lines": lines,
        "evidence": {
            "market_familiarity_score": familiarity_signals.market_familiarity_score,
            "analog_density": density,
            "mean_similarity": familiarity_signals.mean_similarity,
        },
    }
