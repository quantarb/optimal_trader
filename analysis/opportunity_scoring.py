from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from .market_state import feature_family_map_from_columns, percentile_rank


FAMILY_EXPLANATIONS = {
    "analyst_estimates": "earnings revisions improving",
    "prices_div_adj": "positive momentum cluster",
    "economic_indicators": "favorable macro liquidity",
    "treasury_rates": "rate backdrop supportive",
    "income_statement": "fundamental trend supportive",
    "income_statement_growth": "fundamental growth improving",
    "key_metrics": "quality metrics are strengthening",
    "ratios": "valuation and efficiency metrics are supportive",
    "model_signals": "internal model ranking remains supportive",
    "novelty": "setup is historically familiar",
    "oracle_cluster": "state resembles a historically profitable cluster",
}


def _clamp_unit(value: float | None, *, low: float = 0.0, high: float = 1.0) -> float:
    parsed = float(value or 0.0)
    return max(low, min(high, parsed))


def _return_expectation_to_unit(value: float | None) -> float:
    if value is None:
        return 0.5
    return _clamp_unit((float(value) + 0.25) / 0.50)


def _text_band(score: float, *, labels: Sequence[tuple[float, str]]) -> str:
    for threshold, label in labels:
        if score >= threshold:
            return label
    return labels[-1][1]


def _ranking_column(row: dict[str, Any]) -> str:
    for column in ("ranking", "strategy_score", "combined_score", "signal_score", "prediction_score"):
        if column in row:
            return column
    return ""


def _familiarity_column(row: dict[str, Any]) -> str:
    for column in ("ae_familiarity", "mtl_cluster_confidence", "prediction_score"):
        if column in row:
            return column
    return ""


def compute_opportunity_summary(
    *,
    row: dict[str, Any],
    state_frame: pd.DataFrame,
    outcome_summary: dict[str, Any],
    similarity_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    row_dict = dict(row)
    ranking_col = _ranking_column(row_dict)
    familiarity_col = _familiarity_column(row_dict)
    ranking_unit = percentile_rank(state_frame[ranking_col], float(row_dict.get(ranking_col) or 0.0)) if ranking_col else 0.5
    familiarity_unit = percentile_rank(state_frame[familiarity_col], float(row_dict.get(familiarity_col) or 0.0)) if familiarity_col else 0.5
    historical_unit = _return_expectation_to_unit(outcome_summary.get("median_return"))
    opportunity_score = round((0.5 * ranking_unit + 0.3 * historical_unit + 0.2 * familiarity_unit) * 100.0, 2)

    mean_similarity = 0.0
    if similarity_rows:
        mean_similarity = sum(float(item.get("similarity_score") or 0.0) for item in similarity_rows) / float(len(similarity_rows))
    dispersion_penalty = min(1.0, abs(float(outcome_summary.get("best_case") or 0.0) - float(outcome_summary.get("worst_case") or 0.0)))
    confidence_score = round((0.55 * mean_similarity + 0.25 * familiarity_unit + 0.20 * (1.0 - dispersion_penalty)) * 100.0, 2)
    risk_tail = abs(float(outcome_summary.get("worst_case") or 0.0))
    risk_drawdown = abs(float(outcome_summary.get("avg_drawdown") or 0.0))
    risk_score = round(min(100.0, (0.65 * risk_tail + 0.35 * risk_drawdown) * 200.0), 2)

    return {
        "opportunity_score": opportunity_score,
        "confidence_score": confidence_score,
        "confidence_label": _text_band(confidence_score, labels=((80.0, "High"), (65.0, "Medium-High"), (50.0, "Medium"), (0.0, "Low"))),
        "market_familiarity_score": round(familiarity_unit * 100.0, 2),
        "market_familiarity_label": _text_band(familiarity_unit * 100.0, labels=((75.0, "High"), (55.0, "Medium-High"), (40.0, "Medium"), (0.0, "Low"))),
        "risk_score": risk_score,
        "risk_indicator": _text_band(100.0 - risk_score, labels=((75.0, "Low"), (55.0, "Moderate"), (35.0, "Elevated"), (0.0, "High"))),
        "ranking_percentile": round(ranking_unit * 100.0, 2),
        "historical_expectation_percentile": round(historical_unit * 100.0, 2),
        "familiarity_percentile": round(familiarity_unit * 100.0, 2),
    }


def explain_key_drivers(
    *,
    row: dict[str, Any],
    state_frame: pd.DataFrame,
    embedding_columns: Sequence[str],
    limit: int = 3,
) -> list[dict[str, Any]]:
    if state_frame.empty or not embedding_columns:
        return []
    numeric_frame = state_frame[list(embedding_columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    row_series = pd.Series(row)
    means = numeric_frame.mean(axis=0)
    stds = numeric_frame.std(axis=0, ddof=0).replace(0.0, 1.0)
    zscores = ((pd.to_numeric(row_series[list(embedding_columns)], errors="coerce").fillna(0.0) - means) / stds).fillna(0.0)
    family_map = feature_family_map_from_columns(embedding_columns)
    rows: list[dict[str, Any]] = []
    for family, columns in family_map.items():
        if not columns:
            continue
        family_values = zscores[columns].abs()
        contribution = float(family_values.mean()) if not family_values.empty else 0.0
        if contribution <= 0:
            continue
        rows.append(
            {
                "family": family,
                "contribution": round(contribution, 4),
                "explanation": FAMILY_EXPLANATIONS.get(family, f"{family.replace('_', ' ')} is unusually active"),
            }
        )
    rows.sort(key=lambda item: float(item.get("contribution") or 0.0), reverse=True)
    return rows[: max(int(limit), 1)]
