from __future__ import annotations

from typing import Any

from .analog_reasoning import (
    extract_consistent_patterns_from_analogs,
    summarize_analog_outcomes,
    summarize_cross_symbol_analogs,
    summarize_same_symbol_analogs,
)
from .familiarity_reasoning import summarize_familiarity, summarize_novelty_risk
from .feature_reasoning import summarize_feature_extremes, summarize_feature_family
from .llm_prompt_builder import build_market_situation_prompt, build_portfolio_insight_prompt, build_stock_insight_prompt
from .market_insight_schema import (
    MarketInsightInput,
    MarketSituationExplanation,
    PortfolioInsight,
    PortfolioInsightInput,
    StockInsight,
)
from .model_reasoning import detect_signal_conflicts, summarize_model_scores, summarize_signal_alignment


def _score_band(score: float | None) -> str:
    value = float(score or 0.0)
    if value >= 75.0:
        return "high"
    if value >= 55.0:
        return "moderate-to-high"
    if value >= 40.0:
        return "moderate"
    return "low"


def build_stock_insight(insight_input: MarketInsightInput, mode: str = "deterministic") -> StockInsight:
    same_symbol_summary = summarize_same_symbol_analogs(
        insight_input.same_symbol_analogs,
        primary_horizon_days=insight_input.same_symbol_outcome_summary.primary_horizon_days,
    )
    cross_symbol_summary = summarize_cross_symbol_analogs(
        insight_input.cross_symbol_analogs,
        primary_horizon_days=insight_input.cross_symbol_outcome_summary.primary_horizon_days,
    )
    analog_summary = summarize_analog_outcomes(
        list(insight_input.same_symbol_analogs) + list(insight_input.cross_symbol_analogs),
        primary_horizon_days=insight_input.analog_outcome_summary.primary_horizon_days,
    )
    analog_patterns = extract_consistent_patterns_from_analogs(
        list(insight_input.same_symbol_analogs) + list(insight_input.cross_symbol_analogs)
    )
    model_summary = summarize_model_scores(insight_input.model_scores)
    alignment_summary = summarize_signal_alignment(
        insight_input.model_scores,
        same_symbol_summary,
        insight_input.familiarity_signals,
    )
    conflict_summary = detect_signal_conflicts(
        insight_input.model_scores,
        cross_symbol_summary,
        insight_input.familiarity_signals,
    )
    familiarity_summary = summarize_familiarity(insight_input.familiarity_signals)
    novelty_summary = summarize_novelty_risk(
        insight_input.familiarity_signals,
        analog_density=insight_input.familiarity_signals.analog_density,
    )
    family_summaries = [
        summarize_feature_family(family_name, features)
        for family_name, features in insight_input.canonical_features.items()
    ]
    family_summaries = [item for item in family_summaries if list(item.get("summary_lines") or [])]
    family_summaries.sort(
        key=lambda item: max(
            [
                abs(float((feature.get("zscore") or 0.0)))
                for feature in list((item.get("evidence") or {}).get("top_features") or [])
            ]
            or [0.0]
        ),
        reverse=True,
    )
    key_drivers = []
    for item in family_summaries[:3]:
        family_name = str(item.get("family") or "")
        summary_lines = list(item.get("summary_lines") or [])
        if summary_lines:
            key_drivers.append(f"{family_name}: {summary_lines[0]}")
    key_drivers.extend(list(analog_patterns.get("summary_lines") or [])[:2])
    if not key_drivers:
        key_drivers = list(model_summary.get("summary_lines") or [])[:3]

    historical_context = list(same_symbol_summary.get("summary_lines") or [])[:2] + list(cross_symbol_summary.get("summary_lines") or [])[:2]
    opportunity_context = list(analog_summary.get("summary_lines") or [])[:1] + list(alignment_summary.get("summary_lines") or [])[:2]
    risk_flags = list(conflict_summary.get("summary_lines") or []) + list(novelty_summary.get("summary_lines") or [])
    if insight_input.analog_outcome_summary.worst_case is not None:
        risk_flags.insert(
            0,
            f"Historical downside at the primary horizon reached {float(insight_input.analog_outcome_summary.worst_case):+.2%} in the weaker analogs.",
        )
    headline = (
        f"Historical analogs lean {('favorable' if (insight_input.analog_outcome_summary.median_return or 0.0) > 0 else 'mixed')}, "
        f"with {_score_band(next((item.value for item in insight_input.model_scores if item.name == 'opportunity_score'), None))} opportunity context."
    )
    summary = " ".join(
        [
            *(list(analog_summary.get("summary_lines") or [])[:1]),
            *(list(alignment_summary.get("summary_lines") or [])[:1]),
            *(list(familiarity_summary.get("summary_lines") or [])[:1]),
        ]
    ).strip()
    evidence = {
        "same_symbol_summary": same_symbol_summary,
        "cross_symbol_summary": cross_symbol_summary,
        "analog_summary": analog_summary,
        "feature_family_summaries": family_summaries,
        "feature_extremes": summarize_feature_extremes([feature for rows in insight_input.canonical_features.values() for feature in rows]),
        "model_summary": model_summary,
        "alignment_summary": alignment_summary,
        "conflict_summary": conflict_summary,
        "familiarity_summary": familiarity_summary,
        "novelty_summary": novelty_summary,
        "cluster_context": insight_input.cluster_context.to_dict(),
    }
    insight = StockInsight(
        headline=headline,
        summary=summary,
        key_drivers=key_drivers,
        historical_context=historical_context,
        opportunity_context=opportunity_context,
        risk_flags=risk_flags,
        familiarity_comment=list(familiarity_summary.get("summary_lines") or [""])[0],
        confidence_comment=f"Confidence is {insight_input.familiarity_signals.confidence_label.lower() or 'unspecified'} based on analog consistency and familiarity.",
        supporting_evidence=evidence,
        mode=str(mode or "deterministic"),
    )
    if str(mode or "deterministic").lower() != "deterministic":
        return StockInsight(
            **{
                **insight.to_dict(),
                "llm_prompt": build_stock_insight_prompt(insight_input, insight),
            }
        )
    return insight


def build_portfolio_insight(portfolio_input: PortfolioInsightInput, mode: str = "deterministic") -> PortfolioInsight:
    holdings = list(portfolio_input.holdings or [])
    strongest = [item.symbol for item in sorted(holdings, key=lambda row: float(row.opportunity_score or 0.0), reverse=True)[:3]]
    weakest = [item.symbol for item in sorted(holdings, key=lambda row: float(row.opportunity_score or 0.0))[:3]]
    concentration_summary = [
        f"{row.get('cluster_description') or row.get('cluster_id')}: {float(row.get('exposure_pct') or 0.0):.1f}%"
        for row in list(portfolio_input.cluster_exposure_rows or [])[:3]
    ]
    risk_flags: list[str] = []
    if float(portfolio_input.risk_concentration_score or 0.0) >= 55.0:
        risk_flags.append("Risk concentration is elevated because familiarity is uneven across the holdings.")
    if concentration_summary:
        risk_flags.append("Cluster exposure is concentrated in a small number of opportunity regimes.")
    headline = (
        f"The portfolio shows {_score_band(portfolio_input.portfolio_score)} opportunity context, "
        f"with {_score_band(portfolio_input.risk_concentration_score)} concentration risk."
    )
    summary = (
        f"Average opportunity context across the holdings is {float(portfolio_input.portfolio_score or 0.0):.1f}, "
        f"while regime similarity is {float(portfolio_input.regime_similarity_score or 0.0):.1f}."
    )
    insight = PortfolioInsight(
        headline=headline,
        summary=summary,
        strongest_holdings=strongest,
        weakest_holdings=weakest,
        concentration_summary=concentration_summary,
        risk_flags=risk_flags,
        supporting_evidence=portfolio_input.to_dict(),
        mode=str(mode or "deterministic"),
    )
    if str(mode or "deterministic").lower() != "deterministic":
        return PortfolioInsight(
            **{
                **insight.to_dict(),
                "llm_prompt": build_portfolio_insight_prompt(portfolio_input, summary),
            }
        )
    return insight


def build_market_situation_explanation(insight_input: MarketInsightInput, mode: str = "deterministic") -> MarketSituationExplanation:
    cluster = insight_input.cluster_context
    same_summary = summarize_same_symbol_analogs(insight_input.same_symbol_analogs)
    cross_summary = summarize_cross_symbol_analogs(insight_input.cross_symbol_analogs)
    situation_context = []
    if cluster.cluster_id:
        situation_context.append(f"Current situation cluster {cluster.cluster_id} is described as {cluster.description}.")
    situation_context.extend(list(same_summary.get("summary_lines") or [])[:1])
    situation_context.extend(list(cross_summary.get("summary_lines") or [])[:1])
    summary = " ".join(situation_context).strip()
    explanation = MarketSituationExplanation(
        headline="Current market state mapped into a learned historical situation family.",
        summary=summary,
        situation_context=situation_context,
        supporting_evidence={
            "cluster_context": cluster.to_dict(),
            "same_symbol_summary": same_summary,
            "cross_symbol_summary": cross_summary,
        },
        mode=str(mode or "deterministic"),
    )
    if str(mode or "deterministic").lower() != "deterministic":
        return MarketSituationExplanation(
            **{
                **explanation.to_dict(),
                "llm_prompt": build_market_situation_prompt(insight_input, explanation),
            }
        )
    return explanation
