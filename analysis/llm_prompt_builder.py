from __future__ import annotations

from typing import Sequence

from .market_insight_schema import MarketInsightInput, MarketSituationExplanation, PortfolioInsightInput, StockInsight


def _feature_sections_text(insight_input: MarketInsightInput, *, max_features_per_family: int = 5) -> str:
    sections: list[str] = []
    for family_name, rows in insight_input.canonical_features.items():
        sections.append(str(family_name))
        for feature in list(rows or [])[: max(int(max_features_per_family), 1)]:
            sections.append(f"{feature.display_name}: {feature.rendered_value}")
        sections.append("")
    return "\n".join(sections).strip()


def _analog_section_text(label: str, rows: Sequence[dict], *, max_rows: int = 5) -> str:
    lines = [label]
    for row in list(rows or [])[: max(int(max_rows), 1)]:
        parts = [
            str(row.get("symbol") or ""),
            str(row.get("date") or ""),
            f"similarity={row.get('similarity_score')}",
        ]
        if row.get("returns_by_horizon"):
            parts.append(f"returns={row.get('returns_by_horizon')}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def build_stock_insight_prompt(insight_input: MarketInsightInput, stock_insight: StockInsight | None = None) -> str:
    deterministic_summary = stock_insight.summary if stock_insight is not None else ""
    deterministic_headline = stock_insight.headline if stock_insight is not None else ""
    return "\n\n".join(
        [
            "Context",
            f"Symbol: {insight_input.symbol}\nAs Of: {insight_input.as_of_date}",
            "Canonical Features",
            _feature_sections_text(insight_input),
            "Historical Analogs",
            _analog_section_text("Same-Symbol", [item.to_dict() for item in insight_input.same_symbol_analogs])
            + "\n\n"
            + _analog_section_text("Cross-Symbol", [item.to_dict() for item in insight_input.cross_symbol_analogs]),
            "Outcome Summary",
            str(insight_input.analog_outcome_summary.to_dict()),
            "Model Signals",
            "\n".join(f"{item.display_name}: {item.rendered_value}" for item in insight_input.model_scores),
            "Familiarity",
            str(insight_input.familiarity_signals.to_dict()),
            "Cluster Context",
            str(insight_input.cluster_context.to_dict()),
            "Deterministic Draft",
            f"Headline: {deterministic_headline}\nSummary: {deterministic_summary}",
            "Task",
            "Write a concise market-intelligence explanation using only the structured evidence above.",
            "Constraints",
            "\n".join(
                [
                    "- Do not give trading advice.",
                    "- Do not use BUY or SELL language.",
                    "- Do not invent hidden facts.",
                    "- Preserve canonical feature names exactly as written.",
                    "- If evidence is mixed, say it is mixed.",
                    "- Keep the explanation grounded in historical analogs, feature summaries, model alignment, familiarity, and risk context.",
                ]
            ),
        ]
    )


def build_portfolio_insight_prompt(portfolio_input: PortfolioInsightInput, portfolio_summary: str = "") -> str:
    holdings_text = "\n".join(
        f"{item.symbol}: opportunity={item.opportunity_score}, confidence={item.confidence_score}, familiarity={item.familiarity_score}, risk={item.risk_indicator}"
        for item in portfolio_input.holdings
    )
    return "\n\n".join(
        [
            "Context",
            f"Symbols: {', '.join(portfolio_input.symbols)}\nAs Of: {portfolio_input.as_of_date}",
            "Holdings",
            holdings_text,
            "Cluster Exposure",
            str(portfolio_input.cluster_exposure_rows),
            "Portfolio Scores",
            str(
                {
                    "portfolio_score": portfolio_input.portfolio_score,
                    "regime_similarity_score": portfolio_input.regime_similarity_score,
                    "risk_concentration_score": portfolio_input.risk_concentration_score,
                }
            ),
            "Deterministic Draft",
            portfolio_summary,
            "Task",
            "Write a concise portfolio-intelligence explanation using only the structured evidence above.",
            "Constraints",
            "\n".join(
                [
                    "- Do not give trading advice.",
                    "- Do not invent hidden facts.",
                    "- Keep the explanation grounded in holdings, cluster exposure, familiarity, and risk concentration.",
                ]
            ),
        ]
    )


def build_market_situation_prompt(insight_input: MarketInsightInput, explanation: MarketSituationExplanation | None = None) -> str:
    deterministic_summary = explanation.summary if explanation is not None else ""
    return "\n\n".join(
        [
            "Context",
            f"Symbol: {insight_input.symbol}\nAs Of: {insight_input.as_of_date}",
            "Current Situation Cluster",
            str(insight_input.cluster_context.to_dict()),
            "Historical Analogs",
            _analog_section_text("Same-Symbol", [item.to_dict() for item in insight_input.same_symbol_analogs])
            + "\n\n"
            + _analog_section_text("Cross-Symbol", [item.to_dict() for item in insight_input.cross_symbol_analogs]),
            "Outcome Summary",
            str(insight_input.analog_outcome_summary.to_dict()),
            "Deterministic Draft",
            deterministic_summary,
            "Task",
            "Explain the current market situation in plain language using only the evidence above.",
            "Constraints",
            "\n".join(
                [
                    "- Do not give trading advice.",
                    "- Do not invent hidden facts.",
                    "- Use the cluster context and historical analogs as evidence.",
                ]
            ),
        ]
    )
