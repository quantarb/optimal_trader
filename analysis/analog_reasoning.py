from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from .market_insight_schema import HistoricalAnalog


def _primary_return(analog: HistoricalAnalog, horizon_days: int = 60) -> float | None:
    return analog.returns_by_horizon.get(f"return_{int(horizon_days)}d")


def _analog_summary(analogs: Sequence[HistoricalAnalog], *, label: str, primary_horizon_days: int = 60) -> dict[str, Any]:
    rows = list(analogs or [])
    returns = [value for value in (_primary_return(item, primary_horizon_days) for item in rows) if value is not None]
    positive_count = sum(1 for value in returns if float(value) > 0.0)
    summary_lines: list[str] = []
    if rows:
        summary_lines.append(
            f"{label} analogs were favorable in {positive_count} of {len(returns) or len(rows)} comparable {primary_horizon_days}-day outcomes."
        )
        if returns:
            summary_lines.append(
                f"Typical {primary_horizon_days}-day outcomes ranged from {min(returns):+.2%} to {max(returns):+.2%}, with a median of {sorted(returns)[len(returns) // 2]:+.2%}."
            )
    top_examples = [
        {
            "symbol": item.symbol,
            "date": item.date,
            "similarity_score": item.similarity_score,
            f"return_{int(primary_horizon_days)}d": _primary_return(item, primary_horizon_days),
        }
        for item in rows[:3]
    ]
    return {
        "label": label,
        "analog_count": len(rows),
        "comparable_count": len(returns),
        "positive_count": positive_count,
        "positive_rate": float(positive_count / max(len(returns), 1)) if returns else None,
        "summary_lines": summary_lines,
        "evidence": {
            "top_examples": top_examples,
            "returns": returns,
            "primary_horizon_days": int(primary_horizon_days),
        },
    }


def summarize_same_symbol_analogs(analogs: Sequence[HistoricalAnalog], primary_horizon_days: int = 60) -> dict[str, Any]:
    return _analog_summary(analogs, label="Same-symbol", primary_horizon_days=primary_horizon_days)


def summarize_cross_symbol_analogs(analogs: Sequence[HistoricalAnalog], primary_horizon_days: int = 60) -> dict[str, Any]:
    return _analog_summary(analogs, label="Cross-symbol", primary_horizon_days=primary_horizon_days)


def summarize_analog_outcomes(analogs: Sequence[HistoricalAnalog], primary_horizon_days: int = 60) -> dict[str, Any]:
    rows = list(analogs or [])
    returns = [value for value in (_primary_return(item, primary_horizon_days) for item in rows) if value is not None]
    if not returns:
        return {
            "summary_lines": ["Historical analog outcomes are too sparse to summarize confidently."],
            "evidence": {"analog_count": len(rows), "comparable_count": 0},
        }
    sorted_returns = sorted(returns)
    mid_idx = len(sorted_returns) // 2
    median = sorted_returns[mid_idx]
    summary_lines = [
        f"Historical analog outcomes lean {'favorable' if median > 0 else 'mixed'}, with a median {primary_horizon_days}-day return of {median:+.2%}.",
        f"The observed range runs from {min(sorted_returns):+.2%} to {max(sorted_returns):+.2%}.",
    ]
    return {
        "summary_lines": summary_lines,
        "evidence": {
            "analog_count": len(rows),
            "comparable_count": len(sorted_returns),
            "median_return": median,
            "worst_case": min(sorted_returns),
            "best_case": max(sorted_returns),
        },
    }


def extract_consistent_patterns_from_analogs(analogs: Sequence[HistoricalAnalog], limit: int = 3) -> dict[str, Any]:
    rows = list(analogs or [])
    tag_counter: Counter[str] = Counter()
    cluster_counter: Counter[str] = Counter()
    for analog in rows:
        tag_counter.update(tag for tag in analog.explanation_tags if str(tag).strip())
        if analog.cluster_description:
            cluster_counter.update([analog.cluster_description])
    patterns: list[str] = []
    evidence_rows: list[dict[str, Any]] = []
    for value, count in tag_counter.most_common(max(int(limit), 1)):
        patterns.append(f"Repeated analog matches reference {value.lower()}.")
        evidence_rows.append({"pattern": value, "count": count, "source": "explanation_tag"})
    for value, count in cluster_counter.most_common(max(int(limit), 1)):
        if len(patterns) >= max(int(limit), 1):
            break
        patterns.append(f"Several analogs fall into {value.lower()}.")
        evidence_rows.append({"pattern": value, "count": count, "source": "cluster_description"})
    return {
        "summary_lines": patterns,
        "evidence": {"patterns": evidence_rows},
    }
