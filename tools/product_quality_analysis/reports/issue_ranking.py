from __future__ import annotations

from ..models import RankedIssue, Severity


SEVERITY_WEIGHT = {
    Severity.CRITICAL: 5,
    Severity.HIGH: 4,
    Severity.MEDIUM: 2.5,
    Severity.LOW: 1,
}


def rank_issues(issues: list[RankedIssue]) -> list[RankedIssue]:
    ranked: list[RankedIssue] = []
    for issue in issues:
        severity = SEVERITY_WEIGHT.get(issue.severity, 1)
        trust = float(issue.metadata.get("trust_impact", 3))
        frequency = float(issue.metadata.get("frequency", 3))
        scalability = float(issue.metadata.get("scalability_risk", 2))
        usability = float(issue.metadata.get("usability_impact", 3))
        feasibility = float(issue.metadata.get("implementation_feasibility", 3))
        score = 7.0 * severity + 5.0 * trust + 4.0 * frequency + 4.0 * scalability + 4.0 * usability + 3.0 * feasibility
        ranked.append(issue.model_copy(update={"score": round(score * float(issue.confidence or 1.0), 2)}))
    return sorted(ranked, key=lambda item: item.score, reverse=True)


def prioritized_issues_markdown(issues: list[RankedIssue]) -> str:
    lines = ["# Prioritized Issues", ""]
    if not issues:
        lines.extend(["No issues were detected.", ""])
        return "\n".join(lines)
    for index, issue in enumerate(issues, start=1):
        lines.append(f"## {index}. {issue.title}")
        lines.append("")
        lines.append(f"- Severity: `{issue.severity.value}`")
        lines.append(f"- Score: `{issue.score}`")
        lines.append(f"- Category: `{issue.category}`")
        if issue.page:
            lines.append(f"- Page: `{issue.page}`")
        if issue.metric_name:
            lines.append(f"- Metric: `{issue.metric_name}` = `{issue.metric_value}`")
        if issue.recommendation:
            lines.append(f"- Recommendation: {issue.recommendation}")
        if issue.evidence:
            lines.append("- Evidence:")
            lines.extend(f"  - {item}" for item in issue.evidence)
        lines.append("")
    return "\n".join(lines)
