from __future__ import annotations

from ..models import PageSnapshot, RankedIssue, Severity


def analyze_empty_states(page_snapshots: list[PageSnapshot]) -> dict:
    issues: list[RankedIssue] = []
    pages: list[dict] = []
    for snapshot in page_snapshots:
        marker_count = len(snapshot.empty_markers) + len(snapshot.error_markers)
        pages.append({"page": snapshot.name, "empty_markers": snapshot.empty_markers, "error_markers": snapshot.error_markers, "response_error": snapshot.response_error})
        if snapshot.response_error:
            issues.append(
                RankedIssue(
                    issue_id=f"page-load-failure:{snapshot.name}",
                    title=f"{snapshot.name} did not complete a stable render",
                    severity=Severity.CRITICAL,
                    score=0.0,
                    page=snapshot.name,
                    category="empty_state",
                    recommendation="Reduce route work or provide a partial render path instead of blocking the page until all analytics finish.",
                    evidence=[snapshot.response_error, *snapshot.console_errors[:3]],
                    metric_name="empty_section_rate",
                    metric_value=1.0,
                    metadata={"trust_impact": 5, "frequency": 4, "scalability_risk": 5, "usability_impact": 5, "implementation_feasibility": 3},
                )
            )
        elif marker_count and "fmp_universe_screener" not in snapshot.name:
            issues.append(
                RankedIssue(
                    issue_id=f"empty-state:{snapshot.name}",
                    title=f"{snapshot.name} contains visible empty or broken sections",
                    severity=Severity.HIGH if snapshot.error_markers else Severity.MEDIUM,
                    score=0.0,
                    page=snapshot.name,
                    category="empty_state",
                    recommendation="Replace passive placeholders with a recovery path or pre-populated defaults.",
                    evidence=[*(snapshot.error_markers[:3] or snapshot.empty_markers[:3])],
                    metric_name="empty_section_rate",
                    metric_value=round(min(1.0, marker_count / 10.0), 4),
                    metadata={"trust_impact": 4, "frequency": 4, "scalability_risk": 2, "usability_impact": 4, "implementation_feasibility": 4},
                )
            )
    empty_section_rate = round(sum(1 for page in pages if page["empty_markers"] or page["error_markers"]) / max(1, len(pages)), 4)
    return {"pages": pages, "empty_section_rate": empty_section_rate, "issues": [issue.model_dump(mode="json") for issue in issues]}
