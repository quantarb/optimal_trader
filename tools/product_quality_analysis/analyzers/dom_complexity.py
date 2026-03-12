from __future__ import annotations

from ..models import AnalysisConfig, PageSnapshot, RankedIssue, Severity


def analyze_dom_complexity(config: AnalysisConfig, page_snapshots: list[PageSnapshot]) -> dict:
    issues: list[RankedIssue] = []
    rows: list[dict] = []
    for snapshot in page_snapshots:
        rows.append(
            {
                "page": snapshot.name,
                "dom_node_count": snapshot.dom_node_count,
                "interactive_count": snapshot.interactive_count,
                "card_count": snapshot.card_count,
                "chart_like_count": snapshot.chart_like_count,
            }
        )
        if snapshot.dom_node_count > config.dom_critical_threshold:
            issues.append(
                RankedIssue(
                    issue_id=f"dom-critical:{snapshot.name}",
                    title=f"{snapshot.name} exceeds the critical DOM-complexity threshold",
                    severity=Severity.CRITICAL,
                    score=0.0,
                    page=snapshot.name,
                    category="dom_complexity",
                    recommendation="Reduce the first paint surface with paging, lazy sections, or fewer repeated cards.",
                    evidence=[f"DOM nodes: {snapshot.dom_node_count}", f"Interactive elements: {snapshot.interactive_count}"],
                    metric_name="dom_node_count",
                    metric_value=snapshot.dom_node_count,
                    metadata={"trust_impact": 3, "frequency": 4, "scalability_risk": 5, "usability_impact": 4, "implementation_feasibility": 4},
                )
            )
    return {"pages": rows, "issues": [issue.model_dump(mode="json") for issue in issues]}
