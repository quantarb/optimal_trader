from __future__ import annotations

from ..models import PageSnapshot, RankedIssue, Severity


def analyze_table_readability(page_snapshots: list[PageSnapshot]) -> dict:
    issues: list[RankedIssue] = []
    rows: list[dict] = []
    for snapshot in page_snapshots:
        for table in snapshot.table_metrics:
            rows.append(
                {
                    "page": snapshot.name,
                    "table": table.identifier,
                    "rows": table.row_count,
                    "columns": table.column_count,
                    "risk": table.readability_risk_score,
                }
            )
            if table.readability_risk_score >= 4.0:
                issues.append(
                    RankedIssue(
                        issue_id=f"readability-critical:{snapshot.name}:{table.identifier}",
                        title=f"{snapshot.name} table is visually overloaded",
                        severity=Severity.CRITICAL,
                        score=0.0,
                        page=snapshot.name,
                        category="readability",
                        recommendation="Trim columns, lower the visible row count, or split the dataset into focused tabs.",
                        evidence=[f"Rows: {table.row_count}", f"Columns: {table.column_count}", f"Risk score: {table.readability_risk_score}"],
                        metric_name="readability_risk_score",
                        metric_value=table.readability_risk_score,
                        metadata={"trust_impact": 3, "frequency": 3, "scalability_risk": 4, "usability_impact": 5, "implementation_feasibility": 4},
                    )
                )
    average_risk = round(sum(item["risk"] for item in rows) / max(1, len(rows)), 4) if rows else 0.0
    return {"tables": rows, "average_risk": average_risk, "issues": [issue.model_dump(mode="json") for issue in issues]}
