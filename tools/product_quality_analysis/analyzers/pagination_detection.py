from __future__ import annotations

from ..models import AnalysisConfig, PageSnapshot, RankedIssue, Severity


def analyze_pagination(config: AnalysisConfig, page_snapshots: list[PageSnapshot]) -> dict:
    issues: list[RankedIssue] = []
    table_rows: list[dict] = []
    paginated_tables = 0
    large_tables = 0
    for snapshot in page_snapshots:
        for table in snapshot.table_metrics:
            large = table.row_count > config.table_warning_threshold or table.readability_risk_score >= 1.5
            if large:
                large_tables += 1
            if table.has_pagination:
                paginated_tables += 1
            table_rows.append(
                {
                    "page": snapshot.name,
                    "table": table.identifier,
                    "rows": table.row_count,
                    "columns": table.column_count,
                    "has_pagination": table.has_pagination,
                    "page_size": table.page_size,
                    "readability_risk_score": table.readability_risk_score,
                }
            )
            if table.row_count > config.table_critical_threshold and not table.has_pagination:
                issues.append(
                    RankedIssue(
                        issue_id=f"pagination-missing:{snapshot.name}:{table.identifier}",
                        title=f"{snapshot.name} renders a very large table without pagination",
                        severity=Severity.CRITICAL,
                        score=0.0,
                        page=snapshot.name,
                        category="pagination",
                        recommendation="Add paging or virtualization before rendering hundreds of rows into the DOM.",
                        evidence=[f"Rows: {table.row_count}", f"Columns: {table.column_count}"],
                        metric_name="pagination_presence_rate",
                        metric_value=0.0,
                        metadata={"trust_impact": 4, "frequency": 4, "scalability_risk": 5, "usability_impact": 5, "implementation_feasibility": 4},
                    )
                )
    pagination_presence_rate = float(paginated_tables / max(1, large_tables)) if large_tables else 1.0
    return {
        "tables": table_rows,
        "pagination_presence_rate": round(pagination_presence_rate, 4),
        "issues": [issue.model_dump(mode="json") for issue in issues],
    }
