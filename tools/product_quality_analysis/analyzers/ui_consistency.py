from __future__ import annotations

from statistics import mean

from ..models import PageSnapshot, RankedIssue, Severity


def analyze_ui_consistency(page_snapshots: list[PageSnapshot]) -> dict:
    issues: list[RankedIssue] = []
    if not page_snapshots:
        return {"summary": {}, "issues": []}
    color_counts = [snapshot.unique_colors_used for snapshot in page_snapshots]
    font_counts = [snapshot.unique_font_sizes for snapshot in page_snapshots]
    spacing_counts = [snapshot.unique_spacing_values for snapshot in page_snapshots]
    shell_rate = float(sum(1 for snapshot in page_snapshots if "app-shell" in snapshot.layout_signature) / len(page_snapshots))
    layout_variants = sorted({tuple(snapshot.layout_signature) for snapshot in page_snapshots})
    pagination_consistency_rate = float(
        sum(
            1
            for snapshot in page_snapshots
            if not snapshot.table_metrics or all(table.has_pagination for table in snapshot.table_metrics if table.row_count > 0)
        )
        / len(page_snapshots)
    )
    design_token_variance = round((max(color_counts) - min(color_counts)) + (max(font_counts) - min(font_counts)), 4)
    if shell_rate < 0.7 or len(layout_variants) > 4:
        issues.append(
            RankedIssue(
                issue_id="ui-layout-fragmentation",
                title="Major pages do not share a coherent shell or layout pattern",
                severity=Severity.HIGH,
                score=0.0,
                page="global",
                category="ui_consistency",
                recommendation="Normalize the app shell, page header, and filter-bar patterns so adjacent routes feel related.",
                evidence=[f"Shared shell rate: {shell_rate:.0%}", f"Layout variants: {len(layout_variants)}"],
                metric_name="layout_variants",
                metric_value=len(layout_variants),
                metadata={"trust_impact": 4, "frequency": 5, "scalability_risk": 2, "usability_impact": 4, "implementation_feasibility": 3},
            )
        )
    return {
        "summary": {
            "unique_colors_used": round(mean(color_counts), 2),
            "unique_font_sizes": round(mean(font_counts), 2),
            "unique_spacing_values": round(mean(spacing_counts), 2),
            "component_reuse_rate": round(shell_rate, 4),
            "layout_variants": len(layout_variants),
            "pagination_consistency_rate": round(pagination_consistency_rate, 4),
            "design_token_variance": design_token_variance,
        },
        "issues": [issue.model_dump(mode="json") for issue in issues],
    }
