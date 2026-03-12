from __future__ import annotations

from ..models import AnalysisSnapshot


def quality_summary_markdown(snapshot: AnalysisSnapshot) -> str:
    lines = ["# Product Quality Summary", ""]
    lines.append(f"- Label: `{snapshot.label}`")
    lines.append(f"- Generated At: `{snapshot.generated_at}`")
    lines.append(f"- Crawled Pages: `{len(snapshot.page_snapshots)}`")
    lines.append(f"- Issues: `{len(snapshot.issues)}`")
    lines.append("")
    if snapshot.issues:
        lines.append("## Top Issues")
        lines.append("")
        for issue in snapshot.issues[:5]:
            lines.append(f"- `{issue.severity.value}` `{issue.page or 'global'}` {issue.title} (score {issue.score})")
        lines.append("")
    lines.append("## Baseline Metrics")
    lines.append("")
    lines.append(f"- Tier-1 symbol validation rate: `{snapshot.data_quality.get('tier1_symbol_validation_rate', '-')}`")
    lines.append(f"- Pagination presence rate: `{snapshot.pagination.get('pagination_presence_rate', '-')}`")
    lines.append(f"- Empty section rate: `{snapshot.empty_states.get('empty_section_rate', '-')}`")
    lines.append("")
    return "\n".join(lines)
