from __future__ import annotations

from ..models import AnalysisSnapshot


def scalability_report_markdown(snapshot: AnalysisSnapshot) -> str:
    rows = list(snapshot.scalability.get("tiers") or [])
    lines = ["# Scalability Report", ""]
    if not rows:
        lines.extend(["No tiered scalability data was collected.", ""])
        return "\n".join(lines)
    lines.append("| Route | Tier1 Load (ms) | Tier2 Load (ms) | Tier3 Load (ms) | Load Growth | Tier2 DOM | Tier3 DOM | Usability |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(f"| {row.get('route')} | {row.get('tier1_load_time_ms')} | {row.get('tier2_load_time_ms')} | {row.get('tier3_load_time_ms')} | {row.get('load_growth_rate')} | {row.get('tier2_dom_nodes')} | {row.get('tier3_dom_nodes')} | {row.get('ui_usability_score_by_tier')} |")
    lines.append("")
    return "\n".join(lines)
