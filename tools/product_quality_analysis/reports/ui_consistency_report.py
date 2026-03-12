from __future__ import annotations

from ..models import AnalysisSnapshot


def ui_consistency_report_markdown(snapshot: AnalysisSnapshot) -> str:
    summary = dict(snapshot.ui_consistency.get("summary") or {})
    lines = ["# UI Consistency Report", ""]
    if not summary:
        lines.extend(["No UI-consistency summary was generated.", ""])
        return "\n".join(lines)
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    for key in (
        "unique_colors_used",
        "unique_font_sizes",
        "unique_spacing_values",
        "component_reuse_rate",
        "layout_variants",
        "pagination_consistency_rate",
        "design_token_variance",
    ):
        lines.append(f"| {key} | {summary.get(key, '-')} |")
    lines.append("")
    return "\n".join(lines)
