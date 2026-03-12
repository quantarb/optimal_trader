from __future__ import annotations

from ..models import MemoryProfileReport, RuntimeProfileReport
from ..utils.report_utils import markdown_table


def runtime_hotspots_markdown(report: RuntimeProfileReport) -> str:
    lines = ["# Runtime Hotspots", "", f"- Engine: `{report.engine}`", f"- Target: `{report.target}`", f"- Total runtime: `{report.total_seconds:.3f}s`", ""]
    if report.stage_hotspots:
        lines.extend(["## Stage Hotspots", "", markdown_table(["Stage", "Wall (s)", "CPU (s)", "RSS Delta MB"], [(row.get("name", ""), f"{float(row.get('wall_seconds') or 0.0):.3f}", f"{float(row.get('cpu_seconds') or 0.0):.3f}", f"{float(row.get('rss_delta_mb') or 0.0):.3f}") for row in report.stage_hotspots])])
    if report.hotspots:
        lines.extend(["", "## Function Hotspots", "", markdown_table(["Path", "Line", "Function", "Calls", "Cum (s)", "Total (s)", "%"], [(row.path, row.line, row.name, row.ncalls, f"{row.cumulative_seconds:.3f}", f"{row.total_seconds:.3f}", f"{row.percentage:.2f}") for row in report.hotspots])])
    if report.notes:
        lines.extend(["", "## Notes", "", *[f"- {note}" for note in report.notes]])
    return "\n".join(lines)


def memory_hotspots_markdown(report: MemoryProfileReport) -> str:
    lines = ["# Memory Hotspots", "", f"- Engine: `{report.engine}`", f"- Target: `{report.target}`", f"- Peak RSS: `{report.peak_rss_mb:.2f} MB`", f"- Traced peak: `{report.traced_peak_mb:.2f} MB`", ""]
    if report.stage_hotspots:
        lines.extend(["## Stage Hotspots", "", markdown_table(["Stage", "RSS Delta MB", "Wall (s)"], [(row.get("name", ""), f"{float(row.get('rss_delta_mb') or 0.0):.3f}", f"{float(row.get('wall_seconds') or 0.0):.3f}") for row in report.stage_hotspots])])
    if report.hotspots:
        lines.extend(["", "## Allocation Hotspots", "", markdown_table(["Path", "Line", "Size MB", "Count"], [(row.path, row.line, f"{row.size_mb:.4f}", row.count) for row in report.hotspots])])
    if report.notes:
        lines.extend(["", "## Notes", "", *[f"- {note}" for note in report.notes]])
    return "\n".join(lines)
