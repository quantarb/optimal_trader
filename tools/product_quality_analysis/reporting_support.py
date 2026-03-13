from __future__ import annotations

from pathlib import Path

from .models import AnalysisSnapshot
from .reports.data_quality_report import data_quality_report_markdown
from .reports.issue_ranking import prioritized_issues_markdown
from .reports.quality_summary import quality_summary_markdown
from .reports.scalability_report import scalability_report_markdown
from .reports.ui_consistency_report import ui_consistency_report_markdown
from .utils.report_utils import write_json, write_markdown


def _table_markdown(title: str, rows: list[dict], columns: list[str]) -> str:
    lines = [f"# {title}", ""]
    if not rows:
        lines.extend(["No rows were collected.", ""])
        return "\n".join(lines)
    lines.extend(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *["| " + " | ".join(str(row.get(column, "-")) for column in columns) + " |" for row in rows],
            "",
        ]
    )
    return "\n".join(lines)


def _write_reports(snapshot: AnalysisSnapshot, output_dir: Path) -> None:
    write_json(output_dir / f"analysis_{snapshot.label}.json", snapshot)
    write_json(output_dir / "snapshots" / f"analysis_{snapshot.label}.json", snapshot)
    write_markdown(output_dir / "product_quality_summary.md", quality_summary_markdown(snapshot))
    write_markdown(output_dir / "prioritized_issues.md", prioritized_issues_markdown(snapshot.issues))
    write_markdown(output_dir / "data_quality_report.md", data_quality_report_markdown(snapshot))
    write_markdown(output_dir / "ui_consistency_report.md", ui_consistency_report_markdown(snapshot))
    write_markdown(
        output_dir / "pagination_report.md",
        _table_markdown(
            "Pagination Report",
            list(snapshot.pagination.get("tables") or []),
            ["page", "table", "rows", "columns", "has_pagination", "page_size", "readability_risk_score"],
        ),
    )
    write_markdown(
        output_dir / "readability_report.md",
        _table_markdown(
            "Readability Report",
            list(snapshot.readability.get("tables") or []),
            ["page", "table", "rows", "columns", "risk"],
        ),
    )
    write_markdown(output_dir / "scalability_report.md", scalability_report_markdown(snapshot))
