from __future__ import annotations

from ..models import AnalysisSnapshot


def data_quality_report_markdown(snapshot: AnalysisSnapshot) -> str:
    lines = ["# Data Quality Report", ""]
    rows = list(snapshot.data_quality.get("field_coverage") or [])
    if not rows:
        lines.extend(["No data coverage findings were recorded.", ""])
        return "\n".join(lines)
    lines.append("| Tier | Field | Column | Coverage | Page Label | Missing Examples |")
    lines.append("| --- | --- | --- | ---: | --- | --- |")
    for row in rows:
        lines.append(
            "| {tier} | {field} | {column} | {coverage:.0%} | {page_label} | {missing} |".format(
                tier=row.get("dataset_label") or "-",
                field=row.get("field_name") or "-",
                column=row.get("matched_column") or "-",
                coverage=float(row.get("coverage_rate") or 0.0),
                page_label="yes" if row.get("page_label_present") else "no" if row.get("page_label_present") is False else "-",
                missing=", ".join(list(row.get("missing_symbols") or [])[:4]) or "-",
            )
        )
    lines.append("")
    return "\n".join(lines)
