from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RefactoringHintsReport:
    hints: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"hints": list(self.hints)}


def build_refactoring_hints(
    *,
    dependency_report: dict[str, Any],
    duplicate_report: dict[str, Any],
    dead_code_report: dict[str, Any],
    metrics_report: dict[str, Any],
    responsibility_report: dict[str, Any],
) -> RefactoringHintsReport:
    hints: list[dict[str, Any]] = []
    module_rows = list(responsibility_report.get("module_rows") or [])
    if module_rows:
        top_row = next((row for row in module_rows if ".tests" not in str(row.get("module") or "")), module_rows[0])
        hints.append(
            {
                "category": "split_module",
                "priority": "high",
                "target": top_row.get("module"),
                "reason": f"Module mixes {top_row.get('concern_count')} concerns across {top_row.get('line_count')} lines.",
            }
        )
    duplicate_clusters = list(duplicate_report.get("clusters") or [])
    if duplicate_clusters:
        top_cluster = next((row for row in duplicate_clusters if len(list(row.get("modules") or [])) > 1), duplicate_clusters[0])
        hints.append(
            {
                "category": "deduplicate_workflow",
                "priority": "high",
                "target": ", ".join(list(top_cluster.get("members") or [])[:4]),
                "reason": f"Duplicate cluster {top_cluster.get('cluster_id')} groups {top_cluster.get('size')} similar implementations.",
            }
        )
    cycles = list(dependency_report.get("cycles") or [])
    if cycles:
        hints.append(
            {
                "category": "break_cycles",
                "priority": "high",
                "target": "dependency graph",
                "reason": f"Found {len(cycles)} circular dependency groups.",
            }
        )
    unused_functions = list(dead_code_report.get("unused_functions") or [])
    if unused_functions:
        first_unused = unused_functions[0]
        target = ".".join(part for part in [str(first_unused.get("module") or ""), str(first_unused.get("name") or "")] if part)
        hints.append(
            {
                "category": "delete_stale_code",
                "priority": "medium",
                "target": target,
                "reason": f"{len(unused_functions)} private functions look unused under Vulture + import analysis.",
            }
        )
    metric_rows = list(metrics_report.get("module_rows") or [])
    low_mi_rows = sorted(
        (row for row in metric_rows if _as_float(row.get("maintainability_index"), 100.0) < 20.0),
        key=lambda row: (_as_float(row.get("maintainability_index"), 100.0), -int(row.get("max_complexity") or 0), str(row.get("module") or "")),
    )
    if low_mi_rows:
        first = low_mi_rows[0]
        hints.append(
            {
                "category": "reduce_complexity",
                "priority": "medium",
                "target": first.get("module"),
                "reason": f"Maintainability index is {first.get('maintainability_index')} with max complexity {first.get('max_complexity')}.",
            }
        )
    return RefactoringHintsReport(hints=hints)


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def refactoring_hints_markdown(report: RefactoringHintsReport) -> str:
    sections = [
        "# Refactoring Hints",
        "",
    ]
    if not report.hints:
        sections.append("- No strong refactoring hints were generated.")
        return "\n".join(sections)
    for hint in report.hints:
        sections.append(
            f"- [{hint['priority']}] `{hint['category']}` -> `{hint['target']}`: {hint['reason']}"
        )
    return "\n".join(sections)
