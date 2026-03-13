from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RefactorPriorityReport:
    rankings: list[dict[str, Any]]
    summary: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rankings": list(self.rankings),
            "summary": dict(self.summary),
            "notes": list(self.notes),
        }


def build_refactor_priority_report(blast_radius_report: dict[str, Any]) -> RefactorPriorityReport:
    module_rows = [
        row
        for row in list(blast_radius_report.get("module_rows") or [])
        if not _is_test_name(str(row.get("module") or ""))
    ]
    badness_rank = _rank_map(module_rows, "architectural_badness")
    centrality_rank = _rank_map(module_rows, "dependency_centrality_score")
    blast_rank = _rank_map(module_rows, "blast_radius_score")
    leverage_rank = _rank_map(module_rows, "estimated_refactor_leverage")
    risk_rank = _rank_map(module_rows, "change_risk_score")

    rankings: list[dict[str, Any]] = []
    for row in module_rows:
        module = str(row.get("module") or "")
        safest_high_value_score = round(
            float(row.get("estimated_refactor_leverage") or 0.0)
            * max(0.15, (100.0 - float(row.get("change_risk_score") or 0.0)) / 100.0)
            * (0.95 if bool(row.get("critical_execution_path")) else 1.05),
            2,
        )
        overall_priority_score = round(
            (float(row.get("architectural_badness") or 0.0) * 0.30)
            + (float(row.get("dependency_centrality_score") or 0.0) * 0.18)
            + (float(row.get("blast_radius_score") or 0.0) * 0.20)
            + (float(row.get("estimated_refactor_leverage") or 0.0) * 0.32),
            2,
        )
        rankings.append(
            {
                "module": module,
                "path": row.get("path"),
                "architectural_badness": row.get("architectural_badness"),
                "architectural_badness_rank": badness_rank.get(module, 0),
                "centrality_score": row.get("dependency_centrality_score"),
                "centrality_rank": centrality_rank.get(module, 0),
                "blast_radius_score": row.get("blast_radius_score"),
                "blast_radius_rank": blast_rank.get(module, 0),
                "estimated_refactor_leverage": row.get("estimated_refactor_leverage"),
                "estimated_refactor_leverage_rank": leverage_rank.get(module, 0),
                "change_risk_score": row.get("change_risk_score"),
                "change_risk_rank": risk_rank.get(module, 0),
                "critical_execution_path": bool(row.get("critical_execution_path")),
                "god_module": bool(row.get("god_module")),
                "safest_high_value_refactor_score": safest_high_value_score,
                "overall_priority_score": overall_priority_score,
                "suggested_refactor": _suggested_refactor(row),
                "rationale": _rationale(row),
            }
        )
    rankings.sort(
        key=lambda row: (
            -float(row["overall_priority_score"]),
            -float(row["safest_high_value_refactor_score"]),
            row["module"],
        )
    )
    summary = {
        "top_10_architectural_badness": sorted(rankings, key=lambda row: (int(row["architectural_badness_rank"]), row["module"]))[:10],
        "top_10_centrality": sorted(rankings, key=lambda row: (int(row["centrality_rank"]), row["module"]))[:10],
        "top_10_highest_blast_radius_modules": sorted(rankings, key=lambda row: (int(row["blast_radius_rank"]), row["module"]))[:10],
        "top_10_estimated_refactor_leverage": sorted(rankings, key=lambda row: (int(row["estimated_refactor_leverage_rank"]), row["module"]))[:10],
        "top_10_safest_high_value_refactors": sorted(
            rankings,
            key=lambda row: (-float(row["safest_high_value_refactor_score"]), row["module"]),
        )[:10],
        "top_10_highest_risk_modules_to_change": sorted(
            rankings,
            key=lambda row: (-float(row["change_risk_score"]), row["module"]),
        )[:10],
    }
    notes = [
        "Overall priority captures where a refactor would have the highest structural leverage.",
        "Safest high-value refactors combine leverage with lower estimated change risk rather than chasing the absolute worst modules first.",
    ]
    return RefactorPriorityReport(rankings=rankings, summary=summary, notes=notes)


def refactor_priority_markdown(report: RefactorPriorityReport) -> str:
    sections = [
        "# Refactor Priority Report",
        "",
        "## Top 10 Safest High-Value Refactors",
    ]
    sections.extend(
        f"- `{row['module']}`: safe_value={row['safest_high_value_refactor_score']:.2f}, leverage={row['estimated_refactor_leverage']:.2f}, risk={row['change_risk_score']:.2f} -> {row['suggested_refactor']}"
        for row in report.summary.get("top_10_safest_high_value_refactors", [])
    )
    sections.extend(["", "## Top 10 Highest-Risk Modules To Change"])
    sections.extend(
        f"- `{row['module']}`: risk={row['change_risk_score']:.2f}, blast={row['blast_radius_score']:.2f}, badness={row['architectural_badness']:.2f}"
        for row in report.summary.get("top_10_highest_risk_modules_to_change", [])
    )
    sections.extend(["", "## Top 10 Highest-Blast-Radius Modules"])
    sections.extend(
        f"- `{row['module']}`: blast={row['blast_radius_score']:.2f}, centrality={row['centrality_score']:.2f}, leverage={row['estimated_refactor_leverage']:.2f}"
        for row in report.summary.get("top_10_highest_blast_radius_modules", [])
    )
    sections.extend(["", "## Top 10 Architectural Badness"])
    sections.extend(
        f"- `{row['module']}`: badness={row['architectural_badness']:.2f}, violations_rank={row['architectural_badness_rank']}, suggestion={row['suggested_refactor']}"
        for row in report.summary.get("top_10_architectural_badness", [])
    )
    sections.extend(["", "## Ranked Modules"])
    sections.extend(
        f"- `{row['module']}`: overall={row['overall_priority_score']:.2f}, safe_value={row['safest_high_value_refactor_score']:.2f}, ranks=(badness {row['architectural_badness_rank']}, centrality {row['centrality_rank']}, blast {row['blast_radius_rank']}, leverage {row['estimated_refactor_leverage_rank']})"
        for row in report.rankings[:30]
    )
    if report.notes:
        sections.extend(["", "## Notes", *[f"- {note}" for note in report.notes]])
    return "\n".join(sections)


def _rank_map(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    ranked = sorted(rows, key=lambda row: (-float(row.get(key) or 0.0), str(row.get("module") or "")))
    return {str(row.get("module") or ""): index for index, row in enumerate(ranked, start=1)}


def _suggested_refactor(row: dict[str, Any]) -> str:
    architecture_violations = int(row.get("architecture_rule_violations") or 0)
    duplicate_clusters = int(((row.get("quality_context") or {}).get("duplicate_code_clusters") or 0))
    dead_code = int(((row.get("quality_context") or {}).get("dead_code_count") or 0))
    if architecture_violations:
        return "extract boundary adapter and invert imports"
    if bool(row.get("god_module")):
        return "split by concern and extract orchestration seams"
    if duplicate_clusters >= 2:
        return "extract shared workflow or reusable pipeline stage"
    if dead_code:
        return "prune dead branches before deeper changes"
    if bool(row.get("critical_execution_path")):
        return "add characterization tests, then isolate hot path helpers"
    return "extract smaller pure helpers and boundary objects"


def _rationale(row: dict[str, Any]) -> str:
    reasons = list(row.get("risk_reasons") or [])
    if not reasons:
        return "High leverage relative to current structural health."
    return "; ".join(str(reason) for reason in reasons[:4])


def _is_test_name(name: str) -> bool:
    value = str(name or "")
    return value == "tests" or value.startswith("tests.") or ".tests." in value or value.endswith(".tests")
