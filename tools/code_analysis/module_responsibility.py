from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .repository import ModuleRecord, RepositoryInventory


CONCERN_RULES = {
    "web_ui": ("django.http", "django.shortcuts", ".views", "render(", "JsonResponse"),
    "cli_commands": ("BaseCommand", ".management.commands.", "call_command", "add_arguments"),
    "reporting": ("report", "summary", "markdown", "chart"),
    "data_io": ("json", "csv", "Path(", "read_text", "write_text"),
    "pipeline_orchestration": ("PipelineRun", "Artifact", "execute_pipeline_run", "pipeline_run"),
    "ml_modeling": ("sklearn", "torch", "autogluon", "stable_baselines3", "fit_", "predict"),
    "feature_engineering": ("feature", "embedding", "serialize_features", "build_feature"),
    "market_intelligence": ("analog", "opportunity", "insight", "situation", "familiarity"),
    "backtesting": ("backtest", "equity", "turnover", "strategy"),
    "external_data": ("fmp", "endpoint", "requests", "http"),
}


@dataclass
class ModuleResponsibilityReport:
    module_rows: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"module_rows": list(self.module_rows)}


def analyze_module_responsibilities(
    inventory: RepositoryInventory,
    *,
    metrics_report: dict[str, Any] | None = None,
    dependency_report: dict[str, Any] | None = None,
    duplicate_report: dict[str, Any] | None = None,
) -> ModuleResponsibilityReport:
    metric_map = {row["module"]: row for row in list((metrics_report or {}).get("module_rows") or [])}
    indegree = dict((dependency_report or {}).get("indegree") or {})
    outdegree = dict((dependency_report or {}).get("outdegree") or {})
    duplicate_hits = _duplicate_hits_by_module(duplicate_report or {})

    rows: list[dict[str, Any]] = []
    for module_name, module_record in inventory.modules.items():
        evidence = _module_text(module_record)
        concerns: list[dict[str, Any]] = []
        for concern_name, markers in CONCERN_RULES.items():
            score = sum(1 for marker in markers if marker.lower() in evidence)
            if score > 0:
                concerns.append({"concern": concern_name, "score": score})
        concerns.sort(key=lambda row: (-row["score"], row["concern"]))
        if not concerns:
            continue

        metric_row = metric_map.get(module_name, {})
        concern_count = len(concerns)
        max_complexity = int(metric_row.get("max_complexity") or 0)
        maintainability_index = _as_float(metric_row.get("maintainability_index"), 100.0)
        duplicate_count = int(duplicate_hits.get(module_name, 0))
        fan_in = int(indegree.get(module_name, 0))
        fan_out = int(outdegree.get(module_name, 0))

        mixing_score = round(
            concern_count * 6.0
            + min(module_record.line_count / 120.0, 12.0)
            + max_complexity * 0.8
            + duplicate_count * 1.5
            + fan_out * 0.5
            + (5.0 if maintainability_index < 20.0 else 0.0)
            + (2.0 if fan_in > 12 and fan_out > 12 else 0.0),
            2,
        )
        reasons: list[str] = []
        if concern_count >= 3:
            reasons.append(f"matches {concern_count} concern categories")
        if duplicate_count:
            reasons.append(f"appears in {duplicate_count} duplicate clusters/pairs")
        if maintainability_index < 20.0:
            reasons.append(f"maintainability index is low at {maintainability_index:.2f}")
        if fan_out > 12:
            reasons.append(f"high fan-out with {fan_out} imports")
        if module_record.line_count > 500:
            reasons.append(f"large module at {module_record.line_count} lines")
        rows.append(
            {
                "module": module_name,
                "path": module_record.path,
                "line_count": module_record.line_count,
                "function_count": len(module_record.functions),
                "class_count": len(module_record.class_records),
                "concern_count": concern_count,
                "concerns": concerns,
                "max_complexity": max_complexity,
                "maintainability_index": round(maintainability_index, 3),
                "fan_in": fan_in,
                "fan_out": fan_out,
                "duplicate_hits": duplicate_count,
                "mixing_score": mixing_score,
                "reasons": reasons,
            }
        )
    rows.sort(
        key=lambda row: (
            ".tests" in str(row["module"]),
            -float(row["mixing_score"]),
            -int(row["line_count"]),
            row["module"],
        )
    )
    return ModuleResponsibilityReport(module_rows=rows)


def _duplicate_hits_by_module(report: dict[str, Any]) -> dict[str, int]:
    hits: dict[str, int] = {}
    for pair in list(report.get("candidate_pairs") or []):
        for key in ("left", "right"):
            chunk_id = str(pair.get(key) or "")
            module = chunk_id.rsplit(".", 1)[0] if "." in chunk_id else chunk_id
            hits[module] = hits.get(module, 0) + 1
    for cluster in list(report.get("clusters") or []):
        for member in list(cluster.get("members") or []):
            module = str(member).rsplit(".", 1)[0] if "." in str(member) else str(member)
            hits[module] = hits.get(module, 0) + 1
    return hits


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _module_text(module_record: ModuleRecord) -> str:
    parts = [module_record.module, module_record.path]
    parts.extend(module_record.imports)
    parts.extend(class_record.full_name for class_record in module_record.class_records)
    parts.extend(function_record.full_name for function_record in module_record.functions)
    parts.extend(call for function_record in module_record.functions for call in function_record.resolved_calls)
    return " ".join(parts).lower()


def module_responsibility_markdown(report: ModuleResponsibilityReport) -> str:
    sections = [
        "# Module Responsibility Report",
        "",
        "## Highest Mixed-Concern Modules",
    ]
    if not report.module_rows:
        sections.append("- none")
        return "\n".join(sections)
    for row in report.module_rows[:30]:
        concern_text = ", ".join(f"{item['concern']}({item['score']})" for item in row["concerns"][:6])
        reason_text = "; ".join(row["reasons"][:4]) or "multiple concerns detected"
        sections.append(
            f"- `{row['module']}`: mixing_score={row['mixing_score']}, concerns={row['concern_count']}, lines={row['line_count']}, max_complexity={row['max_complexity']} [{concern_text}]"
        )
        sections.append(f"  - reasons: {reason_text}")
    return "\n".join(sections)
