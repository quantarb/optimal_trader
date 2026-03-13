from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .architecture_rules import ArchitectureRulesReport, validate_architecture_rules
from .code_metrics import CodeMetricsReport, analyze_code_metrics
from .dead_code import DeadCodeReport, analyze_dead_code
from .dependency_graph import DependencyGraphReport, analyze_dependency_graph
from .duplicate_code import DuplicateCodeReport
from .patterns.anti_patterns import AntiPatternReport, analyze_anti_patterns
from .patterns.good_patterns import GoodPatternReport, analyze_good_patterns
from .patterns.shared import (
    annotation_text,
    build_repository_ast_context,
    config_like_name,
    cyclomatic_complexity,
    max_nesting_depth,
    numeric_literals,
)
from .repository import RepositoryInventory, build_repository_inventory


@dataclass
class CodeHealthMetricsReport:
    repo_summary: dict[str, Any]
    module_rows: list[dict[str, Any]]
    file_rows: list[dict[str, Any]]
    distributions: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_summary": dict(self.repo_summary),
            "module_rows": list(self.module_rows),
            "file_rows": list(self.file_rows),
            "distributions": dict(self.distributions),
            "notes": list(self.notes),
        }


def analyze_code_health_metrics(
    root: Path,
    *,
    inventory: RepositoryInventory | None = None,
    metrics_report: CodeMetricsReport | dict[str, Any] | None = None,
    dependency_report: DependencyGraphReport | dict[str, Any] | None = None,
    duplicate_report: DuplicateCodeReport | dict[str, Any] | None = None,
    dead_code_report: DeadCodeReport | dict[str, Any] | None = None,
    architecture_report: ArchitectureRulesReport | dict[str, Any] | None = None,
    anti_pattern_report: AntiPatternReport | dict[str, Any] | None = None,
    good_pattern_report: GoodPatternReport | dict[str, Any] | None = None,
) -> CodeHealthMetricsReport:
    inventory = inventory or build_repository_inventory(root)
    context = build_repository_ast_context(root, inventory)
    metrics_payload = _to_payload(metrics_report) or analyze_code_metrics(root).to_dict()
    dependency_payload = _to_payload(dependency_report) or analyze_dependency_graph(root, inventory).to_dict()
    duplicate_payload = _to_payload(duplicate_report) or {"clusters": [], "candidate_pairs": []}
    dead_code_payload = _to_payload(dead_code_report) or analyze_dead_code(root, inventory=inventory).to_dict()
    architecture_payload = _to_payload(architecture_report) or validate_architecture_rules(root, inventory=inventory, dependency_report=dependency_payload).to_dict()
    anti_payload = _to_payload(anti_pattern_report) or analyze_anti_patterns(root, inventory=inventory, duplicate_report=duplicate_payload, architecture_report=architecture_payload).to_dict()
    good_payload = _to_payload(good_pattern_report) or analyze_good_patterns(root, inventory=inventory).to_dict()

    module_metric_map = {row["module"]: row for row in list(metrics_payload.get("module_rows") or [])}
    fan_in = dict(dependency_payload.get("indegree") or {})
    fan_out = dict(dependency_payload.get("outdegree") or {})
    cycle_members = {member for cycle in list(dependency_payload.get("cycles") or []) for member in list(cycle)}
    duplicate_hits = _duplicate_cluster_hits(duplicate_payload)
    dead_code_hits = _dead_code_hits(dead_code_payload)
    architecture_hits = _source_counts(architecture_payload.get("violations") or [], "source_module")
    good_hits = _pattern_hits_by_path(good_payload.get("findings") or [])
    anti_hits = _pattern_hits_by_path(anti_payload.get("findings") or [])
    config_usage = _config_usage_counts(context)

    function_loc_values: list[int] = []
    class_size_values: list[int] = []
    nesting_values: list[int] = []
    complexity_values: list[int] = []

    module_rows: list[dict[str, Any]] = []
    for module_name, module_context in context.modules.items():
        function_stats = [
            {
                "loc": function.loc,
                "complexity": cyclomatic_complexity(function.node),
                "nesting": max_nesting_depth(function.node),
            }
            for function in module_context.functions
        ]
        class_stats = [class_context.loc for class_context in module_context.classes]
        function_loc_values.extend(item["loc"] for item in function_stats)
        class_size_values.extend(class_stats)
        nesting_values.extend(item["nesting"] for item in function_stats)
        complexity_values.extend(item["complexity"] for item in function_stats)

        numeric_count = len(numeric_literals(module_context.tree, module_constants=module_context.top_level_constants))
        function_count = len(module_context.functions)
        type_hint_coverage = _module_type_hint_coverage(module_context.functions)
        module_metric_row = module_metric_map.get(module_name, {})
        module_good_rows = good_hits.get(module_context.path, {})
        module_anti_rows = anti_hits.get(module_context.path, {})
        pure_function_count = int(module_good_rows.get("pure functions", 0))
        artifact_boundary_usage = int(module_good_rows.get("artifact return boundary", 0))
        interface_reuse_count = int(module_good_rows.get("stable base class or protocol", 0))
        explicit_boundary_count = int(module_good_rows.get("explicit boundary objects", 0))
        anti_pattern_count = sum(int(value) for value in module_anti_rows.values())
        high_severity_anti = _severity_hits_for_path(anti_payload.get("findings") or [], module_context.path, "high")
        avg_complexity = round(mean(item["complexity"] for item in function_stats), 3) if function_stats else 0.0
        max_complexity = max((item["complexity"] for item in function_stats), default=0)
        max_nesting = max((item["nesting"] for item in function_stats), default=0)
        llm_editability = _llm_editability_score(
            type_hint_coverage=type_hint_coverage,
            avg_complexity=avg_complexity,
            max_nesting=max_nesting,
            fan_out=int(fan_out.get(module_name, 0)),
            duplicate_hits=int(duplicate_hits.get(module_name, 0)),
            architecture_violations=int(architecture_hits.get(module_name, 0)),
            anti_pattern_count=anti_pattern_count,
            explicit_boundary_count=explicit_boundary_count,
        )
        change_safety = _change_safety_score(
            type_hint_coverage=type_hint_coverage,
            pure_function_count=pure_function_count,
            function_count=function_count,
            avg_complexity=avg_complexity,
            architecture_violations=int(architecture_hits.get(module_name, 0)),
            artifact_boundary_usage=artifact_boundary_usage,
            config_usage=int(config_usage.get(module_name, 0)),
            duplicate_hits=int(duplicate_hits.get(module_name, 0)),
            high_severity_anti=high_severity_anti,
        )
        row = {
            "module": module_name,
            "path": module_context.path,
            "line_count": module_context.line_count,
            "function_count": function_count,
            "class_count": len(module_context.classes),
            "cyclomatic_complexity_avg": avg_complexity,
            "cyclomatic_complexity_max": max_complexity,
            "maintainability_index": round(float(module_metric_row.get("maintainability_index") or 0.0), 3),
            "maintainability_rank": str(module_metric_row.get("maintainability_rank") or "NA"),
            "function_loc_avg": round(mean(item["loc"] for item in function_stats), 3) if function_stats else 0.0,
            "function_loc_max": max((item["loc"] for item in function_stats), default=0),
            "class_size_max": max(class_stats, default=0),
            "nesting_depth_avg": round(mean(item["nesting"] for item in function_stats), 3) if function_stats else 0.0,
            "nesting_depth_max": max_nesting,
            "type_hint_coverage": round(type_hint_coverage, 4),
            "magic_number_count": numeric_count,
            "dependency_fan_in": int(fan_in.get(module_name, 0)),
            "dependency_fan_out": int(fan_out.get(module_name, 0)),
            "import_cycle_count": 1 if module_name in cycle_members else 0,
            "architecture_rule_violations": int(architecture_hits.get(module_name, 0)),
            "duplicate_code_clusters": int(duplicate_hits.get(module_name, 0)),
            "dead_code_count": int(dead_code_hits.get(module_name, 0)),
            "interface_reuse_count": interface_reuse_count,
            "config_object_usage": int(config_usage.get(module_name, 0)),
            "artifact_boundary_usage": artifact_boundary_usage,
            "llm_editability_proxy_score": llm_editability,
            "change_safety_proxy_score": change_safety,
            "good_pattern_count": sum(int(value) for value in module_good_rows.values()),
            "anti_pattern_count": anti_pattern_count,
        }
        module_rows.append(row)

    module_rows.sort(
        key=lambda row: (
            float(row["llm_editability_proxy_score"]),
            float(row["change_safety_proxy_score"]),
            -int(row["anti_pattern_count"]),
            row["module"],
        )
    )
    file_rows = [
        {
            **row,
            "file": row["path"],
        }
        for row in module_rows
    ]
    distributions = {
        "function_loc_distribution": _distribution_summary(function_loc_values),
        "class_size_distribution": _distribution_summary(class_size_values),
        "nesting_depth_distribution": _distribution_summary(nesting_values),
        "cyclomatic_complexity_distribution": _distribution_summary(complexity_values),
    }
    repo_summary = _build_repo_summary(
        metrics_payload=metrics_payload,
        dependency_payload=dependency_payload,
        dead_code_payload=dead_code_payload,
        duplicate_payload=duplicate_payload,
        architecture_payload=architecture_payload,
        anti_payload=anti_payload,
        good_payload=good_payload,
        module_rows=module_rows,
        distributions=distributions,
    )
    notes = [
        "Type-hint coverage and distribution summaries are computed from AST-visible Python definitions in the repository.",
        "LLM editability and change safety are deterministic proxy scores built from structural signals; they are meant for comparison over time rather than absolute truth.",
    ]
    return CodeHealthMetricsReport(
        repo_summary=repo_summary,
        module_rows=module_rows,
        file_rows=file_rows,
        distributions=distributions,
        notes=notes,
    )


def code_health_metrics_markdown(report: CodeHealthMetricsReport) -> str:
    repo = report.repo_summary
    sections = [
        "# Code Health Metrics",
        "",
        f"- cyclomatic complexity avg/max: {repo.get('cyclomatic_complexity_summary', {}).get('avg', 0.0)} / {repo.get('cyclomatic_complexity_summary', {}).get('max', 0)}",
        f"- maintainability avg/min: {repo.get('maintainability_summary', {}).get('avg', 0.0)} / {repo.get('maintainability_summary', {}).get('min', 0.0)}",
        f"- type hint coverage: {repo.get('type_hint_coverage', 0.0):.2%}",
        f"- import cycle count: {repo.get('import_cycle_count', 0)}",
        f"- architecture rule violations: {repo.get('architecture_rule_violations', 0)}",
        f"- duplicate code clusters: {repo.get('duplicate_code_clusters', 0)}",
        f"- dead code count: {repo.get('dead_code_count', 0)}",
        f"- llm editability proxy score: {repo.get('llm_editability_proxy_score', 0.0)}",
        f"- change safety proxy score: {repo.get('change_safety_proxy_score', 0.0)}",
        "",
        "## Lowest Editability Modules",
    ]
    sections.extend(
        f"- `{row['module']}`: llm_editability={row['llm_editability_proxy_score']}, change_safety={row['change_safety_proxy_score']}, anti_patterns={row['anti_pattern_count']}"
        for row in report.module_rows[:30]
    )
    sections.extend(["", "## Distribution Summary"])
    for name, stats in report.distributions.items():
        sections.append(
            f"- `{name}`: count={stats.get('count', 0)}, avg={stats.get('avg', 0.0)}, p50={stats.get('p50', 0.0)}, p90={stats.get('p90', 0.0)}, max={stats.get('max', 0.0)}"
        )
    if report.notes:
        sections.extend(["", "## Notes", *[f"- {note}" for note in report.notes]])
    return "\n".join(sections)


def _build_repo_summary(
    *,
    metrics_payload: dict[str, Any],
    dependency_payload: dict[str, Any],
    dead_code_payload: dict[str, Any],
    duplicate_payload: dict[str, Any],
    architecture_payload: dict[str, Any],
    anti_payload: dict[str, Any],
    good_payload: dict[str, Any],
    module_rows: list[dict[str, Any]],
    distributions: dict[str, Any],
) -> dict[str, Any]:
    maintainability_values = [float(row.get("maintainability_index") or 0.0) for row in list(metrics_payload.get("module_rows") or [])]
    complexity_values = [float(row["cyclomatic_complexity_avg"]) for row in module_rows]
    max_complexity_values = [int(row["cyclomatic_complexity_max"]) for row in module_rows]
    repo_type_hint = mean(float(row["type_hint_coverage"]) for row in module_rows) if module_rows else 0.0
    llm_editability = round(mean(float(row["llm_editability_proxy_score"]) for row in module_rows), 2) if module_rows else 0.0
    change_safety = round(mean(float(row["change_safety_proxy_score"]) for row in module_rows), 2) if module_rows else 0.0
    return {
        "cyclomatic_complexity_summary": {
            "avg": round(mean(complexity_values), 3) if complexity_values else 0.0,
            "p90": _percentile(complexity_values, 0.9),
            "max": max(max_complexity_values, default=0),
        },
        "maintainability_summary": {
            "avg": round(mean(maintainability_values), 3) if maintainability_values else 0.0,
            "p10": _percentile(maintainability_values, 0.1),
            "min": min(maintainability_values, default=0.0),
        },
        "function_loc_distribution": distributions.get("function_loc_distribution", {}),
        "class_size_distribution": distributions.get("class_size_distribution", {}),
        "nesting_depth_distribution": distributions.get("nesting_depth_distribution", {}),
        "type_hint_coverage": round(repo_type_hint, 4),
        "magic_number_count": int(sum(int(row["magic_number_count"]) for row in module_rows)),
        "dependency_fan_in": _distribution_summary([int(row["dependency_fan_in"]) for row in module_rows]),
        "dependency_fan_out": _distribution_summary([int(row["dependency_fan_out"]) for row in module_rows]),
        "import_cycle_count": len(list(dependency_payload.get("cycles") or [])),
        "architecture_rule_violations": len(list(architecture_payload.get("violations") or [])),
        "duplicate_code_clusters": len(list(duplicate_payload.get("clusters") or [])),
        "dead_code_count": len(list(dead_code_payload.get("unused_items") or [])) + len(list(dead_code_payload.get("unused_modules") or [])),
        "interface_reuse_count": sum(1 for finding in list(good_payload.get("findings") or []) if finding.get("pattern") == "stable base class or protocol"),
        "config_object_usage": int(sum(int(row["config_object_usage"]) for row in module_rows)),
        "artifact_boundary_usage": int(sum(int(row["artifact_boundary_usage"]) for row in module_rows)),
        "llm_editability_proxy_score": llm_editability,
        "change_safety_proxy_score": change_safety,
        "good_pattern_strength": round(mean(float(finding.get("strength") or 0.0) for finding in list(good_payload.get("findings") or [])), 3) if list(good_payload.get("findings") or []) else 0.0,
        "anti_pattern_burden": len(list(anti_payload.get("findings") or [])),
    }


def _module_type_hint_coverage(functions) -> float:
    total_slots = 0
    typed_slots = 0
    for function in functions:
        total_slots += function.parameter_count + 1
        typed_slots += function.typed_parameter_count
        typed_slots += 1 if function.has_return_annotation else 0
    if total_slots == 0:
        return 1.0
    return typed_slots / total_slots


def _llm_editability_score(
    *,
    type_hint_coverage: float,
    avg_complexity: float,
    max_nesting: int,
    fan_out: int,
    duplicate_hits: int,
    architecture_violations: int,
    anti_pattern_count: int,
    explicit_boundary_count: int,
) -> float:
    score = 100.0
    score += type_hint_coverage * 15.0
    score += min(explicit_boundary_count * 4.0, 10.0)
    score -= avg_complexity * 2.0
    score -= max_nesting * 4.0
    score -= fan_out * 1.2
    score -= duplicate_hits * 5.0
    score -= architecture_violations * 6.0
    score -= anti_pattern_count * 1.0
    return round(_clamp(score, 0.0, 100.0), 2)


def _change_safety_score(
    *,
    type_hint_coverage: float,
    pure_function_count: int,
    function_count: int,
    avg_complexity: float,
    architecture_violations: int,
    artifact_boundary_usage: int,
    config_usage: int,
    duplicate_hits: int,
    high_severity_anti: int,
) -> float:
    score = 50.0
    score += type_hint_coverage * 28.0
    score += (pure_function_count / max(function_count, 1)) * 18.0
    score += min(artifact_boundary_usage * 4.0, 12.0)
    score += min(config_usage * 2.5, 10.0)
    score -= avg_complexity * 1.6
    score -= architecture_violations * 5.0
    score -= duplicate_hits * 3.0
    score -= high_severity_anti * 4.0
    return round(_clamp(score, 0.0, 100.0), 2)


def _config_usage_counts(context) -> dict[str, int]:
    rows: dict[str, int] = {}
    for module_context in context.modules.values():
        for function in module_context.functions:
            annotations = []
            for arg in list(function.node.args.args) + list(function.node.args.kwonlyargs) + list(function.node.args.posonlyargs):
                annotations.append(annotation_text(arg.annotation))
            annotations.append(annotation_text(function.node.returns))
            if any(config_like_name(annotation) for annotation in annotations if annotation):
                rows[module_context.module] = rows.get(module_context.module, 0) + 1
    return rows


def _duplicate_cluster_hits(duplicate_payload: dict[str, Any]) -> dict[str, int]:
    rows: dict[str, int] = {}
    for cluster in list(duplicate_payload.get("clusters") or []):
        modules = {str(member).rsplit(".", 1)[0] for member in list(cluster.get("members") or []) if str(member)}
        for module in modules:
            rows[module] = rows.get(module, 0) + 1
    return rows


def _dead_code_hits(dead_code_payload: dict[str, Any]) -> dict[str, int]:
    rows: dict[str, int] = {}
    for item in list(dead_code_payload.get("unused_items") or []):
        module = str(item.get("module") or "")
        rows[module] = rows.get(module, 0) + 1
    for item in list(dead_code_payload.get("unused_modules") or []):
        module = str(item.get("module") or "")
        rows[module] = rows.get(module, 0) + 1
    return rows


def _pattern_hits_by_path(findings: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    for finding in findings:
        path = str(finding.get("file") or "")
        pattern = str(finding.get("pattern") or "")
        rows.setdefault(path, {})
        rows[path][pattern] = rows[path].get(pattern, 0) + 1
    return rows


def _severity_hits_for_path(findings: list[dict[str, Any]], path: str, severity: str) -> int:
    return sum(
        1
        for finding in findings
        if str(finding.get("file") or "") == str(path) and str(finding.get("severity") or "").lower() == severity
    )


def _source_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _distribution_summary(values: list[int | float]) -> dict[str, Any]:
    cleaned = sorted(float(value) for value in values)
    if not cleaned:
        return {"count": 0, "avg": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": len(cleaned),
        "avg": round(mean(cleaned), 3),
        "p50": _percentile(cleaned, 0.5),
        "p90": _percentile(cleaned, 0.9),
        "max": round(cleaned[-1], 3),
    }


def _percentile(values: list[int | float], quantile: float) -> float:
    if not values:
        return 0.0
    rows = sorted(float(value) for value in values)
    index = int(round((len(rows) - 1) * quantile))
    index = max(0, min(index, len(rows) - 1))
    return round(rows[index], 3)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _to_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return dict(value)
