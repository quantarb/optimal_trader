from __future__ import annotations

from ..config import OptimizationWeights
from ..models import ComplexityReport, DependencyReport, OptimizationCandidate, OptimizationReport, PatternReport, RuntimeProfileReport, MemoryProfileReport
from ..utils.report_utils import markdown_table, normalize_score_map, utc_timestamp


def _map_sum(items, key_attr: str, value_attr: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        key = str(getattr(item, key_attr))
        out[key] = out.get(key, 0.0) + float(getattr(item, value_attr) or 0.0)
    return out


def build_optimization_candidates(
    *,
    runtime_report: RuntimeProfileReport,
    memory_report: MemoryProfileReport,
    complexity_report: ComplexityReport,
    dependency_report: DependencyReport,
    dataframe_report: PatternReport,
    scaling_report: PatternReport,
    weights: OptimizationWeights | None = None,
) -> OptimizationReport:
    resolved = weights or OptimizationWeights()
    runtime_scores = normalize_score_map(_map_sum(runtime_report.hotspots, "path", "cumulative_seconds"))
    memory_scores = normalize_score_map(_map_sum(memory_report.hotspots, "path", "size_mb"))
    complexity_scores = normalize_score_map(_map_sum(complexity_report.module_findings, "path", "score"))
    centrality_scores = normalize_score_map({row.path: float(row.score) for row in dependency_report.nodes})
    scaling_scores = normalize_score_map({**_map_sum(dataframe_report.findings, "path", "severity"), **{key: _map_sum(scaling_report.findings, "path", "severity").get(key, 0.0) + _map_sum(dataframe_report.findings, "path", "severity").get(key, 0.0) for key in set([row.path for row in dataframe_report.findings] + [row.path for row in scaling_report.findings])}})
    ignored = ("tools/performance_analysis/", "/", "~")
    paths = sorted(path for path in (set(runtime_scores) | set(memory_scores) | set(complexity_scores) | set(centrality_scores) | set(scaling_scores)) if path and not any(path.startswith(prefix) for prefix in ignored))
    candidates: list[OptimizationCandidate] = []
    for path in paths:
        runtime_score = runtime_scores.get(path, 0.0)
        memory_score = memory_scores.get(path, 0.0)
        complexity_score = complexity_scores.get(path, 0.0)
        centrality_score = centrality_scores.get(path, 0.0)
        scaling_score = scaling_scores.get(path, 0.0)
        total = runtime_score * resolved.runtime_weight + memory_score * resolved.memory_weight + complexity_score * resolved.complexity_weight + centrality_score * resolved.centrality_weight + scaling_score * resolved.scaling_weight
        evidence = []
        if runtime_score > 0.0:
            evidence.append("runtime hotspot")
        if memory_score > 0.0:
            evidence.append("memory hotspot")
        if complexity_score > 0.0:
            evidence.append("high complexity")
        if centrality_score > 0.0:
            evidence.append("central dependency")
        if scaling_score > 0.0:
            evidence.append("scaling risk")
        candidates.append(OptimizationCandidate(path, path, round(runtime_score, 4), round(memory_score, 4), round(complexity_score, 4), round(centrality_score, 4), round(scaling_score, 4), round(total, 4), evidence))
    candidates = sorted(candidates, key=lambda row: (-row.total_score, row.path))
    return OptimizationReport(utc_timestamp(), {"runtime_weight": resolved.runtime_weight, "memory_weight": resolved.memory_weight, "complexity_weight": resolved.complexity_weight, "centrality_weight": resolved.centrality_weight, "scaling_weight": resolved.scaling_weight}, candidates, {"count": len(candidates)})


def optimization_candidates_markdown(report: OptimizationReport) -> str:
    return "\n".join(["# Optimization Targets", "", markdown_table(["Path", "Runtime", "Memory", "Complexity", "Centrality", "Scaling", "Total", "Evidence"], [(row.path, f"{row.runtime_score:.2f}", f"{row.memory_score:.2f}", f"{row.complexity_score:.2f}", f"{row.centrality_score:.2f}", f"{row.scaling_score:.2f}", f"{row.total_score:.2f}", ", ".join(row.evidence)) for row in report.candidates[:25]])])
