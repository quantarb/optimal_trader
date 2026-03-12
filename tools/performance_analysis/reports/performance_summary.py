from __future__ import annotations

from ..models import BenchmarkSuiteReport, ComplexityReport, DependencyReport, MemoryProfileReport, OptimizationReport, PatternReport, RuntimeProfileReport, RegressionFinding


def performance_summary_markdown(
    *,
    complexity_report: ComplexityReport,
    dependency_report: DependencyReport,
    runtime_report: RuntimeProfileReport,
    memory_report: MemoryProfileReport,
    dataframe_report: PatternReport,
    scaling_report: PatternReport,
    benchmark_report: BenchmarkSuiteReport,
    optimization_report: OptimizationReport,
    regressions: list[RegressionFinding] | None = None,
) -> str:
    regressions = regressions or []
    best_candidate = optimization_report.candidates[0] if optimization_report.candidates else None
    top_stage = runtime_report.stage_hotspots[0]["name"] if runtime_report.stage_hotspots else ""
    top_module = complexity_report.module_findings[0].path if complexity_report.module_findings else ""
    top_dependency = dependency_report.nodes[0].path if dependency_report.nodes else ""
    return "\n".join([
        "# Performance Summary",
        "",
        f"- Top runtime stage: `{top_stage}`",
        f"- Peak memory: `{memory_report.peak_rss_mb:.2f} MB`",
        f"- Top complexity module: `{top_module}`",
        f"- Most central dependency: `{top_dependency}`",
        f"- DataFrame findings: `{len(dataframe_report.findings)}`",
        f"- Scaling findings: `{len(scaling_report.findings)}`",
        f"- Benchmark measurements: `{len(benchmark_report.measurements)}`",
        f"- Regression rows: `{len(regressions)}`",
        f"- Best optimization target: `{best_candidate.path if best_candidate else ''}`",
        "",
        "## Focus",
        "",
        f"The current benchmark/profile evidence points to `{top_stage}` as the hottest stage, while `{best_candidate.path if best_candidate else top_module}` ranks highest in the weighted optimization score.",
    ])
