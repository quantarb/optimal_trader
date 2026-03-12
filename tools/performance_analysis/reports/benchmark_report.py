from __future__ import annotations

from ..models import BenchmarkSuiteReport, RegressionFinding
from ..utils.report_utils import markdown_table


def benchmark_report_markdown(report: BenchmarkSuiteReport) -> str:
    return "\n".join(["# Scaling Benchmarks", "", f"- Engine: `{report.engine}`", f"- Label: `{report.label}`", f"- Warmup iterations: `{report.warmup_iterations}`", f"- Measured iterations: `{report.measured_iterations}`", "", markdown_table(["Workload", "Tier", "Median (s)", "Mean (s)", "StdDev (s)", "Samples"], [(row.workload, row.tier, f"{row.median_seconds:.3f}", f"{row.mean_seconds:.3f}", f"{row.stdev_seconds:.3f}", ", ".join(f"{sample:.3f}" for sample in row.samples)) for row in report.measurements])])


def regressions_markdown(findings: list[RegressionFinding]) -> str:
    return "\n".join(["# Regressions", "", markdown_table(["Workload", "Tier", "Baseline (s)", "Current (s)", "Delta (s)", "Delta %", "Class"], [(row.workload, row.tier, f"{row.baseline_seconds:.3f}", f"{row.current_seconds:.3f}", f"{row.delta_seconds:.3f}", f"{row.delta_pct:.2f}", row.classification) for row in findings])])
