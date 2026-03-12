from __future__ import annotations

from ..config import BenchmarkDefaults
from ..models import BenchmarkMeasurement, BenchmarkSuiteReport
from ..profilers.profile_targets import run_scalability_target
from ..utils.report_utils import summarize_samples, utc_timestamp
from .benchmark_targets import workload_runtime_from_tier_report


def run_scaling_benchmarks(*, label: str, benchmark: BenchmarkDefaults | None = None) -> BenchmarkSuiteReport:
    defaults = benchmark or BenchmarkDefaults()
    raw_runs: list[dict] = []
    buckets: dict[tuple[str, str], list[float]] = {}
    total_iterations = int(defaults.warmup_iterations) + int(defaults.measured_iterations)
    for tier in defaults.tiers:
        for iteration in range(total_iterations):
            report = run_scalability_target(tier, benchmark=defaults)
            raw_runs.append({"tier": tier, "iteration": iteration, "measured": iteration >= defaults.warmup_iterations, "report": report})
            if iteration < defaults.warmup_iterations:
                continue
            for workload, seconds in workload_runtime_from_tier_report(report).items():
                buckets.setdefault((workload, tier), []).append(float(seconds))
    measurements = []
    for (workload, tier), samples in sorted(buckets.items()):
        summary = summarize_samples(samples)
        measurements.append(BenchmarkMeasurement(workload, tier, len(samples), [round(float(sample), 6) for sample in samples], round(summary["mean_seconds"], 6), round(summary["median_seconds"], 6), round(summary["stdev_seconds"], 6), round(summary["min_seconds"], 6), round(summary["max_seconds"], 6), {}))
    return BenchmarkSuiteReport(utc_timestamp(), "internal_perf_counter", label, int(defaults.warmup_iterations), int(defaults.measured_iterations), measurements, raw_runs, {"tiers": list(defaults.tiers)})
