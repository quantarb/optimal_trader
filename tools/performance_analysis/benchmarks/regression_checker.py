from __future__ import annotations

from ..models import BenchmarkSuiteReport, RegressionFinding


def compare_benchmarks(baseline: BenchmarkSuiteReport, current: BenchmarkSuiteReport, *, neutral_threshold_pct: float = 5.0) -> list[RegressionFinding]:
    current_map = {(row.workload, row.tier): row for row in current.measurements}
    findings: list[RegressionFinding] = []
    for baseline_row in baseline.measurements:
        current_row = current_map.get((baseline_row.workload, baseline_row.tier))
        if current_row is None:
            continue
        delta = float(current_row.median_seconds) - float(baseline_row.median_seconds)
        delta_pct = (delta / float(baseline_row.median_seconds) * 100.0) if baseline_row.median_seconds else 0.0
        if delta_pct <= -abs(neutral_threshold_pct):
            classification = "improved"
        elif delta_pct >= abs(neutral_threshold_pct):
            classification = "regressed"
        else:
            classification = "neutral"
        findings.append(RegressionFinding(baseline_row.workload, baseline_row.tier, round(float(baseline_row.median_seconds), 6), round(float(current_row.median_seconds), 6), round(delta, 6), round(delta_pct, 4), classification))
    return sorted(findings, key=lambda row: (row.classification != "regressed", abs(row.delta_pct) * -1.0))
