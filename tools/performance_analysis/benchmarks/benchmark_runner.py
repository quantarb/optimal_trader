from __future__ import annotations

from ..config import BenchmarkDefaults
from ..models import BenchmarkSuiteReport
from .scaling_benchmarks import run_scaling_benchmarks


def run_benchmarks(*, label: str, benchmark: BenchmarkDefaults | None = None) -> BenchmarkSuiteReport:
    return run_scaling_benchmarks(label=label, benchmark=benchmark)
