from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ComplexityFinding:
    path: str
    name: str
    object_type: str
    complexity: float
    rank: str
    maintainability_index: float | None = None
    loc: int = 0
    halstead_volume: float | None = None
    score: float = 0.0


@dataclass(frozen=True)
class ComplexityReport:
    generated_at: str
    engine: str
    module_findings: list[ComplexityFinding]
    function_findings: list[ComplexityFinding]
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DependencyHotspot:
    module: str
    path: str
    fan_in: int
    fan_out: int
    pagerank: float
    betweenness: float
    score: float


@dataclass(frozen=True)
class DependencyReport:
    generated_at: str
    engine: str
    nodes: list[DependencyHotspot]
    edges: list[tuple[str, str]]
    cycles: list[list[str]]
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PatternFinding:
    path: str
    line: int
    pattern: str
    severity: float
    message: str
    snippet: str = ""
    function_name: str = ""
    in_loop: bool = False


@dataclass(frozen=True)
class PatternReport:
    generated_at: str
    kind: str
    findings: list[PatternFinding]
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeHotspot:
    name: str
    path: str
    line: int
    ncalls: str
    cumulative_seconds: float
    total_seconds: float
    percentage: float


@dataclass(frozen=True)
class RuntimeProfileReport:
    generated_at: str
    engine: str
    target: str
    total_seconds: float
    raw_output_path: str
    hotspots: list[RuntimeHotspot]
    stage_hotspots: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MemoryHotspot:
    path: str
    line: int
    size_mb: float
    count: int
    traceback: str


@dataclass(frozen=True)
class MemoryProfileReport:
    generated_at: str
    engine: str
    target: str
    peak_rss_mb: float
    traced_peak_mb: float
    raw_output_path: str
    hotspots: list[MemoryHotspot]
    stage_hotspots: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkMeasurement:
    workload: str
    tier: str
    iterations: int
    samples: list[float]
    mean_seconds: float
    median_seconds: float
    stdev_seconds: float
    min_seconds: float
    max_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkSuiteReport:
    generated_at: str
    engine: str
    label: str
    warmup_iterations: int
    measured_iterations: int
    measurements: list[BenchmarkMeasurement]
    raw_runs: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegressionFinding:
    workload: str
    tier: str
    baseline_seconds: float
    current_seconds: float
    delta_seconds: float
    delta_pct: float
    classification: str


@dataclass(frozen=True)
class OptimizationCandidate:
    path: str
    label: str
    runtime_score: float
    memory_score: float
    complexity_score: float
    centrality_score: float
    scaling_score: float
    total_score: float
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OptimizationReport:
    generated_at: str
    weights: dict[str, float]
    candidates: list[OptimizationCandidate]
    summary: dict[str, Any] = field(default_factory=dict)
