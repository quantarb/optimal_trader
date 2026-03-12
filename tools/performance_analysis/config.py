from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "performance"
DEFAULT_SNAPSHOT_DIR = DEFAULT_OUTPUT_DIR / "snapshots"
DEFAULT_BENCHMARK_LABEL = "current"
DEFAULT_PROFILE_TARGET = "scalability_tier2"

WORKLOAD_STAGE_PREFIXES: dict[str, tuple[str, ...]] = {
    "feature_generation": ("features.",),
    "label_generation": ("labels.",),
    "model_training": ("model.fit",),
    "model_scoring": ("model.score",),
    "strategy_dataset": ("strategy.",),
    "backtest": ("backtest.",),
}


@dataclass(frozen=True)
class OptimizationWeights:
    runtime_weight: float = 0.35
    memory_weight: float = 0.20
    complexity_weight: float = 0.15
    centrality_weight: float = 0.15
    scaling_weight: float = 0.15


@dataclass(frozen=True)
class BenchmarkDefaults:
    warmup_iterations: int = 1
    measured_iterations: int = 2
    tiers: tuple[str, ...] = ("tier1", "tier2", "tier3")
    feature_profile: str = "baseline"
    start_date: str = "2024-01-02"
    end_date: str = "2024-05-06"
    train_end_date: str = "2024-05-06"
    score_start_date: str = "2024-01-02"
    artifact_storage_format: str = "csv"
    min_profit_pct: float = 0.0
    label_k_params: dict[str, list[int]] = field(default_factory=lambda: {"M": [1]})
    buy_execution: str = "adj_open"
    sell_execution: str = "adj_close"
    short_execution: str = "adj_open"
    cover_execution: str = "adj_close"


@dataclass(frozen=True)
class PerformanceAnalysisConfig:
    root: Path = REPO_ROOT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR
    benchmark_label: str = DEFAULT_BENCHMARK_LABEL
    profile_target: str = DEFAULT_PROFILE_TARGET
    benchmark: BenchmarkDefaults = field(default_factory=BenchmarkDefaults)
    optimization_weights: OptimizationWeights = field(default_factory=OptimizationWeights)
    include_tests: bool = True


def default_config(
    *,
    root: str | Path | None = None,
    output_dir: str | Path | None = None,
    benchmark_label: str = DEFAULT_BENCHMARK_LABEL,
    profile_target: str = DEFAULT_PROFILE_TARGET,
) -> PerformanceAnalysisConfig:
    resolved_root = Path(root).resolve() if root is not None else REPO_ROOT
    resolved_output = Path(output_dir).resolve() if output_dir is not None else DEFAULT_OUTPUT_DIR
    return PerformanceAnalysisConfig(
        root=resolved_root,
        output_dir=resolved_output,
        snapshot_dir=resolved_output / "snapshots",
        benchmark_label=str(benchmark_label or DEFAULT_BENCHMARK_LABEL),
        profile_target=str(profile_target or DEFAULT_PROFILE_TARGET),
    )
