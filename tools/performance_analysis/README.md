# Performance Analysis Toolkit

Thin orchestration around mature profiling and benchmarking tools for this repo. The goal is not to invent a profiler. The goal is to combine repo-specific targets with:

- `radon` for complexity and maintainability
- `networkx` for dependency centrality
- `pyinstrument` when available, with `cProfile` fallback
- `scalene` / `memory_profiler` when available, with `tracemalloc` fallback
- `pytest-benchmark` when available, with an internal `perf_counter` harness fallback
- `typer` + `rich` for CLI UX

## Architecture

- `analyzers/`
  - static repo analysis: complexity, dependency hotspots, DataFrame anti-patterns, scaling risks
- `profilers/`
  - runtime and memory profiling against real repo targets
- `benchmarks/`
  - tiered scalability benchmarks and regression comparison
- `reports/`
  - Markdown/JSON summaries plus weighted optimization ranking
- `cli.py`
  - one-command orchestration and focused reruns

## Install

Required baseline:

```bash
pip install typer rich radon networkx
```

Recommended extras:

```bash
pip install pyinstrument memory_profiler pytest-benchmark
pip install scalene
pip install pydeps
pip install pyan3
pip install line_profiler
pip install asv
```

Repo-local environment example:

```bash
/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m pip install typer rich radon networkx pyinstrument memory_profiler pytest-benchmark
```

## CLI

```bash
python -m tools.performance_analysis.cli analyze
python -m tools.performance_analysis.cli complexity
python -m tools.performance_analysis.cli profile
python -m tools.performance_analysis.cli memory
python -m tools.performance_analysis.cli benchmark --benchmark-label baseline
python -m tools.performance_analysis.cli regressions --baseline-label baseline --current-label after
python -m tools.performance_analysis.cli hotspots
python -m tools.performance_analysis.cli optimize-report
```

## Real Targets

The default runtime/memory/benchmark target is the repo's existing scalability workflow:

- `scalability_tier1`: 10 symbols
- `scalability_tier2`: 100 symbols
- `scalability_tier3`: 1000+ symbols

Synthetic fixture seeding is handled automatically through the repo's existing `pipeline.test_support.ScalabilityFixtureMixin`.

## Outputs

Reports are written under `docs/performance/`:

- `performance_summary.md`
- `complexity_hotspots.md`
- `runtime_hotspots.md`
- `memory_hotspots.md`
- `scaling_benchmarks.md`
- `optimization_targets.md`
- `regressions.md`
- `optimization_changes.md`

Machine-readable JSON is written alongside the Markdown files plus benchmark snapshots under `docs/performance/snapshots/`.

## Baseline vs After

1. Run a baseline snapshot:

```bash
python -m tools.performance_analysis.cli analyze --benchmark-label baseline
```

2. Implement code changes.
3. Run the after snapshot:

```bash
python -m tools.performance_analysis.cli analyze --benchmark-label after
python -m tools.performance_analysis.cli regressions --baseline-label baseline --current-label after
```

## Interpreting Reports

- `complexity_hotspots.md`: modules/functions with the worst complexity and maintainability signals
- `runtime_hotspots.md`: hottest stage timings plus function-level profiler output when available
- `memory_hotspots.md`: peak memory plus top allocation sites
- `scaling_benchmarks.md`: per-tier end-to-end and stage-group timings
- `optimization_targets.md`: weighted ranking combining runtime, memory, complexity, dependency centrality, and scaling risk

## How LLM Agents Should Use It

1. Start with `performance_summary.md` and `optimization_targets.md`.
2. Confirm the top-ranked file is also present in runtime or memory evidence.
3. Prefer removing repeated work, repeated copies, repeated scans, repeated joins, and repeated model fits inside loops.
4. Re-run `analyze` or `benchmark` after each optimization and update `optimization_changes.md` with before/after evidence.
