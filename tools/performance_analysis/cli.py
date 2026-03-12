from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .analyzers.complexity_analysis import analyze_complexity, complexity_markdown
from .analyzers.dataframe_patterns import analyze_dataframe_patterns, dataframe_patterns_markdown
from .analyzers.dependency_hotspots import analyze_dependency_hotspots, dependency_hotspots_markdown
from .analyzers.scaling_risk_analysis import analyze_scaling_risks, scaling_risks_markdown
from .benchmarks.benchmark_runner import run_benchmarks
from .benchmarks.regression_checker import compare_benchmarks
from .config import BenchmarkDefaults, default_config
from .models import BenchmarkMeasurement, BenchmarkSuiteReport
from .profilers.memory_profiler import profile_memory
from .profilers.runtime_profiler import profile_runtime
from .reports.benchmark_report import benchmark_report_markdown, regressions_markdown
from .reports.hotspot_report import memory_hotspots_markdown, runtime_hotspots_markdown
from .reports.optimization_candidates import build_optimization_candidates, optimization_candidates_markdown
from .reports.performance_summary import performance_summary_markdown
from .utils.path_utils import ensure_directory
from .utils.report_utils import load_json, write_json, write_markdown


app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


def _defaults(*, warmup_iterations: int, measured_iterations: int, tiers: str) -> BenchmarkDefaults:
    base = default_config().benchmark
    chosen_tiers = tuple(token.strip() for token in str(tiers).split(",") if token.strip()) or base.tiers
    return BenchmarkDefaults(
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        tiers=chosen_tiers,
        feature_profile=base.feature_profile,
        start_date=base.start_date,
        end_date=base.end_date,
        train_end_date=base.train_end_date,
        score_start_date=base.score_start_date,
        artifact_storage_format=base.artifact_storage_format,
        min_profit_pct=base.min_profit_pct,
        label_k_params=base.label_k_params,
        buy_execution=base.buy_execution,
        sell_execution=base.sell_execution,
        short_execution=base.short_execution,
        cover_execution=base.cover_execution,
    )


def _benchmark_from_payload(payload: dict | None) -> BenchmarkSuiteReport | None:
    if not payload:
        return None
    measurements = [BenchmarkMeasurement(**row) for row in list(payload.get("measurements") or [])]
    return BenchmarkSuiteReport(str(payload.get("generated_at") or ""), str(payload.get("engine") or ""), str(payload.get("label") or ""), int(payload.get("warmup_iterations") or 0), int(payload.get("measured_iterations") or 0), measurements, list(payload.get("raw_runs") or []), dict(payload.get("summary") or {}))


def _run_analysis(*, root: Path, output_dir: Path, benchmark_label: str, profile_target: str, warmup_iterations: int, measured_iterations: int, tiers: str) -> dict:
    benchmark_defaults = _defaults(warmup_iterations=warmup_iterations, measured_iterations=measured_iterations, tiers=tiers)
    output_dir = ensure_directory(output_dir)
    complexity = analyze_complexity(root)
    dependency = analyze_dependency_hotspots(root)
    dataframe = analyze_dataframe_patterns(root)
    scaling = analyze_scaling_risks(root)
    runtime = profile_runtime(profile_target, output_dir=output_dir, benchmark=benchmark_defaults)
    memory = profile_memory(profile_target, output_dir=output_dir, benchmark=benchmark_defaults)
    benchmark = run_benchmarks(label=benchmark_label, benchmark=benchmark_defaults)
    optimization = build_optimization_candidates(runtime_report=runtime, memory_report=memory, complexity_report=complexity, dependency_report=dependency, dataframe_report=dataframe, scaling_report=scaling)
    write_json(output_dir / "complexity_hotspots.json", complexity)
    write_json(output_dir / "dependency_hotspots.json", dependency)
    write_json(output_dir / "dataframe_antipatterns.json", dataframe)
    write_json(output_dir / "scaling_risks.json", scaling)
    write_json(output_dir / "runtime_hotspots.json", runtime)
    write_json(output_dir / "memory_hotspots.json", memory)
    write_json(output_dir / f"scaling_benchmarks_{benchmark_label}.json", benchmark)
    write_json(output_dir / "optimization_targets.json", optimization)
    write_markdown(output_dir / "complexity_hotspots.md", complexity_markdown(complexity))
    write_markdown(output_dir / "dependency_hotspots.md", dependency_hotspots_markdown(dependency))
    write_markdown(output_dir / "dataframe_antipatterns.md", dataframe_patterns_markdown(dataframe))
    write_markdown(output_dir / "scaling_risks.md", scaling_risks_markdown(scaling))
    write_markdown(output_dir / "runtime_hotspots.md", runtime_hotspots_markdown(runtime))
    write_markdown(output_dir / "memory_hotspots.md", memory_hotspots_markdown(memory))
    write_markdown(output_dir / "scaling_benchmarks.md", benchmark_report_markdown(benchmark))
    write_markdown(output_dir / "optimization_targets.md", optimization_candidates_markdown(optimization))
    write_json(output_dir / "snapshots" / f"benchmark_{benchmark_label}.json", benchmark)
    baseline = _benchmark_from_payload(load_json(output_dir / "snapshots" / "benchmark_baseline.json", default={}))
    regressions = compare_benchmarks(baseline, benchmark) if baseline and benchmark_label != "baseline" else []
    write_markdown(output_dir / "regressions.md", regressions_markdown(regressions) if regressions else "# Regressions\n\nNo baseline comparison available.\n")
    write_markdown(output_dir / "performance_summary.md", performance_summary_markdown(complexity_report=complexity, dependency_report=dependency, runtime_report=runtime, memory_report=memory, dataframe_report=dataframe, scaling_report=scaling, benchmark_report=benchmark, optimization_report=optimization, regressions=regressions))
    return {"complexity": complexity, "dependency": dependency, "dataframe": dataframe, "scaling": scaling, "runtime": runtime, "memory": memory, "benchmark": benchmark, "optimization": optimization, "regressions": regressions}


@app.command()
def analyze(root: str = typer.Option(".", help="Repository root."), output: str = typer.Option("docs/performance", help="Output directory."), benchmark_label: str = typer.Option("current", help="Snapshot label."), profile_target: str = typer.Option("scalability_tier2", help="Profile target."), warmup_iterations: int = typer.Option(0, help="Warmup iterations."), measured_iterations: int = typer.Option(1, help="Measured iterations."), tiers: str = typer.Option("tier1,tier2,tier3", help="Comma-separated tiers.")) -> None:
    _run_analysis(root=Path(root).resolve(), output_dir=Path(output).resolve(), benchmark_label=benchmark_label, profile_target=profile_target, warmup_iterations=warmup_iterations, measured_iterations=measured_iterations, tiers=tiers)
    console.print(f"[green]Wrote performance analysis to[/green] {Path(output).resolve()}")


@app.command()
def complexity(root: str = typer.Option("."), output: str = typer.Option("docs/performance")) -> None:
    report = analyze_complexity(Path(root).resolve())
    write_json(Path(output).resolve() / "complexity_hotspots.json", report)
    write_markdown(Path(output).resolve() / "complexity_hotspots.md", complexity_markdown(report))
    console.print("[green]Complexity report written[/green]")


@app.command()
def profile(output: str = typer.Option("docs/performance"), target: str = typer.Option("scalability_tier2")) -> None:
    report = profile_runtime(target, output_dir=Path(output).resolve(), benchmark=_defaults(warmup_iterations=0, measured_iterations=1, tiers="tier1,tier2,tier3"))
    write_json(Path(output).resolve() / "runtime_hotspots.json", report)
    write_markdown(Path(output).resolve() / "runtime_hotspots.md", runtime_hotspots_markdown(report))
    console.print("[green]Runtime profile written[/green]")


@app.command()
def memory(output: str = typer.Option("docs/performance"), target: str = typer.Option("scalability_tier2")) -> None:
    report = profile_memory(target, output_dir=Path(output).resolve(), benchmark=_defaults(warmup_iterations=0, measured_iterations=1, tiers="tier1,tier2,tier3"))
    write_json(Path(output).resolve() / "memory_hotspots.json", report)
    write_markdown(Path(output).resolve() / "memory_hotspots.md", memory_hotspots_markdown(report))
    console.print("[green]Memory profile written[/green]")


@app.command()
def benchmark(output: str = typer.Option("docs/performance"), benchmark_label: str = typer.Option("current"), warmup_iterations: int = typer.Option(0), measured_iterations: int = typer.Option(1), tiers: str = typer.Option("tier1,tier2,tier3")) -> None:
    report = run_benchmarks(label=benchmark_label, benchmark=_defaults(warmup_iterations=warmup_iterations, measured_iterations=measured_iterations, tiers=tiers))
    write_json(Path(output).resolve() / f"scaling_benchmarks_{benchmark_label}.json", report)
    write_json(Path(output).resolve() / "snapshots" / f"benchmark_{benchmark_label}.json", report)
    write_markdown(Path(output).resolve() / "scaling_benchmarks.md", benchmark_report_markdown(report))
    console.print("[green]Benchmarks written[/green]")


@app.command()
def regressions(output: str = typer.Option("docs/performance"), baseline_label: str = typer.Option("baseline"), current_label: str = typer.Option("current")) -> None:
    baseline = _benchmark_from_payload(load_json(Path(output).resolve() / "snapshots" / f"benchmark_{baseline_label}.json", default={}))
    current = _benchmark_from_payload(load_json(Path(output).resolve() / "snapshots" / f"benchmark_{current_label}.json", default={}))
    findings = compare_benchmarks(baseline, current) if baseline and current else []
    write_markdown(Path(output).resolve() / "regressions.md", regressions_markdown(findings) if findings else "# Regressions\n\nNo comparison data available.\n")
    console.print("[green]Regression report written[/green]")


@app.command(name="hotspots")
def hotspots(root: str = typer.Option("."), output: str = typer.Option("docs/performance"), profile_target: str = typer.Option("scalability_tier2")) -> None:
    _run_analysis(root=Path(root).resolve(), output_dir=Path(output).resolve(), benchmark_label="current", profile_target=profile_target, warmup_iterations=0, measured_iterations=1, tiers="tier1,tier2,tier3")
    console.print("[green]Hotspot reports refreshed[/green]")


@app.command(name="optimize-report")
def optimize_report(root: str = typer.Option("."), output: str = typer.Option("docs/performance"), profile_target: str = typer.Option("scalability_tier2")) -> None:
    _run_analysis(root=Path(root).resolve(), output_dir=Path(output).resolve(), benchmark_label="current", profile_target=profile_target, warmup_iterations=0, measured_iterations=1, tiers="tier1,tier2,tier3")
    console.print("[green]Optimization report refreshed[/green]")


if __name__ == "__main__":
    app()
