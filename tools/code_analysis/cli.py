from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .run import (
    analyze_repo_bundle,
    analyze_architecture_rules,
    bootstrap_architecture_rules_file,
    capture_quality_snapshot,
    compare_quality_snapshots,
    analyze_call_graph,
    analyze_blast_radius,
    analyze_code_metrics,
    analyze_dead_code,
    analyze_dependency_graph,
    analyze_duplicate_code,
    build_repo_overview,
    build_semantic_index_bundle,
    build_semantic_search_results,
    generate_refactor_priority_report,
)
from .utils import DEFAULT_EMBEDDING_MODEL, DEFAULT_OUTPUT_DIR


app = typer.Typer(no_args_is_help=True, add_completion=False, rich_markup_mode="markdown")
console = Console()


@app.command(name="analyze_repo")
def analyze_repo_command(
    root: str = typer.Option(".", help="Repository root to analyze."),
    output: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory for generated reports."),
    model_name: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="SentenceTransformer model name for semantic index."),
    rules_path: str = typer.Option("", help="Optional architecture rules file."),
    weights_file: str = typer.Option("", help="Optional scorecard weights file."),
) -> None:
    payload = analyze_repo_bundle(
        root=Path(root).resolve(),
        output_dir=Path(output).resolve(),
        model_name=model_name,
        rules_path=Path(rules_path).resolve() if rules_path else None,
        weights_path=Path(weights_file).resolve() if weights_file else None,
    )
    console.print("[bold green]Analysis complete[/bold green]")
    console.print(payload["summary"])


@app.command(name="generate_dependency_graph")
def generate_dependency_graph_command(
    root: str = typer.Option(".", help="Repository root."),
    output: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory for generated reports."),
) -> None:
    report = analyze_dependency_graph(root=Path(root).resolve(), output_dir=Path(output).resolve())
    console.print(f"[green]Wrote dependency graph[/green] to {report['json_path']}")


@app.command(name="generate_call_graph")
def generate_call_graph_command(
    root: str = typer.Option(".", help="Repository root."),
    output: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory for generated reports."),
) -> None:
    report = analyze_call_graph(root=Path(root).resolve(), output_dir=Path(output).resolve())
    console.print(f"[green]Wrote call graph[/green] to {report['json_path']}")


@app.command(name="detect_duplicate_code")
def detect_duplicate_code_command(
    root: str = typer.Option(".", help="Repository root."),
    output: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory for generated reports."),
    model_name: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="SentenceTransformer model name."),
) -> None:
    report = analyze_duplicate_code(root=Path(root).resolve(), output_dir=Path(output).resolve(), model_name=model_name)
    console.print(f"[green]Wrote duplicate code report[/green] to {report['json_path']}")


@app.command(name="detect_dead_code")
def detect_dead_code_command(
    root: str = typer.Option(".", help="Repository root."),
    output: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory for generated reports."),
) -> None:
    report = analyze_dead_code(root=Path(root).resolve(), output_dir=Path(output).resolve())
    console.print(f"[green]Wrote dead code report[/green] to {report['json_path']}")


@app.command(name="analyze_complexity")
def analyze_complexity_command(
    root: str = typer.Option(".", help="Repository root."),
    output: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory for generated reports."),
) -> None:
    report = analyze_code_metrics(root=Path(root).resolve(), output_dir=Path(output).resolve())
    console.print(f"[green]Wrote code metrics report[/green] to {report['json_path']}")


@app.command(name="analyze_code_health")
def analyze_code_health_command(
    root: str = typer.Option(".", help="Repository root."),
    output: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory for generated reports."),
    model_name: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="SentenceTransformer model name for semantic index."),
    rules_path: str = typer.Option("", help="Optional architecture rules file."),
    weights_file: str = typer.Option("", help="Optional scorecard weights file."),
) -> None:
    payload = analyze_repo_bundle(
        root=Path(root).resolve(),
        output_dir=Path(output).resolve(),
        model_name=model_name,
        rules_path=Path(rules_path).resolve() if rules_path else None,
        weights_path=Path(weights_file).resolve() if weights_file else None,
    )
    console.print(f"[green]Wrote code-health reports[/green] to {Path(output).resolve()}")
    console.print({"repo_score": payload["quality_scorecard"]["report"]["repo_score"]})


@app.command(name="analyze_blast_radius")
def analyze_blast_radius_command(
    root: str = typer.Option(".", help="Repository root."),
    output: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory for generated reports."),
    model_name: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="SentenceTransformer model name for semantic index."),
    rules_path: str = typer.Option("", help="Optional architecture rules file."),
    weights_file: str = typer.Option("", help="Optional scorecard weights file."),
) -> None:
    payload = analyze_repo_bundle(
        root=Path(root).resolve(),
        output_dir=Path(output).resolve(),
        model_name=model_name,
        rules_path=Path(rules_path).resolve() if rules_path else None,
        weights_path=Path(weights_file).resolve() if weights_file else None,
    )
    console.print(f"[green]Wrote blast-radius reports[/green] to {payload['blast_radius']['markdown_path']}")


@app.command(name="generate_refactor_priority_report")
def generate_refactor_priority_report_command(
    output: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory containing generated analysis reports."),
) -> None:
    report = generate_refactor_priority_report(
        output_dir=Path(output).resolve(),
        blast_radius_payload={"report": _load_report(Path(output).resolve() / "blast_radius_report.json")},
    )
    console.print(f"[green]Wrote refactor priority report[/green] to {report['markdown_path']}")


@app.command(name="validate_architecture_rules")
def validate_architecture_rules_command(
    root: str = typer.Option(".", help="Repository root."),
    output: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory for generated reports."),
    rules_path: str = typer.Option("", help="Optional architecture rules file."),
    fail_on_violations: bool = typer.Option(False, help="Exit with status 1 when violations are found."),
) -> None:
    report = analyze_architecture_rules(
        root=Path(root).resolve(),
        output_dir=Path(output).resolve(),
        rules_path=Path(rules_path).resolve() if rules_path else None,
    )
    violation_count = len(report["report"]["violations"])
    console.print(f"[green]Architecture rules checked[/green] with {violation_count} violation(s)")
    if fail_on_violations and violation_count:
        raise typer.Exit(code=1)


@app.command(name="generate_architecture_report")
def generate_architecture_report_command(
    root: str = typer.Option(".", help="Repository root."),
    output: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory for generated reports."),
    rules_path: str = typer.Option("", help="Optional architecture rules file."),
) -> None:
    report = analyze_architecture_rules(
        root=Path(root).resolve(),
        output_dir=Path(output).resolve(),
        rules_path=Path(rules_path).resolve() if rules_path else None,
    )
    console.print(f"[green]Wrote architecture report[/green] to {report['json_path']}")


@app.command(name="bootstrap_architecture_rules")
def bootstrap_architecture_rules_command(
    root: str = typer.Option(".", help="Repository root."),
    output: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory for generated reports."),
    rules_path: str = typer.Option("", help="Where to write the bootstrapped rules file."),
) -> None:
    payload = bootstrap_architecture_rules_file(
        root=Path(root).resolve(),
        output_dir=Path(output).resolve(),
        rules_path=Path(rules_path).resolve() if rules_path else None,
    )
    console.print(f"[green]Bootstrapped architecture rules[/green] at {payload['path']}")


@app.command(name="build_semantic_index")
def build_semantic_index_command(
    root: str = typer.Option(".", help="Repository root."),
    output: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory for generated reports."),
    model_name: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="SentenceTransformer model name."),
) -> None:
    report = build_semantic_index_bundle(root=Path(root).resolve(), output_dir=Path(output).resolve(), model_name=model_name)
    console.print(f"[green]Built semantic index[/green] at {report['index_path']}")


@app.command(name="search_code")
def search_code_command(
    query: str = typer.Argument(..., help="Semantic search query."),
    output: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory containing semantic index."),
    model_name: Optional[str] = typer.Option(None, help="Override model name for the query encoder."),
    top_k: int = typer.Option(8, help="Number of results to return."),
) -> None:
    payload = build_semantic_search_results(
        query=query,
        output_dir=Path(output).resolve(),
        model_name=model_name,
        top_k=top_k,
    )
    table = Table(title=f"Code search: {query}")
    table.add_column("Score", justify="right")
    table.add_column("Chunk")
    table.add_column("Location")
    for row in payload["results"]:
        table.add_row(f"{row['score']:.3f}", row["chunk_id"], f"{row['path']}:{row['lineno']}")
    console.print(table)


@app.command(name="generate_repo_overview")
def generate_repo_overview_command(
    output: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory containing generated analysis reports."),
) -> None:
    report = build_repo_overview(output_dir=Path(output).resolve())
    console.print(f"[green]Wrote repo overview[/green] to {report['markdown_path']}")


@app.command(name="capture_quality_snapshot")
def capture_quality_snapshot_command(
    root: str = typer.Option(".", help="Repository root."),
    output: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory for generated reports."),
    label: str = typer.Option("current", help="Snapshot label."),
    model_name: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="SentenceTransformer model name for semantic index."),
    rules_path: str = typer.Option("", help="Optional architecture rules file."),
    weights_file: str = typer.Option("", help="Optional scorecard weights file."),
) -> None:
    payload = capture_quality_snapshot(
        root=Path(root).resolve(),
        output_dir=Path(output).resolve(),
        label=label,
        model_name=model_name,
        rules_path=Path(rules_path).resolve() if rules_path else None,
        weights_path=Path(weights_file).resolve() if weights_file else None,
    )
    console.print(f"[green]Captured quality snapshot[/green] at {payload['json_path']}")


@app.command(name="compare_quality_snapshots")
def compare_quality_snapshots_command(
    baseline_label: str = typer.Argument(..., help="Baseline snapshot label."),
    current_label: str = typer.Argument(..., help="Current snapshot label."),
    output: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory containing quality snapshots."),
) -> None:
    payload = compare_quality_snapshots(
        output_dir=Path(output).resolve(),
        baseline_label=baseline_label,
        current_label=current_label,
    )
    console.print(f"[green]Wrote quality comparison[/green] to {payload['markdown_path']}")


def _load_report(path: Path) -> dict:
    if not path.exists():
        raise typer.Exit(code=1)
    import json

    return json.loads(path.read_text(encoding="utf-8"))
