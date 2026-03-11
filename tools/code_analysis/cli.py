from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .run import (
    analyze_repo_bundle,
    analyze_call_graph,
    analyze_code_metrics,
    analyze_dead_code,
    analyze_dependency_graph,
    analyze_duplicate_code,
    build_repo_overview,
    build_semantic_index_bundle,
    build_semantic_search_results,
)
from .utils import DEFAULT_EMBEDDING_MODEL, DEFAULT_OUTPUT_DIR


app = typer.Typer(no_args_is_help=True, add_completion=False, rich_markup_mode="markdown")
console = Console()


@app.command(name="analyze_repo")
def analyze_repo_command(
    root: str = typer.Option(".", help="Repository root to analyze."),
    output: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory for generated reports."),
    model_name: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="SentenceTransformer model name for semantic index."),
) -> None:
    payload = analyze_repo_bundle(root=Path(root).resolve(), output_dir=Path(output).resolve(), model_name=model_name)
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
