from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .config import DEFAULT_BASELINE_LABEL, DEFAULT_CURRENT_LABEL
from .reporting_support import _write_reports
from .reports.data_quality_report import data_quality_report_markdown
from .reports.fix_verification import compare_snapshots, fix_verification_markdown
from .reports.issue_ranking import prioritized_issues_markdown, rank_issues
from .reports.scalability_report import scalability_report_markdown
from .reports.ui_consistency_report import ui_consistency_report_markdown
from .snapshot_support import (
    _artifact_inventory,
    _build_snapshot,
    _config,
    _crawl_snapshots,
    _crawl_payload,
    _load_snapshot,
    _selected_routes,
)
from .utils.report_utils import write_json, write_markdown


app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def crawl(
    output: str = typer.Option("docs/product_quality", help="Output directory."),
    label: str = typer.Option(DEFAULT_CURRENT_LABEL, help="Snapshot label."),
    base_url: str = typer.Option("http://127.0.0.1:8000", help="Base URL."),
    routes: str = typer.Option("", help="Comma-separated route names."),
    tiers: str = typer.Option("tier1,tier2,tier3", help="Comma-separated tier labels."),
) -> None:
    config = _config(output, label, tiers, base_url)
    inventories = _artifact_inventory(config)
    selected = _selected_routes(config, routes, inventories=inventories)
    snapshots = _crawl_snapshots(config, selected)
    payload = _crawl_payload(label=label, base_url=base_url, inventories=inventories, snapshots=snapshots)
    write_json(Path(output).resolve() / f"crawl_{label}.json", payload)
    console.print(f"[green]Wrote crawl data to[/green] {Path(output).resolve()}")


@app.command()
def analyze(
    output: str = typer.Option("docs/product_quality", help="Output directory."),
    label: str = typer.Option(DEFAULT_CURRENT_LABEL, help="Snapshot label."),
    base_url: str = typer.Option("http://127.0.0.1:8000", help="Base URL."),
    routes: str = typer.Option("", help="Comma-separated route names."),
    tiers: str = typer.Option("tier1,tier2,tier3", help="Comma-separated tier labels."),
    visual_baseline_dir: str = typer.Option("", help="Directory for screenshot baselines."),
) -> None:
    config = _config(output, label, tiers, base_url)
    snapshot = _build_snapshot(config, route_names=routes, visual_baseline_dir=visual_baseline_dir)
    _write_reports(snapshot, Path(output).resolve())
    console.print(f"[green]Wrote product-quality analysis to[/green] {Path(output).resolve()}")


@app.command(name="data-quality")
def data_quality(
    output: str = typer.Option("docs/product_quality"),
    label: str = typer.Option(DEFAULT_CURRENT_LABEL),
    base_url: str = typer.Option("http://127.0.0.1:8000"),
    tiers: str = typer.Option("tier1,tier2,tier3"),
) -> None:
    config = _config(output, label, tiers, base_url)
    snapshot = _build_snapshot(config)
    write_markdown(Path(output).resolve() / "data_quality_report.md", data_quality_report_markdown(snapshot))
    write_json(Path(output).resolve() / "data_quality_report.json", snapshot.data_quality)
    console.print("[green]Data quality report written[/green]")


@app.command(name="ui-consistency")
def ui_consistency(
    output: str = typer.Option("docs/product_quality"),
    label: str = typer.Option(DEFAULT_CURRENT_LABEL),
    base_url: str = typer.Option("http://127.0.0.1:8000"),
    tiers: str = typer.Option("tier1,tier2,tier3"),
) -> None:
    config = _config(output, label, tiers, base_url)
    snapshot = _build_snapshot(config)
    write_markdown(Path(output).resolve() / "ui_consistency_report.md", ui_consistency_report_markdown(snapshot))
    write_json(Path(output).resolve() / "ui_consistency_report.json", snapshot.ui_consistency)
    console.print("[green]UI consistency report written[/green]")


@app.command()
def scalability(
    output: str = typer.Option("docs/product_quality"),
    label: str = typer.Option(DEFAULT_CURRENT_LABEL),
    base_url: str = typer.Option("http://127.0.0.1:8000"),
    tiers: str = typer.Option("tier1,tier2,tier3"),
) -> None:
    config = _config(output, label, tiers, base_url)
    snapshot = _build_snapshot(config)
    write_markdown(Path(output).resolve() / "scalability_report.md", scalability_report_markdown(snapshot))
    write_json(Path(output).resolve() / "scalability_report.json", snapshot.scalability)
    console.print("[green]Scalability report written[/green]")


@app.command(name="rank-issues")
def rank_issues_command(output: str = typer.Option("docs/product_quality"), label: str = typer.Option(DEFAULT_CURRENT_LABEL)) -> None:
    snapshot = _load_snapshot(output, label)
    if snapshot is None:
        raise typer.Exit(code=1)
    ranked = rank_issues(snapshot.issues)
    write_markdown(Path(output).resolve() / "prioritized_issues.md", prioritized_issues_markdown(ranked))
    write_json(Path(output).resolve() / "prioritized_issues.json", [item.model_dump(mode="json") for item in ranked])
    console.print("[green]Prioritized issues report written[/green]")


@app.command(name="verify-fixes")
def verify_fixes(
    output: str = typer.Option("docs/product_quality"),
    baseline_label: str = typer.Option(DEFAULT_BASELINE_LABEL),
    current_label: str = typer.Option(DEFAULT_CURRENT_LABEL),
) -> None:
    before = _load_snapshot(output, baseline_label)
    after = _load_snapshot(output, current_label)
    findings = compare_snapshots(before, after) if before and after else []
    write_markdown(Path(output).resolve() / "fix_verification.md", fix_verification_markdown(findings))
    write_json(Path(output).resolve() / "fix_verification.json", [item.model_dump(mode="json") for item in findings])
    console.print("[green]Fix verification report written[/green]")


@app.command()
def report(output: str = typer.Option("docs/product_quality"), label: str = typer.Option(DEFAULT_CURRENT_LABEL)) -> None:
    snapshot = _load_snapshot(output, label)
    if snapshot is None:
        raise typer.Exit(code=1)
    _write_reports(snapshot, Path(output).resolve())
    console.print("[green]Markdown reports refreshed[/green]")


if __name__ == "__main__":
    app()
