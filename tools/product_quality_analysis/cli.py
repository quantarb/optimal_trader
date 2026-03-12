from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .analyzers.data_completeness import analyze_data_completeness
from .analyzers.design_token_analysis import analyze_design_tokens
from .analyzers.dom_complexity import analyze_dom_complexity
from .analyzers.empty_state_detection import analyze_empty_states
from .analyzers.layout_similarity import analyze_layout_similarity
from .analyzers.pagination_detection import analyze_pagination
from .analyzers.scalability_analysis import analyze_scalability
from .analyzers.table_readability import analyze_table_readability
from .analyzers.ui_consistency import analyze_ui_consistency
from .config import DEFAULT_BASELINE_LABEL, DEFAULT_CURRENT_LABEL, REPO_ROOT, default_config
from .crawlers.page_crawler import crawl_pages
from .crawlers.route_discovery import discover_routes
from .integrations.axe_runner import run_axe
from .integrations.data_quality_runner import discover_artifact_inventory
from .integrations.lighthouse_runner import run_lighthouse
from .integrations.stylelint_runner import run_stylelint
from .integrations.visual_regression import run_visual_regression
from .models import AnalysisConfig, AnalysisSnapshot, PageSnapshot, RankedIssue
from .reports.data_quality_report import data_quality_report_markdown
from .reports.fix_verification import compare_snapshots, fix_verification_markdown
from .reports.issue_ranking import prioritized_issues_markdown, rank_issues
from .reports.quality_summary import quality_summary_markdown
from .reports.scalability_report import scalability_report_markdown
from .reports.ui_consistency_report import ui_consistency_report_markdown
from .utils.report_utils import load_json, write_json, write_markdown


app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


def _selected_routes(config: AnalysisConfig, route_names: str) -> list:
    routes = discover_routes(config, discover_artifact_inventory(root=REPO_ROOT, tiers=config.symbol_tiers))
    selected = {token.strip() for token in str(route_names or "").split(",") if token.strip()}
    if not selected:
        return routes
    return [route for route in routes if route.name in selected]


def _issues_from_payloads(payloads: list[dict]) -> list[RankedIssue]:
    issues: list[RankedIssue] = []
    for payload in payloads:
        for issue in list(payload.get("issues") or []):
            issues.append(RankedIssue(**issue))
    return rank_issues(issues)


def _table_markdown(title: str, rows: list[dict], columns: list[str]) -> str:
    lines = [f"# {title}", ""]
    if not rows:
        lines.extend(["No rows were collected.", ""])
        return "\n".join(lines)
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "-")) for column in columns) + " |")
    lines.append("")
    return "\n".join(lines)


def _snapshot_from_payload(payload: dict | None) -> AnalysisSnapshot | None:
    if not payload:
        return None
    payload = dict(payload)
    payload["issues"] = [RankedIssue(**item) for item in list(payload.get("issues") or [])]
    payload["page_snapshots"] = [PageSnapshot(**item) for item in list(payload.get("page_snapshots") or [])]
    return AnalysisSnapshot(**payload)


def _build_snapshot(config: AnalysisConfig, *, route_names: str = "", visual_baseline_dir: str = "") -> AnalysisSnapshot:
    inventories = discover_artifact_inventory(root=REPO_ROOT, tiers=config.symbol_tiers)
    routes = _selected_routes(config, route_names)
    page_snapshots = crawl_pages(config, routes)
    data_quality = analyze_data_completeness(config, inventories, page_snapshots)
    pagination = analyze_pagination(config, page_snapshots)
    readability = analyze_table_readability(page_snapshots)
    dom_complexity = analyze_dom_complexity(config, page_snapshots)
    ui_consistency = analyze_ui_consistency(page_snapshots)
    design_tokens = analyze_design_tokens(page_snapshots)
    layout_similarity = analyze_layout_similarity(page_snapshots)
    empty_states = analyze_empty_states(page_snapshots)
    scalability = analyze_scalability(page_snapshots)
    lighthouse = {
        "routes": [
            {"page": page_snapshots[0].name, **run_lighthouse(page_snapshots[0].url, output_dir=config.output_dir / "lighthouse")}
        ] if page_snapshots else []
    }
    axe = run_axe()
    stylelint = run_stylelint(root=REPO_ROOT)
    visual_regression = run_visual_regression(
        page_snapshots,
        baseline_dir=Path(visual_baseline_dir).resolve() if visual_baseline_dir else config.output_dir / "visual_baseline",
        current_label=config.label,
    )
    issues = _issues_from_payloads([data_quality, pagination, readability, dom_complexity, ui_consistency, empty_states, scalability])
    snapshot = AnalysisSnapshot(
        label=config.label,
        base_url=config.base_url,
        routes=routes,
        page_snapshots=page_snapshots,
        artifact_inventory=inventories,
        data_quality=data_quality,
        pagination=pagination,
        readability=readability,
        dom_complexity=dom_complexity,
        ui_consistency=ui_consistency,
        design_tokens=design_tokens,
        layout_similarity=layout_similarity,
        empty_states=empty_states,
        scalability=scalability,
        lighthouse=lighthouse,
        axe=axe,
        stylelint=stylelint,
        visual_regression=visual_regression,
        issues=issues,
    )
    return snapshot


def _write_reports(snapshot: AnalysisSnapshot, output_dir: Path) -> None:
    write_json(output_dir / f"analysis_{snapshot.label}.json", snapshot)
    write_json(output_dir / "snapshots" / f"analysis_{snapshot.label}.json", snapshot)
    write_markdown(output_dir / "product_quality_summary.md", quality_summary_markdown(snapshot))
    write_markdown(output_dir / "prioritized_issues.md", prioritized_issues_markdown(snapshot.issues))
    write_markdown(output_dir / "data_quality_report.md", data_quality_report_markdown(snapshot))
    write_markdown(output_dir / "ui_consistency_report.md", ui_consistency_report_markdown(snapshot))
    write_markdown(
        output_dir / "pagination_report.md",
        _table_markdown("Pagination Report", list(snapshot.pagination.get("tables") or []), ["page", "table", "rows", "columns", "has_pagination", "page_size", "readability_risk_score"]),
    )
    write_markdown(
        output_dir / "readability_report.md",
        _table_markdown("Readability Report", list(snapshot.readability.get("tables") or []), ["page", "table", "rows", "columns", "risk"]),
    )
    write_markdown(output_dir / "scalability_report.md", scalability_report_markdown(snapshot))


def _config(output: str, label: str, tiers: str, base_url: str) -> AnalysisConfig:
    return default_config(base_url=base_url, output_dir=output, label=label, tiers=tuple(token.strip() for token in tiers.split(",") if token.strip()))


@app.command()
def crawl(
    output: str = typer.Option("docs/product_quality", help="Output directory."),
    label: str = typer.Option(DEFAULT_CURRENT_LABEL, help="Snapshot label."),
    base_url: str = typer.Option("http://127.0.0.1:8000", help="Base URL."),
    routes: str = typer.Option("", help="Comma-separated route names."),
    tiers: str = typer.Option("tier1,tier2,tier3", help="Comma-separated tier labels."),
) -> None:
    config = _config(output, label, tiers, base_url)
    inventories = discover_artifact_inventory(root=REPO_ROOT, tiers=config.symbol_tiers)
    selected = _selected_routes(config, routes)
    snapshots = crawl_pages(config, selected)
    payload = {"generated_at": snapshot.generated_at if (snapshot := AnalysisSnapshot(label=label, base_url=base_url)) else "", "pages": [item.model_dump(mode="json") for item in snapshots], "inventory": [item.model_dump(mode="json") for item in inventories]}
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
    snapshot = _snapshot_from_payload(load_json(Path(output).resolve() / "snapshots" / f"analysis_{label}.json", default={}))
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
    before = _snapshot_from_payload(load_json(Path(output).resolve() / "snapshots" / f"analysis_{baseline_label}.json", default={}))
    after = _snapshot_from_payload(load_json(Path(output).resolve() / "snapshots" / f"analysis_{current_label}.json", default={}))
    findings = compare_snapshots(before, after) if before and after else []
    write_markdown(Path(output).resolve() / "fix_verification.md", fix_verification_markdown(findings))
    write_json(Path(output).resolve() / "fix_verification.json", [item.model_dump(mode="json") for item in findings])
    console.print("[green]Fix verification report written[/green]")


@app.command()
def report(output: str = typer.Option("docs/product_quality"), label: str = typer.Option(DEFAULT_CURRENT_LABEL)) -> None:
    snapshot = _snapshot_from_payload(load_json(Path(output).resolve() / "snapshots" / f"analysis_{label}.json", default={}))
    if snapshot is None:
        raise typer.Exit(code=1)
    _write_reports(snapshot, Path(output).resolve())
    console.print("[green]Markdown reports refreshed[/green]")


if __name__ == "__main__":
    app()
