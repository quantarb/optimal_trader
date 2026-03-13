from __future__ import annotations

from pathlib import Path

from .analyzers.data_completeness import analyze_data_completeness
from .analyzers.design_token_analysis import analyze_design_tokens
from .analyzers.dom_complexity import analyze_dom_complexity
from .analyzers.empty_state_detection import analyze_empty_states
from .analyzers.layout_similarity import analyze_layout_similarity
from .analyzers.pagination_detection import analyze_pagination
from .analyzers.scalability_analysis import analyze_scalability
from .analyzers.table_readability import analyze_table_readability
from .analyzers.ui_consistency import analyze_ui_consistency
from .config import REPO_ROOT, default_config
from .crawlers.page_crawler import crawl_pages
from .crawlers.route_discovery import discover_routes
from .integrations.axe_runner import run_axe
from .integrations.data_quality_runner import discover_artifact_inventory
from .integrations.lighthouse_runner import run_lighthouse
from .integrations.stylelint_runner import run_stylelint
from .integrations.visual_regression import run_visual_regression
from .models import AnalysisConfig, AnalysisSnapshot, PageSnapshot, RankedIssue
from .reports.issue_ranking import rank_issues
from .utils.report_utils import load_json


def _artifact_inventory(config: AnalysisConfig) -> list:
    return discover_artifact_inventory(root=REPO_ROOT, tiers=config.symbol_tiers)


def _route_name_filter(route_names: str) -> set[str]:
    return {token.strip() for token in str(route_names or "").split(",") if token.strip()}


def _selected_routes(config: AnalysisConfig, route_names: str, *, inventories: list | None = None) -> list:
    routes = discover_routes(config, inventories if inventories is not None else _artifact_inventory(config))
    selected = _route_name_filter(route_names)
    if not selected:
        return routes
    return [route for route in routes if route.name in selected]


def _issues_from_payloads(payloads: list[dict]) -> list[RankedIssue]:
    return rank_issues(
        [
            RankedIssue(**issue)
            for payload in payloads
            for issue in list(payload.get("issues") or [])
        ]
    )


def _snapshot_from_payload(payload: dict | None) -> AnalysisSnapshot | None:
    if not payload:
        return None
    payload = dict(payload)
    payload["issues"] = [RankedIssue(**item) for item in list(payload.get("issues") or [])]
    payload["page_snapshots"] = [PageSnapshot(**item) for item in list(payload.get("page_snapshots") or [])]
    return AnalysisSnapshot(**payload)


def _page_analysis_payloads(config: AnalysisConfig, inventories: list, page_snapshots: list[PageSnapshot]) -> dict[str, dict]:
    return {
        "data_quality": analyze_data_completeness(config, inventories, page_snapshots),
        "pagination": analyze_pagination(config, page_snapshots),
        "readability": analyze_table_readability(page_snapshots),
        "dom_complexity": analyze_dom_complexity(config, page_snapshots),
        "ui_consistency": analyze_ui_consistency(page_snapshots),
        "design_tokens": analyze_design_tokens(page_snapshots),
        "layout_similarity": analyze_layout_similarity(page_snapshots),
        "empty_states": analyze_empty_states(page_snapshots),
        "scalability": analyze_scalability(page_snapshots),
    }


def _integration_payloads(
    config: AnalysisConfig,
    *,
    page_snapshots: list[PageSnapshot],
    visual_baseline_dir: str,
) -> dict[str, dict]:
    baseline_dir = Path(visual_baseline_dir).resolve() if visual_baseline_dir else config.output_dir / "visual_baseline"
    lighthouse = {
        "routes": [
            {"page": page_snapshots[0].name, **run_lighthouse(page_snapshots[0].url, output_dir=config.output_dir / "lighthouse")}
        ]
        if page_snapshots
        else []
    }
    return {
        "lighthouse": lighthouse,
        "axe": run_axe(),
        "stylelint": run_stylelint(root=REPO_ROOT),
        "visual_regression": run_visual_regression(
            page_snapshots,
            baseline_dir=baseline_dir,
            current_label=config.label,
        ),
    }


def _build_snapshot(config: AnalysisConfig, *, route_names: str = "", visual_baseline_dir: str = "") -> AnalysisSnapshot:
    inventories = _artifact_inventory(config)
    routes = _selected_routes(config, route_names, inventories=inventories)
    page_snapshots = crawl_pages(config, routes)
    analyses = _page_analysis_payloads(config, inventories, page_snapshots)
    integrations = _integration_payloads(
        config,
        page_snapshots=page_snapshots,
        visual_baseline_dir=visual_baseline_dir,
    )
    issues = _issues_from_payloads(
        [
            analyses["data_quality"],
            analyses["pagination"],
            analyses["readability"],
            analyses["dom_complexity"],
            analyses["ui_consistency"],
            analyses["empty_states"],
            analyses["scalability"],
        ]
    )
    return AnalysisSnapshot(
        label=config.label,
        base_url=config.base_url,
        routes=routes,
        page_snapshots=page_snapshots,
        artifact_inventory=inventories,
        data_quality=analyses["data_quality"],
        pagination=analyses["pagination"],
        readability=analyses["readability"],
        dom_complexity=analyses["dom_complexity"],
        ui_consistency=analyses["ui_consistency"],
        design_tokens=analyses["design_tokens"],
        layout_similarity=analyses["layout_similarity"],
        empty_states=analyses["empty_states"],
        scalability=analyses["scalability"],
        lighthouse=integrations["lighthouse"],
        axe=integrations["axe"],
        stylelint=integrations["stylelint"],
        visual_regression=integrations["visual_regression"],
        issues=issues,
    )


def _crawl_snapshots(config: AnalysisConfig, routes: list) -> list[PageSnapshot]:
    return crawl_pages(config, routes)


def _crawl_payload(
    *,
    label: str,
    base_url: str,
    inventories: list,
    snapshots: list[PageSnapshot],
) -> dict[str, object]:
    snapshot = AnalysisSnapshot(label=label, base_url=base_url)
    return {
        "generated_at": snapshot.generated_at,
        "pages": [item.model_dump(mode="json") for item in snapshots],
        "inventory": [item.model_dump(mode="json") for item in inventories],
    }


def _load_snapshot(output: str, label: str) -> AnalysisSnapshot | None:
    return _snapshot_from_payload(load_json(Path(output).resolve() / "snapshots" / f"analysis_{label}.json", default={}))


def _config(output: str, label: str, tiers: str, base_url: str) -> AnalysisConfig:
    return default_config(
        base_url=base_url,
        output_dir=output,
        label=label,
        tiers=tuple(token.strip() for token in tiers.split(",") if token.strip()),
    )
