from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .architecture_rules import architecture_rules_markdown, bootstrap_architecture_rules, validate_architecture_rules
from .baseline_compare import (
    compare_quality_snapshots as build_quality_snapshot_comparison,
    quality_comparison_markdown,
    quality_snapshot_markdown,
    snapshot_from_payload,
    snapshot_from_reports,
)
from .blast_radius import analyze_blast_radius as build_blast_radius_report
from .blast_radius import blast_radius_markdown
from .call_graph import analyze_call_graph as build_call_graph_report
from .call_graph import call_graph_markdown
from .code_metrics import analyze_code_metrics as build_code_metrics_report
from .code_metrics import code_metrics_markdown
from .dead_code import analyze_dead_code as build_dead_code_report
from .dead_code import dead_code_markdown
from .dependency_graph import analyze_dependency_graph as build_dependency_graph_report
from .dependency_graph import dependency_graph_markdown, write_dependency_graph_svg
from .duplicate_code import analyze_duplicate_code as build_duplicate_code_report
from .duplicate_code import duplicate_code_markdown
from .module_responsibility import analyze_module_responsibilities, module_responsibility_markdown
from .pattern_metrics import analyze_code_health_metrics as build_code_health_metrics_report
from .pattern_metrics import code_health_metrics_markdown
from .patterns.anti_patterns import analyze_anti_patterns as build_anti_patterns_report
from .patterns.anti_patterns import anti_patterns_markdown
from .patterns.good_patterns import analyze_good_patterns as build_good_patterns_report
from .patterns.good_patterns import good_patterns_markdown
from .quality_scorecard import build_quality_scorecard, load_score_weights, quality_scorecard_markdown
from .refactor_priority import build_refactor_priority_report, refactor_priority_markdown
from .refactoring_hints import build_refactoring_hints, refactoring_hints_markdown
from .repo_summary import generate_repo_overview, repo_overview_markdown
from .repository import build_repository_inventory
from .semantic_search import (
    build_semantic_index,
    load_semantic_index,
    search_results_markdown,
    search_semantic_index,
    semantic_index_markdown,
)
from .utils import DEFAULT_EMBEDDING_MODEL, ensure_output_dir, slugify_query, write_json, write_markdown


def analyze_dependency_graph(*, root: Path, output_dir: Path, inventory: Any | None = None) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    inventory = inventory or build_repository_inventory(root)
    report = build_dependency_graph_report(root, inventory)
    svg_path = write_dependency_graph_svg(report, output_dir / "dependency_graph.svg")
    json_path = output_dir / "dependency_graph.json"
    markdown_path = output_dir / "dependency_graph.md"
    write_json(json_path, report.to_dict())
    write_markdown(markdown_path, dependency_graph_markdown(report))
    return {
        "report": report.to_dict(),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "svg_path": svg_path,
    }


def analyze_call_graph(*, root: Path, output_dir: Path, inventory: Any | None = None) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    inventory = inventory or build_repository_inventory(root)
    report = build_call_graph_report(inventory)
    json_path = output_dir / "call_graph.json"
    markdown_path = output_dir / "call_graph.md"
    write_json(json_path, report.to_dict())
    write_markdown(markdown_path, call_graph_markdown(report))
    return {
        "report": report.to_dict(),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }


def analyze_code_metrics(*, root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    report = build_code_metrics_report(root)
    json_path = output_dir / "code_metrics_report.json"
    markdown_path = output_dir / "code_metrics_report.md"
    write_json(json_path, report.to_dict())
    write_markdown(markdown_path, code_metrics_markdown(report))
    return {
        "report": report.to_dict(),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }


def build_semantic_index_bundle(
    *,
    root: Path,
    output_dir: Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    include_tests: bool = False,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    report, chunks, embeddings = build_semantic_index(
        root=root,
        output_dir=output_dir,
        model_name=model_name,
        include_tests=include_tests,
    )
    markdown_path = output_dir / "semantic_index.md"
    write_markdown(markdown_path, semantic_index_markdown(report))
    return {
        "report": report.to_dict(),
        "chunks": chunks,
        "embeddings": embeddings,
        "index_path": str(output_dir / "semantic_index.faiss"),
        "chunks_path": str(output_dir / "semantic_chunks.json"),
        "embeddings_path": str(output_dir / "semantic_embeddings.npy"),
        "markdown_path": str(markdown_path),
    }


def analyze_duplicate_code(
    *,
    root: Path,
    output_dir: Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    semantic_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    if semantic_bundle is None:
        if (output_dir / "semantic_index.faiss").exists() and (output_dir / "semantic_chunks.json").exists():
            report, chunks, embeddings, _index = load_semantic_index(output_dir)
            semantic_bundle = {"report": report.to_dict(), "chunks": chunks, "embeddings": embeddings}
        else:
            semantic_bundle = build_semantic_index_bundle(root=root, output_dir=output_dir, model_name=model_name)
    semantic_backend = (
        semantic_bundle.get("report", {}).get("backend")
        if isinstance(semantic_bundle.get("report"), dict)
        else getattr(semantic_bundle.get("report"), "backend", "semantic_embeddings+faiss")
    )
    report = build_duplicate_code_report(
        semantic_bundle["chunks"],
        semantic_bundle["embeddings"],
        backend=str(semantic_backend or "semantic_embeddings+faiss"),
    )
    json_path = output_dir / "duplicate_code_report.json"
    markdown_path = output_dir / "duplicate_code_report.md"
    write_json(json_path, report.to_dict())
    write_markdown(markdown_path, duplicate_code_markdown(report))
    return {
        "report": report.to_dict(),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }


def analyze_dead_code(
    *,
    root: Path,
    output_dir: Path,
    inventory: Any | None = None,
    dependency_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    inventory = inventory or build_repository_inventory(root)
    dependency_report = None
    if dependency_payload:
        dependency_report = build_dependency_graph_report(root, inventory)
    report = build_dead_code_report(root, inventory=inventory, dependency_report=dependency_report)
    json_path = output_dir / "dead_code_report.json"
    markdown_path = output_dir / "dead_code_report.md"
    write_json(json_path, report.to_dict())
    write_markdown(markdown_path, dead_code_markdown(report))
    return {
        "report": report.to_dict(),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }


def analyze_module_responsibility(
    *,
    root: Path,
    output_dir: Path,
    inventory: Any | None = None,
    metrics_payload: dict[str, Any] | None = None,
    dependency_payload: dict[str, Any] | None = None,
    duplicate_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    inventory = inventory or build_repository_inventory(root)
    report = analyze_module_responsibilities(
        inventory,
        metrics_report=(metrics_payload or {}).get("report") or {},
        dependency_report=(dependency_payload or {}).get("report") or {},
        duplicate_report=(duplicate_payload or {}).get("report") or {},
    )
    json_path = output_dir / "module_responsibility_report.json"
    markdown_path = output_dir / "module_responsibility_report.md"
    write_json(json_path, report.to_dict())
    write_markdown(markdown_path, module_responsibility_markdown(report))
    return {
        "report": report.to_dict(),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }


def bootstrap_architecture_rules_file(*, root: Path, output_dir: Path, rules_path: Path | None = None) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    payload = bootstrap_architecture_rules(root, rules_path)
    manifest_path = output_dir / "architecture_rules_bootstrap.json"
    write_json(manifest_path, payload)
    return {
        "path": payload["path"],
        "rules": payload["rules"],
        "json_path": str(manifest_path),
    }


def analyze_architecture_rules(
    *,
    root: Path,
    output_dir: Path,
    rules_path: Path | None = None,
    inventory: Any | None = None,
    dependency_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    inventory = inventory or build_repository_inventory(root)
    report = validate_architecture_rules(
        root,
        rules_path=rules_path,
        inventory=inventory,
        dependency_report=(dependency_payload or {}).get("report") or dependency_payload,
    )
    json_path = output_dir / "architecture_rules_report.json"
    markdown_path = output_dir / "architecture_rules_report.md"
    write_json(json_path, report.to_dict())
    write_markdown(markdown_path, architecture_rules_markdown(report))
    return {
        "report": report.to_dict(),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }


def analyze_anti_patterns(
    *,
    root: Path,
    output_dir: Path,
    inventory: Any | None = None,
    duplicate_payload: dict[str, Any] | None = None,
    architecture_payload: dict[str, Any] | None = None,
    responsibility_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    inventory = inventory or build_repository_inventory(root)
    report = build_anti_patterns_report(
        root,
        inventory=inventory,
        duplicate_report=(duplicate_payload or {}).get("report") or duplicate_payload,
        architecture_report=(architecture_payload or {}).get("report") or architecture_payload,
        responsibility_report=(responsibility_payload or {}).get("report") or responsibility_payload,
    )
    json_path = output_dir / "anti_patterns.json"
    markdown_path = output_dir / "anti_patterns.md"
    write_json(json_path, report.to_dict())
    write_markdown(markdown_path, anti_patterns_markdown(report))
    return {
        "report": report.to_dict(),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }


def analyze_good_patterns(
    *,
    root: Path,
    output_dir: Path,
    inventory: Any | None = None,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    inventory = inventory or build_repository_inventory(root)
    report = build_good_patterns_report(root, inventory=inventory)
    json_path = output_dir / "good_patterns.json"
    markdown_path = output_dir / "good_patterns.md"
    write_json(json_path, report.to_dict())
    write_markdown(markdown_path, good_patterns_markdown(report))
    return {
        "report": report.to_dict(),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }


def analyze_code_health(
    *,
    root: Path,
    output_dir: Path,
    inventory: Any | None = None,
    metrics_payload: dict[str, Any] | None = None,
    dependency_payload: dict[str, Any] | None = None,
    duplicate_payload: dict[str, Any] | None = None,
    dead_code_payload: dict[str, Any] | None = None,
    architecture_payload: dict[str, Any] | None = None,
    anti_pattern_payload: dict[str, Any] | None = None,
    good_pattern_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    inventory = inventory or build_repository_inventory(root)
    report = build_code_health_metrics_report(
        root,
        inventory=inventory,
        metrics_report=(metrics_payload or {}).get("report") or metrics_payload,
        dependency_report=(dependency_payload or {}).get("report") or dependency_payload,
        duplicate_report=(duplicate_payload or {}).get("report") or duplicate_payload,
        dead_code_report=(dead_code_payload or {}).get("report") or dead_code_payload,
        architecture_report=(architecture_payload or {}).get("report") or architecture_payload,
        anti_pattern_report=(anti_pattern_payload or {}).get("report") or anti_pattern_payload,
        good_pattern_report=(good_pattern_payload or {}).get("report") or good_pattern_payload,
    )
    json_path = output_dir / "code_health_metrics.json"
    markdown_path = output_dir / "code_health_metrics.md"
    write_json(json_path, report.to_dict())
    write_markdown(markdown_path, code_health_metrics_markdown(report))
    return {
        "report": report.to_dict(),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }


def generate_quality_scorecard(
    *,
    output_dir: Path,
    metrics_payload: dict[str, Any],
    anti_pattern_payload: dict[str, Any],
    good_pattern_payload: dict[str, Any],
    architecture_payload: dict[str, Any],
    weights_path: Path | None = None,
    rules_path: Path | None = None,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    weights = load_score_weights(weights_path=weights_path, rules_path=rules_path)
    report = build_quality_scorecard(
        metrics_report=(metrics_payload or {}).get("report") or metrics_payload,
        anti_pattern_report=(anti_pattern_payload or {}).get("report") or anti_pattern_payload,
        good_pattern_report=(good_pattern_payload or {}).get("report") or good_pattern_payload,
        architecture_report=(architecture_payload or {}).get("report") or architecture_payload,
        weights=weights,
    )
    json_path = output_dir / "quality_scorecard.json"
    markdown_path = output_dir / "quality_scorecard.md"
    write_json(json_path, report.to_dict())
    write_markdown(markdown_path, quality_scorecard_markdown(report))
    return {
        "report": report.to_dict(),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }


def analyze_blast_radius(
    *,
    root: Path,
    output_dir: Path,
    inventory: Any | None = None,
    dependency_payload: dict[str, Any] | None = None,
    call_graph_payload: dict[str, Any] | None = None,
    code_health_payload: dict[str, Any] | None = None,
    anti_pattern_payload: dict[str, Any] | None = None,
    architecture_payload: dict[str, Any] | None = None,
    responsibility_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    inventory = inventory or build_repository_inventory(root)
    report = build_blast_radius_report(
        root,
        inventory=inventory,
        dependency_report=(dependency_payload or {}).get("report") or dependency_payload,
        call_graph_report=(call_graph_payload or {}).get("report") or call_graph_payload,
        code_health_report=(code_health_payload or {}).get("report") or code_health_payload,
        anti_pattern_report=(anti_pattern_payload or {}).get("report") or anti_pattern_payload,
        architecture_report=(architecture_payload or {}).get("report") or architecture_payload,
        responsibility_report=(responsibility_payload or {}).get("report") or responsibility_payload,
    )
    json_path = output_dir / "blast_radius_report.json"
    markdown_path = output_dir / "blast_radius_report.md"
    write_json(json_path, report.to_dict())
    write_markdown(markdown_path, blast_radius_markdown(report))
    return {
        "report": report.to_dict(),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }


def generate_refactor_priority_report(
    *,
    output_dir: Path,
    blast_radius_payload: dict[str, Any],
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    report = build_refactor_priority_report((blast_radius_payload or {}).get("report") or blast_radius_payload)
    json_path = output_dir / "refactor_priority_report.json"
    markdown_path = output_dir / "refactor_priority_report.md"
    write_json(json_path, report.to_dict())
    write_markdown(markdown_path, refactor_priority_markdown(report))
    return {
        "report": report.to_dict(),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }


def analyze_quality_bundle(
    *,
    root: Path,
    output_dir: Path,
    inventory: Any | None = None,
    metrics_payload: dict[str, Any],
    dependency_payload: dict[str, Any],
    duplicate_payload: dict[str, Any],
    dead_code_payload: dict[str, Any],
    responsibility_payload: dict[str, Any],
    rules_path: Path | None = None,
    weights_path: Path | None = None,
) -> dict[str, Any]:
    architecture_payload = analyze_architecture_rules(
        root=root,
        output_dir=output_dir,
        rules_path=rules_path,
        inventory=inventory,
        dependency_payload=dependency_payload,
    )
    good_pattern_payload = analyze_good_patterns(root=root, output_dir=output_dir, inventory=inventory)
    anti_pattern_payload = analyze_anti_patterns(
        root=root,
        output_dir=output_dir,
        inventory=inventory,
        duplicate_payload=duplicate_payload,
        architecture_payload=architecture_payload,
        responsibility_payload=responsibility_payload,
    )
    code_health_payload = analyze_code_health(
        root=root,
        output_dir=output_dir,
        inventory=inventory,
        metrics_payload=metrics_payload,
        dependency_payload=dependency_payload,
        duplicate_payload=duplicate_payload,
        dead_code_payload=dead_code_payload,
        architecture_payload=architecture_payload,
        anti_pattern_payload=anti_pattern_payload,
        good_pattern_payload=good_pattern_payload,
    )
    scorecard_payload = generate_quality_scorecard(
        output_dir=output_dir,
        metrics_payload=code_health_payload,
        anti_pattern_payload=anti_pattern_payload,
        good_pattern_payload=good_pattern_payload,
        architecture_payload=architecture_payload,
        weights_path=weights_path,
        rules_path=rules_path,
    )
    return {
        "architecture_rules": architecture_payload,
        "anti_patterns": anti_pattern_payload,
        "good_patterns": good_pattern_payload,
        "code_health": code_health_payload,
        "quality_scorecard": scorecard_payload,
    }


def analyze_change_impact_bundle(
    *,
    root: Path,
    output_dir: Path,
    inventory: Any | None = None,
    dependency_payload: dict[str, Any],
    call_graph_payload: dict[str, Any],
    responsibility_payload: dict[str, Any],
    quality_payload: dict[str, Any],
) -> dict[str, Any]:
    blast_radius_payload = analyze_blast_radius(
        root=root,
        output_dir=output_dir,
        inventory=inventory,
        dependency_payload=dependency_payload,
        call_graph_payload=call_graph_payload,
        code_health_payload=quality_payload["code_health"],
        anti_pattern_payload=quality_payload["anti_patterns"],
        architecture_payload=quality_payload["architecture_rules"],
        responsibility_payload=responsibility_payload,
    )
    refactor_priority_payload = generate_refactor_priority_report(
        output_dir=output_dir,
        blast_radius_payload=blast_radius_payload,
    )
    return {
        "blast_radius": blast_radius_payload,
        "refactor_priority": refactor_priority_payload,
    }


def build_repo_overview(*, output_dir: Path, inventory: Any | None = None) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    dependency_report = _load_json(output_dir / "dependency_graph.json")
    call_graph_report = _load_json(output_dir / "call_graph.json")
    duplicate_report = _load_json(output_dir / "duplicate_code_report.json")
    dead_code_report = _load_json(output_dir / "dead_code_report.json")
    metrics_report = _load_json(output_dir / "code_metrics_report.json")
    responsibility_report = _load_json(output_dir / "module_responsibility_report.json")

    if inventory is None:
        root = _discover_root_from_report(output_dir, dependency_report, metrics_report)
        inventory = build_repository_inventory(root)
    report = generate_repo_overview(
        inventory,
        dependency_report=dependency_report,
        call_graph_report=call_graph_report,
        duplicate_report=duplicate_report,
        dead_code_report=dead_code_report,
        metrics_report=metrics_report,
        responsibility_report=responsibility_report,
    )
    json_path = output_dir / "repo_overview.json"
    markdown_path = output_dir / "repo_overview.md"
    write_json(json_path, report.to_dict())
    write_markdown(markdown_path, repo_overview_markdown(report))
    return {
        "report": report.to_dict(),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }


def build_semantic_search_results(
    *,
    query: str,
    output_dir: Path,
    model_name: str | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    payload = search_semantic_index(query=query, output_dir=output_dir, model_name=model_name, top_k=top_k)
    slug = slugify_query(query)
    json_path = output_dir / f"semantic_search_{slug}.json"
    markdown_path = output_dir / f"semantic_search_{slug}.md"
    write_json(json_path, payload)
    write_markdown(markdown_path, search_results_markdown(payload))
    payload["json_path"] = str(json_path)
    payload["markdown_path"] = str(markdown_path)
    return payload


def analyze_repo_bundle(
    *,
    root: Path,
    output_dir: Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    rules_path: Path | None = None,
    weights_path: Path | None = None,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    inventory = build_repository_inventory(root)
    inventory_path = output_dir / "inventory.json"
    write_json(inventory_path, inventory.to_dict())

    dependency_payload = analyze_dependency_graph(root=root, output_dir=output_dir, inventory=inventory)
    call_graph_payload = analyze_call_graph(root=root, output_dir=output_dir, inventory=inventory)
    metrics_payload = analyze_code_metrics(root=root, output_dir=output_dir)
    semantic_payload = build_semantic_index_bundle(root=root, output_dir=output_dir, model_name=model_name)
    duplicate_payload = analyze_duplicate_code(
        root=root,
        output_dir=output_dir,
        model_name=model_name,
        semantic_bundle=semantic_payload,
    )
    dead_code_payload = analyze_dead_code(
        root=root,
        output_dir=output_dir,
        inventory=inventory,
        dependency_payload=dependency_payload,
    )
    responsibility_payload = analyze_module_responsibility(
        root=root,
        output_dir=output_dir,
        inventory=inventory,
        metrics_payload=metrics_payload,
        dependency_payload=dependency_payload,
        duplicate_payload=duplicate_payload,
    )
    quality_payload = analyze_quality_bundle(
        root=root,
        output_dir=output_dir,
        inventory=inventory,
        metrics_payload=metrics_payload,
        dependency_payload=dependency_payload,
        duplicate_payload=duplicate_payload,
        dead_code_payload=dead_code_payload,
        responsibility_payload=responsibility_payload,
        rules_path=rules_path,
        weights_path=weights_path,
    )
    change_impact_payload = analyze_change_impact_bundle(
        root=root,
        output_dir=output_dir,
        inventory=inventory,
        dependency_payload=dependency_payload,
        call_graph_payload=call_graph_payload,
        responsibility_payload=responsibility_payload,
        quality_payload=quality_payload,
    )
    overview_payload = build_repo_overview(output_dir=output_dir, inventory=inventory)
    hints_report = build_refactoring_hints(
        dependency_report=dependency_payload["report"],
        duplicate_report=duplicate_payload["report"],
        dead_code_report=dead_code_payload["report"],
        metrics_report=metrics_payload["report"],
        responsibility_report=responsibility_payload["report"],
    )
    hints_json_path = output_dir / "refactoring_hints.json"
    hints_markdown_path = output_dir / "refactoring_hints.md"
    write_json(hints_json_path, hints_report.to_dict())
    write_markdown(hints_markdown_path, refactoring_hints_markdown(hints_report))

    summary = {
        "root": str(root),
        "output_dir": str(output_dir),
        "modules": len(inventory.modules),
        "functions": len(inventory.functions),
        "classes": len(inventory.classes),
        "cycles": len(dependency_payload["report"]["cycles"]),
        "duplicate_clusters": len(duplicate_payload["report"]["clusters"]),
        "unused_functions": len(dead_code_payload["report"]["unused_functions"]),
        "unused_modules": len(dead_code_payload["report"]["unused_modules"]),
        "semantic_chunks": semantic_payload["report"]["chunk_count"],
        "architecture_violations": len(quality_payload["architecture_rules"]["report"]["violations"]),
        "quality_score": quality_payload["quality_scorecard"]["report"]["repo_score"],
        "highest_blast_radius_module": (
            change_impact_payload["blast_radius"]["report"]["summary"]["top_10_highest_blast_radius_modules"][0]["module"]
            if change_impact_payload["blast_radius"]["report"]["summary"]["top_10_highest_blast_radius_modules"]
            else ""
        ),
        "recommendations": overview_payload["report"]["recommendations"],
    }
    return {
        "summary": summary,
        "inventory_path": str(inventory_path),
        "dependency_graph": dependency_payload,
        "call_graph": call_graph_payload,
        "code_metrics": metrics_payload,
        "semantic_index": {
            "report": semantic_payload["report"],
            "index_path": semantic_payload["index_path"],
            "chunks_path": semantic_payload["chunks_path"],
            "embeddings_path": semantic_payload["embeddings_path"],
            "markdown_path": semantic_payload["markdown_path"],
        },
        "duplicate_code": duplicate_payload,
        "dead_code": dead_code_payload,
        "module_responsibility": responsibility_payload,
        "architecture_rules": quality_payload["architecture_rules"],
        "anti_patterns": quality_payload["anti_patterns"],
        "good_patterns": quality_payload["good_patterns"],
        "code_health": quality_payload["code_health"],
        "quality_scorecard": quality_payload["quality_scorecard"],
        "blast_radius": change_impact_payload["blast_radius"],
        "refactor_priority": change_impact_payload["refactor_priority"],
        "repo_overview": overview_payload,
        "refactoring_hints": {
            "report": hints_report.to_dict(),
            "json_path": str(hints_json_path),
            "markdown_path": str(hints_markdown_path),
        },
    }


def run_code_analysis(root: Path, output_dir: Path) -> dict[str, Any]:
    return analyze_repo_bundle(root=root, output_dir=output_dir)


def capture_quality_snapshot(
    *,
    root: Path,
    output_dir: Path,
    label: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    rules_path: Path | None = None,
    weights_path: Path | None = None,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    ensure_output_dir(output_dir / "snapshots")
    bundle = analyze_repo_bundle(
        root=root,
        output_dir=output_dir,
        model_name=model_name,
        rules_path=rules_path,
        weights_path=weights_path,
    )
    snapshot = snapshot_from_reports(
        label=label,
        root=str(root),
        metrics_report=bundle["code_health"]["report"],
        scorecard_report=bundle["quality_scorecard"]["report"],
        anti_pattern_report=bundle["anti_patterns"]["report"],
        good_pattern_report=bundle["good_patterns"]["report"],
        architecture_report=bundle["architecture_rules"]["report"],
    )
    json_path = output_dir / f"quality_snapshot_{label}.json"
    snapshot_json_path = output_dir / "snapshots" / f"quality_snapshot_{label}.json"
    markdown_path = output_dir / f"quality_snapshot_{label}.md"
    write_json(json_path, snapshot.to_dict())
    write_json(snapshot_json_path, snapshot.to_dict())
    write_markdown(markdown_path, quality_snapshot_markdown(snapshot))
    return {
        "report": snapshot.to_dict(),
        "json_path": str(json_path),
        "snapshot_json_path": str(snapshot_json_path),
        "markdown_path": str(markdown_path),
    }


def compare_quality_snapshots(
    *,
    output_dir: Path,
    baseline_label: str,
    current_label: str,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    baseline = snapshot_from_payload(_load_snapshot(output_dir, baseline_label))
    current = snapshot_from_payload(_load_snapshot(output_dir, current_label))
    if baseline is None or current is None:
        raise FileNotFoundError("Both baseline and current quality snapshots must exist before comparison.")
    comparison = build_quality_snapshot_comparison(baseline, current)
    json_path = output_dir / f"quality_comparison_{baseline_label}_vs_{current_label}.json"
    markdown_path = output_dir / f"quality_comparison_{baseline_label}_vs_{current_label}.md"
    write_json(json_path, comparison.to_dict())
    write_markdown(markdown_path, quality_comparison_markdown(comparison))
    return {
        "report": comparison.to_dict(),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_snapshot(output_dir: Path, label: str) -> dict[str, Any]:
    snapshot_path = output_dir / "snapshots" / f"quality_snapshot_{label}.json"
    if snapshot_path.exists():
        return _load_json(snapshot_path)
    fallback = output_dir / f"quality_snapshot_{label}.json"
    if fallback.exists():
        return _load_json(fallback)
    return {}


def _discover_root_from_report(output_dir: Path, dependency_report: dict[str, Any], metrics_report: dict[str, Any]) -> Path:
    sample_paths = []
    if metrics_report.get("largest_files"):
        sample_paths.extend(item.get("path") for item in metrics_report["largest_files"][:3])
    if dependency_report.get("edges"):
        sample_paths.extend(edge.get("source") for edge in dependency_report["edges"][:3])
    for sample in sample_paths:
        if not sample:
            continue
        path = Path(str(sample))
        if path.exists():
            return Path.cwd()
    return Path.cwd()
