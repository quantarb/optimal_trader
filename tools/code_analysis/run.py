from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
        "repo_overview": overview_payload,
        "refactoring_hints": {
            "report": hints_report.to_dict(),
            "json_path": str(hints_json_path),
            "markdown_path": str(hints_markdown_path),
        },
    }


def run_code_analysis(root: Path, output_dir: Path) -> dict[str, Any]:
    return analyze_repo_bundle(root=root, output_dir=output_dir)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
