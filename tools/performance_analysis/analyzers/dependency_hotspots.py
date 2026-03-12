from __future__ import annotations

import ast
from pathlib import Path

from ..models import DependencyHotspot, DependencyReport
from ..utils.path_utils import iter_python_files, module_name_for_path, path_for_module_name, safe_relative_path
from ..utils.report_utils import markdown_table, utc_timestamp

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


def _modules(root: Path, include_tests: bool) -> dict[str, Path]:
    return {module_name_for_path(root, path): path for path in iter_python_files(root, include_tests=include_tests)}


def _resolve(module_name: str, base: str, level: int) -> str:
    parts = module_name.split(".")[:-1]
    if level > 0:
        parts = parts[: max(0, len(parts) - level + 1)]
    return ".".join([*parts, *([base] if base else [])]).strip(".")


def _imports(path: Path, module_name: str, modules: dict[str, Path]) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                candidate = str(alias.name or "").strip()
                while candidate:
                    if candidate in modules and candidate != module_name:
                        found.add(candidate)
                        break
                    candidate = ".".join(candidate.split(".")[:-1])
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(module_name, str(node.module or "").strip(), int(node.level or 0)) if node.level else str(node.module or "").strip()
            if base in modules and base != module_name:
                found.add(base)
            for alias in node.names:
                full_name = ".".join(part for part in [base, str(alias.name or "").strip()] if part)
                if full_name in modules and full_name != module_name:
                    found.add(full_name)
    return found


def analyze_dependency_hotspots(root: str | Path, *, include_tests: bool = True, limit: int = 25) -> DependencyReport:
    root_path = Path(root).resolve()
    modules = _modules(root_path, include_tests)
    edges = [(name, dep) for name, path in modules.items() for dep in _imports(path, name, modules)]
    if HAS_NETWORKX:
        graph = nx.DiGraph()
        graph.add_nodes_from(modules)
        graph.add_edges_from(edges)
        pagerank = nx.pagerank(graph) if graph.number_of_nodes() else {}
        betweenness = nx.betweenness_centrality(graph) if graph.number_of_nodes() else {}
        cycles = [cycle for cycle in nx.simple_cycles(graph)][:20]
        nodes = [DependencyHotspot(name, safe_relative_path(path_for_module_name(root_path, name), root_path), int(graph.in_degree(name)), int(graph.out_degree(name)), float(pagerank.get(name, 0.0)), float(betweenness.get(name, 0.0)), round(float(pagerank.get(name, 0.0)) * 100.0 + float(betweenness.get(name, 0.0)) * 25.0 + int(graph.in_degree(name)) * 2.0 + int(graph.out_degree(name)), 6)) for name in graph.nodes]
    else:
        inbound = {name: 0 for name in modules}
        outbound = {name: 0 for name in modules}
        for source, target in edges:
            outbound[source] += 1
            inbound[target] += 1
        cycles = []
        nodes = [DependencyHotspot(name, safe_relative_path(path, root_path), inbound[name], outbound[name], 0.0, 0.0, float(inbound[name] * 2 + outbound[name])) for name, path in modules.items()]
    nodes = sorted(nodes, key=lambda row: (-row.score, row.module))[:limit]
    return DependencyReport(utc_timestamp(), "networkx" if HAS_NETWORKX else "fallback_graph", nodes, sorted(set(edges)), cycles, {"modules_analyzed": len(modules), "edges": len(set(edges)), "cycles_detected": len(cycles)})


def dependency_hotspots_markdown(report: DependencyReport) -> str:
    lines = ["# Dependency Hotspots", "", f"- Engine: `{report.engine}`", "", markdown_table(["Module", "Path", "Fan In", "Fan Out", "PageRank", "Betweenness", "Score"], [(row.module, row.path, row.fan_in, row.fan_out, f"{row.pagerank:.4f}", f"{row.betweenness:.4f}", f"{row.score:.3f}") for row in report.nodes])]
    if report.cycles:
        lines.extend(["", "## Cycles", "", *[f"- {' -> '.join(cycle)}" for cycle in report.cycles]])
    return "\n".join(lines)
