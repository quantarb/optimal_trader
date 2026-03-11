from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from grimp import build_graph
    _HAS_GRIMP = True
except ImportError:
    build_graph = None
    _HAS_GRIMP = False

try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    nx = None
    _HAS_NETWORKX = False

from .repository import RepositoryInventory, build_repository_inventory
from .utils import render_svg_graph, top_level_package_names


@dataclass
class DependencyGraphReport:
    backend: str
    nodes: list[str]
    edges: list[dict[str, str]]
    cycles: list[list[str]]
    indegree: dict[str, int]
    outdegree: dict[str, int]
    central_modules: list[dict[str, Any]]
    svg_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "nodes": list(self.nodes),
            "edges": list(self.edges),
            "cycles": [list(item) for item in self.cycles],
            "indegree": dict(self.indegree),
            "outdegree": dict(self.outdegree),
            "central_modules": list(self.central_modules),
            "svg_path": self.svg_path,
        }


def analyze_dependency_graph(root: Path, inventory: RepositoryInventory | None = None) -> DependencyGraphReport:
    inventory = inventory or build_repository_inventory(root)
    nodes: set[str] = set(inventory.modules.keys())
    edge_pairs: set[tuple[str, str]] = set()
    adjacency: dict[str, set[str]] = {module_name: set() for module_name in nodes}

    package_names = top_level_package_names(root)
    grimp_graph = None
    if package_names and _HAS_GRIMP:
        try:
            grimp_graph = build_graph(*package_names, include_external_packages=False)
        except Exception:
            grimp_graph = None
    if grimp_graph is not None:
        for module_name in sorted(grimp_graph.modules):
            nodes.add(module_name)
            adjacency.setdefault(module_name, set())
            for imported_module in sorted(grimp_graph.find_modules_directly_imported_by(module_name)):
                nodes.add(imported_module)
                adjacency.setdefault(imported_module, set())
                edge_pairs.add((module_name, imported_module))
                adjacency[module_name].add(imported_module)

    for module_name, module_record in inventory.modules.items():
        for imported_module in module_record.imports:
            nodes.add(module_name)
            nodes.add(imported_module)
            adjacency.setdefault(module_name, set())
            adjacency.setdefault(imported_module, set())
            edge_pairs.add((module_name, imported_module))
            adjacency[module_name].add(imported_module)

    indegree = {node: 0 for node in nodes}
    outdegree = {node: 0 for node in nodes}
    for source, target in edge_pairs:
        outdegree[source] = outdegree.get(source, 0) + 1
        indegree[target] = indegree.get(target, 0) + 1

    cycles = [
        sorted(component)
        for component in _strongly_connected_components(nodes, adjacency)
        if len(component) > 1
    ]
    cycles = sorted(cycles, key=lambda item: (-len(item), item))
    if _HAS_NETWORKX:
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edge_pairs)
        pagerank = nx.pagerank(graph) if graph.number_of_nodes() else {}
        central_rows = sorted(pagerank.items(), key=lambda item: (-item[1], item[0]))[:30]
        central_modules = [
            {
                "module": module_name,
                "pagerank": round(float(score), 8),
                "indegree": indegree.get(module_name, 0),
                "outdegree": outdegree.get(module_name, 0),
            }
            for module_name, score in central_rows
        ]
    else:
        central_modules = [
            {
                "module": module_name,
                "pagerank": round(float(indegree.get(module_name, 0) + outdegree.get(module_name, 0)), 8),
                "indegree": indegree.get(module_name, 0),
                "outdegree": outdegree.get(module_name, 0),
            }
            for module_name in sorted(
                nodes,
                key=lambda item: (-(indegree.get(item, 0) + outdegree.get(item, 0)), item),
            )[:30]
        ]
    return DependencyGraphReport(
        backend=_dependency_backend_name(),
        nodes=sorted(nodes),
        edges=[{"source": source, "target": target} for source, target in sorted(edge_pairs)],
        cycles=[list(item) for item in cycles],
        indegree=indegree,
        outdegree=outdegree,
        central_modules=central_modules,
    )


def write_dependency_graph_svg(report: DependencyGraphReport, output_path: Path) -> str:
    if _HAS_NETWORKX:
        graph = nx.DiGraph()
        for edge in report.edges:
            graph.add_edge(edge["source"], edge["target"])
        svg_path = render_svg_graph(graph, output_path, title="Repository Dependency Graph")
        report.svg_path = svg_path
        return svg_path
    output_path.write_text(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="840" height="220">'
            '<text x="16" y="32">Repository Dependency Graph</text>'
            f'<text x="16" y="66">nodes: {len(report.nodes)}</text>'
            f'<text x="16" y="92">edges: {len(report.edges)}</text>'
            f'<text x="16" y="118">cycles: {len(report.cycles)}</text>'
            '<text x="16" y="156">Graph rendering unavailable: networkx is not installed.</text>'
            '</svg>'
        ),
        encoding="utf-8",
    )
    report.svg_path = str(output_path)
    return str(output_path)


def dependency_graph_markdown(report: DependencyGraphReport, *, edge_limit: int = 120) -> str:
    sections = [
        "# Dependency Graph",
        "",
        f"- backend: {report.backend}",
        f"- modules: {len(report.nodes)}",
        f"- import edges: {len(report.edges)}",
        f"- circular dependency groups: {len(report.cycles)}",
        *(["- svg: `" + report.svg_path + "`"] if report.svg_path else []),
        "",
        "## Most Central Modules",
        *(f"- `{row['module']}`: pagerank={row['pagerank']}, in={row['indegree']}, out={row['outdegree']}" for row in report.central_modules[:20]),
        "",
        "## Circular Dependencies",
    ]
    if report.cycles:
        sections.extend(f"- {' -> '.join(group)}" for group in report.cycles[:20])
    else:
        sections.append("- none detected")
    sections.extend(
        [
            "",
            "## Sample Import Edges",
            *(f"- `{row['source']}` -> `{row['target']}`" for row in report.edges[:edge_limit]),
        ]
    )
    return "\n".join(sections)


def _dependency_backend_name() -> str:
    parts = []
    parts.append("grimp" if _HAS_GRIMP else "inventory_imports")
    parts.append("networkx" if _HAS_NETWORKX else "custom_graph")
    return "+".join(parts)


def _strongly_connected_components(nodes: set[str], adjacency: dict[str, set[str]]) -> list[list[str]]:
    if _HAS_NETWORKX:
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        for source, targets in adjacency.items():
            for target in targets:
                graph.add_edge(source, target)
        return [sorted(component) for component in nx.strongly_connected_components(graph)]

    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in adjacency.get(node, set()):
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])

        if lowlinks[node] == indices[node]:
            component: list[str] = []
            while stack:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            components.append(component)

    for node in sorted(nodes):
        if node not in indices:
            strongconnect(node)
    return components
