from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from .repository import RepositoryInventory


@dataclass
class CallGraphReport:
    backend: str
    nodes: list[str]
    edges: list[dict[str, str]]
    indegree: dict[str, int]
    outdegree: dict[str, int]
    major_pipelines: list[dict[str, Any]]
    unresolved_call_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "nodes": list(self.nodes),
            "edges": list(self.edges),
            "indegree": dict(self.indegree),
            "outdegree": dict(self.outdegree),
            "major_pipelines": list(self.major_pipelines),
            "unresolved_call_count": self.unresolved_call_count,
        }


def analyze_call_graph(inventory: RepositoryInventory) -> CallGraphReport:
    function_names = set(inventory.functions.keys())
    edges: list[dict[str, str]] = []
    indegree: dict[str, int] = {name: 0 for name in function_names}
    outdegree: dict[str, int] = {name: 0 for name in function_names}
    adjacency: dict[str, set[str]] = defaultdict(set)
    unresolved_call_count = 0
    for function_record in inventory.functions.values():
        unresolved_call_count += len(function_record.unresolved_calls)
        for target in function_record.resolved_calls:
            if target not in function_names:
                continue
            if target == function_record.full_name:
                continue
            if target in adjacency[function_record.full_name]:
                continue
            adjacency[function_record.full_name].add(target)
            edges.append({"source": function_record.full_name, "target": target})
            outdegree[function_record.full_name] += 1
            indegree[target] += 1
    major_pipelines = _build_major_pipelines(function_names, adjacency, indegree, outdegree)
    return CallGraphReport(
        backend="ast",
        nodes=sorted(function_names),
        edges=edges,
        indegree=indegree,
        outdegree=outdegree,
        major_pipelines=major_pipelines,
        unresolved_call_count=unresolved_call_count,
    )


def _build_major_pipelines(
    function_names: set[str],
    adjacency: dict[str, set[str]],
    indegree: dict[str, int],
    outdegree: dict[str, int],
) -> list[dict[str, Any]]:
    candidate_roots = [
        name
        for name in function_names
        if outdegree.get(name, 0) >= 2
        and (
            indegree.get(name, 0) == 0
            or name.endswith(".handle")
            or ".views." in name
            or ".management.commands." in name
        )
        and ".tests" not in name
    ]
    candidate_roots = sorted(
        candidate_roots,
        key=lambda item: (
            ".management.commands." not in item,
            ".views." not in item,
            -outdegree.get(item, 0),
            item,
        ),
    )[:15]
    pipelines: list[dict[str, Any]] = []
    for root in candidate_roots:
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(root, 0)])
        walk: list[dict[str, Any]] = []
        while queue:
            node, depth = queue.popleft()
            if node in visited or depth > 3:
                continue
            visited.add(node)
            children = sorted(adjacency.get(node, set()))[:8]
            walk.append(
                {
                    "function": node,
                    "depth": depth,
                    "children": children,
                }
            )
            for child in children:
                queue.append((child, depth + 1))
        pipelines.append(
            {
                "root": root,
                "reachable_count": len(visited),
                "walk": walk,
            }
        )
    return pipelines


def call_graph_markdown(report: CallGraphReport) -> str:
    top_callers = sorted(report.outdegree.items(), key=lambda item: (-item[1], item[0]))[:20]
    top_callees = sorted(report.indegree.items(), key=lambda item: (-item[1], item[0]))[:20]
    sections = [
        "# Call Graph",
        "",
        f"- backend: {report.backend}",
        f"- functions: {len(report.nodes)}",
        f"- call edges: {len(report.edges)}",
        f"- unresolved call sites: {report.unresolved_call_count}",
        "",
        "## Highest Fan-Out Functions",
        *(f"- `{name}`: {count} outgoing calls" for name, count in top_callers),
        "",
        "## Highest Fan-In Functions",
        *(f"- `{name}`: {count} incoming calls" for name, count in top_callees),
        "",
        "## Major Pipelines",
    ]
    for pipeline in report.major_pipelines:
        sections.append(f"- root `{pipeline['root']}` reaches {pipeline['reachable_count']} functions")
        for row in pipeline["walk"][:10]:
            indent = "  " * int(row["depth"])
            child_text = ", ".join(f"`{item}`" for item in row["children"][:5]) or "leaf"
            sections.append(f"  {indent}- `{row['function']}` -> {child_text}")
    return "\n".join(sections)
