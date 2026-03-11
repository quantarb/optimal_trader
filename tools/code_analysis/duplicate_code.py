from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    faiss = None
    _HAS_FAISS = False

try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    nx = None
    _HAS_NETWORKX = False

from .semantic_search import CodeChunk


@dataclass
class DuplicateCodeReport:
    backend: str
    similarity_threshold: float
    candidate_pairs: list[dict[str, Any]]
    clusters: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "similarity_threshold": self.similarity_threshold,
            "candidate_pairs": list(self.candidate_pairs),
            "clusters": list(self.clusters),
        }


def analyze_duplicate_code(
    chunks: list[CodeChunk],
    embeddings: np.ndarray,
    *,
    backend: str = "semantic_embeddings+faiss",
    similarity_threshold: float = 0.92,
    min_line_count: int = 8,
    top_neighbors: int = 10,
) -> DuplicateCodeReport:
    eligible_indices = [
        idx
        for idx, chunk in enumerate(chunks)
        if not chunk.is_test and _line_count(chunk) >= min_line_count
    ]
    if not eligible_indices:
        return DuplicateCodeReport(
            backend=backend,
            similarity_threshold=similarity_threshold,
            candidate_pairs=[],
            clusters=[],
        )

    subset = np.asarray(embeddings[eligible_indices], dtype="float32")
    neighbor_count = min(max(int(top_neighbors), 2), len(eligible_indices))
    if _HAS_FAISS:
        index = faiss.IndexFlatIP(subset.shape[1])
        index.add(subset)
        scores, neighbors = index.search(subset, neighbor_count)
    else:
        similarity = np.asarray(subset @ subset.T, dtype="float32")
        order = np.argsort(-similarity, axis=1)[:, :neighbor_count]
        scores = np.take_along_axis(similarity, order, axis=1)
        neighbors = order

    adjacency: dict[str, set[str]] = {chunk.chunk_id: set() for chunk in chunks}
    seen_pairs: set[tuple[str, str]] = set()
    candidate_pairs: list[dict[str, Any]] = []

    for local_idx, global_left_idx in enumerate(eligible_indices):
        left = chunks[global_left_idx]
        for score, neighbor_idx in zip(scores[local_idx], neighbors[local_idx], strict=False):
            if int(neighbor_idx) < 0 or int(neighbor_idx) == local_idx:
                continue
            right = chunks[eligible_indices[int(neighbor_idx)]]
            if left.chunk_id == right.chunk_id:
                continue
            if left.module == right.module and left.qualname == right.qualname:
                continue
            if float(score) < similarity_threshold:
                continue
            line_ratio = min(_line_count(left), _line_count(right)) / max(_line_count(left), _line_count(right))
            if line_ratio < 0.5:
                continue
            pair_key = tuple(sorted((left.chunk_id, right.chunk_id)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            adjacency.setdefault(left.chunk_id, set()).add(right.chunk_id)
            adjacency.setdefault(right.chunk_id, set()).add(left.chunk_id)
            candidate_pairs.append(
                {
                    "left": left.chunk_id,
                    "right": right.chunk_id,
                    "left_kind": left.kind,
                    "right_kind": right.kind,
                    "left_path": left.path,
                    "right_path": right.path,
                    "left_lineno": left.lineno,
                    "right_lineno": right.lineno,
                    "similarity": round(float(score), 6),
                    "left_preview": left.preview,
                    "right_preview": right.preview,
                }
            )

    candidate_pairs.sort(key=lambda row: (-float(row["similarity"]), row["left"], row["right"]))
    clusters: list[dict[str, Any]] = []
    for cluster_id, members in enumerate(
        sorted(
            (sorted(component) for component in _connected_components(adjacency) if len(component) > 1),
            key=lambda items: (-len(items), items),
        ),
        start=1,
    ):
        member_chunks = [chunk for chunk in chunks if chunk.chunk_id in set(members)]
        modules = sorted({chunk.module for chunk in member_chunks})
        clusters.append(
            {
                "cluster_id": cluster_id,
                "size": len(members),
                "modules": modules,
                "members": members,
                "kinds": sorted({chunk.kind for chunk in member_chunks}),
                "representative_preview": member_chunks[0].preview if member_chunks else "",
            }
        )

    return DuplicateCodeReport(
        backend=backend if _HAS_FAISS else backend.replace("+faiss", "+numpy"),
        similarity_threshold=similarity_threshold,
        candidate_pairs=candidate_pairs[:200],
        clusters=clusters[:100],
    )


def _line_count(chunk: CodeChunk) -> int:
    return max(0, int(chunk.end_lineno) - int(chunk.lineno) + 1)


def duplicate_code_markdown(report: DuplicateCodeReport) -> str:
    sections = [
        "# Duplicate Code Report",
        "",
        f"- backend: {report.backend}",
        f"- similarity threshold: {report.similarity_threshold}",
        f"- high-similarity pairs: {len(report.candidate_pairs)}",
        f"- duplicate clusters: {len(report.clusters)}",
        "",
        "## Largest Duplicate Clusters",
    ]
    if report.clusters:
        for cluster in report.clusters[:20]:
            sections.append(
                f"- cluster {cluster['cluster_id']} ({cluster['size']} members across {len(cluster['modules'])} modules)"
            )
            sections.append(f"  - modules: {', '.join(f'`{name}`' for name in cluster['modules'][:6])}")
            sections.append(f"  - preview: {cluster['representative_preview']}")
    else:
        sections.append("- none")
    sections.extend(["", "## Top Similar Pairs"])
    if report.candidate_pairs:
        sections.extend(
            f"- `{row['left']}` <-> `{row['right']}` (similarity {row['similarity']:.3f})"
            for row in report.candidate_pairs[:25]
        )
    else:
        sections.append("- none")
    return "\n".join(sections)


def _connected_components(adjacency: dict[str, set[str]]) -> list[list[str]]:
    if _HAS_NETWORKX:
        graph = nx.Graph()
        for node, neighbors in adjacency.items():
            graph.add_node(node)
            for neighbor in neighbors:
                graph.add_edge(node, neighbor)
        return [sorted(component) for component in nx.connected_components(graph)]

    visited: set[str] = set()
    components: list[list[str]] = []
    for node in sorted(adjacency):
        if node in visited:
            continue
        stack = [node]
        component: list[str] = []
        visited.add(node)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components
