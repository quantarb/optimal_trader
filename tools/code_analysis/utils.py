from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    nx = None
    _HAS_NETWORKX = False


DEFAULT_OUTPUT_DIR = "data/code_analysis"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(make_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_markdown(path: Path, content: str) -> None:
    path.write_text(str(content).strip() + "\n", encoding="utf-8")


def make_json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return make_json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return make_json_safe(value.item())
        except Exception:
            return str(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return value


def slugify_query(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or ""))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "query"


def top_level_package_names(root: Path) -> list[str]:
    names: set[str] = set()
    for child in root.iterdir():
        if child.name.startswith("."):
            continue
        if child.is_dir() and (child / "__init__.py").exists():
            names.add(child.name)
    return sorted(names)


def render_svg_graph(graph: Any, path: Path, *, title: str, max_nodes: int = 50) -> str:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "optimal_trader_mpl"))
    if not _HAS_NETWORKX:
        path.write_text(
            (
                '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="180">'
                f'<text x="16" y="36">{title}</text>'
                '<text x="16" y="72">Graph visualization unavailable: networkx is not installed.</text>'
                '</svg>'
            ),
            encoding="utf-8",
        )
        return str(path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if graph.number_of_nodes() > max_nodes:
        pagerank = nx.pagerank(graph) if graph.number_of_nodes() else {}
        nodes = [name for name, _score in sorted(pagerank.items(), key=lambda item: (-item[1], item[0]))[:max_nodes]]
        subgraph = graph.subgraph(nodes).copy()
    else:
        subgraph = graph
    if subgraph.number_of_nodes() == 0:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="320" height="80"><text x="16" y="42">No graph nodes available.</text></svg>', encoding="utf-8")
        return str(path)
    figure = plt.figure(figsize=(16, 12))
    axes = figure.add_subplot(111)
    layout = nx.spring_layout(subgraph, seed=42, k=1.8 / max(subgraph.number_of_nodes(), 1) ** 0.5)
    nx.draw_networkx_edges(subgraph, pos=layout, ax=axes, alpha=0.25, arrows=True, arrowsize=10, edge_color="#5a6b7b")
    nx.draw_networkx_nodes(subgraph, pos=layout, ax=axes, node_size=400, node_color="#0d6b73", alpha=0.85)
    nx.draw_networkx_labels(subgraph, pos=layout, ax=axes, font_size=7, font_color="#172634")
    axes.set_title(title)
    axes.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, format="svg")
    plt.close(figure)
    return str(path)
