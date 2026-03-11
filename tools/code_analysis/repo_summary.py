from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .repository import RepositoryInventory


@dataclass
class RepoOverviewReport:
    overview: dict[str, Any]
    package_rows: list[dict[str, Any]]
    largest_modules: list[dict[str, Any]]
    central_modules: list[dict[str, Any]]
    circular_dependencies: list[list[str]]
    duplicate_clusters: list[dict[str, Any]]
    dead_code_hotspots: list[dict[str, Any]]
    high_complexity_modules: list[dict[str, Any]]
    mixed_responsibility_modules: list[dict[str, Any]]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "overview": dict(self.overview),
            "package_rows": list(self.package_rows),
            "largest_modules": list(self.largest_modules),
            "central_modules": list(self.central_modules),
            "circular_dependencies": [list(item) for item in self.circular_dependencies],
            "duplicate_clusters": list(self.duplicate_clusters),
            "dead_code_hotspots": list(self.dead_code_hotspots),
            "high_complexity_modules": list(self.high_complexity_modules),
            "mixed_responsibility_modules": list(self.mixed_responsibility_modules),
            "recommendations": list(self.recommendations),
        }


def generate_repo_overview(
    inventory: RepositoryInventory,
    *,
    dependency_report: dict[str, Any],
    call_graph_report: dict[str, Any],
    duplicate_report: dict[str, Any],
    dead_code_report: dict[str, Any],
    metrics_report: dict[str, Any],
    responsibility_report: dict[str, Any],
) -> RepoOverviewReport:
    package_counter: dict[str, dict[str, int]] = defaultdict(lambda: {"modules": 0, "functions": 0, "classes": 0, "lines": 0})
    for module_record in inventory.modules.values():
        package = module_record.module.split(".", 1)[0]
        package_counter[package]["modules"] += 1
        package_counter[package]["functions"] += len(module_record.functions)
        package_counter[package]["classes"] += len(module_record.class_records)
        package_counter[package]["lines"] += module_record.line_count

    package_rows = [
        {"package": package, **metrics}
        for package, metrics in sorted(package_counter.items(), key=lambda item: (-item[1]["modules"], item[0]))
    ]
    metric_rows = list(metrics_report.get("module_rows") or [])
    responsibility_rows = list(responsibility_report.get("module_rows") or [])
    duplicate_clusters = list(duplicate_report.get("clusters") or [])
    unused_modules = list(dead_code_report.get("unused_modules") or [])
    unused_functions = list(dead_code_report.get("unused_functions") or [])

    largest_modules = list(metrics_report.get("largest_files") or [])[:20]
    central_modules = list(dependency_report.get("central_modules") or [])[:20]
    circular_dependencies = [list(group) for group in list(dependency_report.get("cycles") or [])[:20]]
    high_complexity_modules = metric_rows[:20]
    mixed_responsibility_modules = responsibility_rows[:20]
    dead_code_hotspots = sorted(
        unused_modules + unused_functions,
        key=lambda row: (-int(row.get("line_count", row.get("size", 0)) or 0), str(row.get("module") or row.get("path") or "")),
    )[:20]

    recommendations: list[str] = []
    if responsibility_rows:
        row = next((item for item in responsibility_rows if ".tests" not in str(item["module"])), responsibility_rows[0])
        recommendations.append(
            f"Split `{row['module']}` because it mixes {row['concern_count']} concerns with score {row['mixing_score']}."
        )
    if duplicate_clusters:
        cluster = next((row for row in duplicate_clusters if len(list(row.get("modules") or [])) > 1), duplicate_clusters[0])
        recommendations.append(
            f"Deduplicate cluster {cluster['cluster_id']} with {cluster['size']} similar code chunks."
        )
    if circular_dependencies:
        recommendations.append(
            f"Break {len(circular_dependencies)} circular dependency groups, starting with the largest strongly connected components."
        )
    if metric_rows:
        row = metric_rows[0]
        recommendations.append(
            f"Reduce complexity in `{row['module']}` (MI {row['maintainability_index']}, max complexity {row['max_complexity']})."
        )
    if unused_modules or unused_functions:
        recommendations.append(
            f"Review {len(unused_modules)} unused modules and {len(unused_functions)} unused functions for deletion."
        )

    overview = {
        "module_count": len(inventory.modules),
        "function_count": len(inventory.functions),
        "class_count": len(inventory.classes),
        "import_edge_count": len(list(dependency_report.get("edges") or [])),
        "call_edge_count": len(list(call_graph_report.get("edges") or [])),
        "cycle_count": len(circular_dependencies),
        "duplicate_cluster_count": len(duplicate_clusters),
        "unused_function_count": len(unused_functions),
        "unused_module_count": len(unused_modules),
    }
    return RepoOverviewReport(
        overview=overview,
        package_rows=package_rows,
        largest_modules=largest_modules,
        central_modules=central_modules,
        circular_dependencies=circular_dependencies,
        duplicate_clusters=duplicate_clusters[:10],
        dead_code_hotspots=dead_code_hotspots,
        high_complexity_modules=high_complexity_modules,
        mixed_responsibility_modules=mixed_responsibility_modules,
        recommendations=recommendations,
    )


def repo_overview_markdown(report: RepoOverviewReport) -> str:
    sections = [
        "# Repository Overview",
        "",
        "## Overview",
        *(f"- {key.replace('_', ' ')}: {value}" for key, value in report.overview.items()),
        "",
        "## Largest Packages",
        *(f"- `{row['package']}`: {row['modules']} modules, {row['functions']} functions, {row['classes']} classes, {row['lines']} lines" for row in report.package_rows[:15]),
        "",
        "## Largest Modules",
        *(f"- `{row['module']}`: {row['line_count']} lines" for row in report.largest_modules[:15]),
        "",
        "## Most Central Modules",
        *(f"- `{row['module']}`: pagerank={row['pagerank']}, in={row['indegree']}, out={row['outdegree']}" for row in report.central_modules[:15]),
        "",
        "## Circular Dependencies",
    ]
    if report.circular_dependencies:
        sections.extend(f"- {' -> '.join(group)}" for group in report.circular_dependencies[:15])
    else:
        sections.append("- none")
    sections.extend(["", "## Duplicate Clusters"])
    if report.duplicate_clusters:
        sections.extend(
            f"- cluster {row['cluster_id']}: {row['size']} members across {len(row['modules'])} modules"
            for row in report.duplicate_clusters[:10]
        )
    else:
        sections.append("- none")
    sections.extend(["", "## High Complexity Modules"])
    sections.extend(
        f"- `{row['module']}`: MI {row['maintainability_index']}, max complexity {row['max_complexity']}, {row['line_count']} lines"
        for row in report.high_complexity_modules[:15]
    )
    sections.extend(["", "## Mixed Responsibility Modules"])
    sections.extend(
        f"- `{row['module']}`: mixing_score {row['mixing_score']}, concerns {row['concern_count']}"
        for row in report.mixed_responsibility_modules[:15]
    )
    sections.extend(["", "## Dead Code Hotspots"])
    if report.dead_code_hotspots:
        sections.extend(
            f"- `{row.get('module') or row.get('path')}`: {row.get('reason') or row.get('message') or 'candidate'}"
            for row in report.dead_code_hotspots[:15]
        )
    else:
        sections.append("- none")
    sections.extend(["", "## Refactoring Recommendations"])
    sections.extend(f"- {item}" for item in report.recommendations)
    return "\n".join(sections)
