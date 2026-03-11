from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from vulture import Vulture
    _HAS_VULTURE = True
except ImportError:
    Vulture = None
    _HAS_VULTURE = False

from .dependency_graph import DependencyGraphReport, analyze_dependency_graph
from .discovery import discover_python_files, module_name_for_path
from .repository import RepositoryInventory, build_repository_inventory


ENTRYPOINT_MODULE_PREFIXES = (
    "pipeline.management.commands.",
    "fmp.endpoints.",
    "tools.code_analysis.",
    "pipeline.templatetags.",
)

ENTRYPOINT_MODULES = {
    "manage",
    "urls",
    "settings",
    "asgi",
    "wsgi",
    "celery_app",
}


@dataclass
class DeadCodeReport:
    backend: str
    unused_items: list[dict[str, Any]]
    unused_functions: list[dict[str, Any]]
    unused_classes: list[dict[str, Any]]
    unused_imports: list[dict[str, Any]]
    unused_modules: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "unused_items": list(self.unused_items),
            "unused_functions": list(self.unused_functions),
            "unused_classes": list(self.unused_classes),
            "unused_imports": list(self.unused_imports),
            "unused_modules": list(self.unused_modules),
        }


def analyze_dead_code(
    root: Path,
    inventory: RepositoryInventory | None = None,
    dependency_report: DependencyGraphReport | None = None,
    *,
    min_confidence: int = 60,
) -> DeadCodeReport:
    inventory = inventory or build_repository_inventory(root)
    dependency_report = dependency_report or analyze_dependency_graph(root, inventory)

    unused_items: list[dict[str, Any]] = []
    if _HAS_VULTURE:
        scanner = Vulture(verbose=False)
        for path in discover_python_files(root):
            try:
                source = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                source = path.read_text(encoding="utf-8", errors="ignore")
            scanner.scan(source, filename=str(path))

        for item in scanner.get_unused_code():
            confidence = int(getattr(item, "confidence", 0) or 0)
            if confidence < min_confidence:
                continue
            path = Path(str(getattr(item, "filename", "")))
            module_name = module_name_for_path(root, path) if path.is_relative_to(root) else path.stem
            if ".tests" in module_name:
                continue
            if _is_entrypoint_module(module_name):
                continue
            if _is_framework_false_positive(module_name, str(getattr(item, "typ", "")), str(getattr(item, "name", ""))):
                continue
            unused_items.append(
                {
                    "type": str(getattr(item, "typ", "unknown")),
                    "name": str(getattr(item, "name", "")),
                    "module": module_name,
                    "path": str(path),
                    "lineno": int(getattr(item, "first_lineno", 0) or 0),
                    "end_lineno": int(getattr(item, "last_lineno", 0) or 0),
                    "size": int(getattr(item, "size", 0) or 0),
                    "confidence": confidence,
                    "message": str(getattr(item, "message", "")).strip(),
                }
            )
    unused_items.sort(key=lambda row: (-int(row["confidence"]), row["path"], row["lineno"], row["name"]))

    unused_modules: list[dict[str, Any]] = []
    for module_name, module_record in inventory.modules.items():
        if ".tests" in module_name or module_name.endswith(".__init__"):
            continue
        if dependency_report.indegree.get(module_name, 0) > 0:
            continue
        if _is_entrypoint_module(module_name):
            continue
        unused_modules.append(
            {
                "module": module_name,
                "path": module_record.path,
                "line_count": module_record.line_count,
                "reason": "module has no inbound internal imports and is not a known entrypoint",
            }
        )
    unused_modules.sort(key=lambda row: (-int(row["line_count"]), row["module"]))

    return DeadCodeReport(
        backend="vulture+import_graph" if _HAS_VULTURE else "import_graph_only",
        unused_items=unused_items,
        unused_functions=[row for row in unused_items if row["type"] in {"function", "method"}],
        unused_classes=[row for row in unused_items if row["type"] == "class"],
        unused_imports=[row for row in unused_items if row["type"] == "import"],
        unused_modules=unused_modules,
    )


def _is_entrypoint_module(module_name: str) -> bool:
    if module_name in ENTRYPOINT_MODULES:
        return True
    if module_name.endswith(".urls") or module_name.endswith(".admin") or module_name.endswith(".apps"):
        return True
    if ".templatetags." in module_name:
        return True
    return any(module_name.startswith(prefix) for prefix in ENTRYPOINT_MODULE_PREFIXES)


def _is_framework_false_positive(module_name: str, item_type: str, name: str) -> bool:
    if module_name.endswith(".admin") or module_name.endswith(".apps"):
        return True
    if module_name.startswith("pipeline.management.commands.") and name in {"handle", "add_arguments", "Command"}:
        return True
    if item_type == "class" and name.endswith("Config"):
        return True
    return False


def dead_code_markdown(report: DeadCodeReport) -> str:
    sections = [
        "# Dead Code Report",
        "",
        f"- backend: {report.backend}",
        f"- unused items: {len(report.unused_items)}",
        f"- unused functions/methods: {len(report.unused_functions)}",
        f"- unused classes: {len(report.unused_classes)}",
        f"- unused imports: {len(report.unused_imports)}",
        f"- unused modules: {len(report.unused_modules)}",
        "",
        "## Top Unused Functions And Methods",
    ]
    if report.unused_functions:
        sections.extend(
            f"- `{row['module']}.{row['name']}` at `{row['path']}:{row['lineno']}` ({row['confidence']}% confidence)"
            for row in report.unused_functions[:30]
        )
    else:
        sections.append("- none")
    sections.extend(["", "## Top Unused Classes"])
    if report.unused_classes:
        sections.extend(
            f"- `{row['module']}.{row['name']}` at `{row['path']}:{row['lineno']}` ({row['confidence']}% confidence)"
            for row in report.unused_classes[:20]
        )
    else:
        sections.append("- none")
    sections.extend(["", "## Unused Module Candidates"])
    if report.unused_modules:
        sections.extend(
            f"- `{row['module']}` ({row['line_count']} lines): {row['reason']}"
            for row in report.unused_modules[:20]
        )
    else:
        sections.append("- none")
    return "\n".join(sections)
