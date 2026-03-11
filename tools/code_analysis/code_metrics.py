from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from radon.complexity import cc_rank, cc_visit
    from radon.metrics import mi_rank, mi_visit
    _HAS_RADON = True
except ImportError:
    cc_rank = None
    cc_visit = None
    mi_rank = None
    mi_visit = None
    _HAS_RADON = False

from .discovery import discover_python_files, module_name_for_path


@dataclass
class CodeMetricsReport:
    backend: str
    module_rows: list[dict[str, Any]]
    high_complexity_functions: list[dict[str, Any]]
    largest_files: list[dict[str, Any]]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "module_rows": list(self.module_rows),
            "high_complexity_functions": list(self.high_complexity_functions),
            "largest_files": list(self.largest_files),
            "notes": list(self.notes),
        }


def analyze_code_metrics(root: Path) -> CodeMetricsReport:
    module_rows: list[dict[str, Any]] = []
    high_complexity_functions: list[dict[str, Any]] = []
    largest_files: list[dict[str, Any]] = []
    notes: list[str] = []
    backend = "radon"
    if not _HAS_RADON:
        backend = "ast_fallback"
        notes.append("radon is not installed; using AST fallback complexity estimates and omitting real maintainability metrics.")
    for path in discover_python_files(root):
        source = path.read_text(encoding="utf-8")
        module_name = module_name_for_path(root, path) or path.stem
        line_count = len(source.splitlines())
        if _HAS_RADON:
            complexity_blocks = cc_visit(source)
            mi_score = float(mi_visit(source, multi=True))
            module_max_complexity = max((int(block.complexity) for block in complexity_blocks), default=0)
            module_avg_complexity = (
                round(
                    sum(int(block.complexity) for block in complexity_blocks) / max(len(complexity_blocks), 1),
                    3,
                )
                if complexity_blocks
                else 0.0
            )
            maintainability_rank = mi_rank(mi_score)
        else:
            complexity_blocks = _fallback_complexity_blocks(module_name, path, source)
            mi_score = 100.0
            module_max_complexity = max((int(block["complexity"]) for block in complexity_blocks), default=0)
            module_avg_complexity = (
                round(
                    sum(int(block["complexity"]) for block in complexity_blocks) / max(len(complexity_blocks), 1),
                    3,
                )
                if complexity_blocks
                else 0.0
            )
            maintainability_rank = "NA"
        module_row = {
            "module": module_name,
            "path": str(path),
            "line_count": line_count,
            "function_count": len(complexity_blocks),
            "max_complexity": module_max_complexity,
            "avg_complexity": module_avg_complexity,
            "maintainability_index": round(mi_score, 3),
            "maintainability_rank": maintainability_rank,
            "analysis_backend": backend,
        }
        module_rows.append(module_row)
        largest_files.append(
            {
                "module": module_name,
                "path": str(path),
                "line_count": line_count,
            }
        )
        for block in complexity_blocks:
            if _HAS_RADON:
                block_name = block.name
                block_lineno = int(block.lineno)
                block_endline = int(getattr(block, "endline", block.lineno))
                block_complexity = int(block.complexity)
                block_rank = cc_rank(block.complexity)
                block_type = block.__class__.__name__
            else:
                block_name = str(block["name"])
                block_lineno = int(block["lineno"])
                block_endline = int(block["endline"])
                block_complexity = int(block["complexity"])
                block_rank = _fallback_cc_rank(block_complexity)
                block_type = str(block["type"])
            high_complexity_functions.append(
                {
                    "module": module_name,
                    "path": str(path),
                    "name": block_name,
                    "lineno": block_lineno,
                    "endline": block_endline,
                    "complexity": block_complexity,
                    "complexity_rank": block_rank,
                    "type": block_type,
                }
            )
    module_rows.sort(key=lambda item: (item["maintainability_index"], -item["max_complexity"], -item["line_count"]))
    largest_files.sort(key=lambda item: (-item["line_count"], item["module"]))
    high_complexity_functions.sort(key=lambda item: (-item["complexity"], item["module"], item["lineno"]))
    return CodeMetricsReport(
        backend=backend,
        module_rows=module_rows,
        high_complexity_functions=high_complexity_functions[:200],
        largest_files=largest_files[:50],
        notes=notes,
    )


def code_metrics_markdown(report: CodeMetricsReport) -> str:
    sections = [
        "# Code Metrics Report",
        "",
        f"- backend: {report.backend}",
    ]
    if report.notes:
        sections.extend(f"- note: {note}" for note in report.notes)
    sections.extend([
        "",
        "## Highest Complexity Functions",
    ])
    sections.extend(
        f"- `{row['module']}.{row['name']}`: complexity {row['complexity']} ({row['complexity_rank']})"
        for row in report.high_complexity_functions[:30]
    )
    sections.extend(["", "## Lowest Maintainability Modules"])
    sections.extend(
        f"- `{row['module']}`: MI {row['maintainability_index']:.2f} ({row['maintainability_rank']}), max complexity {row['max_complexity']}, {row['line_count']} lines"
        for row in report.module_rows[:30]
    )
    sections.extend(["", "## Largest Files"])
    sections.extend(
        f"- `{row['module']}`: {row['line_count']} lines"
        for row in report.largest_files[:20]
    )
    return "\n".join(sections)


_COMPLEXITY_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.IfExp,
    ast.BoolOp,
    ast.comprehension,
    ast.Match,
)


def _fallback_complexity_blocks(module_name: str, path: Path, source: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    blocks: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        complexity = 1 + sum(1 for child in ast.walk(node) if isinstance(child, _COMPLEXITY_NODES))
        blocks.append(
            {
                "module": module_name,
                "name": node.name,
                "lineno": int(getattr(node, "lineno", 0) or 0),
                "endline": int(getattr(node, "end_lineno", getattr(node, "lineno", 0)) or 0),
                "complexity": complexity,
                "type": node.__class__.__name__,
            }
        )
    return blocks


def _fallback_cc_rank(complexity: int) -> str:
    if complexity <= 5:
        return "A"
    if complexity <= 10:
        return "B"
    if complexity <= 20:
        return "C"
    if complexity <= 30:
        return "D"
    if complexity <= 40:
        return "E"
    return "F"
