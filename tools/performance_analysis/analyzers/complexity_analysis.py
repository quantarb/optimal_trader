from __future__ import annotations

from pathlib import Path

from ..models import ComplexityFinding, ComplexityReport
from ..utils.path_utils import iter_python_files, module_name_for_path, safe_relative_path
from ..utils.report_utils import markdown_table, utc_timestamp

try:
    from radon.complexity import cc_rank, cc_visit
    from radon.metrics import h_visit, mi_visit
    from radon.raw import analyze as raw_analyze
    HAS_RADON = True
except ImportError:
    HAS_RADON = False


def _rank(value: float) -> str:
    return str(cc_rank(value)) if HAS_RADON else ("A" if value <= 5 else "B" if value <= 10 else "C" if value <= 20 else "D")


def analyze_complexity(root: str | Path, *, include_tests: bool = True, limit: int = 25) -> ComplexityReport:
    root_path = Path(root).resolve()
    modules: list[ComplexityFinding] = []
    functions: list[ComplexityFinding] = []
    files = iter_python_files(root_path, include_tests=include_tests)
    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
        except Exception:
            continue
        blocks = list(cc_visit(source)) if HAS_RADON else []
        mi = float(mi_visit(source, multi=True)) if HAS_RADON else None
        hal = float(h_visit(source).total.volume) if HAS_RADON else None
        loc = int(raw_analyze(source).loc) if HAS_RADON else sum(1 for line in source.splitlines() if line.strip())
        values = [float(getattr(block, "complexity", 0.0)) for block in blocks]
        max_cc = max(values) if values else 0.0
        avg_cc = sum(values) / len(values) if values else 0.0
        score = round(max_cc * 4.0 + avg_cc * 2.0 + max(0.0, 100.0 - float(mi or 100.0)) / 10.0 + loc / 250.0, 4)
        modules.append(ComplexityFinding(safe_relative_path(path, root_path), module_name_for_path(root_path, path), "module", round(max_cc, 4), _rank(max_cc), round(mi, 4) if mi is not None else None, loc, round(hal, 4) if hal is not None else None, score))
        for block in blocks:
            cc = float(getattr(block, "complexity", 0.0))
            functions.append(ComplexityFinding(safe_relative_path(path, root_path), str(getattr(block, "fullname", None) or getattr(block, "name", "")), str(type(block).__name__).lower(), round(cc, 4), _rank(cc), round(mi, 4) if mi is not None else None, int(getattr(block, "endline", getattr(block, "lineno", 0)) - getattr(block, "lineno", 0) + 1), round(hal, 4) if hal is not None else None, round(cc * 5.0 + max(0.0, 100.0 - float(mi or 100.0)) / 8.0, 4)))
    modules = sorted(modules, key=lambda row: (-row.score, row.path))[:limit]
    functions = sorted(functions, key=lambda row: (-row.score, row.path, row.name))[:limit]
    return ComplexityReport(utc_timestamp(), "radon" if HAS_RADON else "fallback_ast", modules, functions, {"files_analyzed": len(files)})


def complexity_markdown(report: ComplexityReport) -> str:
    return "\n".join([
        "# Complexity Hotspots",
        "",
        f"- Engine: `{report.engine}`",
        f"- Files analyzed: `{report.summary.get('files_analyzed', 0)}`",
        "",
        "## Modules",
        "",
        markdown_table(["Path", "Complexity", "Rank", "MI", "LOC", "Score"], [(row.path, f"{row.complexity:.2f}", row.rank, "" if row.maintainability_index is None else f"{row.maintainability_index:.2f}", row.loc, f"{row.score:.2f}") for row in report.module_findings]),
        "",
        "## Functions",
        "",
        markdown_table(["Path", "Function", "Complexity", "Rank", "Score"], [(row.path, row.name, f"{row.complexity:.2f}", row.rank, f"{row.score:.2f}") for row in report.function_findings]),
    ])
