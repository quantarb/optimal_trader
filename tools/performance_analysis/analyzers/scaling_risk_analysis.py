from __future__ import annotations

import ast
from pathlib import Path

from ..models import PatternFinding, PatternReport
from ..utils.path_utils import iter_python_files, safe_relative_path
from ..utils.report_utils import markdown_table, utc_timestamp


EXPENSIVE_CALLS = {"load_artifact_csv_frame", "read_frame_artifact", "load_adjusted_price_frames", "merge", "groupby", "sort_values", "concat", "fit"}


class ScalingRiskVisitor(ast.NodeVisitor):
    def __init__(self, *, source: str, path: Path, root: Path) -> None:
        self.source = source
        self.path = path
        self.root = root
        self.findings: list[PatternFinding] = []
        self.loop_depth = 0
        self.functions: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    def visit_For(self, node: ast.For) -> None:
        self.loop_depth += 1
        iter_text = (ast.get_source_segment(self.source, node.iter) or "").lower()
        if self.loop_depth >= 2 and any(token in iter_text for token in ("symbol", "date", "horizon", "freq", "row")):
            self.findings.append(PatternFinding(safe_relative_path(self.path, self.root), int(node.lineno), "nested_loop", min(1.0, 0.55 + self.loop_depth * 0.15), "Nested loops over trading dimensions can grow poorly with universe size.", iter_text[:140], self.functions[-1] if self.functions else "", True))
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_Call(self, node: ast.Call) -> None:
        if self.loop_depth > 0:
            if isinstance(node.func, ast.Attribute):
                name = str(node.func.attr or "")
            elif isinstance(node.func, ast.Name):
                name = str(node.func.id or "")
            else:
                name = ""
            if name in EXPENSIVE_CALLS:
                self.findings.append(PatternFinding(safe_relative_path(self.path, self.root), int(node.lineno), f"loop_{name}", 0.8 if name != "fit" else 0.98, f"`{name}` is running inside a loop and likely scales poorly.", (ast.get_source_segment(self.source, node) or "").strip().replace("\n", " ")[:140], self.functions[-1] if self.functions else "", True))
        self.generic_visit(node)


def analyze_scaling_risks(root: str | Path, *, include_tests: bool = True, limit: int = 50) -> PatternReport:
    root_path = Path(root).resolve()
    findings: list[PatternFinding] = []
    for path in iter_python_files(root_path, include_tests=include_tests):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except Exception:
            continue
        visitor = ScalingRiskVisitor(source=source, path=path, root=root_path)
        visitor.visit(tree)
        findings.extend(visitor.findings)
    findings = sorted(findings, key=lambda row: (-row.severity, row.path, row.line))[:limit]
    return PatternReport(utc_timestamp(), "scaling_risks", findings, {"findings": len(findings)})


def scaling_risks_markdown(report: PatternReport) -> str:
    return "\n".join(["# Scaling Risks", "", markdown_table(["Path", "Line", "Pattern", "Severity", "Function", "Message"], [(row.path, row.line, row.pattern, f"{row.severity:.2f}", row.function_name, row.message) for row in report.findings])])
