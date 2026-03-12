from __future__ import annotations

import ast
from pathlib import Path

from ..models import PatternFinding, PatternReport
from ..utils.path_utils import iter_python_files, safe_relative_path
from ..utils.report_utils import markdown_table, utc_timestamp


class DataFramePatternVisitor(ast.NodeVisitor):
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

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_For(self, node: ast.For) -> None:
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_Call(self, node: ast.Call) -> None:
        attr = ""
        if isinstance(node.func, ast.Attribute):
            attr = str(node.func.attr or "")
        elif isinstance(node.func, ast.Name):
            attr = str(node.func.id or "")
        snippet = (ast.get_source_segment(self.source, node) or "").strip().replace("\n", " ")[:140]
        function_name = self.functions[-1] if self.functions else ""
        in_loop = self.loop_depth > 0
        if attr == "iterrows":
            self.findings.append(PatternFinding(safe_relative_path(self.path, self.root), int(node.lineno), "iterrows", 0.95 if in_loop else 0.75, "Row-wise iteration is usually a vectorization candidate.", snippet, function_name, in_loop))
        elif attr == "apply":
            self.findings.append(PatternFinding(safe_relative_path(self.path, self.root), int(node.lineno), "apply", 0.85 if in_loop else 0.6, "DataFrame.apply often hides Python-level loops.", snippet, function_name, in_loop))
        elif attr in {"concat", "merge", "groupby", "sort_values", "sort_index"} and in_loop:
            self.findings.append(PatternFinding(safe_relative_path(self.path, self.root), int(node.lineno), attr, 0.9, f"`{attr}` inside a loop is a scaling risk.", snippet, function_name, True))
        elif attr == "copy" and in_loop:
            self.findings.append(PatternFinding(safe_relative_path(self.path, self.root), int(node.lineno), "copy_in_loop", 0.7, "Repeated DataFrame.copy() inside a loop can amplify memory churn.", snippet, function_name, True))
        elif attr == "to_dict":
            self.findings.append(PatternFinding(safe_relative_path(self.path, self.root), int(node.lineno), "materialization", 0.55 if not in_loop else 0.75, "Converting frames to Python objects materializes data and can inflate memory.", snippet, function_name, in_loop))
        self.generic_visit(node)


def analyze_dataframe_patterns(root: str | Path, *, include_tests: bool = True, limit: int = 50) -> PatternReport:
    root_path = Path(root).resolve()
    findings: list[PatternFinding] = []
    for path in iter_python_files(root_path, include_tests=include_tests):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except Exception:
            continue
        visitor = DataFramePatternVisitor(source=source, path=path, root=root_path)
        visitor.visit(tree)
        findings.extend(visitor.findings)
    findings = sorted(findings, key=lambda row: (-row.severity, row.path, row.line))[:limit]
    return PatternReport(utc_timestamp(), "dataframe_patterns", findings, {"findings": len(findings)})


def dataframe_patterns_markdown(report: PatternReport) -> str:
    return "\n".join(["# DataFrame Anti-Patterns", "", markdown_table(["Path", "Line", "Pattern", "Severity", "Function", "Message"], [(row.path, row.line, row.pattern, f"{row.severity:.2f}", row.function_name, row.message) for row in report.findings])])
