from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..module_responsibility import analyze_module_responsibilities
from ..repository import RepositoryInventory, build_repository_inventory
from .shared import (
    RepositoryAstContext,
    annotation_text,
    build_repository_ast_context,
    child_statement_blocks,
    contains_nested_loop,
    cyclomatic_complexity,
    expression_name,
    handler_swallows_exception,
    has_broad_exception_handler,
    has_hidden_side_effect,
    is_expensive_call,
    is_test_module_name,
    iter_calls,
    iter_loop_nodes,
    max_nesting_depth,
    numeric_literals,
)


@dataclass
class AntiPatternReport:
    findings: list[dict[str, Any]]
    summary: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": list(self.findings),
            "summary": dict(self.summary),
            "notes": list(self.notes),
        }


def analyze_anti_patterns(
    root: Path,
    *,
    inventory: RepositoryInventory | None = None,
    duplicate_report: dict[str, Any] | None = None,
    architecture_report: dict[str, Any] | None = None,
    responsibility_report: dict[str, Any] | None = None,
) -> AntiPatternReport:
    inventory = inventory or build_repository_inventory(root)
    context = build_repository_ast_context(root, inventory)
    responsibility_rows = list(
        (responsibility_report or analyze_module_responsibilities(inventory).to_dict()).get("module_rows")
        or []
    )
    mixed_concern_map = {row["module"]: row for row in responsibility_rows}
    findings: list[dict[str, Any]] = []

    for module_context in context.modules.values():
        if module_context.is_test_module:
            continue
        findings.extend(_module_findings(module_context, mixed_concern_map.get(module_context.module, {})))

    findings.extend(_duplicate_workflow_findings(context, duplicate_report or {}))
    findings.extend(_architecture_violation_findings(context, architecture_report or {}))
    findings.sort(
        key=lambda row: (
            _severity_rank(row.get("severity")),
            row.get("file", ""),
            int(row.get("line_start") or 0),
            row.get("pattern", ""),
        )
    )
    summary = _build_summary(findings)
    notes = [
        "Nested-loop, deep-nesting, long-function, giant-class, magic-number, and exception findings are AST-based heuristics.",
        "Duplicate workflow shape findings reuse the existing duplicate-code analysis clusters so we do not rebuild clone detection from scratch.",
    ]
    return AntiPatternReport(findings=findings, summary=summary, notes=notes)


def anti_patterns_markdown(report: AntiPatternReport) -> str:
    sections = [
        "# Anti-Patterns Report",
        "",
        f"- findings: {report.summary.get('finding_count', 0)}",
        f"- high severity: {report.summary.get('severity_counts', {}).get('high', 0)}",
        f"- medium severity: {report.summary.get('severity_counts', {}).get('medium', 0)}",
        f"- low severity: {report.summary.get('severity_counts', {}).get('low', 0)}",
        "",
        "## Pattern Counts",
    ]
    pattern_counts = list(report.summary.get("pattern_counts", {}).items())
    if pattern_counts:
        sections.extend(f"- `{name}`: {count}" for name, count in pattern_counts)
    else:
        sections.append("- none")
    sections.extend(["", "## Top Findings"])
    if report.findings:
        for finding in report.findings[:40]:
            sections.append(
                f"- `{finding['pattern']}` in `{finding['symbol']}` at `{finding['file']}:{finding['line_start']}` [{finding['severity']}]"
            )
            sections.append(f"  - evidence: {finding['evidence']}")
            sections.append(f"  - suggestion: {finding['suggestion']}")
    else:
        sections.append("- none")
    if report.notes:
        sections.extend(["", "## Notes", *[f"- {note}" for note in report.notes]])
    return "\n".join(sections)


def _module_findings(module_context, mixed_concern_row: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    mixed_concern_finding = _mixed_concern_finding(module_context, mixed_concern_row)
    if mixed_concern_finding:
        findings.append(mixed_concern_finding)

    module_level_magic = numeric_literals(module_context.tree, module_constants=module_context.top_level_constants)
    if len(module_level_magic) >= 7:
        preview = ", ".join(str(item["value"]) for item in module_level_magic[:5])
        findings.append(
            _finding(
                pattern="magic numbers",
                file=module_context.path,
                symbol=module_context.module,
                line_start=int(module_level_magic[0]["lineno"]),
                line_end=int(module_level_magic[min(len(module_level_magic) - 1, 4)]["lineno"]),
                severity="medium" if len(module_level_magic) < 12 else "high",
                evidence=f"{len(module_level_magic)} non-trivial numeric literals found in the module (sample: {preview})",
                suggestion="Promote repeated thresholds into named constants or schema/config objects.",
                metric_value=len(module_level_magic),
            )
        )

    for function in module_context.functions:
        findings.extend(_function_findings(module_context, function))
    for class_context in module_context.classes:
        findings.extend(_class_findings(class_context))
    return findings


def _function_findings(module_context, function) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    complexity = cyclomatic_complexity(function.node)
    nesting = max_nesting_depth(function.node)

    if contains_nested_loop(function.node):
        findings.append(
            _finding(
                pattern="nested loops",
                file=function.path,
                symbol=function.full_name,
                line_start=function.lineno,
                line_end=function.end_lineno,
                severity="medium" if complexity < 15 else "high",
                evidence=f"`{function.name}` contains nested iteration and estimated complexity {complexity}.",
                suggestion="Pre-index inner lookups or split the inner loop into a helper with clearer invariants.",
                metric_value=complexity,
            )
        )

    if nesting >= 4:
        findings.append(
            _finding(
                pattern="deep nesting",
                file=function.path,
                symbol=function.full_name,
                line_start=function.lineno,
                line_end=function.end_lineno,
                severity="medium" if nesting == 4 else "high",
                evidence=f"`{function.name}` reaches nesting depth {nesting}.",
                suggestion="Prefer guard clauses or helper extraction so each branch handles one decision level.",
                metric_value=nesting,
            )
        )

    if function.loc >= 45:
        findings.append(
            _finding(
                pattern="long functions",
                file=function.path,
                symbol=function.full_name,
                line_start=function.lineno,
                line_end=function.end_lineno,
                severity="medium" if function.loc < 80 else "high",
                evidence=f"`{function.name}` spans {function.loc} lines with estimated complexity {complexity}.",
                suggestion="Split orchestration, transformation, and output concerns into smaller helpers.",
                metric_value=function.loc,
            )
        )

    if _loop_append_transform(function.node):
        findings.append(
            _finding(
                pattern="loop append simple transform",
                file=function.path,
                symbol=function.full_name,
                line_start=function.lineno,
                line_end=function.end_lineno,
                severity="low",
                evidence="A loop appends directly into a list without additional control flow.",
                suggestion="A list comprehension or generator pipeline would express the transformation more compactly.",
                metric_value=1,
            )
        )

    dispatch_chain_length = _dispatch_chain_length(function.node)
    if dispatch_chain_length >= 4:
        findings.append(
            _finding(
                pattern="repeated if/elif dispatch chains",
                file=function.path,
                symbol=function.full_name,
                line_start=function.lineno,
                line_end=function.end_lineno,
                severity="medium" if dispatch_chain_length < 6 else "high",
                evidence=f"`{function.name}` uses an if/elif chain with {dispatch_chain_length} branches.",
                suggestion="A registry or dispatch table would make this branch set easier to extend safely.",
                metric_value=dispatch_chain_length,
            )
        )

    magic_literals = numeric_literals(function.node, module_constants=module_context.top_level_constants)
    if len(magic_literals) >= 3:
        preview = ", ".join(str(item["value"]) for item in magic_literals[:4])
        findings.append(
            _finding(
                pattern="magic numbers",
                file=function.path,
                symbol=function.full_name,
                line_start=int(magic_literals[0]["lineno"]),
                line_end=int(magic_literals[min(len(magic_literals) - 1, 3)]["lineno"]),
                severity="low" if len(magic_literals) == 3 else "medium",
                evidence=f"`{function.name}` embeds {len(magic_literals)} numeric literals (sample: {preview}).",
                suggestion="Replace ad hoc thresholds with named constants so future edits preserve intent.",
                metric_value=len(magic_literals),
            )
        )

    broad_handlers = [handler for handler in has_broad_exception_handler(function.node) if handler_swallows_exception(handler)]
    if broad_handlers:
        handler = broad_handlers[0]
        findings.append(
            _finding(
                pattern="broad exception swallowing",
                file=function.path,
                symbol=function.full_name,
                line_start=int(getattr(handler, "lineno", function.lineno) or function.lineno),
                line_end=int(getattr(handler, "end_lineno", function.end_lineno) or function.end_lineno),
                severity="medium" if len(broad_handlers) == 1 else "high",
                evidence=f"`{function.name}` catches `Exception`/bare exceptions and returns a default without re-raising.",
                suggestion="Catch the narrowest expected exception and preserve failure details for callers.",
                metric_value=len(broad_handlers),
            )
        )

    if has_hidden_side_effect(function):
        findings.append(
            _finding(
                pattern="hidden side effects",
                file=function.path,
                symbol=function.full_name,
                line_start=function.lineno,
                line_end=function.end_lineno,
                severity="medium",
                evidence=f"`{function.name}` reads like a query/transform helper but mutates state or performs effectful I/O.",
                suggestion="Rename the function to surface the effect, or split the pure computation from the side effect.",
                metric_value=1,
            )
        )

    expensive_calls = _expensive_loop_calls(function.node)
    if expensive_calls:
        preview = ", ".join(expensive_calls[:4])
        findings.append(
            _finding(
                pattern="possible N+1 expensive calls inside loops",
                file=function.path,
                symbol=function.full_name,
                line_start=function.lineno,
                line_end=function.end_lineno,
                severity="medium" if len(expensive_calls) <= 2 else "high",
                evidence=f"Loop bodies call potentially expensive operations repeatedly (sample: {preview}).",
                suggestion="Hoist lookups outside the loop, batch work, or cache per-iteration dependencies.",
                metric_value=len(expensive_calls),
            )
        )
    return findings


def _class_findings(class_context) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if class_context.loc >= 180 or len(class_context.methods) >= 12:
        findings.append(
            _finding(
                pattern="giant classes",
                file=class_context.path,
                symbol=class_context.full_name,
                line_start=class_context.lineno,
                line_end=class_context.end_lineno,
                severity="medium" if class_context.loc < 260 else "high",
                evidence=f"`{class_context.name}` spans {class_context.loc} lines across {len(class_context.methods)} methods.",
                suggestion="Split orchestration, configuration, and behavior into smaller collaborators or boundary objects.",
                metric_value=max(class_context.loc, len(class_context.methods)),
            )
        )
    return findings


def _duplicate_workflow_findings(context: RepositoryAstContext, duplicate_report: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    function_map = context.functions
    for cluster in list(duplicate_report.get("clusters") or []):
        members = [function_map.get(str(member)) for member in list(cluster.get("members") or []) if function_map.get(str(member))]
        if len(members) < 2:
            continue
        first = members[0]
        module_count = len({member.module for member in members})
        findings.append(
            _finding(
                pattern="duplicate workflow shapes",
                file=first.path,
                symbol=f"cluster:{cluster.get('cluster_id')}",
                line_start=first.lineno,
                line_end=first.end_lineno,
                severity="medium" if len(members) < 4 else "high",
                evidence=f"Duplicate-code analysis found {len(members)} structurally similar members across {module_count} modules.",
                suggestion="Extract the common workflow into a shared helper, stage, or boundary abstraction.",
                metric_value=len(members),
            )
        )
    return findings


def _architecture_violation_findings(context: RepositoryAstContext, architecture_report: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for violation in list(architecture_report.get("violations") or []):
        source_module = str(violation.get("source_module") or "")
        if is_test_module_name(source_module):
            continue
        module_context = context.modules.get(source_module)
        findings.append(
            _finding(
                pattern="architecture layer violations",
                file=module_context.path if module_context else str(violation.get("source_path") or ""),
                symbol=source_module or str(violation.get("source_layer") or "architecture"),
                line_start=int(violation.get("line_start") or (module_context.functions[0].lineno if module_context and module_context.functions else 1)),
                line_end=int(violation.get("line_end") or (module_context.line_count if module_context else 1)),
                severity=str(violation.get("severity") or "medium"),
                evidence=str(violation.get("message") or violation.get("reason") or "Forbidden dependency detected."),
                suggestion="Move the dependency behind an interface or redirect the import through an allowed layer boundary.",
                metric_value=1,
            )
        )
    return findings


def _mixed_concern_finding(module_context, row: dict[str, Any]) -> dict[str, Any] | None:
    concern_count = int(row.get("concern_count") or 0)
    mixing_score = float(row.get("mixing_score") or 0.0)
    if concern_count < 3 or mixing_score < 35.0:
        return None
    concern_names = ", ".join(item["concern"] for item in list(row.get("concerns") or [])[:4]) or "multiple concerns"
    severity = "medium"
    if concern_count >= 4 or mixing_score >= 30.0:
        severity = "high"
    return _finding(
        pattern="mixed concerns modules",
        file=module_context.path,
        symbol=module_context.module,
        line_start=1,
        line_end=module_context.line_count,
        severity=severity,
        evidence=f"`{module_context.module}` spans {concern_count} concern categories ({concern_names}) with mixing score {mixing_score:.1f}.",
        suggestion="Separate data access, orchestration, and reporting logic into narrower modules with one stable responsibility.",
        metric_value=round(mixing_score, 2),
    )


def _loop_append_transform(node: ast.AST) -> bool:
    for loop in iter_loop_nodes(node):
        body = [statement for statement in loop.body if not isinstance(statement, ast.Pass)]
        if len(body) != 1:
            continue
        statement = body[0]
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            continue
        call = statement.value
        if expression_name(call.func).endswith(".append") and len(call.args) == 1:
            if all(not child_statement_blocks(call) for call in [call.args[0]]):
                return True
    return False


def _dispatch_chain_length(node: ast.AST) -> int:
    best = 0
    for child in ast.walk(node):
        if not isinstance(child, ast.If):
            continue
        if isinstance(getattr(child, "_parent", None), ast.If) and child in getattr(getattr(child, "_parent", None), "orelse", []):
            continue
        length = 1
        current = child
        subject = _dispatch_subject(current.test)
        while current.orelse and len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
            next_if = current.orelse[0]
            if subject and _dispatch_subject(next_if.test) != subject:
                break
            if any(len(branch.body) > 2 for branch in [current, next_if]):
                break
            length += 1
            current = next_if
        best = max(best, length)
    return best


def _dispatch_subject(test: ast.AST) -> str:
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], (ast.Eq, ast.Is)):
        return expression_name(test.left)
    return ""


def _expensive_loop_calls(node: ast.AST) -> list[str]:
    labels: list[str] = []
    for loop in iter_loop_nodes(node):
        for call in iter_calls(loop):
            label = expression_name(call.func)
            if label and is_expensive_call(label):
                labels.append(label)
    deduped: list[str] = []
    for label in labels:
        if label not in deduped:
            deduped.append(label)
    return deduped


def _finding(
    *,
    pattern: str,
    file: str,
    symbol: str,
    line_start: int,
    line_end: int,
    severity: str,
    evidence: str,
    suggestion: str,
    metric_value: int | float,
) -> dict[str, Any]:
    return {
        "pattern": pattern,
        "quality_signal": "bad",
        "file": file,
        "symbol": symbol,
        "line_start": int(line_start),
        "line_end": int(line_end),
        "severity": severity,
        "evidence": evidence,
        "suggestion": suggestion,
        "metric_value": metric_value,
    }


def _build_summary(findings: list[dict[str, Any]]) -> dict[str, Any]:
    pattern_counts: dict[str, int] = {}
    severity_counts = {"high": 0, "medium": 0, "low": 0}
    for finding in findings:
        pattern = str(finding.get("pattern") or "")
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        severity = str(finding.get("severity") or "low").lower()
        if severity not in severity_counts:
            severity_counts[severity] = 0
        severity_counts[severity] += 1
    return {
        "finding_count": len(findings),
        "pattern_counts": dict(sorted(pattern_counts.items(), key=lambda item: (-item[1], item[0]))),
        "severity_counts": severity_counts,
    }


def _severity_rank(value: Any) -> tuple[int, str]:
    severity = str(value or "low").lower()
    rank = {"high": 0, "medium": 1, "low": 2}.get(severity, 3)
    return rank, severity
