from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..repository import RepositoryInventory, build_repository_inventory
from .shared import (
    annotation_text,
    artifact_like_name,
    boundary_object_like_name,
    build_repository_ast_context,
    config_like_name,
    cyclomatic_complexity,
    decorator_names,
    expression_name,
    has_guard_clause_shape,
    iter_calls,
    max_nesting_depth,
    reads_like_pure,
)


@dataclass
class GoodPatternReport:
    findings: list[dict[str, Any]]
    summary: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": list(self.findings),
            "summary": dict(self.summary),
            "notes": list(self.notes),
        }


def analyze_good_patterns(
    root: Path,
    *,
    inventory: RepositoryInventory | None = None,
) -> GoodPatternReport:
    inventory = inventory or build_repository_inventory(root)
    context = build_repository_ast_context(root, inventory)
    inheritance = _build_inheritance_index(context.classes.values())
    boundary_usage = _build_boundary_usage_index(context.modules.values())

    findings: list[dict[str, Any]] = []
    for module_context in context.modules.values():
        if module_context.is_test_module:
            continue
        findings.extend(_module_findings(module_context, inheritance, boundary_usage))

    findings.sort(
        key=lambda row: (
            -float(row.get("strength") or 0.0),
            -float(row.get("reusability_score") or 0.0),
            row.get("file", ""),
            int(row.get("line_start") or 0),
        )
    )
    summary = _build_summary(findings)
    notes = [
        "Good-pattern findings favor high-confidence structural signals such as annotations, dataclasses, protocols, and explicit boundary objects.",
        "Pure-function findings intentionally trade recall for precision so the report stays useful during dogfooding.",
    ]
    return GoodPatternReport(findings=findings, summary=summary, notes=notes)


def good_patterns_markdown(report: GoodPatternReport) -> str:
    sections = [
        "# Good Patterns Report",
        "",
        f"- findings: {report.summary.get('finding_count', 0)}",
        "",
        "## Pattern Counts",
    ]
    pattern_counts = list(report.summary.get("pattern_counts", {}).items())
    if pattern_counts:
        sections.extend(f"- `{name}`: {count}" for name, count in pattern_counts)
    else:
        sections.append("- none")
    sections.extend(["", "## Strongest Findings"])
    if report.findings:
        for finding in report.findings[:40]:
            sections.append(
                f"- `{finding['pattern']}` in `{finding['symbol']}` at `{finding['file']}:{finding['line_start']}` (strength {finding['strength']:.2f})"
            )
            sections.append(f"  - why it matters: {finding['why_it_matters']}")
    else:
        sections.append("- none")
    if report.notes:
        sections.extend(["", "## Notes", *[f"- {note}" for note in report.notes]])
    return "\n".join(sections)


def _module_findings(module_context, inheritance: dict[str, list[str]], boundary_usage: dict[str, int]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    registry_finding = _registry_pattern_finding(module_context)
    if registry_finding:
        findings.append(registry_finding)

    pipeline_finding = _build_fit_predict_finding(module_context)
    if pipeline_finding:
        findings.append(pipeline_finding)

    constants_finding = _single_source_of_truth_finding(module_context)
    if constants_finding:
        findings.append(constants_finding)

    reusable_stage_finding = _reusable_pipeline_stage_finding(module_context)
    if reusable_stage_finding:
        findings.append(reusable_stage_finding)

    for function in module_context.functions:
        findings.extend(_function_findings(function))

    for class_context in module_context.classes:
        findings.extend(_class_findings(class_context, inheritance, boundary_usage))
    return findings


def _function_findings(function) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    complexity = cyclomatic_complexity(function.node)
    nesting = max_nesting_depth(function.node)
    annotation_coverage = _annotation_coverage(function)

    if reads_like_pure(function) and function.loc <= 35 and complexity <= 6:
        findings.append(
            _finding(
                pattern="pure functions",
                file=function.path,
                symbol=function.full_name,
                line_start=function.lineno,
                line_end=function.end_lineno,
                strength=round(min(1.0, 0.5 + (annotation_coverage * 0.25) + max(0.0, (6 - complexity) * 0.05)), 2),
                why_it_matters="Deterministic helpers are easy to test, cache, and safely edit in isolation.",
                reusability_score=round(min(1.0, 0.55 + annotation_coverage * 0.2), 2),
                testability_score=round(min(1.0, 0.7 + max(0.0, (5 - nesting) * 0.04)), 2),
                llm_editability_score=round(min(1.0, 0.68 + annotation_coverage * 0.2), 2),
            )
        )

    if function.is_public and annotation_coverage >= 1.0 and function.has_return_annotation:
        findings.append(
            _finding(
                pattern="typed public APIs",
                file=function.path,
                symbol=function.full_name,
                line_start=function.lineno,
                line_end=function.end_lineno,
                strength=round(min(1.0, 0.7 + min(function.parameter_count, 4) * 0.05), 2),
                why_it_matters="Fully annotated public APIs reduce ambiguity for callers, reviewers, and coding agents.",
                reusability_score=0.82,
                testability_score=0.8,
                llm_editability_score=0.9,
            )
        )

    if has_guard_clause_shape(function):
        findings.append(
            _finding(
                pattern="guard clause style",
                file=function.path,
                symbol=function.full_name,
                line_start=function.lineno,
                line_end=function.end_lineno,
                strength=round(min(1.0, 0.55 + max(0.0, (3 - nesting) * 0.1)), 2),
                why_it_matters="Early exits keep the happy path shallow and make future edits less risky.",
                reusability_score=0.7,
                testability_score=0.78,
                llm_editability_score=0.82,
            )
        )

    if _returns_artifact_boundary(function):
        findings.append(
            _finding(
                pattern="artifact return boundary",
                file=function.path,
                symbol=function.full_name,
                line_start=function.lineno,
                line_end=function.end_lineno,
                strength=0.84,
                why_it_matters="Returning explicit boundary objects keeps cross-module contracts inspectable and stable.",
                reusability_score=0.86,
                testability_score=0.81,
                llm_editability_score=0.88,
            )
        )
    return findings


def _class_findings(class_context, inheritance: dict[str, list[str]], boundary_usage: dict[str, int]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    base_names = class_context.base_names
    decorator_list = class_context.decorator_names

    if _is_dataclass_config(class_context):
        findings.append(
            _finding(
                pattern="dataclass config pattern",
                file=class_context.path,
                symbol=class_context.full_name,
                line_start=class_context.lineno,
                line_end=class_context.end_lineno,
                strength=0.86,
                why_it_matters="Structured config/state objects centralize knobs and reduce argument drift across call sites.",
                reusability_score=0.85,
                testability_score=0.8,
                llm_editability_score=0.88,
            )
        )

    if _is_strategy_interface(class_context):
        findings.append(
            _finding(
                pattern="strategy or policy interface",
                file=class_context.path,
                symbol=class_context.full_name,
                line_start=class_context.lineno,
                line_end=class_context.end_lineno,
                strength=0.84,
                why_it_matters="Strategy-style interfaces let behavior vary without spreading conditionals across the repo.",
                reusability_score=0.88,
                testability_score=0.82,
                llm_editability_score=0.86,
            )
        )

    if _has_stable_subclass_family(class_context, inheritance):
        findings.append(
            _finding(
                pattern="stable base class or protocol",
                file=class_context.path,
                symbol=class_context.full_name,
                line_start=class_context.lineno,
                line_end=class_context.end_lineno,
                strength=0.88,
                why_it_matters="Stable extension points keep new variants localized instead of forcing invasive edits.",
                reusability_score=0.92,
                testability_score=0.84,
                llm_editability_score=0.9,
            )
        )

    if _is_explicit_boundary_object(class_context, boundary_usage):
        findings.append(
            _finding(
                pattern="explicit boundary objects",
                file=class_context.path,
                symbol=class_context.full_name,
                line_start=class_context.lineno,
                line_end=class_context.end_lineno,
                strength=0.87,
                why_it_matters="Boundary models make inputs and outputs visible, typed, and easy to refactor safely.",
                reusability_score=0.9,
                testability_score=0.82,
                llm_editability_score=0.92,
            )
        )
    return findings


def _build_fit_predict_finding(module_context) -> dict[str, Any] | None:
    function_names = {function.name for function in module_context.functions}
    method_names = {method.name for class_context in module_context.classes for method in class_context.methods}
    build_names = {name for name in function_names if name.startswith("build_")}
    fit_present = "fit" in function_names or "fit" in method_names
    predict_present = "predict" in function_names or "predict" in method_names
    if not build_names or not fit_present or not predict_present:
        return None
    return _finding(
        pattern="separation of build / fit / predict",
        file=module_context.path,
        symbol=module_context.module,
        line_start=1,
        line_end=module_context.line_count,
        strength=0.82,
        why_it_matters="Separating setup, training, and inference lowers accidental coupling between pipeline stages.",
        reusability_score=0.85,
        testability_score=0.81,
        llm_editability_score=0.84,
    )


def _single_source_of_truth_finding(module_context) -> dict[str, Any] | None:
    constant_usage = 0
    for constant in module_context.top_level_constants:
        references = sum(
            1
            for function in module_context.functions
            if any(isinstance(child, ast.Name) and child.id == constant and isinstance(child.ctx, ast.Load) for child in ast.walk(function.node))
        )
        if references >= 2:
            constant_usage += 1
    schema_classes = [class_context for class_context in module_context.classes if class_context.name.endswith(("Schema", "Spec"))]
    if constant_usage == 0 and not schema_classes:
        return None
    strength = 0.76 if constant_usage == 1 and not schema_classes else 0.84
    return _finding(
        pattern="single source of truth constants or schema",
        file=module_context.path,
        symbol=module_context.module,
        line_start=1,
        line_end=module_context.line_count,
        strength=strength,
        why_it_matters="Centralized constants and schemas keep edits consistent across related workflows.",
        reusability_score=0.84,
        testability_score=0.78,
        llm_editability_score=0.87,
    )


def _registry_pattern_finding(module_context) -> dict[str, Any] | None:
    registry_names: list[str] = []
    register_functions = [function for function in module_context.functions if function.name.startswith("register_")]
    for statement in module_context.tree.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id.endswith("REGISTRY") and isinstance(statement.value, (ast.Dict, ast.Call)):
                    registry_names.append(target.id)
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            if statement.target.id.endswith("REGISTRY") and isinstance(statement.value, (ast.Dict, ast.Call)):
                registry_names.append(statement.target.id)
    if not registry_names and not register_functions:
        return None
    symbol = registry_names[0] if registry_names else register_functions[0].full_name
    return _finding(
        pattern="registry pattern",
        file=module_context.path,
        symbol=symbol,
        line_start=1,
        line_end=module_context.line_count,
        strength=0.83,
        why_it_matters="Registries let new variants plug in without editing a central conditional chain.",
        reusability_score=0.88,
        testability_score=0.76,
        llm_editability_score=0.85,
    )


def _reusable_pipeline_stage_finding(module_context) -> dict[str, Any] | None:
    stage_like = [
        function
        for function in module_context.functions
        if any(token in function.name for token in ("stage", "step", "pipeline", "transform"))
    ]
    stage_like.extend(
        class_context
        for class_context in module_context.classes
        if class_context.name.endswith(("Stage", "Step")) or "pipeline" in class_context.name.lower()
    )
    if len(stage_like) < 2:
        return None
    return _finding(
        pattern="reusable pipeline stages",
        file=module_context.path,
        symbol=module_context.module,
        line_start=1,
        line_end=module_context.line_count,
        strength=0.78,
        why_it_matters="Named pipeline stages make orchestration composable and easier to rearrange or swap.",
        reusability_score=0.86,
        testability_score=0.79,
        llm_editability_score=0.83,
    )


def _annotation_coverage(function) -> float:
    total = max(function.parameter_count, 1)
    params_ratio = function.typed_parameter_count / total if function.parameter_count else 1.0
    return round((params_ratio + (1.0 if function.has_return_annotation else 0.0)) / 2.0, 4)


def _returns_artifact_boundary(function) -> bool:
    if artifact_like_name(annotation_text(function.node.returns)):
        return True
    for child in ast.walk(function.node):
        if isinstance(child, ast.Return):
            value = child.value
            if isinstance(value, ast.Call) and artifact_like_name(expression_name(value.func)):
                return True
    return False


def _is_dataclass_config(class_context) -> bool:
    if not config_like_name(class_context.name):
        return False
    if any(name.endswith("dataclass") or name == "dataclass" for name in class_context.decorator_names):
        return True
    return any(base.endswith("BaseModel") for base in class_context.base_names)


def _is_strategy_interface(class_context) -> bool:
    if class_context.name.endswith(("Strategy", "Policy", "Interface", "Protocol")):
        return True
    if any(base.endswith(("Protocol", "ABC")) for base in class_context.base_names):
        return True
    return any(
        "abstractmethod" in method.decorator_names or _body_is_abstract(method.node.body)
        for method in class_context.methods
    )


def _has_stable_subclass_family(class_context, inheritance: dict[str, list[str]]) -> bool:
    keys = {class_context.full_name, class_context.name}
    if any(len(inheritance.get(key, [])) >= 2 for key in keys):
        return True
    return any(base.endswith(("Protocol", "ABC")) for base in class_context.base_names)


def _is_explicit_boundary_object(class_context, boundary_usage: dict[str, int]) -> bool:
    if not boundary_object_like_name(class_context.name):
        return False
    if any(name.endswith("dataclass") or name == "dataclass" for name in class_context.decorator_names):
        return True
    if any(base.endswith("BaseModel") for base in class_context.base_names):
        return True
    return boundary_usage.get(class_context.full_name, 0) >= 2 or boundary_usage.get(class_context.name, 0) >= 2


def _body_is_abstract(statements: list[ast.stmt]) -> bool:
    if len(statements) != 1:
        return False
    statement = statements[0]
    if isinstance(statement, ast.Pass):
        return True
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and statement.value.value is Ellipsis:
        return True
    if isinstance(statement, ast.Raise):
        label = expression_name(statement.exc) if statement.exc else ""
        return label.endswith("NotImplementedError")
    return False


def _build_inheritance_index(class_contexts) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for class_context in class_contexts:
        for base_name in class_context.base_names:
            rows.setdefault(base_name, []).append(class_context.full_name)
            simple = base_name.rsplit(".", 1)[-1]
            rows.setdefault(simple, []).append(class_context.full_name)
    return rows


def _build_boundary_usage_index(module_contexts) -> dict[str, int]:
    counts: dict[str, int] = {}
    for module_context in module_contexts:
        for function in module_context.functions:
            for annotation in _function_annotation_names(function.node):
                counts[annotation] = counts.get(annotation, 0) + 1
                counts[annotation.rsplit(".", 1)[-1]] = counts.get(annotation.rsplit(".", 1)[-1], 0) + 1
    return counts


def _function_annotation_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    rows: list[str] = []
    for arg in list(node.args.args) + list(node.args.kwonlyargs) + list(node.args.posonlyargs):
        if arg.annotation is not None and expression_name(arg.annotation):
            rows.append(expression_name(arg.annotation))
    if node.returns is not None and expression_name(node.returns):
        rows.append(expression_name(node.returns))
    return rows


def _finding(
    *,
    pattern: str,
    file: str,
    symbol: str,
    line_start: int,
    line_end: int,
    strength: float,
    why_it_matters: str,
    reusability_score: float,
    testability_score: float,
    llm_editability_score: float,
) -> dict[str, Any]:
    return {
        "pattern": pattern,
        "quality_signal": "good",
        "file": file,
        "symbol": symbol,
        "line_start": int(line_start),
        "line_end": int(line_end),
        "strength": round(float(strength), 2),
        "why_it_matters": why_it_matters,
        "reusability_score": round(float(reusability_score), 2),
        "testability_score": round(float(testability_score), 2),
        "llm_editability_score": round(float(llm_editability_score), 2),
    }


def _build_summary(findings: list[dict[str, Any]]) -> dict[str, Any]:
    pattern_counts: dict[str, int] = {}
    for finding in findings:
        pattern = str(finding.get("pattern") or "")
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    return {
        "finding_count": len(findings),
        "pattern_counts": dict(sorted(pattern_counts.items(), key=lambda item: (-item[1], item[0]))),
    }
