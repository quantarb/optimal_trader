from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml

    _HAS_YAML = True
except ImportError:
    yaml = None
    _HAS_YAML = False

from .dependency_graph import DependencyGraphReport, analyze_dependency_graph
from .discovery import module_name_for_path, resolve_import_module
from .repository import RepositoryInventory, build_repository_inventory
from .utils import top_level_package_names


DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "architecture_rules.yaml"


@dataclass
class ArchitectureRulesReport:
    rule_path: str
    rules: dict[str, Any]
    module_layers: dict[str, str]
    violations: list[dict[str, Any]]
    module_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_path": self.rule_path,
            "rules": dict(self.rules),
            "module_layers": dict(self.module_layers),
            "violations": list(self.violations),
            "module_rows": list(self.module_rows),
            "summary": dict(self.summary),
            "notes": list(self.notes),
        }


def load_architecture_rules(path: Path | None = None) -> dict[str, Any]:
    rule_path = Path(path or DEFAULT_RULES_PATH).resolve()
    if not rule_path.exists():
        raise FileNotFoundError(f"Architecture rules file not found: {rule_path}")
    raw = rule_path.read_text(encoding="utf-8")
    if _HAS_YAML:
        payload = yaml.safe_load(raw)
    else:
        payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Architecture rules must parse to a mapping.")
    return payload


def bootstrap_architecture_rules(root: Path, output_path: Path | None = None) -> dict[str, Any]:
    rules = default_architecture_rules(root)
    target_path = Path(output_path or DEFAULT_RULES_PATH).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(rules, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(target_path), "rules": rules}


def default_architecture_rules(root: Path) -> dict[str, Any]:
    packages = top_level_package_names(root)
    entrypoint_modules = [
        name
        for name in ["manage", "asgi", "wsgi", "urls", "settings", "celery_app"]
        if (root / f"{name}.py").exists()
    ]
    assigned: set[str] = set()

    def consume(names: list[str]) -> list[str]:
        rows = [name for name in names if name in packages or name in entrypoint_modules]
        assigned.update(rows)
        return rows

    layers = {
        "entrypoints": consume(["manage", "asgi", "wsgi", "urls", "settings", "celery_app"]),
        "tools": consume(["tools"]),
        "domain": consume(["domain"]),
        "infrastructure": consume(["data", "fmp", "infra"]),
        "shared": consume(["utils"]),
        "tests": consume(["tests"]),
    }
    application_packages = sorted(name for name in packages if name not in assigned)
    if application_packages:
        layers["application"] = application_packages
        assigned.update(application_packages)
    else:
        layers["application"] = []

    layer_names = [name for name, prefixes in layers.items() if prefixes]
    allowed_layer_imports = {
        "entrypoints": [name for name in layer_names if name != "tests"],
        "tools": [name for name in layer_names if name in {"tools", "application", "domain", "infrastructure", "shared"}],
        "application": [name for name in layer_names if name in {"application", "domain", "infrastructure", "shared"}],
        "domain": [name for name in layer_names if name in {"domain", "shared"}],
        "infrastructure": [name for name in layer_names if name in {"infrastructure", "shared"}],
        "shared": [name for name in layer_names if name == "shared"],
        "tests": layer_names,
    }
    rules = {
        "layers": {name: prefixes for name, prefixes in layers.items() if prefixes},
        "allowed_layer_imports": {name: sorted(values) for name, values in allowed_layer_imports.items() if name in layers and layers[name]},
        "forbidden_layer_dependencies": [
            {
                "source_layer": "domain",
                "target_layer": "infrastructure",
                "reason": "Core domain modules should depend on stable abstractions rather than data adapters.",
            },
            {
                "source_layer": "domain",
                "target_layer": "tools",
                "reason": "Domain modules should not depend on repository analysis tooling.",
            },
        ],
        "forbidden_cross_package_imports": [
            {
                "source": "domain",
                "target": "fmp",
                "reason": "Domain rules should not call external market-data clients directly.",
            },
            {
                "source": "domain",
                "target": "pipeline.management.commands",
                "reason": "Domain code should stay independent from CLI and management entrypoints.",
            },
        ],
        "domain_boundaries": [
            {
                "name": "core_domain",
                "packages": ["domain"],
                "allowed_external_imports": ["utils"],
                "reason": "Core domain packages should stay narrow and stable.",
            }
        ],
        "quality_weights": {
            "complexity_health": 0.16,
            "dependency_health": 0.12,
            "duplication_health": 0.1,
            "typing_health": 0.1,
            "architecture_health": 0.16,
            "good_pattern_strength": 0.12,
            "anti_pattern_burden": 0.12,
            "llm_editability": 0.06,
            "change_safety": 0.06,
        },
    }
    return rules


def validate_architecture_rules(
    root: Path,
    *,
    rules_path: Path | None = None,
    inventory: RepositoryInventory | None = None,
    dependency_report: DependencyGraphReport | dict[str, Any] | None = None,
) -> ArchitectureRulesReport:
    inventory = inventory or build_repository_inventory(root)
    dependency = dependency_report
    if dependency is None:
        dependency = analyze_dependency_graph(root, inventory)
    dependency_payload = dependency.to_dict() if hasattr(dependency, "to_dict") else dict(dependency or {})

    rules = load_architecture_rules(rules_path)
    module_layers = classify_modules(inventory.modules.keys(), rules)
    import_lines = _collect_internal_import_lines(root, set(inventory.modules))

    violations: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    for edge in list(dependency_payload.get("edges") or []):
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if not source or not target:
            continue
        source_layer = module_layers.get(source, "unmapped")
        target_layer = module_layers.get(target, "unmapped")
        source_path = inventory.modules.get(source).path if source in inventory.modules else ""
        line_row = import_lines.get((source, target), {})

        for violation in _edge_violations(rules, source, target, source_layer, target_layer, source_path, line_row):
            key = (
                str(violation.get("rule_type") or ""),
                str(violation.get("source_module") or ""),
                str(violation.get("target_module") or ""),
                str(violation.get("message") or ""),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            violations.append(violation)

    violations.sort(
        key=lambda row: (
            _severity_rank(row.get("severity")),
            row.get("source_module", ""),
            row.get("target_module", ""),
            row.get("rule_type", ""),
        )
    )
    module_rows = _module_rows(inventory, module_layers, violations)
    summary = _build_summary(module_layers, violations)
    notes = []
    if not _HAS_YAML:
        notes.append("PyYAML is not installed; architecture rules were parsed through the JSON-compatible subset.")
    return ArchitectureRulesReport(
        rule_path=str(Path(rules_path or DEFAULT_RULES_PATH).resolve()),
        rules=rules,
        module_layers=module_layers,
        violations=violations,
        module_rows=module_rows,
        summary=summary,
        notes=notes,
    )


def classify_modules(modules: Any, rules: dict[str, Any]) -> dict[str, str]:
    layer_prefixes = [(layer, sorted(prefixes, key=len, reverse=True)) for layer, prefixes in dict(rules.get("layers") or {}).items()]
    rows: dict[str, str] = {}
    for module_name in modules:
        module_name = str(module_name)
        assigned = "unmapped"
        for layer, prefixes in layer_prefixes:
            if any(module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in prefixes):
                assigned = layer
                break
        rows[module_name] = assigned
    return rows


def architecture_rules_markdown(report: ArchitectureRulesReport) -> str:
    sections = [
        "# Architecture Rules Report",
        "",
        f"- rules file: `{report.rule_path}`",
        f"- mapped modules: {report.summary.get('mapped_modules', 0)}",
        f"- unmapped modules: {report.summary.get('unmapped_modules', 0)}",
        f"- violations: {report.summary.get('violation_count', 0)}",
        "",
        "## Violations By Type",
    ]
    by_type = list(report.summary.get("violations_by_type", {}).items())
    if by_type:
        sections.extend(f"- `{name}`: {count}" for name, count in by_type)
    else:
        sections.append("- none")
    sections.extend(["", "## Top Violations"])
    if report.violations:
        for violation in report.violations[:40]:
            sections.append(
                f"- `{violation['source_module']}` -> `{violation['target_module']}` [{violation['rule_type']}/{violation['severity']}]"
            )
            sections.append(f"  - {violation['message']}")
    else:
        sections.append("- none")
    sections.extend(["", "## Module Summary"])
    sections.extend(
        f"- `{row['module']}`: layer={row['layer']}, violations={row['violation_count']}"
        for row in report.module_rows[:30]
    )
    if report.notes:
        sections.extend(["", "## Notes", *[f"- {note}" for note in report.notes]])
    return "\n".join(sections)


def _edge_violations(
    rules: dict[str, Any],
    source: str,
    target: str,
    source_layer: str,
    target_layer: str,
    source_path: str,
    line_row: dict[str, Any],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    allowed = dict(rules.get("allowed_layer_imports") or {}).get(source_layer)
    if allowed is not None and target_layer not in set(allowed):
        violations.append(
            _violation(
                rule_type="layer_direction",
                source=source,
                target=target,
                source_layer=source_layer,
                target_layer=target_layer,
                source_path=source_path,
                line_row=line_row,
                message=f"`{source_layer}` is not allowed to import `{target_layer}` according to `allowed_layer_imports`.",
                severity="high" if source_layer in {"domain", "shared"} else "medium",
            )
        )
    for rule in list(rules.get("forbidden_layer_dependencies") or []):
        if source_layer == str(rule.get("source_layer") or "") and target_layer == str(rule.get("target_layer") or ""):
            violations.append(
                _violation(
                    rule_type="forbidden_layer_dependency",
                    source=source,
                    target=target,
                    source_layer=source_layer,
                    target_layer=target_layer,
                    source_path=source_path,
                    line_row=line_row,
                    message=str(rule.get("reason") or f"`{source_layer}` must not depend on `{target_layer}`."),
                    severity="high",
                )
            )
    for rule in list(rules.get("forbidden_cross_package_imports") or []):
        if _matches_prefix(source, str(rule.get("source") or "")) and _matches_prefix(target, str(rule.get("target") or "")):
            violations.append(
                _violation(
                    rule_type="forbidden_cross_package_import",
                    source=source,
                    target=target,
                    source_layer=source_layer,
                    target_layer=target_layer,
                    source_path=source_path,
                    line_row=line_row,
                    message=str(rule.get("reason") or "This cross-package import is forbidden."),
                    severity="high",
                )
            )
    for rule in list(rules.get("domain_boundaries") or []):
        packages = [str(item) for item in list(rule.get("packages") or []) if str(item)]
        allowed_external = [str(item) for item in list(rule.get("allowed_external_imports") or []) if str(item)]
        if packages and any(_matches_prefix(source, package) for package in packages):
            target_inside_boundary = any(_matches_prefix(target, package) for package in packages)
            target_allowed = any(_matches_prefix(target, package) for package in allowed_external)
            if not target_inside_boundary and not target_allowed:
                violations.append(
                    _violation(
                        rule_type="domain_boundary",
                        source=source,
                        target=target,
                        source_layer=source_layer,
                        target_layer=target_layer,
                        source_path=source_path,
                        line_row=line_row,
                        message=str(rule.get("reason") or f"Boundary `{rule.get('name')}` only allows imports within {packages}."),
                        severity="high" if source_layer == "domain" else "medium",
                    )
                )
    return violations


def _module_rows(
    inventory: RepositoryInventory,
    module_layers: dict[str, str],
    violations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for violation in violations:
        source_module = str(violation.get("source_module") or "")
        counts[source_module] = counts.get(source_module, 0) + 1
    rows = [
        {
            "module": module_name,
            "path": module_record.path,
            "layer": module_layers.get(module_name, "unmapped"),
            "violation_count": counts.get(module_name, 0),
        }
        for module_name, module_record in inventory.modules.items()
    ]
    rows.sort(key=lambda row: (-int(row["violation_count"]), row["layer"], row["module"]))
    return rows


def _build_summary(module_layers: dict[str, str], violations: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    by_layer: dict[str, int] = {}
    for violation in violations:
        by_type[str(violation.get("rule_type") or "")] = by_type.get(str(violation.get("rule_type") or ""), 0) + 1
        by_layer[str(violation.get("source_layer") or "unmapped")] = by_layer.get(str(violation.get("source_layer") or "unmapped"), 0) + 1
    return {
        "mapped_modules": sum(1 for layer in module_layers.values() if layer != "unmapped"),
        "unmapped_modules": sum(1 for layer in module_layers.values() if layer == "unmapped"),
        "violation_count": len(violations),
        "violations_by_type": dict(sorted(by_type.items(), key=lambda item: (-item[1], item[0]))),
        "violations_by_source_layer": dict(sorted(by_layer.items(), key=lambda item: (-item[1], item[0]))),
    }


def _collect_internal_import_lines(root: Path, internal_modules: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if ".git" in relative.parts or "__pycache__" in relative.parts or "migrations" in relative.parts:
            continue
        module_name = module_name_for_path(root, path)
        if not module_name:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    matched = _match_internal_module(str(alias.name or ""), internal_modules)
                    if matched:
                        rows.setdefault(
                            (module_name, matched),
                            {
                                "line_start": int(getattr(node, "lineno", 1) or 1),
                                "line_end": int(getattr(node, "end_lineno", getattr(node, "lineno", 1)) or 1),
                            },
                        )
            elif isinstance(node, ast.ImportFrom):
                resolved = resolve_import_module(module_name, node.module, int(node.level or 0))
                matched = _match_internal_module(resolved, internal_modules)
                if matched:
                    rows.setdefault(
                        (module_name, matched),
                        {
                            "line_start": int(getattr(node, "lineno", 1) or 1),
                            "line_end": int(getattr(node, "end_lineno", getattr(node, "lineno", 1)) or 1),
                        },
                    )
    return rows


def _match_internal_module(target: str, internal_modules: set[str]) -> str:
    target = str(target or "").strip()
    if not target:
        return ""
    if target in internal_modules:
        return target
    parts = target.split(".")
    while len(parts) > 1:
        parts = parts[:-1]
        candidate = ".".join(parts)
        if candidate in internal_modules:
            return candidate
    return ""


def _violation(
    *,
    rule_type: str,
    source: str,
    target: str,
    source_layer: str,
    target_layer: str,
    source_path: str,
    line_row: dict[str, Any],
    message: str,
    severity: str,
) -> dict[str, Any]:
    return {
        "rule_type": rule_type,
        "source_module": source,
        "target_module": target,
        "source_layer": source_layer,
        "target_layer": target_layer,
        "source_path": source_path,
        "line_start": int(line_row.get("line_start") or 1),
        "line_end": int(line_row.get("line_end") or int(line_row.get("line_start") or 1)),
        "message": message,
        "reason": message,
        "severity": severity,
    }


def _matches_prefix(module_name: str, prefix: str) -> bool:
    prefix = str(prefix or "").strip()
    if not prefix:
        return False
    return module_name == prefix or module_name.startswith(f"{prefix}.")


def _severity_rank(value: Any) -> tuple[int, str]:
    severity = str(value or "medium").lower()
    rank = {"high": 0, "medium": 1, "low": 2}.get(severity, 3)
    return rank, severity
