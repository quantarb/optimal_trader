from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import networkx as nx

    _HAS_NETWORKX = True
except ImportError:
    nx = None
    _HAS_NETWORKX = False

from .architecture_rules import ArchitectureRulesReport, validate_architecture_rules
from .call_graph import CallGraphReport, analyze_call_graph
from .code_metrics import analyze_code_metrics
from .module_responsibility import ModuleResponsibilityReport, analyze_module_responsibilities
from .pattern_metrics import CodeHealthMetricsReport, analyze_code_health_metrics
from .patterns.anti_patterns import AntiPatternReport, analyze_anti_patterns
from .repository import ClassRecord, RepositoryInventory, build_repository_inventory


@dataclass
class BlastRadiusReport:
    backend: str
    module_rows: list[dict[str, Any]]
    symbol_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "module_rows": list(self.module_rows),
            "symbol_rows": list(self.symbol_rows),
            "summary": dict(self.summary),
            "notes": list(self.notes),
        }


def analyze_blast_radius(
    root: Path,
    *,
    inventory: RepositoryInventory | None = None,
    dependency_report: dict[str, Any] | None = None,
    call_graph_report: CallGraphReport | dict[str, Any] | None = None,
    code_health_report: CodeHealthMetricsReport | dict[str, Any] | None = None,
    anti_pattern_report: AntiPatternReport | dict[str, Any] | None = None,
    architecture_report: ArchitectureRulesReport | dict[str, Any] | None = None,
    responsibility_report: ModuleResponsibilityReport | dict[str, Any] | None = None,
) -> BlastRadiusReport:
    inventory = inventory or build_repository_inventory(root)
    dependency_payload = dict(dependency_report or {})
    if not dependency_payload:
        from .dependency_graph import analyze_dependency_graph

        dependency_payload = analyze_dependency_graph(root, inventory).to_dict()
    call_payload = _to_payload(call_graph_report) or analyze_call_graph(inventory).to_dict()
    health_payload = _to_payload(code_health_report) or analyze_code_health_metrics(root, inventory=inventory).to_dict()
    anti_payload = _to_payload(anti_pattern_report) or analyze_anti_patterns(root, inventory=inventory).to_dict()
    architecture_payload = _to_payload(architecture_report) or validate_architecture_rules(root, inventory=inventory, dependency_report=dependency_payload).to_dict()
    responsibility_payload = _to_payload(responsibility_report) or analyze_module_responsibilities(
        inventory,
        metrics_report=analyze_code_metrics(root).to_dict(),
        dependency_report=dependency_payload,
    ).to_dict()

    module_rows = _build_module_rows(
        inventory=inventory,
        dependency_payload=dependency_payload,
        call_payload=call_payload,
        health_payload=health_payload,
        anti_payload=anti_payload,
        architecture_payload=architecture_payload,
        responsibility_payload=responsibility_payload,
    )
    symbol_rows = _build_symbol_rows(
        inventory=inventory,
        call_payload=call_payload,
        module_rows=module_rows,
    )
    summary = _build_summary(module_rows, symbol_rows)
    notes = [
        "Direct and indirect dependents are estimated from internal import edges for modules and internal call edges for symbols.",
        "Critical execution path status is derived from the existing major pipeline extraction in the call graph plus dependency centrality.",
        "Change risk and refactor leverage are deterministic proxy scores meant for ranking opportunities, not as a substitute for tests.",
    ]
    return BlastRadiusReport(
        backend="dependency_graph+call_graph+quality_metrics",
        module_rows=module_rows,
        symbol_rows=symbol_rows,
        summary=summary,
        notes=notes,
    )


def blast_radius_markdown(report: BlastRadiusReport) -> str:
    sections = [
        "# Blast Radius Report",
        "",
        f"- modules analyzed: {len(report.module_rows)}",
        f"- major symbols analyzed: {len(report.symbol_rows)}",
        "",
        "## Top 10 Highest-Blast-Radius Modules",
    ]
    sections.extend(
        f"- `{row['module']}`: blast={row['blast_radius_score']:.2f}, direct={row['direct_dependents']}, indirect={row['indirect_dependents']}, risk={row['change_risk_level']}"
        for row in report.summary.get("top_10_highest_blast_radius_modules", [])
    )
    sections.extend(["", "## Top 10 Highest-Risk Modules To Change"])
    sections.extend(
        f"- `{row['module']}`: risk={row['change_risk_score']:.2f}, blast={row['blast_radius_score']:.2f}, badness={row['architectural_badness']:.2f}"
        for row in report.summary.get("top_10_highest_risk_modules", [])
    )
    sections.extend(["", "## Top 10 Highest-Blast-Radius Symbols"])
    sections.extend(
        f"- `{row['symbol']}`: blast={row['blast_radius_score']:.2f}, direct={row['direct_dependents']}, indirect={row['indirect_dependents']}, risk={row['change_risk_level']}"
        for row in report.summary.get("top_10_highest_blast_radius_symbols", [])
    )
    sections.extend(["", "## Top God Modules"])
    god_modules = [row for row in report.module_rows if row["god_module"]][:10]
    if god_modules:
        sections.extend(
            f"- `{row['module']}`: {', '.join(row['god_module_reasons'][:4])}"
            for row in god_modules
        )
    else:
        sections.append("- none")
    if report.notes:
        sections.extend(["", "## Notes", *[f"- {note}" for note in report.notes]])
    return "\n".join(sections)


def _build_module_rows(
    *,
    inventory: RepositoryInventory,
    dependency_payload: dict[str, Any],
    call_payload: dict[str, Any],
    health_payload: dict[str, Any],
    anti_payload: dict[str, Any],
    architecture_payload: dict[str, Any],
    responsibility_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    modules = sorted(inventory.modules)
    dep_reverse = _reverse_adjacency(_adjacency_from_edges(dependency_payload.get("edges") or []))
    dep_centrality = _module_dependency_centrality(modules, dependency_payload.get("edges") or [])
    module_pipeline_hits, module_root_hits, symbol_pipeline_hits, symbol_root_hits = _pipeline_hits(call_payload)
    health_map = {row["module"]: row for row in list(health_payload.get("module_rows") or [])}
    anti_by_path = _anti_findings_by_path(list(anti_payload.get("findings") or []))
    architecture_counts = _count_by_key(list(architecture_payload.get("violations") or []), "source_module")
    responsibility_map = {row["module"]: row for row in list(responsibility_payload.get("module_rows") or [])}

    direct_counts = {module: len(dep_reverse.get(module, set())) for module in modules}
    indirect_counts = {module: len(_reverse_reach(dep_reverse, module)) - direct_counts.get(module, 0) for module in modules}
    critical_raw = {
        module: float(module_pipeline_hits.get(module, 0)) + float(module_root_hits.get(module, 0)) * 1.5
        for module in modules
    }

    max_direct = max(direct_counts.values(), default=1)
    max_indirect = max(indirect_counts.values(), default=1)
    max_centrality = max((float(dep_centrality.get(module, 0.0)) for module in modules), default=1.0) or 1.0
    max_critical = max(critical_raw.values(), default=1.0) or 1.0

    rows: list[dict[str, Any]] = []
    for module in modules:
        health_row = health_map.get(module, {})
        anti_rows = anti_by_path.get(str(inventory.modules[module].path), [])
        architecture_violations = int(architecture_counts.get(module, 0))
        responsibility_row = responsibility_map.get(module, {})
        mixed_concerns = sum(1 for row in anti_rows if row.get("pattern") == "mixed concerns modules")
        high_severity_anti = sum(1 for row in anti_rows if str(row.get("severity") or "").lower() == "high")
        god_module, god_reasons = _is_god_module(
            line_count=int(health_row.get("line_count") or inventory.modules[module].line_count),
            function_count=int(health_row.get("function_count") or len(inventory.modules[module].functions)),
            class_count=int(health_row.get("class_count") or len(inventory.modules[module].class_records)),
            fan_in=int(health_row.get("dependency_fan_in") or 0),
            fan_out=int(health_row.get("dependency_fan_out") or 0),
            architecture_violations=architecture_violations,
            anti_pattern_count=len(anti_rows),
            mixing_score=float(responsibility_row.get("mixing_score") or 0.0),
        )
        architectural_badness = _architectural_badness(
            architecture_violations=architecture_violations,
            import_cycle_count=int(health_row.get("import_cycle_count") or 0),
            mixed_concerns=mixed_concerns,
            fan_in=int(health_row.get("dependency_fan_in") or 0),
            fan_out=int(health_row.get("dependency_fan_out") or 0),
            max_complexity=float(health_row.get("cyclomatic_complexity_max") or 0.0),
            anti_pattern_count=len(anti_rows),
            high_severity_anti=high_severity_anti,
            god_module=god_module,
            mixing_score=float(responsibility_row.get("mixing_score") or 0.0),
        )
        direct_norm = float(direct_counts.get(module, 0)) / max_direct if max_direct else 0.0
        indirect_norm = float(indirect_counts.get(module, 0)) / max_indirect if max_indirect else 0.0
        centrality_norm = float(dep_centrality.get(module, 0.0)) / max_centrality if max_centrality else 0.0
        critical_norm = float(critical_raw.get(module, 0.0)) / max_critical if max_critical else 0.0
        blast_radius_score = round(
            100.0
            * (
                (indirect_norm * 0.36)
                + (direct_norm * 0.26)
                + (centrality_norm * 0.24)
                + (critical_norm * 0.14)
            ),
            2,
        )
        critical_execution_path = bool(module_pipeline_hits.get(module, 0) or module_root_hits.get(module, 0)) and (
            direct_counts.get(module, 0) >= 2 or dep_centrality.get(module, 0.0) >= mean(dep_centrality.values() or [0.0])
        )
        change_safety = float(health_row.get("change_safety_proxy_score") or 0.0)
        change_risk_score = round(
            _clamp(
                (blast_radius_score * 0.34)
                + (architectural_badness * 0.28)
                + ((100.0 - change_safety) * 0.23)
                + (12.0 if critical_execution_path else 0.0)
                + (8.0 if god_module else 0.0)
            ),
            2,
        )
        centrality_score = round(centrality_norm * 100.0, 2)
        estimated_refactor_leverage = round(
            _clamp((architectural_badness * 0.48) + (centrality_score * 0.24) + (blast_radius_score * 0.28)),
            2,
        )
        rows.append(
            {
                "module": module,
                "path": inventory.modules[module].path,
                "line_count": int(health_row.get("line_count") or inventory.modules[module].line_count),
                "function_count": int(health_row.get("function_count") or len(inventory.modules[module].functions)),
                "class_count": int(health_row.get("class_count") or len(inventory.modules[module].class_records)),
                "direct_dependents": int(direct_counts.get(module, 0)),
                "indirect_dependents": int(max(indirect_counts.get(module, 0), 0)),
                "dependency_centrality": round(float(dep_centrality.get(module, 0.0)), 6),
                "dependency_centrality_score": centrality_score,
                "critical_execution_path": critical_execution_path,
                "critical_path_hits": int(module_pipeline_hits.get(module, 0)),
                "critical_path_root_hits": int(module_root_hits.get(module, 0)),
                "god_module": god_module,
                "god_module_reasons": god_reasons,
                "architectural_badness": architectural_badness,
                "blast_radius_score": blast_radius_score,
                "change_risk_score": change_risk_score,
                "change_risk_level": _risk_level(change_risk_score),
                "estimated_refactor_leverage": estimated_refactor_leverage,
                "anti_pattern_count": len(anti_rows),
                "high_severity_anti_patterns": high_severity_anti,
                "architecture_rule_violations": architecture_violations,
                "change_safety_proxy_score": round(change_safety, 2),
                "quality_context": {
                    "llm_editability_proxy_score": round(float(health_row.get("llm_editability_proxy_score") or 0.0), 2),
                    "duplicate_code_clusters": int(health_row.get("duplicate_code_clusters") or 0),
                    "dead_code_count": int(health_row.get("dead_code_count") or 0),
                    "mixing_score": round(float(responsibility_row.get("mixing_score") or 0.0), 2),
                },
                "risk_reasons": _module_risk_reasons(
                    blast_radius_score=blast_radius_score,
                    architectural_badness=architectural_badness,
                    critical_execution_path=critical_execution_path,
                    god_module=god_module,
                    architecture_violations=architecture_violations,
                    direct_dependents=int(direct_counts.get(module, 0)),
                    indirect_dependents=int(max(indirect_counts.get(module, 0), 0)),
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["blast_radius_score"]),
            -float(row["change_risk_score"]),
            row["module"],
        )
    )
    return rows


def _build_symbol_rows(
    *,
    inventory: RepositoryInventory,
    call_payload: dict[str, Any],
    module_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    call_edges = list(call_payload.get("edges") or [])
    adjacency = _adjacency_from_edges(call_edges)
    reverse = _reverse_adjacency(adjacency)
    centrality = _symbol_centrality(list(call_payload.get("nodes") or []), call_edges)
    module_risk_map = {row["module"]: row for row in module_rows}
    _module_pipeline_hits, _module_root_hits, symbol_pipeline_hits, symbol_root_hits = _pipeline_hits(call_payload)

    major_rows: list[dict[str, Any]] = []
    max_direct = 1
    max_indirect = 1
    max_centrality = max((float(value) for value in centrality.values()), default=1.0) or 1.0
    max_critical = max(
        (
            float(symbol_pipeline_hits.get(symbol, 0)) + float(symbol_root_hits.get(symbol, 0)) * 1.5
            for symbol in call_payload.get("nodes") or []
        ),
        default=1.0,
    ) or 1.0

    function_records = inventory.functions
    class_records = inventory.classes
    class_method_map = {
        class_full_name: [name for name in function_records if name.startswith(f"{class_full_name}.")]
        for class_full_name in class_records
    }

    symbol_candidates: list[dict[str, Any]] = []
    for symbol, record in function_records.items():
        indegree = len(reverse.get(symbol, set()))
        outdegree = len(adjacency.get(symbol, set()))
        if not _is_major_function(symbol, record, indegree, outdegree, symbol_pipeline_hits):
            continue
        indirect = len(_reverse_reach(reverse, symbol)) - indegree
        max_direct = max(max_direct, indegree)
        max_indirect = max(max_indirect, max(indirect, 0))
        symbol_candidates.append(
            {
                "symbol": symbol,
                "symbol_type": "function",
                "module": record.module,
                "path": record.path,
                "line_start": int(record.lineno),
                "line_end": int(record.end_lineno),
                "direct_dependents": indegree,
                "indirect_dependents": max(indirect, 0),
                "centrality_raw": float(centrality.get(symbol, 0.0)),
                "critical_raw": float(symbol_pipeline_hits.get(symbol, 0)) + float(symbol_root_hits.get(symbol, 0)) * 1.5,
            }
        )
    for symbol, record in class_records.items():
        method_names = class_method_map.get(symbol, [])
        if not _is_major_class(record):
            continue
        direct_callers = {caller for method_name in method_names for caller in reverse.get(method_name, set())}
        indirect_callers = {caller for method_name in method_names for caller in _reverse_reach(reverse, method_name)}
        max_direct = max(max_direct, len(direct_callers))
        max_indirect = max(max_indirect, max(len(indirect_callers) - len(direct_callers), 0))
        symbol_candidates.append(
            {
                "symbol": symbol,
                "symbol_type": "class",
                "module": record.module,
                "path": record.path,
                "line_start": int(record.lineno),
                "line_end": int(record.end_lineno),
                "direct_dependents": len(direct_callers),
                "indirect_dependents": max(len(indirect_callers) - len(direct_callers), 0),
                "centrality_raw": _class_centrality(method_names, centrality),
                "critical_raw": _class_centrality(method_names, {**symbol_pipeline_hits, **{k: float(v) for k, v in symbol_root_hits.items()}}),
                "god_like_symbol": bool(record.line_count >= 250 or len(record.methods) >= 10),
            }
        )

    for row in symbol_candidates:
        direct_norm = float(row["direct_dependents"]) / max_direct if max_direct else 0.0
        indirect_norm = float(row["indirect_dependents"]) / max_indirect if max_indirect else 0.0
        centrality_norm = float(row["centrality_raw"]) / max_centrality if max_centrality else 0.0
        critical_norm = float(row["critical_raw"]) / max_critical if max_critical else 0.0
        blast_radius_score = round(
            100.0
            * (
                (indirect_norm * 0.34)
                + (direct_norm * 0.27)
                + (centrality_norm * 0.27)
                + (critical_norm * 0.12)
            ),
            2,
        )
        module_risk = float(module_risk_map.get(row["module"], {}).get("change_risk_score") or 0.0)
        change_risk_score = round(_clamp((blast_radius_score * 0.58) + (module_risk * 0.42)), 2)
        major_rows.append(
            {
                "symbol": row["symbol"],
                "symbol_type": row["symbol_type"],
                "module": row["module"],
                "path": row["path"],
                "line_start": row["line_start"],
                "line_end": row["line_end"],
                "direct_dependents": int(row["direct_dependents"]),
                "indirect_dependents": int(row["indirect_dependents"]),
                "centrality": round(float(row["centrality_raw"]), 6),
                "centrality_score": round(centrality_norm * 100.0, 2),
                "critical_execution_path": bool(row["critical_raw"]),
                "critical_path_hits": int(symbol_pipeline_hits.get(row["symbol"], 0)),
                "critical_path_root_hits": int(symbol_root_hits.get(row["symbol"], 0)),
                "blast_radius_score": blast_radius_score,
                "change_risk_score": change_risk_score,
                "change_risk_level": _risk_level(change_risk_score),
                "god_like_symbol": bool(row.get("god_like_symbol", False)),
            }
        )
    major_rows.sort(
        key=lambda row: (
            -float(row["blast_radius_score"]),
            -float(row["change_risk_score"]),
            row["symbol"],
        )
    )
    return major_rows


def _module_dependency_centrality(nodes: list[str], edges: list[dict[str, str]]) -> dict[str, float]:
    if _HAS_NETWORKX:
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from((edge["source"], edge["target"]) for edge in edges)
        pagerank = nx.pagerank(graph) if graph.number_of_nodes() else {}
        betweenness = nx.betweenness_centrality(graph, normalized=True) if graph.number_of_nodes() else {}
        return {
            node: round((float(pagerank.get(node, 0.0)) * 0.72) + (float(betweenness.get(node, 0.0)) * 0.28), 8)
            for node in nodes
        }
    indegree = {node: 0 for node in nodes}
    outdegree = {node: 0 for node in nodes}
    for edge in edges:
        indegree[edge["target"]] = indegree.get(edge["target"], 0) + 1
        outdegree[edge["source"]] = outdegree.get(edge["source"], 0) + 1
    max_degree = max((indegree[node] + outdegree[node] for node in nodes), default=1) or 1
    return {
        node: round((indegree.get(node, 0) + outdegree.get(node, 0)) / max_degree, 8)
        for node in nodes
    }


def _symbol_centrality(nodes: list[str], edges: list[dict[str, str]]) -> dict[str, float]:
    if _HAS_NETWORKX:
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from((edge["source"], edge["target"]) for edge in edges)
        pagerank = nx.pagerank(graph) if graph.number_of_nodes() else {}
        betweenness = nx.betweenness_centrality(graph, normalized=True) if graph.number_of_nodes() else {}
        return {
            node: round((float(pagerank.get(node, 0.0)) * 0.7) + (float(betweenness.get(node, 0.0)) * 0.3), 8)
            for node in nodes
        }
    indegree = {node: 0 for node in nodes}
    outdegree = {node: 0 for node in nodes}
    for edge in edges:
        indegree[edge["target"]] = indegree.get(edge["target"], 0) + 1
        outdegree[edge["source"]] = outdegree.get(edge["source"], 0) + 1
    max_degree = max((indegree[node] + outdegree[node] for node in nodes), default=1) or 1
    return {
        node: round((indegree.get(node, 0) + outdegree.get(node, 0)) / max_degree, 8)
        for node in nodes
    }


def _anti_findings_by_path(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    payload: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        payload.setdefault(str(row.get("file") or ""), []).append(row)
    return payload


def _count_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _pipeline_hits(call_payload: dict[str, Any]) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    module_hits: dict[str, int] = {}
    module_root_hits: dict[str, int] = {}
    symbol_hits: dict[str, int] = {}
    symbol_root_hits: dict[str, int] = {}
    for pipeline in list(call_payload.get("major_pipelines") or []):
        root = str(pipeline.get("root") or "")
        if root:
            symbol_root_hits[root] = symbol_root_hits.get(root, 0) + 1
            root_module = root.rsplit(".", 1)[0] if "." in root else root
            module_root_hits[root_module] = module_root_hits.get(root_module, 0) + 1
        pipeline_symbols: set[str] = set()
        pipeline_modules: set[str] = set()
        for row in list(pipeline.get("walk") or []):
            symbol = str(row.get("function") or "")
            if not symbol:
                continue
            pipeline_symbols.add(symbol)
            pipeline_modules.add(symbol.rsplit(".", 1)[0] if "." in symbol else symbol)
        for symbol in pipeline_symbols:
            symbol_hits[symbol] = symbol_hits.get(symbol, 0) + 1
        for module in pipeline_modules:
            module_hits[module] = module_hits.get(module, 0) + 1
    return module_hits, module_root_hits, symbol_hits, symbol_root_hits


def _is_god_module(
    *,
    line_count: int,
    function_count: int,
    class_count: int,
    fan_in: int,
    fan_out: int,
    architecture_violations: int,
    anti_pattern_count: int,
    mixing_score: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if line_count >= 500:
        reasons.append(f"large module ({line_count} lines)")
    if function_count >= 15:
        reasons.append(f"many functions ({function_count})")
    if class_count >= 5:
        reasons.append(f"many classes ({class_count})")
    if fan_in >= 4 and fan_out >= 6:
        reasons.append(f"hub dependency profile (in={fan_in}, out={fan_out})")
    if architecture_violations >= 3:
        reasons.append(f"multiple architecture violations ({architecture_violations})")
    if anti_pattern_count >= 10:
        reasons.append(f"high anti-pattern burden ({anti_pattern_count})")
    if mixing_score >= 60.0:
        reasons.append(f"mixed-concern score {mixing_score:.1f}")
    return len(reasons) >= 3 or (line_count >= 900 and anti_pattern_count >= 8), reasons


def _architectural_badness(
    *,
    architecture_violations: int,
    import_cycle_count: int,
    mixed_concerns: int,
    fan_in: int,
    fan_out: int,
    max_complexity: float,
    anti_pattern_count: int,
    high_severity_anti: int,
    god_module: bool,
    mixing_score: float,
) -> float:
    score = 0.0
    score += architecture_violations * 14.0
    score += import_cycle_count * 20.0
    score += mixed_concerns * 18.0
    score += min(mixing_score / 4.0, 18.0)
    score += max(0.0, fan_out - 6.0) * 2.2
    score += max(0.0, fan_in - 10.0) * 1.1
    score += max(0.0, max_complexity - 15.0) * 0.85
    score += min(anti_pattern_count, 15) * 1.4
    score += min(high_severity_anti, 10) * 2.3
    if god_module:
        score += 10.0
    return round(_clamp(score), 2)


def _module_risk_reasons(
    *,
    blast_radius_score: float,
    architectural_badness: float,
    critical_execution_path: bool,
    god_module: bool,
    architecture_violations: int,
    direct_dependents: int,
    indirect_dependents: int,
) -> list[str]:
    reasons: list[str] = []
    if blast_radius_score >= 70.0:
        reasons.append(f"blast radius score {blast_radius_score:.1f}")
    if direct_dependents >= 5:
        reasons.append(f"{direct_dependents} direct dependents")
    if indirect_dependents >= 15:
        reasons.append(f"{indirect_dependents} indirect dependents")
    if architectural_badness >= 60.0:
        reasons.append(f"architectural badness {architectural_badness:.1f}")
    if architecture_violations:
        reasons.append(f"{architecture_violations} architecture violations")
    if critical_execution_path:
        reasons.append("appears on a major execution path")
    if god_module:
        reasons.append("behaves like a god module")
    return reasons


def _is_major_function(
    symbol: str,
    record,
    indegree: int,
    outdegree: int,
    pipeline_hits: dict[str, int],
) -> bool:
    if pipeline_hits.get(symbol):
        return True
    if indegree + outdegree >= 2:
        return True
    if record.line_count >= 25:
        return True
    return not record.name.startswith("_") and not record.class_name


def _is_major_class(record: ClassRecord) -> bool:
    return bool(record.methods) and (record.line_count >= 60 or len(record.methods) >= 3 or not record.name.startswith("_"))


def _class_centrality(method_names: list[str], values: dict[str, float]) -> float:
    if not method_names:
        return 0.0
    return round(max(float(values.get(name, 0.0)) for name in method_names), 8)


def _build_summary(module_rows: list[dict[str, Any]], symbol_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked_modules = [row for row in module_rows if not _is_test_name(str(row.get("module") or ""))]
    ranked_symbols = [row for row in symbol_rows if not _is_test_name(str(row.get("module") or ""))]
    highest_blast = sorted(ranked_modules, key=lambda row: (-float(row["blast_radius_score"]), row["module"]))[:10]
    highest_risk = sorted(ranked_modules, key=lambda row: (-float(row["change_risk_score"]), row["module"]))[:10]
    highest_symbol_blast = sorted(ranked_symbols, key=lambda row: (-float(row["blast_radius_score"]), row["symbol"]))[:10]
    return {
        "top_10_highest_blast_radius_modules": highest_blast,
        "top_10_highest_risk_modules": highest_risk,
        "top_10_highest_blast_radius_symbols": highest_symbol_blast,
    }


def _adjacency_from_edges(edges: list[dict[str, str]]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source and target:
            adjacency[source].add(target)
            adjacency.setdefault(target, set())
    return adjacency


def _reverse_adjacency(adjacency: dict[str, set[str]]) -> dict[str, set[str]]:
    reverse: dict[str, set[str]] = defaultdict(set)
    for source, targets in adjacency.items():
        reverse.setdefault(source, set())
        for target in targets:
            reverse[target].add(source)
            reverse.setdefault(target, set())
    return reverse


def _reverse_reach(reverse: dict[str, set[str]], node: str) -> set[str]:
    visited: set[str] = set()
    queue: deque[str] = deque(reverse.get(node, set()))
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        for parent in reverse.get(current, set()):
            if parent not in visited:
                queue.append(parent)
    return visited


def _risk_level(score: float) -> str:
    if score >= 75.0:
        return "critical"
    if score >= 60.0:
        return "high"
    if score >= 40.0:
        return "medium"
    return "low"


def _is_test_name(name: str) -> bool:
    value = str(name or "")
    return value == "tests" or value.startswith("tests.") or ".tests." in value or value.endswith(".tests")


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def _to_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return dict(value)
