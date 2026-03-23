from __future__ import annotations

from typing import Any


PATTERN_METADATA = {
    "artifact return boundary": {
        "why": "Explicit artifact-style outputs keep downstream contracts visible and make side effects easier to isolate.",
        "expected_benefit": "Safer edits around pipeline and reporting boundaries with fewer hidden output changes.",
        "refactor_action": "introduce an explicit artifact return object around the module boundary",
    },
    "dataclass config pattern": {
        "why": "Structured config objects reduce argument drift and make tuning knobs easier to evolve safely.",
        "expected_benefit": "Fewer scattered literals and less fragile parameter plumbing across related helpers.",
        "refactor_action": "replace scattered knobs with a typed dataclass config object",
    },
    "explicit boundary objects": {
        "why": "Boundary objects make data movement explicit when a module mixes orchestration, IO, and transformation work.",
        "expected_benefit": "Lower change risk because inputs and outputs stop leaking through ad hoc dictionaries and side effects.",
        "refactor_action": "extract typed boundary objects for the cross-module inputs and outputs",
    },
    "guard clause style": {
        "why": "Guard clauses flatten deeply nested logic and keep the happy path easier to reason about.",
        "expected_benefit": "Lower nesting and smaller edit surfaces in long control-flow-heavy functions.",
        "refactor_action": "flatten control flow with early-return guard clauses",
    },
    "registry pattern": {
        "why": "Registries fit modules where behavior variants are currently routed through central dispatch logic.",
        "expected_benefit": "Adding a new variant becomes localized instead of editing another branch in a shared switchboard.",
        "refactor_action": "replace central dispatch with a registry-based lookup",
    },
    "reusable pipeline stages": {
        "why": "Named pipeline stages fit modules with repeated orchestration steps and duplicated workflow shapes.",
        "expected_benefit": "Common workflow chunks become easier to compose, test, and reuse across related flows.",
        "refactor_action": "extract repeated orchestration into reusable pipeline stages",
    },
    "single source of truth constants or schema": {
        "why": "Centralizing repeated literals and schema-like values lowers drift across related logic.",
        "expected_benefit": "Fewer accidental inconsistencies when changing thresholds, field names, or shared semantics.",
        "refactor_action": "promote repeated literals into shared constants or schema objects",
    },
    "stable base class or protocol": {
        "why": "Stable interfaces help central, widely reused modules evolve without forcing invasive cross-repo edits.",
        "expected_benefit": "Call sites can depend on a stable contract while implementations change behind the boundary.",
        "refactor_action": "introduce a stable protocol or base abstraction for the shared contract",
    },
    "strategy or policy interface": {
        "why": "Strategy-style seams fit modules where variant behavior is expressed through repeated branching.",
        "expected_benefit": "Behavior changes become additive and isolated instead of expanding branch-heavy orchestrators.",
        "refactor_action": "extract a strategy or policy interface around the variant behavior",
    },
    "typed public APIs": {
        "why": "Typed APIs matter most where many dependents rely on a module with low annotation coverage.",
        "expected_benefit": "Clearer contracts for humans and coding agents, plus safer call-site refactors.",
        "refactor_action": "add full type hints to the public entry points and boundary types",
    },
}


def build_pattern_recommendations(
    *,
    blast_radius_report: dict[str, Any],
    code_health_report: dict[str, Any] | None = None,
    anti_pattern_report: dict[str, Any] | None = None,
    good_pattern_report: dict[str, Any] | None = None,
    responsibility_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    health_by_module = {row["module"]: row for row in list((code_health_report or {}).get("module_rows") or [])}
    responsibility_by_module = {row["module"]: row for row in list((responsibility_report or {}).get("module_rows") or [])}
    anti_by_path = _group_pattern_stats(list((anti_pattern_report or {}).get("findings") or []), "file", kind="bad")
    anti_by_symbol = _group_pattern_stats(list((anti_pattern_report or {}).get("findings") or []), "symbol", kind="bad")
    good_by_path = _group_pattern_stats(list((good_pattern_report or {}).get("findings") or []), "file", kind="good")
    good_by_symbol = _group_pattern_stats(list((good_pattern_report or {}).get("findings") or []), "symbol", kind="good")

    module_recommendations: list[dict[str, Any]] = []
    for row in list(blast_radius_report.get("module_rows") or []):
        module = str(row.get("module") or "")
        if _is_test_name(module):
            continue
        path = str(row.get("path") or "")
        candidates = _module_candidates(
            blast_row=row,
            health_row=health_by_module.get(module, {}),
            responsibility_row=responsibility_by_module.get(module, {}),
            anti_stats=anti_by_path.get(path, _empty_pattern_stats()),
            good_stats=good_by_path.get(path, _empty_pattern_stats()),
        )
        module_recommendations.append(
            {
                "module": module,
                "path": path,
                "recommended_pattern": candidates[0]["pattern"] if candidates else "",
                "recommended_pattern_fit_score": candidates[0]["fit_score"] if candidates else 0.0,
                "recommended_pattern_safe_adoption_score": candidates[0]["safe_adoption_score"] if candidates else 0.0,
                "pattern_candidates": candidates,
            }
        )
    module_recommendations.sort(
        key=lambda row: (
            -float(row.get("recommended_pattern_fit_score") or 0.0),
            -float(row.get("recommended_pattern_safe_adoption_score") or 0.0),
            str(row.get("module") or ""),
        )
    )

    symbol_recommendations: list[dict[str, Any]] = []
    for row in list(blast_radius_report.get("symbol_rows") or []):
        symbol = str(row.get("symbol") or "")
        module = str(row.get("module") or "")
        if _is_test_name(module) or _is_test_name(symbol):
            continue
        candidates = _symbol_candidates(
            symbol_row=row,
            module_health_row=health_by_module.get(module, {}),
            anti_stats=anti_by_symbol.get(symbol, _empty_pattern_stats()),
            good_stats=good_by_symbol.get(symbol, _empty_pattern_stats()),
        )
        if not candidates:
            continue
        symbol_recommendations.append(
            {
                "symbol": symbol,
                "module": module,
                "path": row.get("path"),
                "line_start": row.get("line_start"),
                "line_end": row.get("line_end"),
                "recommended_pattern": candidates[0]["pattern"],
                "recommended_pattern_fit_score": candidates[0]["fit_score"],
                "recommended_pattern_safe_adoption_score": candidates[0]["safe_adoption_score"],
                "blast_radius_score": row.get("blast_radius_score"),
                "change_risk_score": row.get("change_risk_score"),
                "pattern_candidates": candidates,
            }
        )
    symbol_recommendations.sort(
        key=lambda row: (
            -float(row.get("recommended_pattern_fit_score") or 0.0),
            -float(row.get("recommended_pattern_safe_adoption_score") or 0.0),
            str(row.get("symbol") or ""),
        )
    )

    all_top_candidates = [
        {
            "target_type": "module",
            "target": row["module"],
            "module": row["module"],
            "path": row["path"],
            "recommended_pattern": row["recommended_pattern"],
            "fit_score": row["recommended_pattern_fit_score"],
            "safe_adoption_score": row["recommended_pattern_safe_adoption_score"],
        }
        for row in module_recommendations
        if row["recommended_pattern"]
    ]
    all_top_candidates.extend(
        {
            "target_type": "symbol",
            "target": row["symbol"],
            "module": row["module"],
            "path": row["path"],
            "recommended_pattern": row["recommended_pattern"],
            "fit_score": row["recommended_pattern_fit_score"],
            "safe_adoption_score": row["recommended_pattern_safe_adoption_score"],
        }
        for row in symbol_recommendations
        if row["recommended_pattern"]
    )
    all_top_candidates.sort(
        key=lambda row: (
            -float(row.get("fit_score") or 0.0),
            -float(row.get("safe_adoption_score") or 0.0),
            str(row.get("target") or ""),
        )
    )

    pattern_type_counts: dict[str, int] = {}
    for row in all_top_candidates:
        pattern = str(row.get("recommended_pattern") or "")
        if not pattern:
            continue
        pattern_type_counts[pattern] = pattern_type_counts.get(pattern, 0) + 1

    return {
        "module_recommendations": module_recommendations,
        "symbol_recommendations": symbol_recommendations,
        "summary": {
            "pattern_type_counts": dict(sorted(pattern_type_counts.items(), key=lambda item: (-item[1], item[0]))),
            "top_10_module_pattern_candidates": module_recommendations[:10],
            "top_10_symbol_pattern_candidates": symbol_recommendations[:10],
            "top_10_safest_pattern_adoptions": sorted(
                all_top_candidates,
                key=lambda row: (-float(row.get("safe_adoption_score") or 0.0), row.get("target", "")),
            )[:10],
        },
    }


def _module_candidates(
    *,
    blast_row: dict[str, Any],
    health_row: dict[str, Any],
    responsibility_row: dict[str, Any],
    anti_stats: dict[str, Any],
    good_stats: dict[str, Any],
) -> list[dict[str, Any]]:
    type_hint_coverage = float(health_row.get("type_hint_coverage") or 0.0)
    fan_in = int(health_row.get("dependency_fan_in") or 0)
    fan_out = int(health_row.get("dependency_fan_out") or 0)
    duplicate_clusters = int(health_row.get("duplicate_code_clusters") or 0)
    config_usage = int(health_row.get("config_object_usage") or 0)
    artifact_usage = int(health_row.get("artifact_boundary_usage") or 0)
    interface_reuse = int(health_row.get("interface_reuse_count") or 0)
    architecture_violations = int(blast_row.get("architecture_rule_violations") or health_row.get("architecture_rule_violations") or 0)
    family_count = int(responsibility_row.get("concern_family_count") or 0)
    dominant_share = float(responsibility_row.get("dominant_concern_share") or 1.0)
    mixing_score = float((blast_row.get("quality_context") or {}).get("mixing_score") or responsibility_row.get("mixing_score") or 0.0)
    blast_score = float(blast_row.get("blast_radius_score") or 0.0)
    change_risk = float(blast_row.get("change_risk_score") or 0.0)
    centrality_score = float(blast_row.get("dependency_centrality_score") or 0.0)
    leverage_score = float(blast_row.get("estimated_refactor_leverage") or 0.0)
    change_safety = float(blast_row.get("change_safety_proxy_score") or health_row.get("change_safety_proxy_score") or 0.0)
    class_count = int(blast_row.get("class_count") or health_row.get("class_count") or 0)

    dispatch_count = _count(anti_stats, "repeated if/elif dispatch chains")
    duplicate_workflows = _count(anti_stats, "duplicate workflow shapes")
    long_functions = _count(anti_stats, "long functions")
    deep_nesting = _count(anti_stats, "deep nesting")
    magic_numbers = _count(anti_stats, "magic numbers")
    hidden_side_effects = _count(anti_stats, "hidden side effects")
    broad_exceptions = _count(anti_stats, "broad exception swallowing")
    layer_violations = _count(anti_stats, "architecture layer violations")
    mixed_concerns = _count(anti_stats, "mixed concerns modules")

    candidates = [
        _candidate(
            pattern="strategy or policy interface",
            fit_score=(
                dispatch_count * 28.0
                + duplicate_workflows * 18.0
                + fan_out * 1.6
                + centrality_score * 0.22
                + leverage_score * 0.08
                - _good_penalty(good_stats, "strategy or policy interface", 26.0)
                - _good_penalty(good_stats, "registry pattern", 8.0)
            )
            if (dispatch_count > 0 or duplicate_workflows >= 2)
            else 0.0,
            adoption_risk_score=change_risk,
            leverage_score=leverage_score,
            metric_drivers={
                "dispatch_chain_count": dispatch_count,
                "duplicate_workflow_shapes": duplicate_workflows,
                "dependency_fan_out": fan_out,
                "dependency_centrality_score": centrality_score,
            },
            confidence_base=0.62,
        ),
        _candidate(
            pattern="registry pattern",
            fit_score=(
                dispatch_count * 24.0
                + duplicate_workflows * 16.0
                + fan_in * 1.5
                + centrality_score * 0.18
                + blast_score * 0.1
                - _good_penalty(good_stats, "registry pattern", 28.0)
            )
            if (dispatch_count > 0 or duplicate_workflows >= 2)
            else 0.0,
            adoption_risk_score=change_risk,
            leverage_score=leverage_score,
            metric_drivers={
                "dispatch_chain_count": dispatch_count,
                "duplicate_workflow_shapes": duplicate_workflows,
                "dependency_fan_in": fan_in,
                "blast_radius_score": blast_score,
            },
            confidence_base=0.6,
        ),
        _candidate(
            pattern="reusable pipeline stages",
            fit_score=(
                duplicate_workflows * 24.0
                + long_functions * 10.0
                + deep_nesting * 8.0
                + duplicate_clusters * 7.0
                + mixed_concerns * 16.0
                + mixing_score * 0.55
                + (8.0 if bool(blast_row.get("critical_execution_path")) else 0.0)
                - _good_penalty(good_stats, "reusable pipeline stages", 26.0)
            )
            if (duplicate_workflows > 0 or long_functions >= 2 or deep_nesting >= 2 or mixed_concerns > 0)
            else 0.0,
            adoption_risk_score=change_risk,
            leverage_score=leverage_score,
            metric_drivers={
                "duplicate_workflow_shapes": duplicate_workflows,
                "long_function_count": long_functions,
                "deep_nesting_count": deep_nesting,
                "duplicate_code_clusters": duplicate_clusters,
                "mixing_score": mixing_score,
            },
            confidence_base=0.65,
        ),
        _candidate(
            pattern="explicit boundary objects",
            fit_score=(
                hidden_side_effects * 24.0
                + broad_exceptions * 10.0
                + mixed_concerns * 12.0
                + family_count * 8.0
                + fan_out * 1.1
                + max(0.0, 100.0 - change_safety) * 0.18
                + blast_score * 0.1
                - _good_penalty(good_stats, "explicit boundary objects", 28.0)
            )
            if (hidden_side_effects > 0 or broad_exceptions > 0 or mixed_concerns > 0 or family_count >= 3)
            else 0.0,
            adoption_risk_score=change_risk,
            leverage_score=leverage_score,
            metric_drivers={
                "hidden_side_effect_count": hidden_side_effects,
                "broad_exception_count": broad_exceptions,
                "concern_family_count": family_count,
                "dependency_fan_out": fan_out,
                "change_safety_proxy_score": change_safety,
            },
            confidence_base=0.66,
        ),
        _candidate(
            pattern="artifact return boundary",
            fit_score=(
                hidden_side_effects * 20.0
                + broad_exceptions * 8.0
                + blast_score * 0.18
                + (12.0 if bool(blast_row.get("critical_execution_path")) else 0.0)
                + max(0, 2 - artifact_usage) * 10.0
                + fan_out * 0.8
                - _good_penalty(good_stats, "artifact return boundary", 24.0)
            )
            if (hidden_side_effects > 0 or broad_exceptions > 0 or bool(blast_row.get("critical_execution_path")))
            else 0.0,
            adoption_risk_score=change_risk,
            leverage_score=leverage_score,
            metric_drivers={
                "hidden_side_effect_count": hidden_side_effects,
                "broad_exception_count": broad_exceptions,
                "artifact_boundary_usage": artifact_usage,
                "blast_radius_score": blast_score,
                "critical_execution_path": 1 if bool(blast_row.get("critical_execution_path")) else 0,
            },
            confidence_base=0.62,
        ),
        _candidate(
            pattern="typed public APIs",
            fit_score=(
                (1.0 - type_hint_coverage) * 72.0
                + fan_in * 1.8
                + centrality_score * 0.18
                + blast_score * 0.1
                + architecture_violations * 6.0
                - _good_penalty(good_stats, "typed public APIs", 28.0)
            )
            if type_hint_coverage < 0.92
            else 0.0,
            adoption_risk_score=change_risk,
            leverage_score=leverage_score,
            metric_drivers={
                "type_hint_coverage": type_hint_coverage,
                "dependency_fan_in": fan_in,
                "dependency_centrality_score": centrality_score,
                "architecture_rule_violations": architecture_violations,
            },
            confidence_base=0.7,
        ),
        _candidate(
            pattern="dataclass config pattern",
            fit_score=(
                magic_numbers * 13.0
                + duplicate_workflows * 10.0
                + long_functions * 5.0
                + max(0, 2 - config_usage) * 14.0
                + fan_out * 0.9
                + (6.0 if family_count >= 2 else 0.0)
                - _good_penalty(good_stats, "dataclass config pattern", 26.0)
            )
            if (magic_numbers > 0 or duplicate_workflows > 0)
            else 0.0,
            adoption_risk_score=change_risk,
            leverage_score=leverage_score,
            metric_drivers={
                "magic_number_count": magic_numbers,
                "duplicate_workflow_shapes": duplicate_workflows,
                "config_object_usage": config_usage,
                "dependency_fan_out": fan_out,
            },
            confidence_base=0.58,
        ),
        _candidate(
            pattern="single source of truth constants or schema",
            fit_score=(
                magic_numbers * 15.0
                + duplicate_clusters * 8.0
                + fan_in * 1.2
                + blast_score * 0.08
                + duplicate_workflows * 8.0
                - _good_penalty(good_stats, "single source of truth constants or schema", 28.0)
            )
            if (magic_numbers > 0 or duplicate_clusters > 0)
            else 0.0,
            adoption_risk_score=change_risk,
            leverage_score=leverage_score,
            metric_drivers={
                "magic_number_count": magic_numbers,
                "duplicate_code_clusters": duplicate_clusters,
                "dependency_fan_in": fan_in,
                "blast_radius_score": blast_score,
            },
            confidence_base=0.63,
        ),
        _candidate(
            pattern="stable base class or protocol",
            fit_score=(
                fan_in * 2.4
                + centrality_score * 0.24
                + blast_score * 0.1
                + class_count * 5.0
                + layer_violations * 8.0
                + architecture_violations * 5.0
                + (6.0 if dominant_share < 0.5 else 0.0)
                - interface_reuse * 14.0
                - _good_penalty(good_stats, "stable base class or protocol", 20.0)
            )
            if (class_count >= 2 or fan_in >= 8 or layer_violations > 0)
            else 0.0,
            adoption_risk_score=change_risk,
            leverage_score=leverage_score,
            metric_drivers={
                "dependency_fan_in": fan_in,
                "dependency_centrality_score": centrality_score,
                "class_count": class_count,
                "architecture_rule_violations": architecture_violations,
                "concern_family_count": family_count,
            },
            confidence_base=0.57,
        ),
    ]
    return _finalize_candidates(candidates)


def _symbol_candidates(
    *,
    symbol_row: dict[str, Any],
    module_health_row: dict[str, Any],
    anti_stats: dict[str, Any],
    good_stats: dict[str, Any],
) -> list[dict[str, Any]]:
    type_hint_coverage = float(module_health_row.get("type_hint_coverage") or 0.0)
    fan_in = int(module_health_row.get("dependency_fan_in") or 0)
    blast_score = float(symbol_row.get("blast_radius_score") or 0.0)
    change_risk = float(symbol_row.get("change_risk_score") or 0.0)
    direct_dependents = int(symbol_row.get("direct_dependents") or 0)
    centrality_score = float(symbol_row.get("centrality_score") or 0.0)

    dispatch_metric = _metric(anti_stats, "repeated if/elif dispatch chains", default=float(_count(anti_stats, "repeated if/elif dispatch chains") * 3))
    duplicate_workflows = _count(anti_stats, "duplicate workflow shapes")
    long_metric = _metric(anti_stats, "long functions", default=float(_count(anti_stats, "long functions") * 20))
    deep_metric = _metric(anti_stats, "deep nesting", default=float(_count(anti_stats, "deep nesting") * 3))
    hidden_side_effects = _count(anti_stats, "hidden side effects")
    broad_exceptions = _count(anti_stats, "broad exception swallowing")
    loop_append = _count(anti_stats, "loop append simple transform")
    magic_metric = _metric(anti_stats, "magic numbers", default=float(_count(anti_stats, "magic numbers") * 3))

    candidates = [
        _candidate(
            pattern="strategy or policy interface",
            fit_score=(
                dispatch_metric * 18.0
                + duplicate_workflows * 14.0
                + blast_score * 0.18
                + centrality_score * 0.14
                - _good_penalty(good_stats, "strategy or policy interface", 28.0)
            )
            if (dispatch_metric > 0 or duplicate_workflows > 0)
            else 0.0,
            adoption_risk_score=change_risk,
            leverage_score=blast_score,
            metric_drivers={
                "dispatch_chain_branches": dispatch_metric,
                "duplicate_workflow_shapes": duplicate_workflows,
                "blast_radius_score": blast_score,
                "centrality_score": centrality_score,
            },
            confidence_base=0.64,
        ),
        _candidate(
            pattern="registry pattern",
            fit_score=(
                dispatch_metric * 15.0
                + duplicate_workflows * 14.0
                + direct_dependents * 4.0
                + blast_score * 0.15
                - _good_penalty(good_stats, "registry pattern", 28.0)
            )
            if (dispatch_metric > 0 or duplicate_workflows > 0)
            else 0.0,
            adoption_risk_score=change_risk,
            leverage_score=blast_score,
            metric_drivers={
                "dispatch_chain_branches": dispatch_metric,
                "duplicate_workflow_shapes": duplicate_workflows,
                "direct_dependents": direct_dependents,
                "blast_radius_score": blast_score,
            },
            confidence_base=0.6,
        ),
        _candidate(
            pattern="guard clause style",
            fit_score=(
                deep_metric * 12.0
                + long_metric * 0.8
                + loop_append * 14.0
                + blast_score * 0.12
                - _good_penalty(good_stats, "guard clause style", 28.0)
            )
            if (deep_metric >= 3.0 or long_metric >= 55.0 or loop_append > 0)
            else 0.0,
            adoption_risk_score=change_risk,
            leverage_score=blast_score,
            metric_drivers={
                "nesting_depth": deep_metric,
                "function_loc": long_metric,
                "loop_append_transform_count": loop_append,
                "blast_radius_score": blast_score,
            },
            confidence_base=0.7,
        ),
        _candidate(
            pattern="reusable pipeline stages",
            fit_score=(
                duplicate_workflows * 18.0
                + long_metric * 0.7
                + loop_append * 18.0
                + (10.0 if bool(symbol_row.get("critical_execution_path")) else 0.0)
                + blast_score * 0.1
                - _good_penalty(good_stats, "reusable pipeline stages", 24.0)
            )
            if (duplicate_workflows > 0 or loop_append > 0 or long_metric >= 80.0)
            else 0.0,
            adoption_risk_score=change_risk,
            leverage_score=blast_score,
            metric_drivers={
                "duplicate_workflow_shapes": duplicate_workflows,
                "function_loc": long_metric,
                "loop_append_transform_count": loop_append,
                "critical_execution_path": 1 if bool(symbol_row.get("critical_execution_path")) else 0,
            },
            confidence_base=0.66,
        ),
        _candidate(
            pattern="explicit boundary objects",
            fit_score=(
                hidden_side_effects * 24.0
                + broad_exceptions * 10.0
                + blast_score * 0.14
                + (6.0 if bool(symbol_row.get("critical_execution_path")) else 0.0)
                - _good_penalty(good_stats, "explicit boundary objects", 28.0)
            )
            if (hidden_side_effects > 0 or broad_exceptions > 0)
            else 0.0,
            adoption_risk_score=change_risk,
            leverage_score=blast_score,
            metric_drivers={
                "hidden_side_effect_count": hidden_side_effects,
                "broad_exception_count": broad_exceptions,
                "blast_radius_score": blast_score,
                "critical_execution_path": 1 if bool(symbol_row.get("critical_execution_path")) else 0,
            },
            confidence_base=0.64,
        ),
        _candidate(
            pattern="artifact return boundary",
            fit_score=(
                hidden_side_effects * 18.0
                + broad_exceptions * 8.0
                + blast_score * 0.12
                + (8.0 if bool(symbol_row.get("critical_execution_path")) else 0.0)
                - _good_penalty(good_stats, "artifact return boundary", 24.0)
            )
            if (hidden_side_effects > 0 or broad_exceptions > 0 or bool(symbol_row.get("critical_execution_path")))
            else 0.0,
            adoption_risk_score=change_risk,
            leverage_score=blast_score,
            metric_drivers={
                "hidden_side_effect_count": hidden_side_effects,
                "broad_exception_count": broad_exceptions,
                "blast_radius_score": blast_score,
                "critical_execution_path": 1 if bool(symbol_row.get("critical_execution_path")) else 0,
            },
            confidence_base=0.58,
        ),
        _candidate(
            pattern="typed public APIs",
            fit_score=(
                (1.0 - type_hint_coverage) * 68.0
                + direct_dependents * 5.0
                + fan_in * 1.2
                + blast_score * 0.15
                - _good_penalty(good_stats, "typed public APIs", 26.0)
            )
            if type_hint_coverage < 0.9 and (direct_dependents >= 2 or blast_score >= 15.0)
            else 0.0,
            adoption_risk_score=change_risk,
            leverage_score=blast_score,
            metric_drivers={
                "type_hint_coverage": type_hint_coverage,
                "direct_dependents": direct_dependents,
                "module_dependency_fan_in": fan_in,
                "blast_radius_score": blast_score,
            },
            confidence_base=0.68,
        ),
        _candidate(
            pattern="single source of truth constants or schema",
            fit_score=(
                magic_metric * 6.0
                + blast_score * 0.08
                + direct_dependents * 2.0
                - _good_penalty(good_stats, "single source of truth constants or schema", 24.0)
            )
            if magic_metric > 0
            else 0.0,
            adoption_risk_score=change_risk,
            leverage_score=blast_score,
            metric_drivers={
                "magic_number_metric": magic_metric,
                "direct_dependents": direct_dependents,
                "blast_radius_score": blast_score,
            },
            confidence_base=0.56,
        ),
    ]
    return _finalize_candidates(candidates)


def _finalize_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in candidates if float(row.get("fit_score") or 0.0) >= 35.0]
    rows.sort(
        key=lambda row: (
            -float(row.get("fit_score") or 0.0),
            -float(row.get("safe_adoption_score") or 0.0),
            str(row.get("pattern") or ""),
        )
    )
    return rows[:3]


def _candidate(
    *,
    pattern: str,
    fit_score: float,
    adoption_risk_score: float,
    leverage_score: float,
    metric_drivers: dict[str, Any],
    confidence_base: float,
) -> dict[str, Any]:
    fit = round(_clamp(fit_score), 2)
    safe_adoption_score = round(_clamp((fit * 0.58) + (leverage_score * 0.27) + ((100.0 - adoption_risk_score) * 0.15)), 2)
    active_driver_count = sum(1 for value in metric_drivers.values() if _driver_active(value))
    confidence = round(min(0.97, confidence_base + active_driver_count * 0.06), 2)
    metadata = PATTERN_METADATA[pattern]
    return {
        "pattern": pattern,
        "fit_score": fit,
        "safe_adoption_score": safe_adoption_score,
        "confidence": confidence,
        "why": metadata["why"],
        "expected_benefit": metadata["expected_benefit"],
        "refactor_action": metadata["refactor_action"],
        "metric_drivers": _clean_metric_drivers(metric_drivers),
        "adoption_risk_level": _risk_level(adoption_risk_score),
    }


def _group_pattern_stats(rows: list[dict[str, Any]], key_field: str, *, kind: str) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get(key_field) or "")
        pattern = str(row.get("pattern") or "")
        if not key or not pattern:
            continue
        entry = stats.setdefault(key, _empty_pattern_stats())
        entry["counts"][pattern] = int(entry["counts"].get(pattern, 0)) + 1
        if kind == "bad":
            metric_value = row.get("metric_value")
            if isinstance(metric_value, (int, float)):
                entry["metrics"][pattern] = max(float(entry["metrics"].get(pattern, 0.0)), float(metric_value))
        if kind == "good":
            strength = row.get("strength")
            if isinstance(strength, (int, float)):
                entry["strengths"][pattern] = max(float(entry["strengths"].get(pattern, 0.0)), float(strength))
    return stats


def _empty_pattern_stats() -> dict[str, Any]:
    return {"counts": {}, "metrics": {}, "strengths": {}}


def _count(stats: dict[str, Any], pattern: str) -> int:
    return int((stats.get("counts") or {}).get(pattern, 0))


def _metric(stats: dict[str, Any], pattern: str, *, default: float = 0.0) -> float:
    return float((stats.get("metrics") or {}).get(pattern, default) or default)


def _good_penalty(stats: dict[str, Any], pattern: str, base: float) -> float:
    count = _count(stats, pattern)
    strength = float((stats.get("strengths") or {}).get(pattern, 0.0))
    return min(base, (count * (base * 0.55)) + (strength * (base * 0.45)))


def _clean_metric_drivers(metric_drivers: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in metric_drivers.items():
        if not _driver_active(value):
            continue
        if isinstance(value, float):
            cleaned[key] = round(value, 2)
        else:
            cleaned[key] = value
    return cleaned


def _driver_active(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    return bool(value)


def _clamp(value: float, *, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def _risk_level(score: float) -> str:
    if score >= 75.0:
        return "high"
    if score >= 45.0:
        return "medium"
    return "low"


def _is_test_name(name: str) -> bool:
    value = str(name or "")
    return value == "tests" or value.startswith("tests.") or ".tests." in value or value.endswith(".tests")
