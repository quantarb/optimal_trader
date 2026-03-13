from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .architecture_rules import DEFAULT_RULES_PATH, load_architecture_rules


DEFAULT_SCORE_WEIGHTS = {
    "complexity_health": 0.16,
    "dependency_health": 0.12,
    "duplication_health": 0.1,
    "typing_health": 0.1,
    "architecture_health": 0.16,
    "good_pattern_strength": 0.12,
    "anti_pattern_burden": 0.12,
    "llm_editability": 0.06,
    "change_safety": 0.06,
}


@dataclass
class QualityScorecardReport:
    weights: dict[str, float]
    repo_score: float
    repo_dimensions: dict[str, float]
    module_scores: list[dict[str, Any]]
    file_scores: list[dict[str, Any]]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "repo_score": self.repo_score,
            "repo_dimensions": dict(self.repo_dimensions),
            "module_scores": list(self.module_scores),
            "file_scores": list(self.file_scores),
            "notes": list(self.notes),
        }


def load_score_weights(*, weights_path: Path | None = None, rules_path: Path | None = None) -> dict[str, float]:
    if weights_path is not None:
        payload = json.loads(Path(weights_path).read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "quality_weights" in payload:
            payload = payload["quality_weights"]
        return _normalize_weights(payload)
    try:
        rules = load_architecture_rules(rules_path or DEFAULT_RULES_PATH)
    except Exception:
        return dict(DEFAULT_SCORE_WEIGHTS)
    payload = dict(rules.get("quality_weights") or {})
    if not payload:
        return dict(DEFAULT_SCORE_WEIGHTS)
    return _normalize_weights(payload)


def build_quality_scorecard(
    *,
    metrics_report: dict[str, Any],
    anti_pattern_report: dict[str, Any],
    good_pattern_report: dict[str, Any],
    architecture_report: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> QualityScorecardReport:
    weights = _normalize_weights(weights or DEFAULT_SCORE_WEIGHTS)
    module_rows = list(metrics_report.get("module_rows") or [])
    path_to_module = {str(row.get("path") or ""): str(row.get("module") or "") for row in module_rows}
    good_by_path = _group_good_by_path(list(good_pattern_report.get("findings") or []))
    anti_by_path = _group_anti_by_path(list(anti_pattern_report.get("findings") or []))
    architecture_counts = _source_counts(list(architecture_report.get("violations") or []))

    module_scores: list[dict[str, Any]] = []
    for row in module_rows:
        path = str(row.get("path") or "")
        module = str(row.get("module") or "")
        good_rows = good_by_path.get(path, [])
        anti_rows = anti_by_path.get(path, [])
        dimensions = {
            "complexity_health": _complexity_health(row),
            "dependency_health": _dependency_health(row),
            "duplication_health": _duplication_health(row),
            "typing_health": round(float(row.get("type_hint_coverage") or 0.0) * 100.0, 2),
            "architecture_health": _architecture_health(row, architecture_counts.get(module, 0)),
            "good_pattern_strength": _good_pattern_strength(good_rows),
            "anti_pattern_burden": _anti_pattern_burden(anti_rows),
            "llm_editability": round(float(row.get("llm_editability_proxy_score") or 0.0), 2),
            "change_safety": round(float(row.get("change_safety_proxy_score") or 0.0), 2),
        }
        module_scores.append(
            {
                "module": module,
                "path": path,
                "score": _weighted_score(dimensions, weights),
                "dimensions": dimensions,
                "line_count": int(row.get("line_count") or 0),
            }
        )

    module_scores.sort(key=lambda row: (float(row["score"]), row["module"]))
    repo_dimensions = _repo_dimensions(module_scores)
    repo_score = _weighted_score(repo_dimensions, weights)
    file_scores = [{**row, "file": row["path"]} for row in module_scores]
    notes = [
        "The scorecard is a weighted summary on top of the raw metrics; use it to compare snapshots and modules, not as a substitute for the underlying findings.",
    ]
    return QualityScorecardReport(
        weights=weights,
        repo_score=repo_score,
        repo_dimensions=repo_dimensions,
        module_scores=module_scores,
        file_scores=file_scores,
        notes=notes,
    )


def quality_scorecard_markdown(report: QualityScorecardReport) -> str:
    sections = [
        "# Quality Scorecard",
        "",
        f"- repo score: {report.repo_score:.2f}",
        "",
        "## Repo Dimensions",
    ]
    sections.extend(f"- `{name}`: {value:.2f}" for name, value in report.repo_dimensions.items())
    sections.extend(["", "## Lowest Scoring Modules"])
    sections.extend(
        f"- `{row['module']}`: score={row['score']:.2f}, complexity={row['dimensions']['complexity_health']:.2f}, architecture={row['dimensions']['architecture_health']:.2f}, editability={row['dimensions']['llm_editability']:.2f}"
        for row in report.module_scores[:30]
    )
    sections.extend(["", "## Weights"])
    sections.extend(f"- `{name}`: {value:.3f}" for name, value in report.weights.items())
    if report.notes:
        sections.extend(["", "## Notes", *[f"- {note}" for note in report.notes]])
    return "\n".join(sections)


def _complexity_health(row: dict[str, Any]) -> float:
    avg_complexity = float(row.get("cyclomatic_complexity_avg") or 0.0)
    max_complexity = float(row.get("cyclomatic_complexity_max") or 0.0)
    max_nesting = float(row.get("nesting_depth_max") or 0.0)
    score = 100.0 - (avg_complexity * 4.0) - max(0.0, max_complexity - 15.0) * 1.8 - max_nesting * 4.0
    return round(_clamp(score), 2)


def _dependency_health(row: dict[str, Any]) -> float:
    fan_in = float(row.get("dependency_fan_in") or 0.0)
    fan_out = float(row.get("dependency_fan_out") or 0.0)
    cycles = float(row.get("import_cycle_count") or 0.0)
    architecture = float(row.get("architecture_rule_violations") or 0.0)
    score = 100.0 - max(0.0, fan_out - 6.0) * 4.0 - max(0.0, fan_in - 18.0) * 1.5 - cycles * 20.0 - architecture * 4.0
    return round(_clamp(score), 2)


def _duplication_health(row: dict[str, Any]) -> float:
    duplicates = float(row.get("duplicate_code_clusters") or 0.0)
    score = 100.0 - duplicates * 12.0
    return round(_clamp(score), 2)


def _architecture_health(row: dict[str, Any], violation_count: int) -> float:
    cycles = float(row.get("import_cycle_count") or 0.0)
    score = 100.0 - float(violation_count) * 18.0 - cycles * 18.0
    return round(_clamp(score), 2)


def _good_pattern_strength(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    avg_strength = sum(float(row.get("strength") or 0.0) for row in rows) / len(rows)
    return round(_clamp(avg_strength * 100.0 + min(len(rows) * 2.0, 12.0)), 2)


def _anti_pattern_burden(rows: list[dict[str, Any]]) -> float:
    penalty = 0.0
    for row in rows:
        severity = str(row.get("severity") or "low").lower()
        penalty += {"high": 15.0, "medium": 7.0, "low": 3.0}.get(severity, 4.0)
    return round(_clamp(100.0 - penalty), 2)


def _weighted_score(dimensions: dict[str, float], weights: dict[str, float]) -> float:
    total = sum(weights.values()) or 1.0
    score = sum(float(dimensions.get(name) or 0.0) * float(weight) for name, weight in weights.items()) / total
    return round(score, 2)


def _repo_dimensions(module_scores: list[dict[str, Any]]) -> dict[str, float]:
    if not module_scores:
        return {name: 0.0 for name in DEFAULT_SCORE_WEIGHTS}
    total_lines = sum(max(1, int(row.get("line_count") or 1)) for row in module_scores)
    dimensions: dict[str, float] = {}
    for name in DEFAULT_SCORE_WEIGHTS:
        weighted = sum(
            float(row["dimensions"].get(name) or 0.0) * max(1, int(row.get("line_count") or 1))
            for row in module_scores
        )
        dimensions[name] = round(weighted / total_lines, 2)
    return dimensions


def _group_good_by_path(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    payload: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        payload.setdefault(str(row.get("file") or ""), []).append(row)
    return payload


def _group_anti_by_path(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    payload: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        payload.setdefault(str(row.get("file") or ""), []).append(row)
    return payload


def _source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("source_module") or "")
        counts[source] = counts.get(source, 0) + 1
    return counts


def _normalize_weights(rows: Any) -> dict[str, float]:
    payload = dict(DEFAULT_SCORE_WEIGHTS)
    if isinstance(rows, dict):
        for key, value in rows.items():
            if key in payload:
                try:
                    payload[key] = float(value)
                except (TypeError, ValueError):
                    continue
    total = sum(payload.values()) or 1.0
    return {key: round(value / total, 6) for key, value in payload.items()}


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))
