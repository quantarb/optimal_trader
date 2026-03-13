from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class QualitySnapshot:
    label: str
    root: str
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    repo_score: float = 0.0
    repo_dimensions: dict[str, float] = field(default_factory=dict)
    repo_metrics: dict[str, Any] = field(default_factory=dict)
    module_scores: list[dict[str, Any]] = field(default_factory=list)
    file_scores: list[dict[str, Any]] = field(default_factory=list)
    anti_pattern_summary: dict[str, Any] = field(default_factory=dict)
    good_pattern_summary: dict[str, Any] = field(default_factory=dict)
    architecture_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "root": self.root,
            "generated_at": self.generated_at,
            "repo_score": round(float(self.repo_score), 2),
            "repo_dimensions": dict(self.repo_dimensions),
            "repo_metrics": dict(self.repo_metrics),
            "module_scores": list(self.module_scores),
            "file_scores": list(self.file_scores),
            "anti_pattern_summary": dict(self.anti_pattern_summary),
            "good_pattern_summary": dict(self.good_pattern_summary),
            "architecture_summary": dict(self.architecture_summary),
        }


@dataclass
class QualityComparison:
    baseline_label: str
    current_label: str
    overall_score_delta: float
    improved: list[dict[str, Any]]
    regressed: list[dict[str, Any]]
    unchanged: list[dict[str, Any]]
    module_deltas: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_label": self.baseline_label,
            "current_label": self.current_label,
            "overall_score_delta": round(float(self.overall_score_delta), 2),
            "improved": list(self.improved),
            "regressed": list(self.regressed),
            "unchanged": list(self.unchanged),
            "module_deltas": list(self.module_deltas),
        }


NUMERIC_DIRECTIONS = {
    "repo_score": "higher",
    "architecture_rule_violations": "lower",
    "anti_pattern_burden": "lower",
    "artifact_boundary_usage": "higher",
    "class_size_distribution.avg": "lower",
    "class_size_distribution.max": "lower",
    "class_size_distribution.p50": "lower",
    "class_size_distribution.p90": "lower",
    "change_safety_proxy_score": "higher",
    "cyclomatic_complexity_summary.avg": "lower",
    "cyclomatic_complexity_summary.max": "lower",
    "cyclomatic_complexity_summary.p90": "lower",
    "complexity_health": "higher",
    "dead_code_count": "lower",
    "dependency_health": "higher",
    "dependency_fan_in.avg": "lower",
    "dependency_fan_in.max": "lower",
    "dependency_fan_in.p50": "lower",
    "dependency_fan_in.p90": "lower",
    "dependency_fan_out.avg": "lower",
    "dependency_fan_out.max": "lower",
    "dependency_fan_out.p50": "lower",
    "dependency_fan_out.p90": "lower",
    "duplicate_code_clusters": "lower",
    "duplication_health": "higher",
    "function_loc_distribution.avg": "lower",
    "function_loc_distribution.max": "lower",
    "function_loc_distribution.p50": "lower",
    "function_loc_distribution.p90": "lower",
    "good_pattern_strength": "higher",
    "import_cycle_count": "lower",
    "llm_editability": "higher",
    "llm_editability_proxy_score": "higher",
    "magic_number_count": "lower",
    "maintainability_summary.avg": "higher",
    "maintainability_summary.min": "higher",
    "maintainability_summary.p10": "higher",
    "nesting_depth_distribution.avg": "lower",
    "nesting_depth_distribution.max": "lower",
    "nesting_depth_distribution.p50": "lower",
    "nesting_depth_distribution.p90": "lower",
    "typing_health": "higher",
    "type_hint_coverage": "higher",
    "change_safety": "higher",
}


def snapshot_from_reports(
    *,
    label: str,
    root: str,
    metrics_report: dict[str, Any],
    scorecard_report: dict[str, Any],
    anti_pattern_report: dict[str, Any],
    good_pattern_report: dict[str, Any],
    architecture_report: dict[str, Any],
) -> QualitySnapshot:
    return QualitySnapshot(
        label=label,
        root=root,
        repo_score=float(scorecard_report.get("repo_score") or 0.0),
        repo_dimensions={key: float(value) for key, value in dict(scorecard_report.get("repo_dimensions") or {}).items()},
        repo_metrics=dict(metrics_report.get("repo_summary") or {}),
        module_scores=list(scorecard_report.get("module_scores") or []),
        file_scores=list(scorecard_report.get("file_scores") or []),
        anti_pattern_summary=dict(anti_pattern_report.get("summary") or {}),
        good_pattern_summary=dict(good_pattern_report.get("summary") or {}),
        architecture_summary=dict(architecture_report.get("summary") or {}),
    )


def snapshot_from_payload(payload: dict[str, Any] | None) -> QualitySnapshot | None:
    if not payload:
        return None
    return QualitySnapshot(
        label=str(payload.get("label") or ""),
        root=str(payload.get("root") or ""),
        generated_at=str(payload.get("generated_at") or ""),
        repo_score=float(payload.get("repo_score") or 0.0),
        repo_dimensions={key: float(value) for key, value in dict(payload.get("repo_dimensions") or {}).items()},
        repo_metrics=dict(payload.get("repo_metrics") or {}),
        module_scores=list(payload.get("module_scores") or []),
        file_scores=list(payload.get("file_scores") or []),
        anti_pattern_summary=dict(payload.get("anti_pattern_summary") or {}),
        good_pattern_summary=dict(payload.get("good_pattern_summary") or {}),
        architecture_summary=dict(payload.get("architecture_summary") or {}),
    )


def compare_quality_snapshots(baseline: QualitySnapshot, current: QualitySnapshot) -> QualityComparison:
    metrics = _comparison_metrics(baseline, current)
    improved: list[dict[str, Any]] = []
    regressed: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []
    for metric in metrics:
        if metric["status"] == "improved":
            improved.append(metric)
        elif metric["status"] == "regressed":
            regressed.append(metric)
        else:
            unchanged.append(metric)
    module_deltas = _module_score_deltas(baseline.module_scores, current.module_scores)
    return QualityComparison(
        baseline_label=baseline.label,
        current_label=current.label,
        overall_score_delta=round(float(current.repo_score) - float(baseline.repo_score), 2),
        improved=sorted(improved, key=lambda row: (-abs(float(row["delta"])), row["metric"])),
        regressed=sorted(regressed, key=lambda row: (-abs(float(row["delta"])), row["metric"])),
        unchanged=sorted(unchanged, key=lambda row: row["metric"]),
        module_deltas=module_deltas,
    )


def quality_snapshot_markdown(snapshot: QualitySnapshot) -> str:
    sections = [
        f"# Quality Snapshot: {snapshot.label}",
        "",
        f"- generated at: `{snapshot.generated_at}`",
        f"- root: `{snapshot.root}`",
        f"- repo score: {snapshot.repo_score:.2f}",
        f"- architecture violations: {snapshot.architecture_summary.get('violation_count', 0)}",
        f"- anti-pattern findings: {snapshot.anti_pattern_summary.get('finding_count', 0)}",
        f"- good-pattern findings: {snapshot.good_pattern_summary.get('finding_count', 0)}",
        "",
        "## Repo Dimensions",
    ]
    sections.extend(f"- `{name}`: {value:.2f}" for name, value in snapshot.repo_dimensions.items())
    sections.extend(["", "## Lowest Scoring Modules"])
    sections.extend(
        f"- `{row['module']}`: {float(row.get('score') or 0.0):.2f}"
        for row in list(snapshot.module_scores)[:20]
    )
    return "\n".join(sections)


def quality_comparison_markdown(comparison: QualityComparison) -> str:
    sections = [
        f"# Quality Comparison: {comparison.baseline_label} vs {comparison.current_label}",
        "",
        f"- overall score delta: {comparison.overall_score_delta:+.2f}",
        "",
        "## Improved",
    ]
    if comparison.improved:
        sections.extend(
            f"- `{row['metric']}`: {row['before']} -> {row['after']} ({row['delta']:+.2f})"
            for row in comparison.improved[:20]
        )
    else:
        sections.append("- none")
    sections.extend(["", "## Regressed"])
    if comparison.regressed:
        sections.extend(
            f"- `{row['metric']}`: {row['before']} -> {row['after']} ({row['delta']:+.2f})"
            for row in comparison.regressed[:20]
        )
    else:
        sections.append("- none")
    sections.extend(["", "## Unchanged"])
    if comparison.unchanged:
        sections.extend(f"- `{row['metric']}`: {row['after']}" for row in comparison.unchanged[:20])
    else:
        sections.append("- none")
    sections.extend(["", "## Module Score Deltas"])
    if comparison.module_deltas:
        sections.extend(
            f"- `{row['module']}`: {row['before_score']:.2f} -> {row['after_score']:.2f} ({row['delta']:+.2f})"
            for row in comparison.module_deltas[:20]
        )
    else:
        sections.append("- none")
    return "\n".join(sections)


def _comparison_metrics(baseline: QualitySnapshot, current: QualitySnapshot) -> list[dict[str, Any]]:
    rows = []
    baseline_metrics = {"repo_score": baseline.repo_score}
    baseline_metrics.update(baseline.repo_dimensions)
    baseline_metrics.update(_flatten_repo_metrics(baseline.repo_metrics))
    current_metrics = {"repo_score": current.repo_score}
    current_metrics.update(current.repo_dimensions)
    current_metrics.update(_flatten_repo_metrics(current.repo_metrics))
    metric_names = sorted(set(baseline_metrics) | set(current_metrics))
    for name in metric_names:
        before = baseline_metrics.get(name)
        after = current_metrics.get(name)
        if before is None or after is None:
            continue
        if name.endswith(".count"):
            continue
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            continue
        direction = NUMERIC_DIRECTIONS.get(name, "higher")
        delta = round(float(after) - float(before), 2)
        if abs(delta) < 0.01:
            status = "unchanged"
        elif (direction == "higher" and delta > 0) or (direction == "lower" and delta < 0):
            status = "improved"
        else:
            status = "regressed"
        rows.append(
            {
                "metric": name,
                "before": round(float(before), 2),
                "after": round(float(after), 2),
                "delta": delta,
                "preferred_direction": direction,
                "status": status,
            }
        )
    return rows


def _module_score_deltas(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    before_map = {str(row.get("module") or ""): row for row in before_rows}
    after_map = {str(row.get("module") or ""): row for row in after_rows}
    rows: list[dict[str, Any]] = []
    for module in sorted(set(before_map) & set(after_map)):
        before_score = float(before_map.get(module, {}).get("score") or 0.0)
        after_score = float(after_map.get(module, {}).get("score") or 0.0)
        rows.append(
            {
                "module": module,
                "before_score": before_score,
                "after_score": after_score,
                "delta": round(after_score - before_score, 2),
            }
        )
    rows.sort(key=lambda row: (row["delta"], row["module"]))
    return rows


def _flatten_repo_metrics(repo_metrics: dict[str, Any]) -> dict[str, float]:
    rows: dict[str, float] = {}
    for key, value in repo_metrics.items():
        if isinstance(value, (int, float)):
            rows[key] = float(value)
        elif isinstance(value, dict):
            for child_key, child_value in value.items():
                if isinstance(child_value, (int, float)):
                    rows[f"{key}.{child_key}"] = float(child_value)
    return rows
