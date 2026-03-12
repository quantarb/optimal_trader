from __future__ import annotations

from ..config import WORKLOAD_STAGE_PREFIXES


def workload_stage_names() -> dict[str, tuple[str, ...]]:
    return dict(WORKLOAD_STAGE_PREFIXES)


def workload_runtime_from_tier_report(report: dict) -> dict[str, float]:
    stages = list((report.get("performance") or {}).get("stages") or [])
    totals = {"end_to_end": float(report.get("total_runtime_seconds") or 0.0)}
    for workload, prefixes in workload_stage_names().items():
        totals[workload] = round(sum(float(stage.get("wall_seconds") or 0.0) for stage in stages if any(str(stage.get("name") or "").startswith(prefix) for prefix in prefixes)), 6)
    return totals
