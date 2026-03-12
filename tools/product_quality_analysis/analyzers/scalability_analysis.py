from __future__ import annotations

from collections import defaultdict

from ..models import PageSnapshot, RankedIssue, Severity


def analyze_scalability(page_snapshots: list[PageSnapshot]) -> dict:
    grouped: dict[str, dict[str, PageSnapshot]] = defaultdict(dict)
    for snapshot in page_snapshots:
        if snapshot.tier:
            base_name = snapshot.name
            for tier in ("tier1", "tier2", "tier3"):
                base_name = base_name.replace(f"_{tier}", "")
            grouped[base_name][str(snapshot.tier)] = snapshot
    rows: list[dict] = []
    issues: list[RankedIssue] = []
    for route, tier_map in grouped.items():
        tier1 = tier_map.get("tier1")
        tier2 = tier_map.get("tier2")
        tier3 = tier_map.get("tier3")
        if tier1 is None or tier2 is None:
            continue
        load_growth = round((tier2.load_time_ms or 0.0) / max(1.0, tier1.load_time_ms or 1.0), 4)
        dom_growth = round(tier2.dom_node_count / max(1, tier1.dom_node_count or 1), 4)
        usability_score = 100.0
        for snapshot in (tier1, tier2, tier3):
            if snapshot is None:
                continue
            if snapshot.response_error:
                usability_score -= 35.0
            usability_score -= min(40.0, (snapshot.load_time_ms or 0.0) / 1000.0)
            usability_score -= min(20.0, snapshot.dom_node_count / 250.0)
        rows.append(
            {
                "route": route,
                "tier1_load_time_ms": tier1.load_time_ms,
                "tier2_load_time_ms": tier2.load_time_ms,
                "tier3_load_time_ms": tier3.load_time_ms if tier3 else None,
                "load_growth_rate": load_growth,
                "tier1_dom_nodes": tier1.dom_node_count,
                "tier2_dom_nodes": tier2.dom_node_count,
                "tier3_dom_nodes": tier3.dom_node_count if tier3 else None,
                "dom_growth_rate": dom_growth,
                "ui_usability_score_by_tier": round(max(0.0, usability_score), 2),
            }
        )
        if tier2.response_error or (tier3 and tier3.response_error):
            issues.append(
                RankedIssue(
                    issue_id=f"scalability-timeout:{route}",
                    title=f"{route} stops being usable at larger symbol tiers",
                    severity=Severity.CRITICAL,
                    score=0.0,
                    page=route,
                    category="scalability",
                    recommendation="Cap the expensive candidate set and avoid recomputing analog search for every symbol in the full tier.",
                    evidence=[f"tier2 load ms: {tier2.load_time_ms}", f"tier3 load ms: {tier3.load_time_ms if tier3 else 'n/a'}", f"tier2 error: {tier2.response_error or 'none'}", f"tier3 error: {tier3.response_error if tier3 else 'n/a'}"],
                    metric_name="ui_usability_score_by_tier",
                    metric_value=round(max(0.0, usability_score), 2),
                    metadata={"trust_impact": 5, "frequency": 4, "scalability_risk": 5, "usability_impact": 5, "implementation_feasibility": 4},
                )
            )
    return {"tiers": rows, "issues": [issue.model_dump(mode="json") for issue in issues]}
