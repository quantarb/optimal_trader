from __future__ import annotations

from django import template
from django.urls import NoReverseMatch, reverse

from pipeline.feature_presentation import (
    format_feature_value,
    get_feature_definition,
    render_feature,
    render_feature_family_name,
    render_feature_family_signature,
)


register = template.Library()


SIDEBAR_PAGE_GROUPS = [
    {
        "title": "Data",
        "links": [
            {"label": "Forms", "route": "form-tabs"},
            {
                "label": "Runs",
                "route": "pipeline-ui",
                "active_routes": [
                    "pipeline-ui",
                    "pipeline-run-list",
                    "pipeline-run-status",
                    "pipeline-job-catalog",
                ],
            },
            {
                "label": "Artifacts",
                "route": "pipeline-cohorts",
                "active_routes": [
                    "pipeline-cohorts",
                    "pipeline-artifact-detail",
                    "pipeline-artifact-preview",
                    "pipeline-artifact-symbol-breakdown",
                    "pipeline-artifact-list",
                    "pipeline-artifact-latest",
                ],
            },
            {"label": "Status", "route": "pipeline-status-board"},
        ],
    },
    {
        "title": "Models",
        "links": [
            {"label": "Experiments", "route": "pipeline-lab"},
            {
                "label": "Models",
                "route": "pipeline-strategies",
                "active_routes": ["pipeline-strategies", "pipeline-strategy-detail"],
            },
            {
                "label": "Definitions",
                "route": "pipeline-strategy-definitions",
                "active_routes": [
                    "pipeline-strategy-definitions",
                    "pipeline-strategy-definition-edit",
                ],
            },
        ],
    },
    {
        "title": "Reports",
        "links": [
            {
                "label": "Research",
                "route": "pipeline-research-reports",
                "active_routes": ["pipeline-research-reports", "pipeline-backtest-detail"],
            },
            {"label": "Oracle", "route": "pipeline-oracle-reports"},
            {"label": "Feature Attribution", "route": "pipeline-feature-attribution-reports"},
            {"label": "RL Policies", "route": "pipeline-rl-policy-reports"},
        ],
    },
    {
        "title": "Insights",
        "links": [
            {"label": "Market Situations", "route": "pipeline-market-situations"},
            {
                "label": "Opportunities",
                "route": "pipeline-opportunities",
                "active_routes": ["pipeline-opportunities", "pipeline-top-opportunities"],
            },
            {
                "label": "Stock Analysis",
                "route": "pipeline-stock-intelligence",
                "active_routes": [
                    "pipeline-stock-intelligence",
                    "pipeline-stock-intelligence-symbol",
                    "pipeline-symbol-research",
                ],
            },
            {"label": "Portfolio Analysis", "route": "pipeline-portfolio-analysis"},
        ],
    },
    {
        "title": "Trading",
        "links": [
            {
                "label": "Cockpit",
                "route": "trading-leaderboard",
                "active_routes": ["trading-leaderboard", "trading-similar-trades"],
            },
        ],
    },
]


@register.simple_tag(takes_context=True)
def sidebar_page_groups(context) -> list[dict[str, object]]:
    request = context.get("request")
    current_route = getattr(getattr(request, "resolver_match", None), "url_name", "") or ""
    groups: list[dict[str, object]] = []
    for group in SIDEBAR_PAGE_GROUPS:
        links = []
        for link in group["links"]:
            route = str(link["route"])
            try:
                url = reverse(route)
            except NoReverseMatch:
                continue
            active_routes = set(link.get("active_routes") or [route])
            links.append(
                {
                    "label": str(link["label"]),
                    "url": url,
                    "is_active": current_route in active_routes,
                }
            )
        if links:
            groups.append({"title": str(group["title"]), "links": links})
    return groups


@register.filter
def compact_number(value, digits: int = 2) -> str:
    try:
        number = float(value)
    except Exception:
        return "-"
    sign = "-" if number < 0 else ""
    number = abs(number)
    suffixes = [
        (1_000_000_000_000, "T"),
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "k"),
    ]
    for threshold, suffix in suffixes:
        if number >= threshold:
            return f"{sign}{number / threshold:.{int(digits)}f}{suffix}"
    if number >= 100:
        return f"{sign}{number:.0f}"
    if number >= 10:
        return f"{sign}{number:.1f}"
    return f"{sign}{number:.{int(digits)}f}"


@register.filter
def pct(value, digits: int = 1) -> str:
    try:
        number = float(value) * 100.0
    except Exception:
        return "-"
    return f"{number:.{int(digits)}f}%"


@register.filter
def signed_pct(value, digits: int = 1) -> str:
    try:
        number = float(value) * 100.0
    except Exception:
        return "-"
    return f"{number:+.{int(digits)}f}%"


@register.filter
def signed_number(value, digits: int = 2) -> str:
    try:
        number = float(value)
    except Exception:
        return "-"
    return f"{number:+.{int(digits)}f}"


@register.simple_tag
def feature_label(feature_name: str) -> str:
    return get_feature_definition(str(feature_name or "")).display_name


@register.simple_tag
def feature_value(feature_name: str, value) -> str:
    return format_feature_value(str(feature_name or ""), value)


@register.simple_tag
def feature_line(feature_name: str, value) -> str:
    return render_feature(str(feature_name or ""), value, mode="canonical")


@register.filter
def feature_family_label(value: str) -> str:
    return render_feature_family_name(str(value or ""))


@register.filter
def feature_family_signature_label(value: str) -> str:
    return render_feature_family_signature(str(value or ""))
