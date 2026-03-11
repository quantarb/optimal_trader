from __future__ import annotations

from django import template

from pipeline.feature_presentation import (
    format_feature_value,
    get_feature_definition,
    render_feature,
    render_feature_family_name,
    render_feature_family_signature,
)


register = template.Library()


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
