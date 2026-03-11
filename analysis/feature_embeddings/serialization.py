from __future__ import annotations

from collections.abc import Mapping
from datetime import date as date_type
from datetime import datetime
import math
from numbers import Integral, Real
import re
from typing import Any


_ACRONYMS = {
    "api",
    "cfo",
    "ebit",
    "ebitda",
    "eps",
    "ev",
    "fcf",
    "fx",
    "gdp",
    "ipo",
    "roa",
    "roe",
    "roic",
    "rsi",
    "sec",
    "usd",
}

_TOKEN_SPLIT_RE = re.compile(r"[_\-\s]+")
_BOUNDARY_RE = re.compile(r"(?<=[a-z])(?=[A-Z0-9])")


def serialize_family(symbol: str, date: Any, family_name: str, features: Mapping[str, Any]) -> str:
    """Render one feature family into deterministic structured text."""
    if not isinstance(features, Mapping):
        raise TypeError("features must be a mapping of feature names to values")

    lines = [
        f"[FAMILY={_normalize_header_value(family_name).upper()}]",
        f"[SYMBOL={_normalize_header_value(symbol).upper()}]",
        f"[DATE={_normalize_date(date)}]",
    ]
    for raw_name, value in sorted(features.items(), key=lambda item: str(item[0]).lower()):
        if is_missing_feature_value(value):
            continue
        formatted = format_feature_value(value)
        if not formatted:
            continue
        lines.append(f"{humanize_feature_name(str(raw_name))}: {formatted}")
    return "\n".join(lines)


def humanize_feature_name(feature_name: str) -> str:
    raw = str(feature_name or "").strip()
    if not raw:
        return "Unknown Feature"
    raw = raw.replace("dividedby", " / ")
    raw = _BOUNDARY_RE.sub(" ", raw)
    tokens = [token for token in _TOKEN_SPLIT_RE.split(raw) if token]
    rendered: list[str] = []
    for token in tokens:
        lower = token.lower()
        if token == "/":
            rendered.append(token)
            continue
        if lower in _ACRONYMS:
            rendered.append(lower.upper())
            continue
        numeric_suffix = re.fullmatch(r"(\d+)([a-z]+)", token)
        if numeric_suffix:
            rendered.append(f"{numeric_suffix.group(1)}{numeric_suffix.group(2).upper()}")
            continue
        if token.isupper():
            rendered.append(token)
            continue
        rendered.append(token.capitalize())
    text = " ".join(rendered).replace(" / ", " / ")
    return re.sub(r"\s+", " ", text).strip()


def format_feature_value(value: Any) -> str:
    if is_missing_feature_value(value):
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date_type):
        return value.isoformat()
    if isinstance(value, Mapping):
        rendered_items = [
            f"{str(key).strip()}={format_feature_value(item)}"
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]).lower())
            if not is_missing_feature_value(item)
        ]
        return ", ".join(item for item in rendered_items if item)
    if isinstance(value, Integral) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, Real):
        if not math.isfinite(float(value)):
            return ""
        decimals = 4 if abs(value) < 1 else 2
        return f"{float(value):.{decimals}f}"
    if isinstance(value, (list, tuple, set)):
        rendered_items = [format_feature_value(item) for item in value if not is_missing_feature_value(item)]
        return ", ".join(item for item in rendered_items if item)
    return str(value).strip()


def _normalize_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date_type):
        return value.isoformat()
    return str(value).strip()


def _normalize_header_value(value: Any) -> str:
    return str(value or "").strip()


def is_missing_feature_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return str(value).strip().lower() in {"", "nan", "none", "null", "<na>", "n/a", "na"}
    if isinstance(value, Mapping):
        return not any(not is_missing_feature_value(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return not any(not is_missing_feature_value(item) for item in value)
    if isinstance(value, Real) and not math.isfinite(float(value)):
        return True
    return False
