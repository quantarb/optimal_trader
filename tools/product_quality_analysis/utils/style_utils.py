from __future__ import annotations

from collections import Counter


def normalize_style_value(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"", "initial", "normal", "auto", "none"}:
        return ""
    if text.startswith("rgba(") and text.endswith(", 0)"):
        return ""
    return text


def top_values(values: list[str], *, limit: int = 8) -> list[str]:
    counts = Counter(value for value in values if value)
    return [value for value, _count in counts.most_common(limit)]
