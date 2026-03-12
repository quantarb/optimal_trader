from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return json_safe(value.item())
        except Exception:
            return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return value


def write_json(path: str | Path, payload: Any) -> Path:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    return resolved


def write_markdown(path: str | Path, content: str) -> Path:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(str(content).rstrip() + "\n", encoding="utf-8")
    return resolved


def load_json(path: str | Path, *, default: Any = None) -> Any:
    resolved = Path(path).resolve()
    if not resolved.exists():
        return default
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:
        return default


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, separator, *body])


def summarize_samples(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {
            "mean_seconds": 0.0,
            "median_seconds": 0.0,
            "stdev_seconds": 0.0,
            "min_seconds": 0.0,
            "max_seconds": 0.0,
        }
    return {
        "mean_seconds": float(mean(samples)),
        "median_seconds": float(median(samples)),
        "stdev_seconds": float(pstdev(samples)) if len(samples) > 1 else 0.0,
        "min_seconds": float(min(samples)),
        "max_seconds": float(max(samples)),
    }


def normalize_score_map(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    max_value = max(values.values())
    min_value = min(values.values())
    if max_value <= min_value:
        return {key: 0.0 for key in values}
    return {key: (float(value) - min_value) / (max_value - min_value) for key, value in values.items()}
