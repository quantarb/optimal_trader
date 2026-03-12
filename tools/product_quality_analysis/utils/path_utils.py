from __future__ import annotations

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    target = Path(path).resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def safe_slug(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value))
    while "--" in text:
        text = text.replace("--", "-")
    return text.strip("-") or "item"
