from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .path_utils import ensure_directory


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path).resolve()
    ensure_directory(target.parent)
    if hasattr(payload, "model_dump"):
        data = payload.model_dump(mode="json")
    else:
        data = payload
    target.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return target


def write_markdown(path: str | Path, content: str) -> Path:
    target = Path(path).resolve()
    ensure_directory(target.parent)
    target.write_text(str(content), encoding="utf-8")
    return target


def load_json(path: str | Path, *, default: Any = None) -> Any:
    target = Path(path).resolve()
    if not target.exists():
        return default
    return json.loads(target.read_text(encoding="utf-8"))


def bullet_list(items: list[str]) -> str:
    if not items:
        return "- None\n"
    return "".join(f"- {item}\n" for item in items)
