from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def load_cached_payload(path: Path, required_keys: Sequence[str], *, schema_version: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("schema_version") or 0) != int(schema_version):
        return None
    if any(key not in payload for key in required_keys):
        return None
    return payload


__all__ = ["load_cached_payload"]
