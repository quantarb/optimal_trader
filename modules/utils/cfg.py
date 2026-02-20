from __future__ import annotations

from typing import Any, Mapping, Optional


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safe getter for config objects or dict-like containers.

    Supports:
      - dict / Mapping
      - objects with attributes (dataclasses / pydantic / SimpleNamespace)
    """
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)
