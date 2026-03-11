from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np


DEFAULT_EMBEDDING_STORE_DIR = Path("embedding_store")
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class EmbeddingKey:
    symbol: str
    date: str
    family: str
    model_name: str
    model_version: str


def save_embedding(
    key: EmbeddingKey | dict[str, Any] | tuple[Any, ...],
    embedding: np.ndarray,
    *,
    store_dir: str | Path = DEFAULT_EMBEDDING_STORE_DIR,
) -> Path:
    path = embedding_path(key, store_dir=store_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(embedding, dtype="float32"))
    return path


def load_embedding(
    key: EmbeddingKey | dict[str, Any] | tuple[Any, ...],
    *,
    store_dir: str | Path = DEFAULT_EMBEDDING_STORE_DIR,
) -> np.ndarray | None:
    path = embedding_path(key, store_dir=store_dir)
    if not path.exists():
        return None
    return np.load(path, allow_pickle=False).astype("float32")


def embedding_path(
    key: EmbeddingKey | dict[str, Any] | tuple[Any, ...],
    *,
    store_dir: str | Path = DEFAULT_EMBEDDING_STORE_DIR,
) -> Path:
    normalized = normalize_key(key)
    family = _slugify(normalized.family)
    symbol = _slugify(normalized.symbol)
    date = _slugify(normalized.date)
    model_name = _slugify(normalized.model_name)
    model_version = _slugify(normalized.model_version)
    filename = f"{date}__{symbol}__{model_name}__{model_version}.npy"
    return Path(store_dir) / family / symbol / filename


def normalize_key(key: EmbeddingKey | dict[str, Any] | tuple[Any, ...]) -> EmbeddingKey:
    if isinstance(key, EmbeddingKey):
        return key
    if isinstance(key, dict):
        return EmbeddingKey(
            symbol=str(key["symbol"]),
            date=str(key["date"]),
            family=str(key["family"]),
            model_name=str(key["model_name"]),
            model_version=str(key["model_version"]),
        )
    if isinstance(key, tuple) and len(key) == 5:
        symbol, date, family, model_name, model_version = key
        return EmbeddingKey(
            symbol=str(symbol),
            date=str(date),
            family=str(family),
            model_name=str(model_name),
            model_version=str(model_version),
        )
    raise TypeError("key must be an EmbeddingKey, a dict, or a 5-tuple")


def _slugify(value: str) -> str:
    cleaned = _SAFE_RE.sub("_", str(value).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"
