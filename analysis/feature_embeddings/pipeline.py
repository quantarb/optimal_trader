from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .embedding_store import DEFAULT_EMBEDDING_STORE_DIR, EmbeddingKey, load_embedding, save_embedding
from .encoder import encode_family, fit_text_to_token_limit, resolve_encoder_identity
from .pooling import l2_normalize, pool_embeddings
from .serialization import is_missing_feature_value, serialize_family


def build_row_embedding(
    symbol: str,
    date: Any,
    family_feature_dict: Mapping[str, Mapping[str, Any]],
    *,
    encoder: Any | None = None,
    store_dir: str = str(DEFAULT_EMBEDDING_STORE_DIR),
) -> np.ndarray:
    if not isinstance(family_feature_dict, Mapping):
        raise TypeError("family_feature_dict must map family names to feature dictionaries")

    model_name, model_version = resolve_encoder_identity(encoder)
    family_embeddings: list[np.ndarray] = []
    for family_name, features in sorted(family_feature_dict.items(), key=lambda item: str(item[0]).lower()):
        cleaned_features = _clean_family_features(features)
        if not cleaned_features:
            continue
        key = EmbeddingKey(
            symbol=str(symbol),
            date=str(date),
            family=str(family_name),
            model_name=model_name,
            model_version=model_version,
        )
        cached = load_embedding(key, store_dir=store_dir)
        if cached is None:
            serialized = fit_text_to_token_limit(
                serialize_family(symbol, date, family_name, cleaned_features),
                encoder=encoder,
            )
            cached = l2_normalize(encode_family(serialized, encoder=encoder))
            save_embedding(key, cached, store_dir=store_dir)
        else:
            cached = l2_normalize(cached)
        family_embeddings.append(cached)
    if not family_embeddings:
        raise ValueError(f"No non-empty feature families were available for {symbol} on {date}.")
    return pool_embeddings(family_embeddings)


def build_dataset_embeddings(
    dataset: Iterable[Mapping[str, Any]],
    *,
    encoder: Any | None = None,
    store_dir: str = str(DEFAULT_EMBEDDING_STORE_DIR),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in dataset:
        symbol = str(row["symbol"])
        date = row["date"]
        families = _extract_family_feature_dict(row)
        rows.append(
            {
                "symbol": symbol,
                "date": str(date),
                "embedding_vector": build_row_embedding(
                    symbol,
                    date,
                    families,
                    encoder=encoder,
                    store_dir=store_dir,
                ),
            }
        )
    return rows


def _extract_family_feature_dict(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    explicit = row.get("families")
    if isinstance(explicit, Mapping):
        return {str(name): value for name, value in explicit.items() if isinstance(value, Mapping)}
    families: dict[str, Mapping[str, Any]] = {}
    for key, value in row.items():
        if key in {"symbol", "date"}:
            continue
        if isinstance(value, Mapping):
            families[str(key)] = value
    return families


def _clean_family_features(features: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in dict(features or {}).items():
        if _is_missing_value(value):
            continue
        cleaned[str(key)] = value
    return cleaned


def _is_missing_value(value: Any) -> bool:
    if is_missing_feature_value(value):
        return True
    try:
        return bool(np.isnan(value))
    except Exception:
        return False
