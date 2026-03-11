from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import blake2b
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from pipeline.feature_presentation import serialize_features_for_embedding


DEFAULT_NON_FEATURE_KEYS = {
    "date",
    "symbol",
    "label",
    "market_position",
    "trade_return",
    "hold_days",
    "side",
    "freq",
    "k",
    "match_type",
}


@dataclass
class TextEmbeddingConfig:
    backend: str = "auto"
    model_name: str = ""
    embedding_dim: int = 128
    normalize: bool = True
    max_features_per_family: int = 24


@dataclass
class MarketStateRepresentation:
    method: str
    vector: np.ndarray
    numeric_vector: np.ndarray
    embedding_vector: np.ndarray
    feature_columns: list[str]
    feature_family_map: dict[str, list[str]]
    family_texts: dict[str, str] = field(default_factory=dict)
    family_embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _coerce_state_series(state: pd.Series | Mapping[str, Any]) -> pd.Series:
    if isinstance(state, pd.Series):
        return state.copy()
    return pd.Series(dict(state))


def _infer_numeric_feature_columns(
    state: pd.Series | Mapping[str, Any],
    *,
    feature_columns: Sequence[str] | None = None,
    excluded_keys: Sequence[str] = (),
) -> list[str]:
    if feature_columns:
        return [str(column) for column in list(feature_columns or []) if str(column).strip()]
    row = _coerce_state_series(state)
    excluded = {str(value) for value in DEFAULT_NON_FEATURE_KEYS}
    excluded.update(str(value) for value in list(excluded_keys or []))
    columns: list[str] = []
    for key, value in row.items():
        name = str(key)
        if name in excluded:
            continue
        parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(parsed):
            columns.append(name)
    return columns


def build_numeric_state_vector(
    state: pd.Series | Mapping[str, Any],
    *,
    feature_columns: Sequence[str] | None = None,
    excluded_keys: Sequence[str] = (),
    fill_value: float = 0.0,
) -> np.ndarray:
    row = _coerce_state_series(state)
    columns = _infer_numeric_feature_columns(row, feature_columns=feature_columns, excluded_keys=excluded_keys)
    values: list[float] = []
    for column in columns:
        parsed = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        values.append(float(fill_value if pd.isna(parsed) else parsed))
    vector = np.asarray(values, dtype=np.float64)
    return np.nan_to_num(vector, nan=float(fill_value), posinf=float(fill_value), neginf=float(fill_value))


def normalize_numeric_vector(
    vector: Sequence[float],
    *,
    means: Sequence[float] | None = None,
    stds: Sequence[float] | None = None,
    l2_normalize: bool = True,
    clip_value: float = 1_000.0,
) -> np.ndarray:
    arr = np.asarray(list(vector), dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if means is not None and stds is not None:
        mean_arr = np.asarray(list(means), dtype=np.float64)
        std_arr = np.asarray(list(stds), dtype=np.float64)
        std_arr[std_arr == 0.0] = 1.0
        arr = np.nan_to_num((arr - mean_arr) / std_arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, -abs(float(clip_value)), abs(float(clip_value)))
    if not l2_normalize:
        return arr
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr
    return np.nan_to_num(arr / norm, nan=0.0, posinf=0.0, neginf=0.0)


def serialize_feature_family_to_text(
    family: Mapping[str, Any] | tuple[str, Mapping[str, Any]],
    *,
    family_name: str | None = None,
    max_features: int | None = None,
) -> str:
    if isinstance(family, tuple):
        resolved_name = str(family[0])
        values = dict(family[1] or {})
    else:
        resolved_name = str(family_name or "features")
        values = dict(family or {})
    limit = max(1, int(max_features or len(values) or 1))
    limited_values = {str(key): values.get(key) for idx, key in enumerate(sorted(values.keys(), key=str)) if idx < limit}
    return serialize_features_for_embedding({resolved_name: limited_values})


def _project_embedding(vector: np.ndarray, target_dim: int, *, seed_text: str) -> np.ndarray:
    source = np.nan_to_num(np.asarray(vector, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if source.shape[0] == target_dim:
        return source
    digest = blake2b(seed_text.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, "little", signed=False)
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((source.shape[0], target_dim), dtype=np.float64) / max(np.sqrt(target_dim), 1.0)
    projected = source @ projection
    return np.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)


@lru_cache(maxsize=8)
def _hashing_vectorizer(n_features: int) -> HashingVectorizer:
    return HashingVectorizer(
        n_features=max(8, int(n_features)),
        alternate_sign=False,
        norm=None,
        analyzer="word",
        lowercase=False,
    )


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_feature_family(
    text: str,
    *,
    config: TextEmbeddingConfig | None = None,
) -> np.ndarray:
    resolved = config or TextEmbeddingConfig()
    backend = str(resolved.backend or "auto").strip().lower()
    model_name = str(resolved.model_name or "").strip()
    if backend in {"auto", "sentence_transformer", "sentence-transformer"} and model_name:
        try:
            model = _load_sentence_transformer(model_name)
            raw = np.asarray(model.encode([text], show_progress_bar=False)[0], dtype=np.float64)
            vector = _project_embedding(raw, max(8, int(resolved.embedding_dim or raw.shape[0])), seed_text=model_name)
            return normalize_numeric_vector(vector, l2_normalize=bool(resolved.normalize), clip_value=1_000.0)
        except Exception:
            pass
    hashed = _hashing_vectorizer(max(8, int(resolved.embedding_dim or 128))).transform([text]).toarray()[0]
    return normalize_numeric_vector(hashed, l2_normalize=bool(resolved.normalize), clip_value=1_000.0)


def combine_family_embeddings(
    embeddings: Sequence[Sequence[float]],
    *,
    normalize: bool = True,
) -> np.ndarray:
    vectors = [np.asarray(list(vector), dtype=np.float64) for vector in list(embeddings or []) if len(vector) > 0]
    if not vectors:
        return np.zeros(0, dtype=np.float64)
    matrix = np.vstack(vectors)
    combined = np.nan_to_num(matrix.mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    return normalize_numeric_vector(combined, l2_normalize=normalize, clip_value=1_000.0)


def _family_value_map(
    state: pd.Series | Mapping[str, Any],
    *,
    feature_family_map: Mapping[str, Sequence[str]] | None = None,
    feature_columns: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    row = _coerce_state_series(state)
    if feature_family_map:
        grouped: dict[str, dict[str, Any]] = {}
        for family_name, columns in dict(feature_family_map or {}).items():
            values = {str(column): row.get(column) for column in list(columns or []) if str(column) in row.index}
            if values:
                grouped[str(family_name)] = values
        if grouped:
            return grouped
    columns = _infer_numeric_feature_columns(row, feature_columns=feature_columns)
    return {"numeric_state": {column: row.get(column) for column in columns}}


def generate_state_embedding(
    state: pd.Series | Mapping[str, Any],
    *,
    feature_family_map: Mapping[str, Sequence[str]] | None = None,
    feature_columns: Sequence[str] | None = None,
    config: TextEmbeddingConfig | None = None,
) -> np.ndarray:
    resolved = config or TextEmbeddingConfig()
    family_values = _family_value_map(
        state,
        feature_family_map=feature_family_map,
        feature_columns=feature_columns,
    )
    family_embeddings = [
        embed_feature_family(
            serialize_feature_family_to_text(
                (family_name, values),
                max_features=int(resolved.max_features_per_family or 24),
            ),
            config=resolved,
        )
        for family_name, values in family_values.items()
        if values
    ]
    if not family_embeddings:
        return np.zeros(max(8, int(resolved.embedding_dim or 128)), dtype=np.float64)
    return combine_family_embeddings(family_embeddings, normalize=bool(resolved.normalize))


def build_market_state_representation(
    state: pd.Series | Mapping[str, Any],
    *,
    method: str = "numeric",
    feature_columns: Sequence[str] | None = None,
    feature_family_map: Mapping[str, Sequence[str]] | None = None,
    numeric_means: Sequence[float] | None = None,
    numeric_stds: Sequence[float] | None = None,
    text_embedding_config: TextEmbeddingConfig | None = None,
) -> MarketStateRepresentation:
    resolved_method = str(method or "numeric").strip().lower()
    numeric_columns = _infer_numeric_feature_columns(state, feature_columns=feature_columns)
    numeric_vector = build_numeric_state_vector(state, feature_columns=numeric_columns)
    normalized_numeric = normalize_numeric_vector(
        numeric_vector,
        means=numeric_means,
        stds=numeric_stds,
        l2_normalize=True,
        clip_value=1_000.0,
    )
    family_map = {
        str(name): [str(column) for column in list(columns or [])]
        for name, columns in dict(feature_family_map or {}).items()
        if list(columns or [])
    }
    embedding_vector = generate_state_embedding(
        state,
        feature_family_map=family_map or None,
        feature_columns=numeric_columns,
        config=text_embedding_config,
    )
    row = _coerce_state_series(state)
    family_values = _family_value_map(row, feature_family_map=family_map or None, feature_columns=numeric_columns)
    family_texts = {
        family_name: serialize_feature_family_to_text(
            (family_name, values),
            max_features=int((text_embedding_config or TextEmbeddingConfig()).max_features_per_family or 24),
        )
        for family_name, values in family_values.items()
    }
    family_embeddings = {
        family_name: embed_feature_family(text, config=text_embedding_config or TextEmbeddingConfig())
        for family_name, text in family_texts.items()
    }
    if resolved_method == "numeric":
        vector = normalized_numeric
    elif resolved_method in {"text_embedding", "embedding"}:
        vector = embedding_vector
        resolved_method = "text_embedding"
    elif resolved_method == "hybrid":
        vector = np.concatenate([normalized_numeric, embedding_vector]).astype(np.float64)
    else:
        raise ValueError(f"Unsupported market state representation method: {method}")
    return MarketStateRepresentation(
        method=resolved_method,
        vector=np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0),
        numeric_vector=normalized_numeric,
        embedding_vector=embedding_vector,
        feature_columns=numeric_columns,
        feature_family_map=family_map,
        family_texts=family_texts,
        family_embeddings=family_embeddings,
        metadata={
            "text_embedding_backend": str((text_embedding_config or TextEmbeddingConfig()).backend or "auto"),
            "text_embedding_model": str((text_embedding_config or TextEmbeddingConfig()).model_name or ""),
            "embedding_dim": int((text_embedding_config or TextEmbeddingConfig()).embedding_dim or len(embedding_vector)),
        },
    )


def append_embedding_features(
    X: pd.DataFrame | np.ndarray,
    embedding_matrix: pd.DataFrame | np.ndarray,
    *,
    prefix: str = "emb_",
) -> pd.DataFrame | np.ndarray:
    if isinstance(embedding_matrix, pd.DataFrame):
        emb_df = embedding_matrix.copy()
    else:
        emb_array = np.asarray(embedding_matrix, dtype=np.float64)
        if emb_array.ndim != 2:
            raise ValueError("Embedding matrix must be two-dimensional.")
        emb_df = pd.DataFrame(emb_array, columns=[f"{prefix}{idx}" for idx in range(emb_array.shape[1])])
    if isinstance(X, pd.DataFrame):
        out = X.reset_index(drop=True).copy()
        emb_reset = emb_df.reset_index(drop=True).copy()
        if list(emb_reset.columns) == list(range(len(emb_reset.columns))):
            emb_reset.columns = [f"{prefix}{idx}" for idx in range(emb_reset.shape[1])]
        return pd.concat([out, emb_reset], axis=1)
    base = np.asarray(X, dtype=np.float64)
    emb = emb_df.to_numpy(dtype=np.float64)
    if base.ndim != 2 or emb.ndim != 2:
        raise ValueError("Both feature and embedding matrices must be two-dimensional.")
    return np.concatenate([base, emb], axis=1)


def export_embedding_features(
    states: pd.DataFrame,
    *,
    feature_family_map: Mapping[str, Sequence[str]] | None = None,
    feature_columns: Sequence[str] | None = None,
    config: TextEmbeddingConfig | None = None,
    id_columns: Sequence[str] = ("date", "symbol"),
    prefix: str = "emb_",
) -> pd.DataFrame:
    if states.empty:
        columns = [str(column) for column in list(id_columns or [])]
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for row in states.to_dict(orient="records"):
        vector = generate_state_embedding(
            row,
            feature_family_map=feature_family_map,
            feature_columns=feature_columns,
            config=config,
        )
        item = {str(column): row.get(column) for column in list(id_columns or [])}
        for idx, value in enumerate(vector.tolist()):
            item[f"{prefix}{idx}"] = float(value)
        rows.append(item)
    return pd.DataFrame(rows)
