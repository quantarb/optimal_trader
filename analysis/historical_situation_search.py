from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from pipeline.feature_presentation import render_feature_family_name
from .historical_outcomes import aggregate_outcome_distribution, enrich_similarity_matches_with_outcomes
from .state_representations import (
    MarketStateRepresentation,
    TextEmbeddingConfig,
    build_market_state_representation,
    build_numeric_state_vector,
    export_embedding_features,
    normalize_numeric_vector,
)


DEFAULT_SEARCH_HORIZONS = (5, 20, 60, 90, 180)


@dataclass
class StateSearchIndex:
    frame: pd.DataFrame
    vector_columns: list[str]
    matrix: np.ndarray
    normalized_matrix: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    method: str


@dataclass
class HistoricalSituationSearchBundle:
    frame: pd.DataFrame
    feature_columns: list[str]
    feature_family_map: dict[str, list[str]]
    numeric_index: StateSearchIndex
    embedding_index: StateSearchIndex
    text_embedding_config: TextEmbeddingConfig


def _infer_frame_feature_columns(frame: pd.DataFrame, feature_columns: Sequence[str] | None = None) -> list[str]:
    if feature_columns:
        return [str(column) for column in list(feature_columns or []) if str(column) in frame.columns]
    columns: list[str] = []
    for column in frame.columns:
        if str(column) in {"date", "symbol"}:
            continue
        parsed = pd.to_numeric(frame[column], errors="coerce")
        if parsed.notna().any():
            columns.append(str(column))
    return columns


def _normalized_matrix(matrix: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(matrix, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, -1_000.0, 1_000.0)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized = np.nan_to_num(arr / norms, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(normalized, -1_000.0, 1_000.0)


def _safe_similarity_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    base = np.nan_to_num(np.asarray(matrix, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    probe = np.nan_to_num(np.asarray(query, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    base = np.clip(base, -1_000.0, 1_000.0)
    probe = np.clip(probe, -1_000.0, 1_000.0)
    return np.nan_to_num(np.sum(base * probe[None, :], axis=1), nan=-1.0, posinf=-1.0, neginf=-1.0)


def build_numeric_index(
    state_vectors: pd.DataFrame | np.ndarray,
    *,
    frame: pd.DataFrame | None = None,
    feature_columns: Sequence[str] | None = None,
) -> StateSearchIndex:
    if isinstance(state_vectors, pd.DataFrame):
        base_frame = state_vectors.reset_index(drop=True).copy()
        columns = _infer_frame_feature_columns(base_frame, feature_columns=feature_columns)
        matrix_df = base_frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    else:
        if frame is None:
            raise ValueError("A frame is required when building a numeric index from a raw matrix.")
        base_frame = frame.reset_index(drop=True).copy()
        matrix = np.asarray(state_vectors, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("Numeric state vectors must be two-dimensional.")
        columns = [f"numeric_{idx}" for idx in range(matrix.shape[1])]
        matrix_df = pd.DataFrame(matrix, columns=columns)
    means = matrix_df.mean(axis=0).to_numpy(dtype=np.float64)
    stds = matrix_df.std(axis=0, ddof=0).replace(0.0, 1.0).to_numpy(dtype=np.float64)
    scaled = np.vstack([
        normalize_numeric_vector(row, means=means, stds=stds, l2_normalize=False, clip_value=1_000.0)
        for row in matrix_df.to_numpy(dtype=np.float64)
    ])
    return StateSearchIndex(
        frame=base_frame,
        vector_columns=columns,
        matrix=scaled,
        normalized_matrix=_normalized_matrix(scaled),
        means=means,
        stds=stds,
        method="numeric",
    )


def build_embedding_index(
    embeddings: pd.DataFrame | np.ndarray,
    *,
    frame: pd.DataFrame | None = None,
) -> StateSearchIndex:
    if isinstance(embeddings, pd.DataFrame):
        base_frame = embeddings.reset_index(drop=True).copy() if frame is None else frame.reset_index(drop=True).copy()
        vector_columns = [str(column) for column in embeddings.columns if str(column).startswith("emb_") or str(column).startswith("embedding_")]
        if not vector_columns:
            vector_columns = [str(column) for column in embeddings.columns if str(column) not in {"date", "symbol"}]
        matrix = embeddings[vector_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    else:
        if frame is None:
            raise ValueError("A frame is required when building an embedding index from a raw matrix.")
        base_frame = frame.reset_index(drop=True).copy()
        matrix = np.asarray(embeddings, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("Embedding matrix must be two-dimensional.")
        vector_columns = [f"emb_{idx}" for idx in range(matrix.shape[1])]
    means = np.nan_to_num(matrix.mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    stds = np.nan_to_num(matrix.std(axis=0, ddof=0), nan=1.0, posinf=1.0, neginf=1.0)
    stds[stds == 0.0] = 1.0
    normalized = _normalized_matrix(matrix)
    return StateSearchIndex(
        frame=base_frame,
        vector_columns=vector_columns,
        matrix=np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0),
        normalized_matrix=normalized,
        means=means,
        stds=stds,
        method="text_embedding",
    )


def _filter_search_candidates(
    frame: pd.DataFrame,
    *,
    query_symbol: str | None,
    query_date: str | None,
    search_mode: str,
    exclude_exact: bool = True,
) -> pd.Series:
    symbol = str(query_symbol or "").strip().upper()
    date_value = pd.Timestamp(str(query_date)) if query_date else None
    candidate = pd.Series(True, index=frame.index)
    if date_value is not None and "date" in frame.columns:
        candidate &= pd.to_datetime(frame["date"], errors="coerce") < date_value
    if not symbol or "symbol" not in frame.columns:
        return candidate
    symbols = frame["symbol"].astype(str).str.strip().str.upper()
    mode = str(search_mode or "mixed").strip().lower()
    if mode == "same_symbol":
        candidate &= symbols == symbol
    elif mode == "cross_symbol":
        candidate &= symbols != symbol
    elif exclude_exact and date_value is not None:
        candidate &= ~((symbols == symbol) & (pd.to_datetime(frame["date"], errors="coerce") == date_value))
    return candidate


def _match_type(symbol: str, query_symbol: str | None) -> str:
    if not query_symbol:
        return "mixed"
    return "same_symbol" if str(symbol).strip().upper() == str(query_symbol).strip().upper() else "cross_symbol"


def search_numeric_neighbors(
    query_vector: Sequence[float],
    numeric_index: StateSearchIndex,
    *,
    top_k: int = 10,
    search_mode: str = "mixed",
    query_symbol: str | None = None,
    query_date: str | None = None,
) -> list[dict[str, Any]]:
    normalized_query = normalize_numeric_vector(
        query_vector,
        means=numeric_index.means,
        stds=numeric_index.stds,
        l2_normalize=True,
        clip_value=1_000.0,
    )
    scores = _safe_similarity_scores(numeric_index.normalized_matrix, normalized_query)
    candidate = numeric_index.frame.copy()
    candidate["numeric_similarity"] = scores
    mask = _filter_search_candidates(
        candidate,
        query_symbol=query_symbol,
        query_date=query_date,
        search_mode=search_mode,
        exclude_exact=True,
    )
    candidate = candidate[mask].copy()
    candidate["similarity_score"] = candidate["numeric_similarity"]
    candidate = candidate.sort_values("similarity_score", ascending=False).head(max(int(top_k), 1))
    rows: list[dict[str, Any]] = []
    for row in candidate.to_dict(orient="records"):
        item = dict(row)
        item["date"] = str(pd.Timestamp(item["date"]).date()) if "date" in item else ""
        item["similarity_score"] = round(float(item.get("similarity_score") or 0.0), 6)
        item["numeric_similarity"] = round(float(item.get("numeric_similarity") or 0.0), 6)
        item["match_type"] = _match_type(str(item.get("symbol") or ""), query_symbol)
        rows.append(item)
    return rows


def search_embedding_neighbors(
    query_embedding: Sequence[float],
    embedding_index: StateSearchIndex,
    *,
    top_k: int = 10,
    search_mode: str = "mixed",
    query_symbol: str | None = None,
    query_date: str | None = None,
) -> list[dict[str, Any]]:
    query = normalize_numeric_vector(query_embedding, l2_normalize=True, clip_value=1_000.0)
    scores = _safe_similarity_scores(embedding_index.normalized_matrix, query)
    candidate = embedding_index.frame.copy()
    candidate["embedding_similarity"] = scores
    mask = _filter_search_candidates(
        candidate,
        query_symbol=query_symbol,
        query_date=query_date,
        search_mode=search_mode,
        exclude_exact=True,
    )
    candidate = candidate[mask].copy()
    candidate["similarity_score"] = candidate["embedding_similarity"]
    candidate = candidate.sort_values("similarity_score", ascending=False).head(max(int(top_k), 1))
    rows: list[dict[str, Any]] = []
    for row in candidate.to_dict(orient="records"):
        item = dict(row)
        item["date"] = str(pd.Timestamp(item["date"]).date()) if "date" in item else ""
        item["similarity_score"] = round(float(item.get("similarity_score") or 0.0), 6)
        item["embedding_similarity"] = round(float(item.get("embedding_similarity") or 0.0), 6)
        item["match_type"] = _match_type(str(item.get("symbol") or ""), query_symbol)
        rows.append(item)
    return rows


def compute_hybrid_similarity(
    query_state: MarketStateRepresentation | Mapping[str, Any],
    candidate_state: MarketStateRepresentation | Mapping[str, Any],
    *,
    numeric_weight: float = 0.5,
    embedding_weight: float = 0.5,
) -> dict[str, float]:
    def _resolve_vectors(item: MarketStateRepresentation | Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(item, MarketStateRepresentation):
            return item.numeric_vector, item.embedding_vector
        payload = dict(item)
        return (
            normalize_numeric_vector(payload.get("numeric_vector") or [], l2_normalize=True, clip_value=1_000.0),
            normalize_numeric_vector(payload.get("embedding_vector") or [], l2_normalize=True, clip_value=1_000.0),
        )

    query_numeric, query_embedding = _resolve_vectors(query_state)
    cand_numeric, cand_embedding = _resolve_vectors(candidate_state)
    numeric_similarity = float(_safe_similarity_scores(np.asarray([query_numeric]), cand_numeric)[0]) if len(query_numeric) and len(cand_numeric) else 0.0
    embedding_similarity = float(_safe_similarity_scores(np.asarray([query_embedding]), cand_embedding)[0]) if len(query_embedding) and len(cand_embedding) else 0.0
    weight_total = max(float(numeric_weight) + float(embedding_weight), 1e-9)
    hybrid_similarity = (
        float(numeric_weight) * numeric_similarity + float(embedding_weight) * embedding_similarity
    ) / weight_total
    return {
        "numeric_similarity": round(float(numeric_similarity), 6),
        "embedding_similarity": round(float(embedding_similarity), 6),
        "hybrid_similarity": round(float(hybrid_similarity), 6),
    }


def _family_similarity_rows(
    query_state: Mapping[str, Any],
    candidate_state: Mapping[str, Any],
    *,
    feature_family_map: Mapping[str, Sequence[str]],
    limit: int = 3,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_name, columns in dict(feature_family_map or {}).items():
        usable = [str(column) for column in list(columns or []) if str(column) in query_state and str(column) in candidate_state]
        if not usable:
            continue
        query_vector = normalize_numeric_vector(build_numeric_state_vector(query_state, feature_columns=usable), l2_normalize=True, clip_value=1_000.0)
        candidate_vector = normalize_numeric_vector(build_numeric_state_vector(candidate_state, feature_columns=usable), l2_normalize=True, clip_value=1_000.0)
        if len(query_vector) == 0 or len(candidate_vector) == 0:
            continue
        similarity = float(_safe_similarity_scores(np.asarray([query_vector]), candidate_vector)[0])
        rows.append(
            {
                "family": str(family_name),
                "similarity": round(similarity, 6),
                "explanation": f"{render_feature_family_name(str(family_name))} features",
            }
        )
    rows.sort(key=lambda item: float(item.get("similarity") or 0.0), reverse=True)
    return rows[: max(int(limit), 1)]


def build_historical_situation_search_bundle(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    feature_family_map: Mapping[str, Sequence[str]] | None = None,
    text_embedding_config: TextEmbeddingConfig | None = None,
) -> HistoricalSituationSearchBundle:
    work = frame.copy().reset_index(drop=True)
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
    if "symbol" in work.columns:
        work["symbol"] = work["symbol"].astype(str).str.strip().str.upper()
    columns = _infer_frame_feature_columns(work, feature_columns=feature_columns)
    family_map = {
        str(name): [str(column) for column in list(values or []) if str(column) in work.columns]
        for name, values in dict(feature_family_map or {}).items()
        if list(values or [])
    }
    numeric_index = build_numeric_index(work, feature_columns=columns)
    embedding_frame = export_embedding_features(
        work,
        feature_family_map=family_map or None,
        feature_columns=columns,
        config=text_embedding_config or TextEmbeddingConfig(),
        id_columns=("date", "symbol"),
        prefix="emb_",
    )
    embedding_index = build_embedding_index(embedding_frame.drop(columns=[col for col in ("date", "symbol") if col in embedding_frame.columns]), frame=work)
    return HistoricalSituationSearchBundle(
        frame=work,
        feature_columns=columns,
        feature_family_map=family_map,
        numeric_index=numeric_index,
        embedding_index=embedding_index,
        text_embedding_config=text_embedding_config or TextEmbeddingConfig(),
    )


def search_hybrid_neighbors(
    query_state: pd.Series | Mapping[str, Any] | MarketStateRepresentation,
    search_bundle: HistoricalSituationSearchBundle,
    *,
    top_k: int = 10,
    search_mode: str = "mixed",
    query_symbol: str | None = None,
    query_date: str | None = None,
    numeric_weight: float = 0.5,
    embedding_weight: float = 0.5,
) -> list[dict[str, Any]]:
    if isinstance(query_state, MarketStateRepresentation):
        representation = query_state
        query_row = {}
    else:
        representation = build_market_state_representation(
            query_state,
            method="hybrid",
            feature_columns=search_bundle.feature_columns,
            feature_family_map=search_bundle.feature_family_map,
            numeric_means=search_bundle.numeric_index.means,
            numeric_stds=search_bundle.numeric_index.stds,
            text_embedding_config=search_bundle.text_embedding_config,
        )
        query_row = dict(query_state)
    numeric_scores = _safe_similarity_scores(search_bundle.numeric_index.normalized_matrix, representation.numeric_vector)
    embedding_scores = _safe_similarity_scores(search_bundle.embedding_index.normalized_matrix, representation.embedding_vector)
    weight_total = max(float(numeric_weight) + float(embedding_weight), 1e-9)
    hybrid_scores = (
        float(numeric_weight) * numeric_scores + float(embedding_weight) * embedding_scores
    ) / weight_total
    candidate = search_bundle.frame.copy()
    candidate["numeric_similarity"] = numeric_scores
    candidate["embedding_similarity"] = embedding_scores
    candidate["similarity_score"] = hybrid_scores
    mask = _filter_search_candidates(
        candidate,
        query_symbol=query_symbol,
        query_date=query_date,
        search_mode=search_mode,
        exclude_exact=True,
    )
    candidate = candidate[mask].copy()
    candidate = candidate.sort_values("similarity_score", ascending=False).head(max(int(top_k), 1))
    rows: list[dict[str, Any]] = []
    for row in candidate.to_dict(orient="records"):
        item = dict(row)
        item["date"] = str(pd.Timestamp(item["date"]).date()) if "date" in item else ""
        item["similarity_score"] = round(float(item.get("similarity_score") or 0.0), 6)
        item["numeric_similarity"] = round(float(item.get("numeric_similarity") or 0.0), 6)
        item["embedding_similarity"] = round(float(item.get("embedding_similarity") or 0.0), 6)
        item["match_type"] = _match_type(str(item.get("symbol") or ""), query_symbol)
        item["explanations"] = _family_similarity_rows(
            query_row,
            item,
            feature_family_map=search_bundle.feature_family_map,
            limit=3,
        ) if query_row else []
        rows.append(item)
    return rows


def search_market_state_neighbors(
    query_state: pd.Series | Mapping[str, Any],
    search_bundle: HistoricalSituationSearchBundle,
    *,
    method: str = "hybrid",
    top_k: int = 10,
    search_mode: str = "mixed",
    query_symbol: str | None = None,
    query_date: str | None = None,
    numeric_weight: float = 0.5,
    embedding_weight: float = 0.5,
) -> list[dict[str, Any]]:
    resolved_method = str(method or "hybrid").strip().lower()
    if resolved_method == "numeric":
        query_vector = build_numeric_state_vector(query_state, feature_columns=search_bundle.feature_columns)
        rows = search_numeric_neighbors(
            query_vector,
            search_bundle.numeric_index,
            top_k=top_k,
            search_mode=search_mode,
            query_symbol=query_symbol,
            query_date=query_date,
        )
        for row in rows:
            row["explanations"] = _family_similarity_rows(
                dict(query_state),
                row,
                feature_family_map=search_bundle.feature_family_map,
                limit=3,
            )
        return rows
    if resolved_method in {"text_embedding", "embedding"}:
        query_embedding = build_market_state_representation(
            query_state,
            method="text_embedding",
            feature_columns=search_bundle.feature_columns,
            feature_family_map=search_bundle.feature_family_map,
            numeric_means=search_bundle.numeric_index.means,
            numeric_stds=search_bundle.numeric_index.stds,
            text_embedding_config=search_bundle.text_embedding_config,
        ).embedding_vector
        rows = search_embedding_neighbors(
            query_embedding,
            search_bundle.embedding_index,
            top_k=top_k,
            search_mode=search_mode,
            query_symbol=query_symbol,
            query_date=query_date,
        )
        for row in rows:
            row["explanations"] = _family_similarity_rows(
                dict(query_state),
                row,
                feature_family_map=search_bundle.feature_family_map,
                limit=3,
            )
        return rows
    return search_hybrid_neighbors(
        query_state,
        search_bundle,
        top_k=top_k,
        search_mode=search_mode,
        query_symbol=query_symbol,
        query_date=query_date,
        numeric_weight=numeric_weight,
        embedding_weight=embedding_weight,
    )


def summarize_historical_outcomes(
    matches: Sequence[dict[str, Any]],
    frame: pd.DataFrame,
    *,
    price_col: str,
    horizons: Sequence[int] = DEFAULT_SEARCH_HORIZONS,
) -> dict[str, Any]:
    enriched = enrich_similarity_matches_with_outcomes(matches, frame, price_col=price_col, horizons=horizons)
    summary = aggregate_outcome_distribution(enriched, horizons=horizons)
    return {
        "matches": enriched,
        "summary": summary,
    }
