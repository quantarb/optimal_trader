from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass
class SimilarityIndex:
    frame: pd.DataFrame
    embedding_columns: list[str]
    scaled_matrix: np.ndarray
    normalized_matrix: np.ndarray
    means: np.ndarray
    stds: np.ndarray


def build_similarity_index(frame: pd.DataFrame, embedding_columns: Sequence[str]) -> SimilarityIndex:
    embedding_cols = [str(col) for col in list(embedding_columns or []) if str(col) in frame.columns]
    if not embedding_cols:
        raise ValueError("No embedding columns were available for similarity search.")
    matrix_df = frame[embedding_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    means = matrix_df.mean(axis=0).to_numpy(dtype=float)
    stds = matrix_df.std(axis=0, ddof=0).replace(0.0, 1.0).to_numpy(dtype=float)
    scaled = (matrix_df.to_numpy(dtype=float) - means) / stds
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = np.clip(scaled, -1_000_000.0, 1_000_000.0)
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized = np.nan_to_num(scaled / norms, nan=0.0, posinf=0.0, neginf=0.0)
    return SimilarityIndex(
        frame=frame.reset_index(drop=True).copy(),
        embedding_columns=embedding_cols,
        scaled_matrix=scaled,
        normalized_matrix=normalized,
        means=means,
        stds=stds,
    )


def encode_state_vector(state_vector: Sequence[float], similarity_index: SimilarityIndex) -> np.ndarray:
    arr = np.asarray(list(state_vector), dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.shape[0] != len(similarity_index.embedding_columns):
        raise ValueError("State vector length does not match the similarity index embedding columns.")
    scaled = arr.copy()
    norm = np.linalg.norm(scaled)
    if norm == 0.0:
        return scaled
    return np.nan_to_num(scaled / norm, nan=0.0, posinf=0.0, neginf=0.0)


def find_similar_market_states(
    state_vector: Sequence[float],
    similarity_index: SimilarityIndex,
    *,
    k: int = 10,
    query_date: str | None = None,
    exclude_symbol: str | None = None,
    exclude_date: str | None = None,
) -> list[dict[str, Any]]:
    normalized_query = encode_state_vector(state_vector, similarity_index)
    base_matrix = np.nan_to_num(similarity_index.normalized_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    normalized_query = np.nan_to_num(normalized_query, nan=0.0, posinf=0.0, neginf=0.0)
    similarities = np.nan_to_num(np.sum(base_matrix * normalized_query[None, :], axis=1), nan=-1.0, posinf=-1.0, neginf=-1.0)
    candidate = similarity_index.frame.copy()
    candidate["similarity_score"] = similarities
    if query_date:
        candidate = candidate[candidate["date"] < pd.Timestamp(str(query_date))].copy()
    if exclude_symbol and exclude_date:
        candidate = candidate[
            ~(
                (candidate["symbol"] == str(exclude_symbol).strip().upper())
                & (candidate["date"] == pd.Timestamp(str(exclude_date)))
            )
        ].copy()
    candidate = candidate.sort_values("similarity_score", ascending=False).head(max(int(k), 1))
    rows: list[dict[str, Any]] = []
    for row in candidate.to_dict(orient="records"):
        item = dict(row)
        item["date"] = str(pd.Timestamp(item["date"]).date())
        item["similarity_score"] = round(float(item["similarity_score"]), 6)
        rows.append(item)
    return rows
