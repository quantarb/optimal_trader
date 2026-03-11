from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .market_state import embedding_columns_from_frame


@dataclass
class StateEmbeddingModel:
    feature_columns: list[str]
    fill_values: list[float]
    means: list[float]
    stds: list[float]
    embedding_columns: list[str]
    pca_components: list[list[float]]
    pca_mean: list[float]
    explained_variance_ratio: list[float]


def _numeric_frame(frame: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    usable = [str(column) for column in list(feature_columns or []) if str(column) in frame.columns]
    if not usable:
        raise ValueError("No usable feature columns were provided for state embedding.")
    numeric = frame[usable].apply(pd.to_numeric, errors="coerce")
    fill_values = numeric.median(axis=0, skipna=True).fillna(0.0)
    return numeric.fillna(fill_values), fill_values


def fit_state_embedding_model(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    pca_components: int = 8,
) -> StateEmbeddingModel:
    columns = [str(column) for column in list(feature_columns or []) if str(column).strip()]
    if not columns:
        columns = embedding_columns_from_frame(frame)
    numeric, fill_values = _numeric_frame(frame, columns)
    matrix = numeric.to_numpy(dtype=float)
    means = np.nan_to_num(matrix.mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    stds = np.nan_to_num(matrix.std(axis=0, ddof=0), nan=1.0, posinf=1.0, neginf=1.0)
    stds[stds == 0.0] = 1.0
    scaled = np.nan_to_num((matrix - means) / stds, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = np.clip(scaled, -1_000_000.0, 1_000_000.0)

    pca_matrix: np.ndarray | None = None
    pca_mean: np.ndarray = np.asarray([], dtype=float)
    explained_variance_ratio: np.ndarray = np.asarray([], dtype=float)
    if int(pca_components or 0) > 0 and scaled.shape[0] >= 2 and scaled.shape[1] >= 2:
        component_count = max(1, min(int(pca_components), scaled.shape[0], scaled.shape[1]))
        pca_mean = np.nan_to_num(scaled.mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        centered = np.nan_to_num(scaled - pca_mean[None, :], nan=0.0, posinf=0.0, neginf=0.0)
        try:
            _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
            pca_matrix = np.asarray(vt[:component_count], dtype=float)
            if centered.shape[0] > 1:
                explained_variance = (singular_values ** 2) / float(centered.shape[0] - 1)
                total_variance = float(explained_variance.sum())
                explained_variance_ratio = (
                    np.asarray(explained_variance[:component_count] / total_variance, dtype=float)
                    if total_variance > 0.0
                    else np.zeros(component_count, dtype=float)
                )
            else:
                explained_variance_ratio = np.zeros(component_count, dtype=float)
        except Exception:
            pca_matrix = None
            pca_mean = np.asarray([], dtype=float)
            explained_variance_ratio = np.asarray([], dtype=float)
        embedding_cols = [f"emb_{idx}" for idx in range(component_count if pca_matrix is not None else scaled.shape[1])]
    else:
        embedding_cols = [f"emb_{idx}" for idx in range(scaled.shape[1])]
    return StateEmbeddingModel(
        feature_columns=list(columns),
        fill_values=[float(value) for value in fill_values.reindex(columns).tolist()],
        means=[float(value) for value in means.tolist()],
        stds=[float(value) for value in stds.tolist()],
        embedding_columns=embedding_cols,
        pca_components=[list(map(float, row)) for row in pca_matrix.tolist()] if pca_matrix is not None else [],
        pca_mean=[float(value) for value in pca_mean.tolist()],
        explained_variance_ratio=[float(value) for value in explained_variance_ratio.tolist()],
    )


def serialize_state_embedding_model(model: StateEmbeddingModel) -> dict[str, Any]:
    return {
        "feature_columns": list(model.feature_columns),
        "fill_values": list(model.fill_values),
        "means": list(model.means),
        "stds": list(model.stds),
        "embedding_columns": list(model.embedding_columns),
        "pca_components": [list(row) for row in list(model.pca_components)],
        "pca_mean": list(model.pca_mean),
        "explained_variance_ratio": list(model.explained_variance_ratio),
    }


def deserialize_state_embedding_model(payload: dict[str, Any]) -> StateEmbeddingModel:
    return StateEmbeddingModel(
        feature_columns=[str(value) for value in list(payload.get("feature_columns") or [])],
        fill_values=[float(value) for value in list(payload.get("fill_values") or [])],
        means=[float(value) for value in list(payload.get("means") or [])],
        stds=[float(value) for value in list(payload.get("stds") or [])],
        embedding_columns=[str(value) for value in list(payload.get("embedding_columns") or [])],
        pca_components=[list(map(float, row)) for row in list(payload.get("pca_components") or [])],
        pca_mean=[float(value) for value in list(payload.get("pca_mean") or [])],
        explained_variance_ratio=[float(value) for value in list(payload.get("explained_variance_ratio") or [])],
    )


def transform_scaled_state_frame(frame: pd.DataFrame, model: StateEmbeddingModel) -> pd.DataFrame:
    usable = [str(column) for column in model.feature_columns if str(column) in frame.columns]
    numeric = frame[usable].apply(pd.to_numeric, errors="coerce")
    fill_series = pd.Series(model.fill_values[: len(usable)], index=usable, dtype=float)
    numeric = numeric.fillna(fill_series)
    matrix = numeric.to_numpy(dtype=float)
    means = np.asarray(model.means[: len(usable)], dtype=float)
    stds = np.asarray(model.stds[: len(usable)], dtype=float)
    stds[stds == 0.0] = 1.0
    scaled = np.nan_to_num((matrix - means) / stds, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = np.clip(scaled, -1_000_000.0, 1_000_000.0)
    return pd.DataFrame(scaled, columns=usable, index=frame.index)


def compute_state_embedding(row: pd.Series | dict[str, Any], model: StateEmbeddingModel) -> np.ndarray:
    row_series = pd.Series(dict(row))
    values: list[float] = []
    for idx, column in enumerate(model.feature_columns):
        parsed = pd.to_numeric(pd.Series([row_series.get(column)]), errors="coerce").iloc[0]
        if pd.isna(parsed):
            parsed = float(model.fill_values[idx])
        values.append(float(parsed))
    vector = np.asarray(values, dtype=float)
    means = np.asarray(model.means, dtype=float)
    stds = np.asarray(model.stds, dtype=float)
    stds[stds == 0.0] = 1.0
    scaled = np.nan_to_num((vector - means) / stds, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = np.clip(scaled, -1_000_000.0, 1_000_000.0)
    if model.pca_components:
        components = np.nan_to_num(np.asarray(model.pca_components, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        pca_mean = np.nan_to_num(
            np.asarray(model.pca_mean, dtype=np.float64) if model.pca_mean else np.zeros(scaled.shape[0], dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        centered = np.nan_to_num(np.asarray(scaled, dtype=np.float64) - pca_mean, nan=0.0, posinf=0.0, neginf=0.0)
        centered = np.clip(centered, -10_000.0, 10_000.0)
        components = np.clip(components, -10.0, 10.0)
        embedded = np.nan_to_num(np.sum(centered[None, :] * components, axis=1), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(embedded, -1_000.0, 1_000.0)
    return np.clip(scaled, -1_000.0, 1_000.0)


def transform_state_frame(frame: pd.DataFrame, model: StateEmbeddingModel) -> pd.DataFrame:
    scaled_df = transform_scaled_state_frame(frame, model)
    matrix = scaled_df.to_numpy(dtype=float)
    if model.pca_components:
        components = np.nan_to_num(np.asarray(model.pca_components, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        pca_mean = np.nan_to_num(
            np.asarray(model.pca_mean, dtype=np.float64) if model.pca_mean else np.zeros(matrix.shape[1], dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        centered = np.nan_to_num(np.asarray(matrix, dtype=np.float64) - pca_mean[None, :], nan=0.0, posinf=0.0, neginf=0.0)
        centered = np.clip(centered, -10_000.0, 10_000.0)
        components = np.clip(components, -10.0, 10.0)
        embedded = np.nan_to_num(np.sum(centered[:, None, :] * components[None, :, :], axis=2), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        embedded = matrix
    embedded = np.clip(embedded, -1_000.0, 1_000.0)
    return pd.DataFrame(embedded, columns=model.embedding_columns, index=frame.index)
