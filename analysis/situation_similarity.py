from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.models import Artifact
from pipeline.service_runtime import read_frame_artifact
from .state_embedding import StateEmbeddingModel, compute_state_embedding, deserialize_state_embedding_model


@dataclass
class MarketSituationClusterBundle:
    artifact: Artifact
    assignments: pd.DataFrame
    summary: dict[str, Any]
    embedding_model: StateEmbeddingModel
    embedding_columns: list[str]
    cluster_rows: list[dict[str, Any]]
    cluster_lookup: dict[str, dict[str, Any]]


def _read_summary_payload(artifact: Artifact) -> dict[str, Any]:
    summary_uri = str((artifact.metadata or {}).get("summary_json_uri") or "").strip()
    if summary_uri:
        summary_path = Path(summary_uri)
        if summary_path.exists() and summary_path.is_file():
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload
            except Exception:
                pass
    if artifact.content and isinstance(artifact.content, dict):
        return dict(artifact.content)
    return {}


def resolve_market_situation_artifact(*, artifact_id: int = 0) -> Artifact | None:
    if int(artifact_id or 0) > 0:
        return Artifact.objects.filter(pk=int(artifact_id), artifact_type="MARKET_SITUATION_CLUSTER").select_related("pipeline_run").first()
    return Artifact.objects.filter(artifact_type="MARKET_SITUATION_CLUSTER").select_related("pipeline_run").order_by("-created_at", "-id").first()


def load_market_situation_cluster_artifact(artifact: Artifact) -> MarketSituationClusterBundle:
    if str(artifact.artifact_type) != "MARKET_SITUATION_CLUSTER":
        raise ValueError(f"Artifact #{artifact.id} is not a MARKET_SITUATION_CLUSTER artifact.")
    uri = str(artifact.uri or "").strip()
    if not uri:
        raise ValueError(f"Artifact #{artifact.id} has no assignments CSV path.")
    if not Path(uri).exists():
        raise ValueError(f"Artifact #{artifact.id} assignments file does not exist.")
    assignments = read_frame_artifact(artifact)
    if assignments.empty:
        raise ValueError(f"Artifact #{artifact.id} contains no market situation assignments.")
    assignments["date"] = pd.to_datetime(assignments["date"], errors="coerce")
    assignments["symbol"] = assignments["symbol"].astype(str).str.strip().str.upper()
    assignments = assignments.dropna(subset=["date", "symbol", "cluster_id"]).reset_index(drop=True)
    summary = _read_summary_payload(artifact)
    embedding_payload = dict(summary.get("embedding") or {})
    embedding_model = deserialize_state_embedding_model(embedding_payload)
    embedding_columns = [str(column) for column in assignments.columns if str(column).startswith("emb_")]
    cluster_rows = [dict(row) for row in list(summary.get("clusters") or []) if isinstance(row, dict)]
    cluster_lookup = {str(row.get("cluster_id") or ""): dict(row) for row in cluster_rows if str(row.get("cluster_id") or "")}
    return MarketSituationClusterBundle(
        artifact=artifact,
        assignments=assignments,
        summary=summary,
        embedding_model=embedding_model,
        embedding_columns=embedding_columns,
        cluster_rows=cluster_rows,
        cluster_lookup=cluster_lookup,
    )


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def find_nearest_clusters(
    state_embedding: np.ndarray,
    bundle: MarketSituationClusterBundle,
    *,
    side: str | None = None,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    cluster_rows = list(bundle.cluster_rows or [])
    if side:
        cluster_rows = [row for row in cluster_rows if str(row.get("side") or "").strip().lower() == str(side).strip().lower()]
    if not cluster_rows:
        return []
    centroid_matrix = np.asarray([list(row.get("centroid") or []) for row in cluster_rows], dtype=float)
    centroid_matrix = _normalize_matrix(centroid_matrix)
    query = np.asarray(state_embedding, dtype=float).reshape(1, -1)
    query = _normalize_matrix(query)[0]
    similarities = np.nan_to_num(np.sum(centroid_matrix * query[None, :], axis=1), nan=-1.0, posinf=-1.0, neginf=-1.0)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(cluster_rows):
        item = dict(row)
        item["similarity_score"] = round(float(similarities[idx]), 6)
        rows.append(item)
    rows.sort(key=lambda item: float(item.get("similarity_score") or -1.0), reverse=True)
    return rows[: max(int(top_n), 1)]


def find_similar_historical_states(
    cluster_id: str,
    state_embedding: np.ndarray,
    bundle: MarketSituationClusterBundle,
    *,
    k: int = 10,
    before_date: str | None = None,
    exclude_symbol: str | None = None,
    exclude_date: str | None = None,
) -> list[dict[str, Any]]:
    assignments = bundle.assignments[bundle.assignments["cluster_id"].astype(str) == str(cluster_id)].copy()
    if before_date:
        assignments = assignments[assignments["date"] < pd.Timestamp(str(before_date))].copy()
    if exclude_symbol and exclude_date:
        assignments = assignments[
            ~(
                (assignments["symbol"] == str(exclude_symbol).strip().upper())
                & (assignments["date"] == pd.Timestamp(str(exclude_date)))
            )
        ].copy()
    if assignments.empty:
        return []
    matrix = assignments[bundle.embedding_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    matrix = _normalize_matrix(matrix)
    query = _normalize_matrix(np.asarray(state_embedding, dtype=float).reshape(1, -1))[0]
    similarities = np.nan_to_num(np.sum(matrix * query[None, :], axis=1), nan=-1.0, posinf=-1.0, neginf=-1.0)
    assignments["similarity_score"] = similarities
    assignments = assignments.sort_values("similarity_score", ascending=False).head(max(int(k), 1))
    rows: list[dict[str, Any]] = []
    for row in assignments.to_dict(orient="records"):
        item = dict(row)
        item["date"] = str(pd.Timestamp(item["date"]).date())
        item["similarity_score"] = round(float(item.get("similarity_score") or 0.0), 6)
        cluster_row = dict(bundle.cluster_lookup.get(str(item.get("cluster_id") or "")) or {})
        item["cluster_description"] = str(cluster_row.get("description") or item.get("cluster_description") or "")
        rows.append(item)
    return rows


def assign_market_situation_cluster(
    row: pd.Series | dict[str, Any],
    bundle: MarketSituationClusterBundle,
    *,
    side: str | None = None,
) -> dict[str, Any]:
    embedding = compute_state_embedding(row, bundle.embedding_model)
    nearest = find_nearest_clusters(embedding, bundle, side=side, top_n=1)
    return dict(nearest[0]) if nearest else {}
