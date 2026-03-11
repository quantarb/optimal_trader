from __future__ import annotations

import csv
import json
import math
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from django.utils import timezone
from sklearn.cluster import MiniBatchKMeans

from settings import BASE_DIR

from .cluster_explanations import build_cluster_feature_explanations
from .cluster_outcomes import compute_cluster_outcome_stats
from pipeline.models import Artifact, PipelineRun
from .oracle_state_dataset import build_oracle_state_dataset
from .state_embedding import (
    compute_state_embedding,
    fit_state_embedding_model,
    serialize_state_embedding_model,
    transform_scaled_state_frame,
    transform_state_frame,
)


ARTIFACT_DIR = Path(BASE_DIR) / "data" / "pipeline_artifacts"
MARKET_SITUATION_SCHEMA_VERSION = 1


def _ensure_artifact_dir() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    _ensure_artifact_dir()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return str(path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    _ensure_artifact_dir()
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return str(path)


def _cluster_count(side_rows: int, *, max_clusters: int, min_cluster_size: int) -> int:
    if side_rows <= 0:
        return 0
    if side_rows <= max(2, int(min_cluster_size)):
        return 1
    heuristic = int(round(math.sqrt(float(side_rows) / float(max(min_cluster_size, 5)))))
    heuristic = max(1, heuristic * 2)
    return max(1, min(int(max_clusters), heuristic))


def _initial_centroids(matrix: np.ndarray, cluster_count: int) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("Cannot initialize market-situation centroids from an empty matrix.")
    if int(cluster_count) <= 1:
        return np.asarray([matrix.mean(axis=0)], dtype=float)
    unique_rows = np.unique(np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0), axis=0)
    source = unique_rows if unique_rows.shape[0] >= cluster_count else matrix
    if source.shape[0] <= cluster_count:
        return np.asarray(source[:cluster_count], dtype=float)
    picks = np.linspace(0, source.shape[0] - 1, num=cluster_count, dtype=int)
    return np.asarray(source[picks], dtype=float)


def fit_market_situation_clusters(
    *,
    strategy_artifact: Artifact | None = None,
    feature_artifact: Artifact | None = None,
    label_artifact: Artifact | None = None,
    prediction_artifacts: Sequence[Artifact] = (),
    pca_components: int = 8,
    max_clusters: int = 12,
    min_cluster_size: int = 25,
    random_state: int = 42,
    start_date: str | None = None,
    end_date: str | None = None,
    label_ks: Sequence[int] = (),
    min_abs_trade_return: float | None = None,
) -> dict[str, Any]:
    oracle_df, dataset_meta = build_oracle_state_dataset(
        strategy_artifact=strategy_artifact,
        feature_artifact=feature_artifact,
        label_artifact=label_artifact,
        prediction_artifacts=prediction_artifacts,
        start_date=start_date,
        end_date=end_date,
        label_ks=label_ks,
        min_abs_trade_return=min_abs_trade_return,
    )
    embedding_model = fit_state_embedding_model(
        oracle_df,
        feature_columns=dataset_meta.get("feature_columns") or [],
        pca_components=int(pca_components or 0),
    )
    embedded_df = transform_state_frame(oracle_df, embedding_model)
    scaled_df = transform_scaled_state_frame(oracle_df, embedding_model)

    assignments = oracle_df.copy().reset_index(drop=True)
    for column in embedded_df.columns:
        assignments[column] = embedded_df[column].to_numpy()

    cluster_code_base = 0
    centroid_rows: list[dict[str, Any]] = []
    side_values = assignments.get("side", pd.Series("long", index=assignments.index)).fillna("long").astype(str).str.strip().str.lower()
    assignments["side"] = side_values.replace("", "long")

    for side, side_index in assignments.groupby("side", observed=True).groups.items():
        subset = assignments.loc[list(side_index)].copy()
        subset_matrix = subset[embedding_model.embedding_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        subset_matrix = np.nan_to_num(subset_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        subset_matrix = np.clip(subset_matrix, -1_000.0, 1_000.0)
        cluster_count = _cluster_count(len(subset), max_clusters=int(max_clusters), min_cluster_size=int(min_cluster_size))
        if cluster_count <= 1:
            labels = np.zeros(len(subset), dtype=int)
            centroids = np.asarray([subset_matrix.mean(axis=0)], dtype=float)
        else:
            initial_centroids = _initial_centroids(subset_matrix, cluster_count)
            model = MiniBatchKMeans(
                n_clusters=cluster_count,
                random_state=int(random_state),
                batch_size=min(max(32, len(subset)), 4096),
                init=initial_centroids,
                n_init=1,
            )
            labels = model.fit_predict(subset_matrix)
            centroids = np.asarray(model.cluster_centers_, dtype=float)
        centroids = np.nan_to_num(centroids, nan=0.0, posinf=0.0, neginf=0.0)
        centroids = np.clip(centroids, -1_000.0, 1_000.0)
        distances = np.linalg.norm(subset_matrix - centroids[labels], axis=1)
        similarities = 1.0 / (1.0 + np.nan_to_num(distances, nan=10.0, posinf=10.0, neginf=10.0))
        for local_label in sorted(set(int(value) for value in labels.tolist())):
            cluster_id = f"{side}_{int(local_label):02d}"
            cluster_code = int(cluster_code_base + local_label)
            centroid_rows.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_code": cluster_code,
                    "side": side,
                    "centroid": [float(value) for value in centroids[int(local_label)].tolist()],
                }
            )
        cluster_code_base += max(1, cluster_count) * 100
        row_cluster_ids = [f"{side}_{int(value):02d}" for value in labels.tolist()]
        row_cluster_codes = [int(cluster_code_base - max(1, cluster_count) * 100 + int(value)) for value in labels.tolist()]
        assignments.loc[subset.index, "cluster_id"] = row_cluster_ids
        assignments.loc[subset.index, "cluster_code"] = row_cluster_codes
        assignments.loc[subset.index, "cluster_distance"] = distances
        assignments.loc[subset.index, "cluster_similarity"] = similarities

    assignments["cluster_code"] = pd.to_numeric(assignments["cluster_code"], errors="coerce").fillna(0).astype(int)
    assignments["cluster_distance"] = pd.to_numeric(assignments["cluster_distance"], errors="coerce").fillna(0.0)
    assignments["cluster_similarity"] = pd.to_numeric(assignments["cluster_similarity"], errors="coerce").fillna(0.0)

    outcome_stats = compute_cluster_outcome_stats(assignments)
    explanation_map = build_cluster_feature_explanations(
        scaled_feature_df=scaled_df,
        assignments_df=assignments,
        feature_columns=embedding_model.feature_columns,
        top_n=3,
    )
    stats_lookup = {str(row["cluster_id"]): dict(row) for row in outcome_stats.to_dict(orient="records")}

    cluster_rows: list[dict[str, Any]] = []
    cluster_lookup: dict[str, dict[str, Any]] = {}
    for centroid_row in centroid_rows:
        cluster_id = str(centroid_row["cluster_id"])
        stats_row = dict(stats_lookup.get(cluster_id) or {})
        explain_row = dict(explanation_map.get(cluster_id) or {})
        cluster_assignments = assignments[assignments["cluster_id"].astype(str) == cluster_id].copy()
        examples = []
        if not cluster_assignments.empty:
            preview = cluster_assignments.sort_values(["trade_return", "cluster_similarity"], ascending=[False, False]).head(3)
            for row in preview.to_dict(orient="records"):
                examples.append(
                    {
                        "date": str(pd.Timestamp(row["date"]).date()),
                        "symbol": str(row.get("symbol") or ""),
                        "trade_return": round(float(row.get("trade_return") or 0.0), 6),
                    }
                )
        item = {
            "cluster_id": cluster_id,
            "cluster_code": int(centroid_row["cluster_code"]),
            "side": str(centroid_row["side"]),
            "cluster_size": int(stats_row.get("sample_size") or len(cluster_assignments)),
            "description": str(explain_row.get("description") or "Market Situation Cluster"),
            "feature_signature": list(explain_row.get("feature_signature") or []),
            "feature_signature_rows": list(explain_row.get("family_rows") or []),
            "typical_features": list(explain_row.get("typical_features") or []),
            "outcome_statistics": {
                "median_return": float(stats_row.get("median_return") or 0.0),
                "mean_return": float(stats_row.get("mean_return") or 0.0),
                "win_rate": float(stats_row.get("win_rate") or 0.0),
                "worst_case": float(stats_row.get("worst_case") or 0.0),
                "best_case": float(stats_row.get("best_case") or 0.0),
                "avg_hold_days": float(stats_row.get("avg_hold_days") or 0.0),
                "return_std": float(stats_row.get("return_std") or 0.0),
                "yearly_median_return_std": float(stats_row.get("yearly_median_return_std") or 0.0),
                "yearly_win_rate_std": float(stats_row.get("yearly_win_rate_std") or 0.0),
            },
            "example_historical_dates": examples,
            "centroid": list(centroid_row["centroid"]),
        }
        cluster_rows.append(item)
        cluster_lookup[cluster_id] = item

    assignments["cluster_description"] = assignments["cluster_id"].map(lambda value: str((cluster_lookup.get(str(value)) or {}).get("description") or ""))
    assignments["cluster_family_signature"] = assignments["cluster_id"].map(lambda value: ", ".join((cluster_lookup.get(str(value)) or {}).get("feature_signature") or []))

    summary = {
        "kind": "market_situation_cluster",
        "schema_version": MARKET_SITUATION_SCHEMA_VERSION,
        "summary": {
            "rows": int(len(assignments)),
            "clusters": int(len(cluster_rows)),
            "symbols": int(assignments["symbol"].nunique()),
            "date_start": str(pd.to_datetime(assignments["date"], errors="coerce").min().date()),
            "date_end": str(pd.to_datetime(assignments["date"], errors="coerce").max().date()),
            "embedding_dims": int(len(embedding_model.embedding_columns)),
            "sides": sorted({str(value) for value in assignments["side"].dropna().astype(str).tolist()}),
        },
        "dataset": dict(dataset_meta),
        "embedding": serialize_state_embedding_model(embedding_model),
        "clusters": cluster_rows,
    }
    return {
        "assignments": assignments,
        "summary": summary,
    }


def assign_market_situation_cluster(
    row: pd.Series | dict[str, Any],
    *,
    summary_payload: dict[str, Any],
) -> dict[str, Any]:
    from .situation_similarity import find_nearest_clusters
    from .situation_similarity import MarketSituationClusterBundle
    from .state_embedding import deserialize_state_embedding_model

    cluster_rows = [dict(item) for item in list(summary_payload.get("clusters") or []) if isinstance(item, dict)]
    embedding_model = deserialize_state_embedding_model(dict(summary_payload.get("embedding") or {}))
    dummy_bundle = MarketSituationClusterBundle(
        artifact=Artifact(artifact_type="MARKET_SITUATION_CLUSTER", key="memory"),
        assignments=pd.DataFrame(),
        summary=summary_payload,
        embedding_model=embedding_model,
        embedding_columns=list(embedding_model.embedding_columns),
        cluster_rows=cluster_rows,
        cluster_lookup={str(item.get("cluster_id") or ""): dict(item) for item in cluster_rows},
    )
    nearest = find_nearest_clusters(
        compute_state_embedding(row, embedding_model),
        dummy_bundle,
        side=str(dict(row).get("side") or "").strip().lower() or None,
        top_n=1,
    )
    return dict(nearest[0]) if nearest else {}


def materialize_market_situation_cluster_artifact(
    *,
    output_basename: str,
    clustering_payload: dict[str, Any],
    strategy_artifact: Artifact | None = None,
    feature_artifact: Artifact | None = None,
    label_artifact: Artifact | None = None,
    prediction_artifacts: Sequence[Artifact] = (),
) -> Artifact:
    summary = dict(clustering_payload.get("summary") or {})
    assignments_df = clustering_payload.get("assignments")
    if not isinstance(assignments_df, pd.DataFrame) or assignments_df.empty:
        raise ValueError("Clustering payload did not include any assignments rows.")

    basename = str(output_basename).strip() or f"market_situation_cluster_{uuid.uuid4().hex[:8]}"
    summary_path = ARTIFACT_DIR / f"{basename}.json"
    assignments_path = ARTIFACT_DIR / f"{basename}.csv"

    assignment_rows = assignments_df.copy()
    assignment_rows["date"] = pd.to_datetime(assignment_rows["date"], errors="coerce").dt.date.astype(str)
    fieldnames = list(assignment_rows.columns)
    _write_csv(assignments_path, assignment_rows.to_dict(orient="records"), fieldnames)
    _write_json(summary_path, summary)

    pipeline_run = PipelineRun.objects.create(
        name=basename,
        requested_job="market_situation_clusters",
        mode=PipelineRun.Mode.STRICT,
        status=PipelineRun.Status.SUCCEEDED,
        started_at=timezone.now(),
        finished_at=timezone.now(),
        config={
            "strategy_artifact_id": int(strategy_artifact.id) if strategy_artifact is not None else 0,
            "feature_artifact_id": int(feature_artifact.id) if feature_artifact is not None else 0,
            "label_artifact_id": int(label_artifact.id) if label_artifact is not None else 0,
            "prediction_artifact_ids": [int(artifact.id) for artifact in prediction_artifacts],
        },
    )
    return Artifact.objects.create(
        pipeline_run=pipeline_run,
        artifact_type="MARKET_SITUATION_CLUSTER",
        key=f"market_situation_cluster_{uuid.uuid4().hex}",
        uri=str(assignments_path),
        content={
            "kind": "market_situation_cluster",
            "schema_version": MARKET_SITUATION_SCHEMA_VERSION,
            **dict(summary.get("summary") or {}),
        },
        metadata={
            "summary_json_uri": str(summary_path),
            "strategy_artifact_id": int(strategy_artifact.id) if strategy_artifact is not None else 0,
            "feature_artifact_id": int(feature_artifact.id) if feature_artifact is not None else 0,
            "label_artifact_id": int(label_artifact.id) if label_artifact is not None else 0,
            "prediction_artifact_ids": [int(artifact.id) for artifact in prediction_artifacts],
            "schema_version": MARKET_SITUATION_SCHEMA_VERSION,
        },
    )
