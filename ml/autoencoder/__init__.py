from .config import AutoEncoderConfig
from .adapter import TorchAutoEncoder
from .vector_db import (
    LatentVectorDB,
    build_latent_vector_frame,
    build_latent_vector_db,
    query_latent_neighbors,
    select_natural_k_by_silhouette,
    select_natural_k_fast_elbow,
    cluster_latent_kmeans,
    summarize_cluster_performance,
    build_ae_cluster_report,
    explain_cluster_feature_uniqueness,
    explain_cluster_ae_feature_breaks,
)

__all__ = [
    "AutoEncoderConfig",
    "TorchAutoEncoder",
    "LatentVectorDB",
    "build_latent_vector_frame",
    "build_latent_vector_db",
    "query_latent_neighbors",
    "select_natural_k_by_silhouette",
    "select_natural_k_fast_elbow",
    "cluster_latent_kmeans",
    "summarize_cluster_performance",
    "build_ae_cluster_report",
    "explain_cluster_feature_uniqueness",
    "explain_cluster_ae_feature_breaks",
    "run_ae_manifold_event_diagnostics",
]

from .diagnostics import run_ae_manifold_event_diagnostics
