from __future__ import annotations

from importlib import import_module

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



_LAZY_EXPORTS = {
    "AutoEncoderConfig": ("ml.autoencoder.config", "AutoEncoderConfig"),
    "TorchAutoEncoder": ("ml.autoencoder.adapter", "TorchAutoEncoder"),
    "LatentVectorDB": ("ml.autoencoder.vector_db", "LatentVectorDB"),
    "build_latent_vector_frame": ("ml.autoencoder.vector_db", "build_latent_vector_frame"),
    "build_latent_vector_db": ("ml.autoencoder.vector_db", "build_latent_vector_db"),
    "query_latent_neighbors": ("ml.autoencoder.vector_db", "query_latent_neighbors"),
    "select_natural_k_by_silhouette": ("ml.autoencoder.vector_db", "select_natural_k_by_silhouette"),
    "select_natural_k_fast_elbow": ("ml.autoencoder.vector_db", "select_natural_k_fast_elbow"),
    "cluster_latent_kmeans": ("ml.autoencoder.vector_db", "cluster_latent_kmeans"),
    "summarize_cluster_performance": ("ml.autoencoder.vector_db", "summarize_cluster_performance"),
    "build_ae_cluster_report": ("ml.autoencoder.vector_db", "build_ae_cluster_report"),
    "explain_cluster_feature_uniqueness": ("ml.autoencoder.vector_db", "explain_cluster_feature_uniqueness"),
    "explain_cluster_ae_feature_breaks": ("ml.autoencoder.vector_db", "explain_cluster_ae_feature_breaks"),
    "run_ae_manifold_event_diagnostics": ("ml.autoencoder.diagnostics", "run_ae_manifold_event_diagnostics"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(str(name))
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
