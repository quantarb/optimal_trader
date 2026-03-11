from __future__ import annotations

from importlib import import_module


__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_MODEL_VERSION",
    "EmbeddingKey",
    "SentenceTransformerEncoder",
    "build_dataset_embeddings",
    "build_row_embedding",
    "count_text_tokens",
    "encode_family",
    "fit_text_to_token_limit",
    "load_embedding",
    "max_text_tokens",
    "measure_text_tokens",
    "pool_embeddings",
    "save_embedding",
    "serialize_family",
]

_EXPORT_TO_MODULE = {
    "DEFAULT_MODEL_NAME": "encoder",
    "DEFAULT_MODEL_VERSION": "encoder",
    "EmbeddingKey": "embedding_store",
    "SentenceTransformerEncoder": "encoder",
    "build_dataset_embeddings": "pipeline",
    "build_row_embedding": "pipeline",
    "count_text_tokens": "encoder",
    "encode_family": "encoder",
    "fit_text_to_token_limit": "encoder",
    "load_embedding": "embedding_store",
    "max_text_tokens": "encoder",
    "measure_text_tokens": "encoder",
    "pool_embeddings": "pooling",
    "save_embedding": "embedding_store",
    "serialize_family": "serialization",
}


def __getattr__(name: str):
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{module_name}")
    return getattr(module, name)
