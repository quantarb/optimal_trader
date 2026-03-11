"""Compatibility layer for feature-panel workflows."""

from typing import Any

import pandas as pd

from domain.features.panel import (
    REPRESENTATION_EMBEDDING_FAMILY_GROUPS,
    REPRESENTATION_EMBEDDING_MODEL_VERSION,
    SECTION_LABELS,
    SECTION_ORDER,
    _resolve_representation_embedding_backend as _domain_resolve_representation_embedding_backend,
    build_feature_family_coverage_row as _build_feature_family_coverage_row,
    feature_toggle_data,
    representation_embedding_config,
    representation_embedding_dataset_rows as _representation_embedding_dataset_rows,
    representation_embedding_grouped_feature_columns as _representation_embedding_grouped_feature_columns,
    representation_embedding_missing_value as _representation_embedding_missing_value,
    resolve_feature_date_window as _resolve_feature_date_window,
)
from domain.features.specs import RepresentationEmbeddingSpec
from workflows.features import build_feature_panel_for_symbols, build_feature_panel_frame_for_symbols


def _resolve_representation_embedding_backend(config: dict[str, Any]):
    spec = config if isinstance(config, RepresentationEmbeddingSpec) else RepresentationEmbeddingSpec(**dict(config or {}))
    return _domain_resolve_representation_embedding_backend(spec)


def _append_representation_embedding_columns(
    symbol_df: pd.DataFrame,
    grouped_feature_columns: dict[str, list[str]],
    *,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    embedding_spec = config if isinstance(config, RepresentationEmbeddingSpec) else RepresentationEmbeddingSpec(**dict(config or {}))
    if symbol_df.empty or not embedding_spec.enabled:
        return symbol_df, [], {
            "enabled": False,
            "columns": [],
            "dimension": 0,
            "model_name": str(embedding_spec.model_name),
            "model_version": str(embedding_spec.model_version),
            "store_dir": str(embedding_spec.store_dir),
        }

    build_dataset_embeddings, encoder = _resolve_representation_embedding_backend(embedding_spec.to_dict())
    dataset_rows = _representation_embedding_dataset_rows(symbol_df, grouped_feature_columns)
    if not dataset_rows:
        return symbol_df, [], {
            "enabled": False,
            "columns": [],
            "dimension": 0,
            "model_name": str(getattr(encoder, "model_name", embedding_spec.model_name)),
            "model_version": str(getattr(encoder, "model_version", embedding_spec.model_version)),
            "store_dir": str(embedding_spec.store_dir),
        }

    embedding_rows = build_dataset_embeddings(
        dataset_rows,
        encoder=encoder,
        store_dir=str(embedding_spec.store_dir),
    )
    first_vector_value = embedding_rows[0].get("embedding_vector")
    first_vector = list(first_vector_value) if first_vector_value is not None else []
    embedding_columns = [f"{embedding_spec.column_prefix}{idx}" for idx in range(len(first_vector))]
    embedding_df = pd.DataFrame(
        [
            {column: float(vector[idx]) for idx, column in enumerate(embedding_columns)}
            for vector in [
                list(item.get("embedding_vector")) if item.get("embedding_vector") is not None else []
                for item in embedding_rows
            ]
        ]
    )
    augmented = pd.concat([symbol_df.reset_index(drop=True), embedding_df], axis=1)
    return augmented, embedding_columns, {
        "enabled": bool(embedding_columns),
        "columns": list(embedding_columns),
        "dimension": len(embedding_columns),
        "model_name": str(getattr(encoder, "model_name", embedding_spec.model_name)),
        "model_version": str(getattr(encoder, "model_version", embedding_spec.model_version)),
        "store_dir": str(embedding_spec.store_dir),
        "family_groups": {key: list(value) for key, value in REPRESENTATION_EMBEDDING_FAMILY_GROUPS.items()},
    }

__all__ = [
    "REPRESENTATION_EMBEDDING_FAMILY_GROUPS",
    "REPRESENTATION_EMBEDDING_MODEL_VERSION",
    "SECTION_LABELS",
    "SECTION_ORDER",
    "_append_representation_embedding_columns",
    "_build_feature_family_coverage_row",
    "_resolve_representation_embedding_backend",
    "_representation_embedding_dataset_rows",
    "_representation_embedding_grouped_feature_columns",
    "_representation_embedding_missing_value",
    "_resolve_feature_date_window",
    "build_feature_panel_for_symbols",
    "build_feature_panel_frame_for_symbols",
    "feature_toggle_data",
    "representation_embedding_config",
]
