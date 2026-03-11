"""Feature-engineering domain contracts and pure implementations."""

from domain.features.composition import merge_feature_sets
from domain.features.interfaces import FeatureBuilder, FeaturePanelBuilder
from domain.features.panel import (
    REPRESENTATION_EMBEDDING_FAMILY_GROUPS,
    REPRESENTATION_EMBEDDING_MODEL_VERSION,
    SECTION_LABELS,
    SECTION_ORDER,
    append_representation_embedding_columns,
    build_feature_family_coverage_row,
    feature_toggle_data,
    needed_sparse_sections,
    representation_embedding_config,
    representation_embedding_dataset_rows,
    representation_embedding_grouped_feature_columns,
    resolve_feature_date_window,
)
from domain.features.specs import BuiltFeatureSet, FeatureBuildSpec, FeatureToggleSpec, RepresentationEmbeddingSpec
from domain.features.technical import (
    BASE_PRICE_COLS,
    FeaturesResult,
    build_price_technical_features,
    compute_features_worldclass,
    load_or_compute_features_daily,
)

__all__ = [
    "BASE_PRICE_COLS",
    "BuiltFeatureSet",
    "FeatureBuildSpec",
    "FeatureBuilder",
    "FeaturePanelBuilder",
    "FeatureToggleSpec",
    "FeaturesResult",
    "REPRESENTATION_EMBEDDING_FAMILY_GROUPS",
    "REPRESENTATION_EMBEDDING_MODEL_VERSION",
    "RepresentationEmbeddingSpec",
    "SECTION_LABELS",
    "SECTION_ORDER",
    "append_representation_embedding_columns",
    "build_feature_family_coverage_row",
    "build_price_technical_features",
    "compute_features_worldclass",
    "feature_toggle_data",
    "load_or_compute_features_daily",
    "merge_feature_sets",
    "needed_sparse_sections",
    "representation_embedding_config",
    "representation_embedding_dataset_rows",
    "representation_embedding_grouped_feature_columns",
    "resolve_feature_date_window",
]
