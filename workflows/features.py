from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from data.historical_prices import load_adjusted_price_frames
from domain.features import FeatureBuildSpec, REPRESENTATION_EMBEDDING_MODEL_VERSION
from settings import BASE_DIR
from workflows.feature_runtime import (
    FeaturePanelDependencies,
    build_feature_panel_environment,
    build_feature_panel_frame,
)


def build_feature_panel_frame_for_symbols(
    *,
    symbols: list[str],
    spec: FeatureBuildSpec | None = None,
    config: dict[str, Any] | None = None,
    progress_callback=None,
    performance_tracer=None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Build a multi-symbol feature panel using typed research specs."""

    build_spec = spec or FeatureBuildSpec.from_mapping(
        config,
        default_store_dir=str(Path(BASE_DIR) / "data" / "embedding_store"),
        default_model_version=REPRESENTATION_EMBEDDING_MODEL_VERSION,
    )
    environment = build_feature_panel_environment(
        symbols=symbols,
        build_spec=build_spec,
        dependencies=FeaturePanelDependencies(load_price_frames=load_adjusted_price_frames),
        performance_tracer=performance_tracer,
    )
    return build_feature_panel_frame(
        environment=environment,
        progress_callback=progress_callback,
        performance_tracer=performance_tracer,
    )


def build_feature_panel_for_symbols(
    *,
    symbols: list[str],
    spec: FeatureBuildSpec | None = None,
    config: dict[str, Any] | None = None,
    progress_callback=None,
    performance_tracer=None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    frame, fieldnames, metadata = build_feature_panel_frame_for_symbols(
        symbols=symbols,
        spec=spec,
        config=config,
        progress_callback=progress_callback,
        performance_tracer=performance_tracer,
    )
    return frame.to_dict(orient="records"), fieldnames, metadata
