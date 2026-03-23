from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from domain.models import ArtifactTrainingDatasetSpec
from domain.models.datasets import dedupe_label_frame, feature_columns_from_frame, filter_frame_by_date
from domain.models.feature_families import infer_feature_family_columns
from pipeline.models import Artifact
from pipeline.service_runtime import read_frame_artifact

from .multitask import derive_oracle_cluster_labels


def load_artifact_csv_frame(artifact: Artifact) -> pd.DataFrame:
    """Load a saved artifact frame with canonical date/symbol guards."""

    if not str(artifact.uri or "").strip():
        raise ValueError(f"Artifact #{artifact.id} has no file path.")
    df = read_frame_artifact(artifact)
    if df.empty:
        return df
    if "date" not in df.columns or "symbol" not in df.columns:
        raise ValueError(f"Artifact #{artifact.id} must contain 'date' and 'symbol' columns.")
    return df.dropna(subset=["date", "symbol"])


def _coverage_metadata(df: pd.DataFrame, feature_cols: Sequence[str]) -> dict[str, Any]:
    usable_cols = [str(col) for col in list(feature_cols) if str(col) in df.columns]
    if df.empty or not usable_cols or "date" not in df.columns:
        return {"coverage_start_date": "", "coverage_end_date": "", "coverage_rows": 0}
    mask = df[usable_cols].notna().any(axis=1)
    if not mask.any():
        return {"coverage_start_date": "", "coverage_end_date": "", "coverage_rows": 0}
    dates = pd.to_datetime(df.loc[mask, "date"], errors="coerce").dropna()
    if dates.empty:
        return {
            "coverage_start_date": "",
            "coverage_end_date": "",
            "coverage_rows": int(mask.sum()),
        }
    return {
        "coverage_start_date": str(dates.min().date().isoformat()),
        "coverage_end_date": str(dates.max().date().isoformat()),
        "coverage_rows": int(mask.sum()),
    }


def _rename_panel_columns(df: pd.DataFrame, *, prefix: str) -> tuple[pd.DataFrame, list[str]]:
    rename_map: dict[str, str] = {}
    feature_cols: list[str] = []
    for col in df.columns:
        if col in {"date", "symbol"}:
            continue
        renamed = f"{prefix}{col}"
        rename_map[col] = renamed
        feature_cols.append(renamed)
    if not rename_map:
        return df[["date", "symbol"]].copy(), []
    return df.rename(columns=rename_map), feature_cols


def _join_feature_panels(
    base_feature_artifact: Artifact,
    *,
    extra_artifacts: Sequence[Artifact] = (),
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    base_df = load_artifact_csv_frame(base_feature_artifact)
    if base_df.empty:
        raise ValueError("Selected feature artifact has no rows.")
    joined = base_df.copy()
    feature_cols = feature_columns_from_frame(base_df)
    source_artifact_ids = [int(base_feature_artifact.id)]
    extra_sources: list[dict[str, Any]] = []

    for artifact in extra_artifacts:
        panel_df = load_artifact_csv_frame(artifact)
        if panel_df.empty:
            continue
        prefix = f"{str(artifact.artifact_type or '').strip().lower()}_{int(artifact.id)}__"
        renamed_df, renamed_cols = _rename_panel_columns(panel_df, prefix=prefix)
        joined = joined.merge(renamed_df, on=["date", "symbol"], how="left")
        feature_cols.extend(renamed_cols)
        source_artifact_ids.append(int(artifact.id))
        extra_sources.append(
            {
                "artifact_id": int(artifact.id),
                "artifact_type": str(artifact.artifact_type),
                "prefix": prefix,
                "columns": renamed_cols,
            }
        )

    feature_cols = list(dict.fromkeys(feature_cols))
    return joined, feature_cols, {
        "base_feature_artifact_id": int(base_feature_artifact.id),
        "panel_artifact_ids": source_artifact_ids,
        "extra_panel_sources": extra_sources,
    }


def build_feature_frame_from_artifacts(
    *,
    base_feature_artifact: Artifact,
    extra_panel_artifacts: Sequence[Artifact] = (),
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Load and join a base feature artifact with optional state panels."""

    return _join_feature_panels(base_feature_artifact, extra_artifacts=extra_panel_artifacts)


def _select_feature_columns(
    *,
    feature_cols: Sequence[str],
    spec: ArtifactTrainingDatasetSpec,
) -> tuple[list[str], list[str], list[str], dict[str, list[str]]]:
    family_map = infer_feature_family_columns(feature_cols)
    all_feature_cols = list(feature_cols)
    selected_feature_cols = list(feature_cols)
    selected_families = list(spec.selected_feature_families())
    if selected_families:
        selected_feature_cols = []
        for family_name in selected_families:
            selected_feature_cols.extend(list(family_map.get(family_name) or []))
        selected_feature_cols = list(dict.fromkeys(selected_feature_cols))
        if not selected_feature_cols:
            raise ValueError(f"Feature artifact does not contain usable columns for families {selected_families!r}.")
    return all_feature_cols, selected_feature_cols, selected_families, family_map


def _apply_label_filters(
    *,
    label_df: pd.DataFrame,
    spec: ArtifactTrainingDatasetSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected_label_ks = list(spec.selected_label_ks())
    if selected_label_ks and "k" in label_df.columns:
        label_df = label_df[pd.to_numeric(label_df["k"], errors="coerce").isin(selected_label_ks)].copy()
    label_rows_before_trade_filters = int(len(label_df))
    if spec.min_abs_trade_return not in (None, "") and "trade_return" in label_df.columns:
        min_abs_value = max(0.0, float(spec.min_abs_trade_return))
        label_df["trade_return"] = pd.to_numeric(label_df["trade_return"], errors="coerce")
        label_df = label_df[label_df["trade_return"].abs() >= min_abs_value].copy()
    if spec.max_hold_days not in (None, "") and "hold_days" in label_df.columns:
        max_hold_value = max(1, int(spec.max_hold_days))
        label_df["hold_days"] = pd.to_numeric(label_df["hold_days"], errors="coerce")
        label_df = label_df[label_df["hold_days"].fillna(max_hold_value + 1) <= max_hold_value].copy()
    label_df = dedupe_label_frame(label_df)
    selected_cluster_keys = list(spec.selected_oracle_cluster_keys())
    cluster_rows_before_filter = int(len(label_df))
    if selected_cluster_keys:
        label_df["oracle_cluster_key"] = derive_oracle_cluster_labels(label_df)
        label_df = label_df[label_df["oracle_cluster_key"].isin(selected_cluster_keys)].copy()
        if label_df.empty:
            raise ValueError("Selected oracle cluster keys produced no label rows in the requested training window.")
    if label_df.empty:
        raise ValueError("Selected label artifact has no rows.")
    return label_df, {
        "label_k": spec.label_k,
        "label_ks": selected_label_ks,
        "label_rows_before_trade_filters": label_rows_before_trade_filters,
        "label_rows_after_filters": int(len(label_df)),
        "oracle_cluster_keys": selected_cluster_keys,
        "oracle_cluster_scope": "specialist" if selected_cluster_keys else "generalist",
        "cluster_rows_before_filter": cluster_rows_before_filter,
        "cluster_rows_after_filter": int(len(label_df)),
    }


def _apply_sample_weights(joined: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    if "sample_weight" not in joined.columns:
        joined["sample_weight"] = 1.0
    if mode == "trade_return_abs" and "trade_return" in joined.columns:
        weights = pd.to_numeric(joined["trade_return"], errors="coerce").abs().fillna(0.0)
        joined["sample_weight"] = (1.0 + weights).astype(float)
    return joined


def build_training_frame_from_panel_artifacts(
    *,
    base_feature_artifact: Artifact,
    label_artifact: Artifact,
    spec: ArtifactTrainingDatasetSpec,
    extra_panel_artifacts: Sequence[Artifact] = (),
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Build a trainable panel from saved feature and label artifacts."""

    joined_features, feature_cols, panel_meta = _join_feature_panels(
        base_feature_artifact,
        extra_artifacts=extra_panel_artifacts,
    )
    label_df = load_artifact_csv_frame(label_artifact)
    all_feature_cols, selected_feature_cols, selected_families, family_map = _select_feature_columns(
        feature_cols=feature_cols,
        spec=spec,
    )
    coverage_before = _coverage_metadata(joined_features, selected_feature_cols)
    joined_features = filter_frame_by_date(joined_features, start_date=spec.start_date, end_date=spec.end_date)
    label_df = filter_frame_by_date(label_df, start_date=spec.start_date, end_date=spec.end_date)
    label_df, label_meta = _apply_label_filters(label_df=label_df, spec=spec)
    joined = joined_features.merge(label_df, on=["date", "symbol"], how="inner", suffixes=("", "_label"))
    if joined.empty:
        raise ValueError("Selected feature and label artifacts have no overlapping (date, symbol) rows.")
    usable_feature_cols = [col for col in selected_feature_cols if col in joined.columns]
    missing_feature_policy = spec.normalized_missing_feature_policy()
    if usable_feature_cols:
        if missing_feature_policy == "complete_case":
            joined = joined.dropna(subset=usable_feature_cols).copy()
        else:
            joined = joined[joined[usable_feature_cols].notna().any(axis=1)].copy()
        if joined.empty:
            raise ValueError("Selected training window has no rows with usable feature-family coverage.")
    weight_mode = spec.normalized_sample_weight_mode()
    joined = _apply_sample_weights(joined, mode=weight_mode)
    joined = joined.sort_values(["symbol", "date"]).reset_index(drop=True)
    symbols = sorted(set(joined["symbol"].astype(str).tolist()))
    coverage_after = _coverage_metadata(joined, selected_feature_cols)
    return (
        joined,
        selected_feature_cols,
        {
            "feature_artifact_id": int(base_feature_artifact.id),
            "label_artifact_id": int(label_artifact.id),
            "symbols": symbols,
            "symbols_count": len(symbols),
            "joined_rows": int(len(joined)),
            "feature_df": joined_features,
            "label_df": label_df,
            "panel_artifact_ids": list(panel_meta["panel_artifact_ids"]),
            "extra_panel_sources": list(panel_meta["extra_panel_sources"]),
            "start_date": str(spec.start_date or ""),
            "end_date": str(spec.end_date or ""),
            "feature_family": ",".join(selected_families),
            "feature_families": list(selected_families),
            "feature_family_columns": list(selected_feature_cols),
            "available_feature_families": sorted(family_map.keys()),
            "all_feature_columns": all_feature_cols,
            "coverage_before": coverage_before,
            "coverage_after": coverage_after,
            "min_abs_trade_return": None if spec.min_abs_trade_return in (None, "") else float(spec.min_abs_trade_return),
            "max_hold_days": None if spec.max_hold_days in (None, "") else int(spec.max_hold_days),
            "sample_weight_mode": weight_mode,
            "missing_feature_policy": missing_feature_policy,
            **label_meta,
        },
    )
