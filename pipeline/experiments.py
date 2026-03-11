from __future__ import annotations

from itertools import product
from typing import Any

from ml.execution import infer_feature_family_columns, load_artifact_csv_frame
from .models import Artifact


def available_feature_families(feature_artifact: Artifact) -> list[str]:
    feature_df = load_artifact_csv_frame(feature_artifact)
    family_map = infer_feature_family_columns([str(col) for col in feature_df.columns if str(col) not in {"date", "symbol"}])
    return sorted([family for family, cols in family_map.items() if cols])


def available_label_ks(label_artifact: Artifact) -> list[int]:
    label_df = load_artifact_csv_frame(label_artifact)
    if label_df.empty or "k" not in label_df.columns:
        return []
    values = (
        label_df["k"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    ks: list[int] = []
    seen: set[int] = set()
    for raw in values:
        try:
            parsed = int(raw)
        except Exception:
            continue
        if parsed <= 0 or parsed in seen:
            continue
        seen.add(parsed)
        ks.append(parsed)
    return sorted(ks)


def expand_model_cohort_configs(
    *,
    base_config: dict[str, Any],
    feature_artifact: Artifact,
    label_artifact: Artifact,
) -> list[dict[str, Any]]:
    config = dict(base_config or {})
    feature_family_mode = str(config.get("feature_family_mode") or "all_features").strip().lower()
    label_horizon_mode = str(config.get("label_horizon_mode") or "all_k").strip().lower()

    requested_families = [str(value).strip() for value in list(config.get("feature_families") or []) if str(value).strip()]
    requested_family_groups = [
        [str(item).strip() for item in list(group or []) if str(item).strip()]
        for group in list(config.get("feature_family_groups") or [])
    ]
    requested_family_groups = [group for group in requested_family_groups if group]
    requested_ks: list[int] = []
    for value in list(config.get("label_ks") or []):
        try:
            parsed = int(value)
        except Exception:
            continue
        if parsed > 0:
            requested_ks.append(parsed)
    requested_k_groups: list[list[int]] = []
    for group in list(config.get("label_k_groups") or []):
        parsed_group: list[int] = []
        for value in list(group or []):
            try:
                parsed = int(value)
            except Exception:
                continue
            if parsed > 0 and parsed not in parsed_group:
                parsed_group.append(parsed)
        if parsed_group:
            requested_k_groups.append(parsed_group)

    if feature_family_mode == "per_family":
        family_values = [[family] for family in (requested_families or available_feature_families(feature_artifact))]
    elif feature_family_mode == "grouped_family":
        family_values = requested_family_groups or [[family] for family in available_feature_families(feature_artifact)]
    else:
        direct_group = [str(value).strip() for value in list(config.get("feature_families") or []) if str(value).strip()]
        if not direct_group:
            single_family = str(config.get("feature_family") or "").strip()
            direct_group = [single_family] if single_family else []
        family_values = [direct_group]

    if label_horizon_mode == "per_k":
        k_values = [[k] for k in (requested_ks or available_label_ks(label_artifact))]
    elif label_horizon_mode == "grouped_k":
        k_values = requested_k_groups or [[k] for k in available_label_ks(label_artifact)]
    else:
        raw_label_k = config.get("label_k")
        try:
            parsed = int(raw_label_k)
            k_values = [[parsed]] if parsed > 0 else [[]]
        except Exception:
            direct_group = []
            for value in list(config.get("label_ks") or []):
                try:
                    parsed = int(value)
                except Exception:
                    continue
                if parsed > 0 and parsed not in direct_group:
                    direct_group.append(parsed)
            k_values = [direct_group]

    variants: list[dict[str, Any]] = []
    for feature_group, label_k_group in product(family_values or [[]], k_values or [[]]):
        variant = dict(config)
        cleaned_feature_group = [str(value).strip() for value in list(feature_group or []) if str(value).strip()]
        cleaned_label_group = []
        for value in list(label_k_group or []):
            try:
                parsed = int(value)
            except Exception:
                continue
            if parsed > 0 and parsed not in cleaned_label_group:
                cleaned_label_group.append(parsed)
        variant["feature_family"] = cleaned_feature_group[0] if len(cleaned_feature_group) == 1 else ""
        variant["feature_families"] = cleaned_feature_group
        variant["label_k"] = cleaned_label_group[0] if len(cleaned_label_group) == 1 else None
        variant["label_ks"] = cleaned_label_group
        suffix_parts = []
        if cleaned_feature_group:
            suffix_parts.append("+".join(cleaned_feature_group))
        if cleaned_label_group:
            suffix_parts.append("k" + "-".join(str(value) for value in cleaned_label_group))
        if suffix_parts:
            base_name = str(config.get("model_name") or config.get("name") or "model").strip()
            variant["model_name"] = f"{base_name}__{'__'.join(suffix_parts)}"
        variants.append(variant)
    return variants
