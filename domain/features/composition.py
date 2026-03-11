from __future__ import annotations

import pandas as pd

from domain.features.specs import BuiltFeatureSet


def merge_feature_sets(parts: list[BuiltFeatureSet], target_index: pd.MultiIndex) -> BuiltFeatureSet:
    """Join multiple feature-family outputs on a shared target index."""

    merged = pd.DataFrame(index=target_index)
    feature_cols: list[str] = []
    for part in parts:
        if not part.df.empty and part.feature_cols:
            merged = merged.join(part.df[part.feature_cols], how="left")
            feature_cols.extend(part.feature_cols)
    feature_cols = list(dict.fromkeys(feature_cols))
    return BuiltFeatureSet(df=merged, feature_cols=feature_cols)

