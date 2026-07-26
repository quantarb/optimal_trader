"""Build versioned same-issuer and same-asset-class HITS rank labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from multirate_transformer.ranking import add_hits_rank_targets


def build_rank_labels(
    labels: pd.DataFrame,
    hit_columns: list[str],
    *,
    version: str,
    min_issuer_group_size: int = 2,
    min_asset_class_group_size: int = 5,
    metadata: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {"date", "issuer", "asset_class", *hit_columns}
    missing = required.difference(labels.columns)
    if missing:
        raise KeyError(f"labels missing required rank columns: {sorted(missing)}")
    result = add_hits_rank_targets(
        labels,
        hit_columns,
        min_issuer_group_size=min_issuer_group_size,
        min_asset_class_group_size=min_asset_class_group_size,
    )
    manifest = {
        "label_version": version,
        "hit_columns": hit_columns,
        "ranking_method": "average_percentile",
        "descending": True,
        "minimum_same_issuer_group_size": min_issuer_group_size,
        "minimum_same_asset_class_group_size": min_asset_class_group_size,
        "date_range": [str(result.date.min()), str(result.date.max())],
        "rows": len(result),
        "issuer_count": int(result.issuer.nunique()),
        "asset_classes": sorted(result.asset_class.dropna().astype(str).unique().tolist()),
        **(metadata or {}),
    }
    return result, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--hit-column", action="append", required=True)
    parser.add_argument("--min-issuer-group-size", type=int, default=2)
    parser.add_argument("--min-asset-class-group-size", type=int, default=5)
    args = parser.parse_args()
    labels = pd.read_parquet(args.input) if args.input.suffix == ".parquet" else pd.read_csv(args.input)
    result, manifest = build_rank_labels(
        labels,
        args.hit_column,
        version=args.version,
        min_issuer_group_size=args.min_issuer_group_size,
        min_asset_class_group_size=args.min_asset_class_group_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".parquet":
        result.to_parquet(args.output, index=False)
    else:
        result.to_csv(args.output, index=False)
    args.manifest.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()

