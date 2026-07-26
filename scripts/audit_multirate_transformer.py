"""Audit the authoritative feature-family and MTL registries.

The feature-family index is the persisted registry produced by the warehouse
workflow. This script reports family frequency, dimensions, missingness, and
the exact TA exclusion decision without creating a second feature registry.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd


TA_FAMILIES = {
    "equity-historical-price-eod",
    "technical_candles",
    "technical_cycles",
    "technical_math",
    "technical_momentum",
    "technical_overlap",
    "technical_performance",
}
TA_PREFIXES = ("ta_", "technical_")


def _load_gnn(path: Path):
    spec = importlib.util.spec_from_file_location("gnn_registry", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frequency(family: str, columns: list[str]) -> str:
    text = f"{family} {' '.join(columns)}".lower()
    if "quarter" in text or "quarterly" in text:
        return "quarterly"
    if "annual" in text or "income" in text or "balance" in text or "cash" in text:
        return "annual_or_quarterly"
    if "event" in text or text.startswith("is_"):
        return "event"
    if "calendar" in text or "sector" in text or "industry" in text:
        return "static_or_macro"
    if "daily" in text or "price" in text or "volume" in text or "mcap" in text:
        return "daily"
    return "static_or_unknown"


def audit(index_path: Path, output_path: Path | None = None) -> pd.DataFrame:
    index = pd.read_csv(index_path)
    rows: list[dict[str, object]] = []
    for _, metadata in index.iterrows():
        family = str(metadata.family)
        panel_path = Path(str(metadata.panel_path))
        metadata_path = Path(str(metadata.metadata_path))
        panel = pd.read_parquet(panel_path)
        meta = pd.read_parquet(metadata_path)
        columns = [str(column) for column in meta.feature if str(column) in panel.columns]
        missingness = float(panel[columns].isna().mean().mean()) if columns else 1.0
        excluded = family.lower() in TA_FAMILIES or family.lower().startswith(TA_PREFIXES)
        rows.append({
            "feature_family": family,
            "frequency": _frequency(family, columns),
            "included": not excluded,
            "exclusion_reason": "manual_TA_family" if excluded else "",
            "input_dimension": len(columns),
            "missingness_rate": missingness,
            "rows": len(panel),
            "asset_classes": "equity;related_non_option_when_present",
            "source": str(metadata.get("source", "")),
            "panel_path": str(panel_path),
        })
    result = pd.DataFrame(rows).sort_values("feature_family").reset_index(drop=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.index, args.output)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
