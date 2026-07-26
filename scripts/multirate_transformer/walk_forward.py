"""Anchored walk-forward fold utilities and artifact manifests."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    test_year: int
    train_start: str
    train_end: str
    validation_start: str | None
    validation_end: str | None
    test_start: str
    test_end: str


def anchored_folds(
    dates: Iterable[pd.Timestamp],
    *,
    first_test_year: int,
    last_test_year: int,
    validation_years: int = 0,
) -> list[WalkForwardFold]:
    values = pd.to_datetime(list(dates), errors="raise")
    if len(values) == 0:
        raise ValueError("dates cannot be empty")
    minimum, maximum = values.min(), values.max()
    folds = []
    for test_year in range(first_test_year, last_test_year + 1):
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year}-12-31")
        validation_start = pd.Timestamp(f"{test_year - validation_years}-01-01") if validation_years else None
        validation_end = pd.Timestamp(f"{test_year - 1}-12-31") if validation_years else None
        train_end = validation_start - pd.Timedelta(days=1) if validation_start is not None else test_start - pd.Timedelta(days=1)
        if train_end < minimum or test_start > maximum:
            continue
        folds.append(WalkForwardFold(
            test_year=test_year,
            train_start=str(minimum.date()),
            train_end=str(train_end.date()),
            validation_start=str(validation_start.date()) if validation_start is not None else None,
            validation_end=str(validation_end.date()) if validation_end is not None else None,
            test_start=str(test_start.date()),
            test_end=str(min(test_end, maximum).date()),
        ))
    return folds


def persist_fold_manifest(path: Path, fold: WalkForwardFold, *, feature_families: list[str], tasks: list[str], assets: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fold": asdict(fold), "feature_families": feature_families, "tasks": tasks, "asset_classes": assets}, indent=2) + "\n")

