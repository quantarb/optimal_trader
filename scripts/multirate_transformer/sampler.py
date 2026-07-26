"""Date-aware batches for cross-sectional ranking supervision."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


class CrossSectionalDateBatchSampler:
    """Yield date groups while retaining enough issuer and class peers."""

    def __init__(self, frame: pd.DataFrame, batch_dates: int = 1, seed: int = 20260725):
        required = {"date", "issuer", "asset_class"}
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"batch frame missing columns: {sorted(missing)}")
        if batch_dates <= 0:
            raise ValueError("batch_dates must be positive")
        self.frame = frame.copy()
        self.frame["date"] = pd.to_datetime(self.frame["date"], errors="raise").dt.normalize()
        self.dates = np.array(sorted(self.frame.date.unique()), dtype="datetime64[ns]")
        self.batch_dates = batch_dates
        self.seed = seed

    def __iter__(self) -> Iterator[pd.DataFrame]:
        rng = np.random.default_rng(self.seed)
        dates = self.dates.copy()
        rng.shuffle(dates)
        for start in range(0, len(dates), self.batch_dates):
            selected = set(dates[start:start + self.batch_dates])
            yield self.frame.loc[self.frame.date.isin(selected)].copy()

    def __len__(self) -> int:
        return int(np.ceil(len(self.dates) / self.batch_dates))

    @staticmethod
    def coverage(frame: pd.DataFrame) -> dict[str, int]:
        issuer_counts = frame.groupby(["date", "issuer"], dropna=False).size()
        class_counts = frame.groupby(["date", "asset_class"], dropna=False).size()
        return {
            "same_issuer_pairs": int((issuer_counts * (issuer_counts - 1) // 2).sum()),
            "same_asset_class_pairs": int((class_counts * (class_counts - 1) // 2).sum()),
            "unique_issuers": int(frame.issuer.nunique()),
            "unique_asset_classes": int(frame.asset_class.nunique()),
            "valid_issuer_groups": int((issuer_counts >= 2).sum()),
            "valid_asset_class_groups": int((class_counts >= 2).sum()),
        }

