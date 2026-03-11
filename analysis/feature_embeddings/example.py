from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from .pipeline import build_dataset_embeddings


class DeterministicDemoEncoder:
    """
    Offline-safe encoder used only for the synthetic example and tests.

    The production path still defaults to SentenceTransformerEncoder.
    """

    model_name = "demo-deterministic-encoder"
    model_version = "1"

    def __init__(self, dimension: int = 8):
        self.dimension = dimension

    def encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype="float32")
        payload = str(text).encode("utf-8")
        for idx, byte in enumerate(payload):
            vector[idx % self.dimension] += ((byte % 31) + 1) / 31.0
        return vector


def build_synthetic_dataset() -> list[dict[str, object]]:
    return [
        {
            "symbol": "AAPL",
            "date": "2024-01-31",
            "families": {
                "technical": {
                    "rsi": 65.4,
                    "momentum_20d": 0.12,
                    "volatility_30d": 0.21,
                },
                "fundamental": {
                    "ev_dividedby_ebitda": 8.0,
                    "roic": 0.14,
                    "debt_dividedby_ebitda": 2.1,
                },
                "macro": {
                    "fed_funds_rate": 0.0525,
                    "yield_curve_10y_2y": -0.0032,
                },
                "metadata": {
                    "company_name": "Apple Inc",
                    "sector": "Information Technology",
                    "exchange": "NASDAQ",
                    "country": "United States",
                    "currency": "USD",
                },
            },
        },
        {
            "symbol": "MSFT",
            "date": "2024-01-31",
            "families": {
                "technical": {
                    "rsi": 59.1,
                    "momentum_20d": 0.08,
                    "volatility_30d": 0.18,
                },
                "macro": {
                    "fed_funds_rate": 0.0525,
                    "yield_curve_10y_2y": -0.0032,
                },
                "metadata": {
                    "company_name": "Microsoft Corp",
                    "sector": "Information Technology",
                    "exchange": "NASDAQ",
                    "country": "United States",
                    "currency": "USD",
                },
            },
        },
    ]


def run_synthetic_example(store_dir: str | Path | None = None) -> list[dict[str, object]]:
    dataset = build_synthetic_dataset()
    encoder = DeterministicDemoEncoder()
    if store_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        try:
            return build_dataset_embeddings(dataset, encoder=encoder, store_dir=temp_dir.name)
        finally:
            temp_dir.cleanup()
    return build_dataset_embeddings(dataset, encoder=encoder, store_dir=store_dir)


if __name__ == "__main__":
    rows = run_synthetic_example()
    for row in rows:
        preview = np.round(np.asarray(row["embedding_vector"])[:8], 4).tolist()
        print(f"{row['symbol']} {row['date']} {preview}")
