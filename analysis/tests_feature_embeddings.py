from __future__ import annotations

import tempfile
from unittest import TestCase, skipUnless

from analysis.feature_embeddings.serialization import serialize_family

try:
    import numpy as np

    from analysis.feature_embeddings.embedding_store import EmbeddingKey, load_embedding
    from analysis.feature_embeddings.encoder import count_text_tokens, fit_text_to_token_limit, max_text_tokens, measure_text_tokens
    from analysis.feature_embeddings.example import build_synthetic_dataset, run_synthetic_example
    from analysis.feature_embeddings.pipeline import build_dataset_embeddings, build_row_embedding
    from analysis.feature_embeddings.pooling import l2_normalize, pool_embeddings
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
    np = None
    EmbeddingKey = None
    load_embedding = None
    count_text_tokens = None
    fit_text_to_token_limit = None
    max_text_tokens = None
    measure_text_tokens = None
    build_synthetic_dataset = None
    run_synthetic_example = None
    build_dataset_embeddings = None
    build_row_embedding = None
    l2_normalize = None
    pool_embeddings = None


class CountingEncoder:
    model_name = "counting-demo"
    model_version = "1"

    def __init__(self, dimension: int = 6):
        self.dimension = dimension
        self.calls = 0

    def encode(self, text: str) -> np.ndarray:
        self.calls += 1
        vector = np.zeros(self.dimension, dtype="float32")
        for idx, byte in enumerate(str(text).encode("utf-8")):
            vector[idx % self.dimension] += float(byte % 19)
        return vector


class TokenBudgetEncoder(CountingEncoder):
    def __init__(self, *, max_tokens: int = 9, dimension: int = 6):
        super().__init__(dimension=dimension)
        self._max_tokens = max_tokens

    def token_count(self, text: str) -> int:
        return len(str(text).split())

    def max_tokens(self) -> int:
        return self._max_tokens


@skipUnless(np is not None, "numpy is required for feature embedding pipeline tests")
class FeatureEmbeddingPipelineTests(TestCase):
    def test_serialize_family_is_deterministic_and_readable(self):
        text = serialize_family(
            "AAPL",
            "2024-01-31",
            "technical",
            {
                "volatility_30d": 0.21,
                "momentum_20d": 0.12,
                "rsi": 65.4,
                "missing_value": None,
            },
        )
        self.assertEqual(
            text,
            "\n".join(
                [
                    "[FAMILY=TECHNICAL]",
                    "[SYMBOL=AAPL]",
                    "[DATE=2024-01-31]",
                    "Momentum 20D: 0.1200",
                    "RSI: 65.40",
                    "Volatility 30D: 0.2100",
                ]
            ),
        )

    def test_serialize_family_omits_blank_and_missing_container_values(self):
        text = serialize_family(
            "AAPL",
            "2024-01-31",
            "metadata",
            {
                "company_name": "Apple Inc",
                "blank_value": "   ",
                "null_text": "NaN",
                "tags": [None, "", "Consumer Electronics"],
                "empty_list": [None, ""],
            },
        )
        self.assertEqual(
            text,
            "\n".join(
                [
                    "[FAMILY=METADATA]",
                    "[SYMBOL=AAPL]",
                    "[DATE=2024-01-31]",
                    "Company Name: Apple Inc",
                    "Tags: Consumer Electronics",
                ]
            ),
        )

    def test_pool_embeddings_requires_matching_dimensions(self):
        with self.assertRaises(ValueError):
            pool_embeddings([np.array([1.0, 2.0]), np.array([1.0])])

    def test_build_row_embedding_uses_disk_cache(self):
        encoder = CountingEncoder()
        families = {
            "technical": {"rsi": 65.4, "momentum_20d": 0.12},
            "metadata": {"exchange": "NASDAQ", "currency": "USD"},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            first = build_row_embedding("AAPL", "2024-01-31", families, encoder=encoder, store_dir=temp_dir)
            second = build_row_embedding("AAPL", "2024-01-31", families, encoder=encoder, store_dir=temp_dir)
            self.assertEqual(encoder.calls, 2)
            np.testing.assert_allclose(first, second, atol=1e-6, rtol=1e-6)

            technical_key = EmbeddingKey(
                symbol="AAPL",
                date="2024-01-31",
                family="technical",
                model_name=encoder.model_name,
                model_version=encoder.model_version,
            )
            cached = load_embedding(technical_key, store_dir=temp_dir)
            self.assertIsNotNone(cached)
            np.testing.assert_allclose(np.linalg.norm(cached), 1.0, atol=1e-6)

    def test_build_row_embedding_skips_blank_only_family_values(self):
        encoder = CountingEncoder()
        families = {
            "technical": {"rsi": 65.4},
            "metadata": {"exchange": "   ", "country": None},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            build_row_embedding("AAPL", "2024-01-31", families, encoder=encoder, store_dir=temp_dir)
        self.assertEqual(encoder.calls, 1)

    def test_build_dataset_embeddings_returns_fixed_size_vectors(self):
        encoder = CountingEncoder(dimension=8)
        with tempfile.TemporaryDirectory() as temp_dir:
            rows = build_dataset_embeddings(build_synthetic_dataset(), encoder=encoder, store_dir=temp_dir)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["symbol"], "AAPL")
        self.assertEqual(rows[1]["symbol"], "MSFT")
        self.assertEqual(rows[0]["embedding_vector"].shape, (8,))
        self.assertEqual(rows[1]["embedding_vector"].shape, (8,))

    def test_l2_normalize_preserves_zero_vector(self):
        vector = l2_normalize(np.zeros(4, dtype="float32"))
        np.testing.assert_array_equal(vector, np.zeros(4, dtype="float32"))

    def test_synthetic_example_runs_offline(self):
        rows = run_synthetic_example()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(isinstance(row["embedding_vector"], np.ndarray) for row in rows))

    def test_fit_text_to_token_limit_keeps_headers_and_respects_budget(self):
        encoder = TokenBudgetEncoder(max_tokens=9)
        text = "\n".join(
            [
                "[FAMILY=TECHNICAL]",
                "[SYMBOL=AAPL]",
                "[DATE=2024-01-31]",
                "RSI: sixty five",
                "Momentum: strong trend",
                "Volatility: elevated now",
            ]
        )
        measured = measure_text_tokens(text, encoder=encoder)
        self.assertEqual(measured["token_count"], 12)
        self.assertEqual(max_text_tokens(encoder=encoder), 9)
        self.assertFalse(measured["within_limit"])

        trimmed = fit_text_to_token_limit(text, encoder=encoder)
        self.assertEqual(
            trimmed,
            "\n".join(
                [
                    "[FAMILY=TECHNICAL]",
                    "[SYMBOL=AAPL]",
                    "[DATE=2024-01-31]",
                    "RSI: sixty five",
                    "Momentum: strong trend",
                ]
            ),
        )
        self.assertLessEqual(count_text_tokens(trimmed, encoder=encoder), max_text_tokens(encoder=encoder))
