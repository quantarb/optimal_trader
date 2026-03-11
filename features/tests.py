from unittest.mock import patch
from unittest import TestCase, skipUnless

try:
    from features.naming import feature_display_name
    import pandas as pd

    from features.pipeline_builders import (
        _append_representation_embedding_columns,
        _representation_embedding_dataset_rows,
        REPRESENTATION_EMBEDDING_MODEL_VERSION,
        representation_embedding_config,
    )
    from ml.execution import infer_feature_family_columns
except ModuleNotFoundError as exc:
    if exc.name not in {"pandas", "numpy"}:
        raise
    feature_display_name = None
    pd = None
    _append_representation_embedding_columns = None
    _representation_embedding_dataset_rows = None
    REPRESENTATION_EMBEDDING_MODEL_VERSION = None
    representation_embedding_config = None
    infer_feature_family_columns = None


@skipUnless(feature_display_name is not None, "numpy is required for feature naming tests in this environment")
class FeatureNamingTests(TestCase):
    def test_vendor_prefixed_feature_names_are_humanized(self):
        self.assertEqual(feature_display_name("rt__grossprofitmargin"), "Gross Profit Margin")

    def test_internal_feature_names_drop_internal_prefixes(self):
        self.assertEqual(feature_display_name("own__market_cap_log"), "Market Cap Log")

    def test_growth_endpoint_prefixes_are_humanized(self):
        self.assertEqual(feature_display_name("bsg__totalassetsgrowth"), "Total Assets Growth")

    def test_raw_fmp_column_names_match_feature_display_names(self):
        self.assertEqual(feature_display_name("adjOpen"), "Adjusted Open")
        self.assertEqual(feature_display_name("rsi_14"), "RSI 14")

    def test_treasury_labels_use_ust_acronym(self):
        self.assertEqual(feature_display_name("macro__ust_month1"), "UST Month1")


@skipUnless(pd is not None, "pandas is required for representation embedding builder tests")
class RepresentationEmbeddingBuilderTests(TestCase):
    def test_representation_embedding_config_defaults(self):
        config = representation_embedding_config({})
        self.assertFalse(config["enabled"])
        self.assertEqual(config["model_name"], "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(config["model_version"], REPRESENTATION_EMBEDDING_MODEL_VERSION)
        self.assertEqual(config["column_prefix"], "embedding_")

    def test_representation_embedding_dataset_rows_groups_features_by_semantic_family(self):
        frame = pd.DataFrame(
            [
                {
                    "date": "2024-01-01",
                    "symbol": "AAPL",
                    "close": 100.0,
                    "ret_1": 0.05,
                    "km__ev_to_ebitda": 8.2,
                    "evt__ae_revision": None,
                }
            ]
        )
        grouped = {
            "prices_div_adj": ["close", "ret_1"],
            "key_metrics": ["km__ev_to_ebitda"],
            "analyst_estimates": ["evt__ae_revision"],
            "representation_embedding": [],
        }
        rows = _representation_embedding_dataset_rows(frame, grouped)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], "AAPL")
        self.assertEqual(rows[0]["date"], "2024-01-01")
        self.assertEqual(
            rows[0]["families"],
            {
                "price_technical": {"Close": 100.0, "Ret 1": 0.05},
                "valuation_quality": {"EV To EBITDA": 8.2},
            },
        )

    def test_append_representation_embedding_columns_appends_fixed_width_columns(self):
        frame = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAPL", "close": 100.0, "ret_1": 0.05},
                {"date": "2024-01-02", "symbol": "AAPL", "close": 101.0, "ret_1": -0.02},
            ]
        )
        grouped = {
            "prices_div_adj": ["close", "ret_1"],
            "representation_embedding": [],
        }

        class DummyEncoder:
            model_name = "demo-model"
            model_version = "7"

        def fake_build_dataset_embeddings(dataset_rows, *, encoder, store_dir):
            self.assertEqual(encoder.model_name, "demo-model")
            self.assertEqual(store_dir, "/tmp/demo-store")
            self.assertEqual(len(dataset_rows), 2)
            return [
                {"symbol": "AAPL", "date": "2024-01-01", "embedding_vector": [0.25, 0.75]},
                {"symbol": "AAPL", "date": "2024-01-02", "embedding_vector": [0.40, 0.60]},
            ]

        with patch(
            "features.pipeline_builders._resolve_representation_embedding_backend",
            return_value=(fake_build_dataset_embeddings, DummyEncoder()),
        ):
            augmented, columns, meta = _append_representation_embedding_columns(
                frame,
                grouped,
                config={
                    "enabled": True,
                    "model_name": "demo-model",
                    "model_version": "7",
                    "store_dir": "/tmp/demo-store",
                    "column_prefix": "embedding_",
                },
            )
        self.assertEqual(columns, ["embedding_0", "embedding_1"])
        self.assertEqual(list(augmented[columns].iloc[0]), [0.25, 0.75])
        self.assertEqual(meta["dimension"], 2)
        self.assertTrue(meta["enabled"])

    def test_infer_feature_family_columns_recognizes_representation_embeddings(self):
        grouped = infer_feature_family_columns(["close", "embedding_0", "embedding_1"])
        self.assertEqual(grouped["prices_div_adj"], ["close"])
        self.assertEqual(grouped["representation_embedding"], ["embedding_0", "embedding_1"])
