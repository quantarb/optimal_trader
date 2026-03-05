from django.test import SimpleTestCase

from features.naming import feature_display_name


class FeatureNamingTests(SimpleTestCase):
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
