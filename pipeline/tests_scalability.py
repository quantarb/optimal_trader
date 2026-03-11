from __future__ import annotations

import os
from tempfile import TemporaryDirectory
from unittest import skipUnless

from django.test import TestCase

from .scalability import run_scalability_benchmark_suite
from .test_support import ScalabilityFixtureMixin


def _enabled_tiers() -> set[str]:
    raw = str(os.getenv("SCALABILITY_TEST_TIERS") or "").strip()
    if not raw:
        return {"tier1", "tier2", "tier3"}
    return {token.strip().lower() for token in raw.split(",") if token.strip()}


def _tier_enabled(tier_name: str) -> bool:
    return str(os.getenv("RUN_SCALABILITY_TESTS") or "").strip() == "1" and tier_name in _enabled_tiers()


class ScalabilityBenchmarkSuiteTests(ScalabilityFixtureMixin, TestCase):
    @classmethod
    def setUpTestData(cls):
        super().setUpTestData()
        cls.seed_scalability_universe(start_date="2024-01-02", business_days=90)

    @skipUnless(_tier_enabled("tier1"), "Enable with RUN_SCALABILITY_TESTS=1 and SCALABILITY_TEST_TIERS=tier1")
    def test_tier1_benchmark_runs(self):
        report = self._run_suite(["tier1"])
        tier_report = report["tiers"][0]
        self.assertEqual(tier_report["tier"]["actual_symbol_count"], 10)
        self.assertGreater(int(tier_report["artifacts"]["features"]["content"].get("rows") or 0), 0)
        self.assertGreater(int(tier_report["artifacts"]["backtest"]["content"].get("days") or 0), 0)

    @skipUnless(_tier_enabled("tier2"), "Enable with RUN_SCALABILITY_TESTS=1 and SCALABILITY_TEST_TIERS=tier2")
    def test_tier2_benchmark_runs(self):
        report = self._run_suite(["tier2"])
        tier_report = report["tiers"][0]
        self.assertEqual(tier_report["tier"]["actual_symbol_count"], 100)
        self.assertGreater(int(tier_report["artifacts"]["predictions"]["content"].get("rows") or 0), 0)
        self.assertTrue((tier_report["performance"] or {}).get("stages"))

    @skipUnless(_tier_enabled("tier3"), "Enable with RUN_SCALABILITY_TESTS=1 and SCALABILITY_TEST_TIERS=tier3")
    def test_tier3_benchmark_runs(self):
        report = self._run_suite(["tier3"])
        tier_report = report["tiers"][0]
        self.assertEqual(tier_report["tier"]["actual_symbol_count"], 1000)
        self.assertGreater(float(tier_report.get("total_runtime_seconds") or 0.0), 0.0)
        self.assertGreater(int(tier_report["artifacts"]["strategy_dataset"]["content"].get("rows") or 0), 0)

    def _run_suite(self, tiers: list[str]) -> dict:
        with TemporaryDirectory() as temp_dir:
            return run_scalability_benchmark_suite(
                tiers=tiers,
                output_dir=temp_dir,
                feature_profile="baseline",
                start_date="2024-01-02",
                end_date="2024-05-06",
                train_end_date="2024-05-06",
                score_start_date="2024-01-02",
                artifact_storage_format="csv",
                max_tier2_runtime_seconds=9999.0,
                min_profit_pct=0.0,
                label_k_params={"M": [1]},
                buy_execution="adj_open",
                sell_execution="adj_close",
                short_execution="adj_open",
                cover_execution="adj_close",
            )
