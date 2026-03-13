from __future__ import annotations

import unittest

from tools.code_analysis.quality_scorecard import build_quality_scorecard


class QualityScorecardTests(unittest.TestCase):
    def test_quality_scorecard_prefers_healthier_module(self) -> None:
        metrics_report = {
            "module_rows": [
                {
                    "module": "pkg.clean",
                    "path": "/repo/pkg/clean.py",
                    "line_count": 100,
                    "cyclomatic_complexity_avg": 2.0,
                    "cyclomatic_complexity_max": 4,
                    "nesting_depth_max": 1,
                    "dependency_fan_in": 2,
                    "dependency_fan_out": 2,
                    "import_cycle_count": 0,
                    "architecture_rule_violations": 0,
                    "duplicate_code_clusters": 0,
                    "type_hint_coverage": 1.0,
                    "llm_editability_proxy_score": 88.0,
                    "change_safety_proxy_score": 90.0,
                },
                {
                    "module": "pkg.messy",
                    "path": "/repo/pkg/messy.py",
                    "line_count": 100,
                    "cyclomatic_complexity_avg": 12.0,
                    "cyclomatic_complexity_max": 40,
                    "nesting_depth_max": 5,
                    "dependency_fan_in": 10,
                    "dependency_fan_out": 14,
                    "import_cycle_count": 1,
                    "architecture_rule_violations": 3,
                    "duplicate_code_clusters": 4,
                    "type_hint_coverage": 0.2,
                    "llm_editability_proxy_score": 22.0,
                    "change_safety_proxy_score": 18.0,
                },
            ]
        }
        anti_report = {
            "findings": [
                {"file": "/repo/pkg/messy.py", "severity": "high"},
                {"file": "/repo/pkg/messy.py", "severity": "medium"},
            ]
        }
        good_report = {
            "findings": [
                {"file": "/repo/pkg/clean.py", "strength": 0.9},
                {"file": "/repo/pkg/clean.py", "strength": 0.8},
            ]
        }
        architecture_report = {
            "violations": [
                {"source_module": "pkg.messy"},
                {"source_module": "pkg.messy"},
                {"source_module": "pkg.messy"},
            ]
        }

        report = build_quality_scorecard(
            metrics_report=metrics_report,
            anti_pattern_report=anti_report,
            good_pattern_report=good_report,
            architecture_report=architecture_report,
        ).to_dict()

        clean = next(row for row in report["module_scores"] if row["module"] == "pkg.clean")
        messy = next(row for row in report["module_scores"] if row["module"] == "pkg.messy")

        self.assertGreater(clean["score"], messy["score"])
        self.assertGreater(report["repo_score"], 0)
        self.assertEqual(clean["dimensions"]["typing_health"], 100.0)
        self.assertLess(messy["dimensions"]["anti_pattern_burden"], clean["dimensions"]["anti_pattern_burden"])
