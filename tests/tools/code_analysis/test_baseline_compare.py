from __future__ import annotations

import unittest

from tools.code_analysis.baseline_compare import compare_quality_snapshots, snapshot_from_reports


class BaselineCompareTests(unittest.TestCase):
    def test_compare_quality_snapshots_classifies_metric_changes(self) -> None:
        before = snapshot_from_reports(
            label="baseline",
            root="/repo",
            metrics_report={
                "repo_summary": {
                    "architecture_rule_violations": 8,
                    "type_hint_coverage": 0.6,
                    "llm_editability_proxy_score": 40.0,
                }
            },
            scorecard_report={
                "repo_score": 55.0,
                "repo_dimensions": {"complexity_health": 45.0, "typing_health": 60.0},
                "module_scores": [{"module": "pkg.a", "score": 50.0}],
                "file_scores": [{"file": "/repo/pkg/a.py", "score": 50.0}],
            },
            anti_pattern_report={"summary": {"finding_count": 10}},
            good_pattern_report={"summary": {"finding_count": 5}},
            architecture_report={"summary": {"violation_count": 8}},
        )
        after = snapshot_from_reports(
            label="after",
            root="/repo",
            metrics_report={
                "repo_summary": {
                    "architecture_rule_violations": 3,
                    "type_hint_coverage": 0.8,
                    "llm_editability_proxy_score": 55.0,
                }
            },
            scorecard_report={
                "repo_score": 67.0,
                "repo_dimensions": {"complexity_health": 55.0, "typing_health": 80.0},
                "module_scores": [{"module": "pkg.a", "score": 62.0}],
                "file_scores": [{"file": "/repo/pkg/a.py", "score": 62.0}],
            },
            anti_pattern_report={"summary": {"finding_count": 6}},
            good_pattern_report={"summary": {"finding_count": 8}},
            architecture_report={"summary": {"violation_count": 3}},
        )

        comparison = compare_quality_snapshots(before, after).to_dict()
        improved_metrics = {row["metric"] for row in comparison["improved"]}

        self.assertEqual(comparison["overall_score_delta"], 12.0)
        self.assertIn("repo_score", improved_metrics)
        self.assertIn("type_hint_coverage", improved_metrics)
        self.assertIn("architecture_rule_violations", improved_metrics)
