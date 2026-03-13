from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.code_analysis.architecture_rules import bootstrap_architecture_rules, validate_architecture_rules
from tools.code_analysis.blast_radius import analyze_blast_radius
from tools.code_analysis.call_graph import analyze_call_graph
from tools.code_analysis.module_responsibility import analyze_module_responsibilities
from tools.code_analysis.pattern_metrics import analyze_code_health_metrics
from tools.code_analysis.patterns.anti_patterns import analyze_anti_patterns
from tools.code_analysis.patterns.good_patterns import analyze_good_patterns
from tools.code_analysis.refactor_priority import build_refactor_priority_report
from tools.code_analysis.repository import build_repository_inventory
from tools.code_analysis.dependency_graph import analyze_dependency_graph


class BlastRadiusTests(unittest.TestCase):
    def test_blast_radius_and_refactor_priority_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write(root / "sample" / "__init__.py", "")
            _write(
                root / "sample" / "core.py",
                """
def helper(value: int) -> int:
    return value + 1


def normalize(values: list[int]) -> list[int]:
    rows = []
    for value in values:
        rows.append(helper(value))
    return rows
""".strip()
                + "\n",
            )
            _write(
                root / "sample" / "service.py",
                """
from sample.core import helper


def run(values: list[int]) -> int:
    total = 0
    for value in values:
        total += helper(value)
    return total
""".strip()
                + "\n",
            )
            _write(
                root / "sample" / "api.py",
                """
from sample.core import normalize
from sample.service import run


def handle(values: list[int]) -> int:
    return run(normalize(values))
""".strip()
                + "\n",
            )
            _write(
                root / "sample" / "cli.py",
                """
from sample.service import run


def handle(values: list[int]) -> int:
    return run(values)
""".strip()
                + "\n",
            )
            rules_path = root / "architecture_rules.yaml"
            bootstrap_architecture_rules(root, rules_path)

            inventory = build_repository_inventory(root)
            dependency = analyze_dependency_graph(root, inventory)
            call_graph = analyze_call_graph(inventory)
            architecture = validate_architecture_rules(root, rules_path=rules_path, inventory=inventory, dependency_report=dependency)
            anti = analyze_anti_patterns(root, inventory=inventory, architecture_report=architecture.to_dict())
            good = analyze_good_patterns(root, inventory=inventory)
            responsibility = analyze_module_responsibilities(
                inventory,
                dependency_report=dependency.to_dict(),
            )
            health = analyze_code_health_metrics(
                root,
                inventory=inventory,
                dependency_report=dependency.to_dict(),
                architecture_report=architecture.to_dict(),
                anti_pattern_report=anti.to_dict(),
                good_pattern_report=good.to_dict(),
            )

            blast = analyze_blast_radius(
                root,
                inventory=inventory,
                dependency_report=dependency.to_dict(),
                call_graph_report=call_graph.to_dict(),
                code_health_report=health.to_dict(),
                anti_pattern_report=anti.to_dict(),
                architecture_report=architecture.to_dict(),
                responsibility_report=responsibility.to_dict(),
            ).to_dict()
            priority = build_refactor_priority_report(blast).to_dict()

            core_row = next(row for row in blast["module_rows"] if row["module"] == "sample.core")
            self.assertEqual(core_row["direct_dependents"], 2)
            self.assertGreaterEqual(core_row["indirect_dependents"], 1)
            self.assertTrue(core_row["critical_execution_path"])

            summary_modules = {row["module"] for row in blast["summary"]["top_10_highest_blast_radius_modules"]}
            self.assertIn("sample.core", summary_modules)

            ranking_row = next(row for row in priority["rankings"] if row["module"] == "sample.core")
            self.assertGreaterEqual(ranking_row["blast_radius_rank"], 1)
            self.assertIn("suggested_refactor", ranking_row)

            symbol_names = {row["symbol"] for row in blast["symbol_rows"]}
            self.assertIn("sample.core.helper", symbol_names)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
