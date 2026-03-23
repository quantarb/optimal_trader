from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from typer.testing import CliRunner

from tools.code_analysis.cli import app


class CodeAnalysisCliTests(unittest.TestCase):
    def test_cli_quality_snapshot_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write(root / "sample" / "__init__.py", "")
            _write(
                root / "sample" / "logic.py",
                """
from dataclasses import dataclass


@dataclass
class Config:
    value: int = 1


def compute_total(values: list[int]) -> int:
    return sum(values)
""".strip()
                + "\n",
            )
            output_dir = root / "reports"
            runner = CliRunner()

            result = runner.invoke(
                app,
                [
                    "bootstrap_architecture_rules",
                    "--root",
                    str(root),
                    "--output",
                    str(output_dir),
                    "--rules-path",
                    str(root / "architecture_rules.yaml"),
                ],
                catch_exceptions=False,
            )
            self.assertEqual(result.exit_code, 0)

            result = runner.invoke(
                app,
                [
                    "validate_architecture_rules",
                    "--root",
                    str(root),
                    "--output",
                    str(output_dir),
                    "--rules-path",
                    str(root / "architecture_rules.yaml"),
                ],
                catch_exceptions=False,
            )
            self.assertEqual(result.exit_code, 0)

            result = runner.invoke(
                app,
                [
                    "capture_quality_snapshot",
                    "--root",
                    str(root),
                    "--output",
                    str(output_dir),
                    "--label",
                    "baseline",
                    "--rules-path",
                    str(root / "architecture_rules.yaml"),
                ],
                catch_exceptions=False,
            )
            self.assertEqual(result.exit_code, 0)

            result = runner.invoke(
                app,
                [
                    "capture_quality_snapshot",
                    "--root",
                    str(root),
                    "--output",
                    str(output_dir),
                    "--label",
                    "current",
                    "--rules-path",
                    str(root / "architecture_rules.yaml"),
                ],
                catch_exceptions=False,
            )
            self.assertEqual(result.exit_code, 0)

            result = runner.invoke(
                app,
                [
                    "compare_quality_snapshots",
                    "baseline",
                    "current",
                    "--output",
                    str(output_dir),
                    "--paths",
                    "sample/logic.py",
                ],
                catch_exceptions=False,
            )
            self.assertEqual(result.exit_code, 0)
            self.assertTrue((output_dir / "quality_snapshot_baseline.json").exists())
            self.assertTrue((output_dir / "quality_snapshot_current.json").exists())
            self.assertTrue((output_dir / "quality_comparison_baseline_vs_current.md").exists())
            self.assertTrue((output_dir / "blast_radius_report.json").exists())
            self.assertTrue((output_dir / "refactor_priority_report.md").exists())
            priority_payload = json.loads((output_dir / "refactor_priority_report.json").read_text(encoding="utf-8"))
            self.assertIn("symbol_recommendations", priority_payload)
            self.assertIn("pattern_type_counts", priority_payload["summary"])


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
