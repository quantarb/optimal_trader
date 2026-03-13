from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.code_analysis.architecture_rules import bootstrap_architecture_rules, validate_architecture_rules


class ArchitectureRuleTests(unittest.TestCase):
    def test_architecture_rules_detect_layer_and_boundary_violations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write(root / "domain" / "__init__.py", "")
            _write(root / "data" / "__init__.py", "")
            _write(root / "utils" / "__init__.py", "")
            _write(root / "data" / "schema.py", "SCHEMA_VERSION = 1\n")
            _write(
                root / "domain" / "panel.py",
                """
from data import schema


def build_panel():
    return schema.SCHEMA_VERSION
""".strip()
                + "\n",
            )
            rules_path = root / "architecture_rules.yaml"
            rules_path.write_text(
                json.dumps(
                    {
                        "layers": {
                            "domain": ["domain"],
                            "infrastructure": ["data"],
                            "shared": ["utils"],
                        },
                        "allowed_layer_imports": {
                            "domain": ["domain", "shared"],
                            "infrastructure": ["infrastructure", "shared"],
                            "shared": ["shared"],
                        },
                        "forbidden_layer_dependencies": [
                            {
                                "source_layer": "domain",
                                "target_layer": "infrastructure",
                                "reason": "Domain must not depend on infrastructure.",
                            }
                        ],
                        "forbidden_cross_package_imports": [
                            {
                                "source": "domain",
                                "target": "data",
                                "reason": "Domain must not import data directly.",
                            }
                        ],
                        "domain_boundaries": [
                            {
                                "name": "core_domain",
                                "packages": ["domain"],
                                "allowed_external_imports": ["utils"],
                                "reason": "Domain boundary broken.",
                            }
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            report = validate_architecture_rules(root, rules_path=rules_path).to_dict()
            rule_types = {row["rule_type"] for row in report["violations"]}

            self.assertIn("layer_direction", rule_types)
            self.assertIn("forbidden_layer_dependency", rule_types)
            self.assertIn("forbidden_cross_package_import", rule_types)
            self.assertIn("domain_boundary", rule_types)

    def test_bootstrap_architecture_rules_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write(root / "domain" / "__init__.py", "")
            _write(root / "utils" / "__init__.py", "")

            output = bootstrap_architecture_rules(root, root / "generated_rules.yaml")

            self.assertTrue(Path(output["path"]).exists())
            payload = json.loads(Path(output["path"]).read_text(encoding="utf-8"))
            self.assertIn("layers", payload)
            self.assertIn("quality_weights", payload)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
