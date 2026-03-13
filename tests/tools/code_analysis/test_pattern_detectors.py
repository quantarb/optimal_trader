from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.code_analysis.patterns.anti_patterns import analyze_anti_patterns
from tools.code_analysis.patterns.good_patterns import analyze_good_patterns


class PatternDetectorTests(unittest.TestCase):
    def test_pattern_detectors_find_expected_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write(root / "sample" / "__init__.py", "")
            _write(
                root / "sample" / "demo.py",
                """
from dataclasses import dataclass

REGISTRY = {}


def register_handler(name, handler):
    REGISTRY[name] = handler


class PricingStrategy:
    def apply(self, price):
        raise NotImplementedError


@dataclass
class ModelConfig:
    window: int = 12


@dataclass
class TradeReport:
    symbol: str


def pure_total(values: list[int]) -> int:
    return sum(values)


def transform(items):
    rows = []
    for item in items:
        rows.append(item + 5)
    return rows


def dispatcher(kind):
    if kind == "a":
        return 11
    elif kind == "b":
        return 13
    elif kind == "c":
        return 17
    elif kind == "d":
        return 19
    return 23


def magic(values):
    threshold = 42
    scale = 17
    divisor = 29
    return [(value * threshold) / divisor for value in values if value > scale]


def broad(values):
    try:
        return values[0]
    except Exception:
        return None


def nested(matrix):
    total = 0
    for row in matrix:
        for value in row:
            total += value
    return total


def build_report(cfg: ModelConfig) -> TradeReport:
    return TradeReport(symbol=str(cfg.window))
""".strip()
                + "\n",
            )

            anti_report = analyze_anti_patterns(root).to_dict()
            good_report = analyze_good_patterns(root).to_dict()

            anti_patterns = {row["pattern"] for row in anti_report["findings"]}
            good_patterns = {row["pattern"] for row in good_report["findings"]}

            self.assertIn("nested loops", anti_patterns)
            self.assertIn("broad exception swallowing", anti_patterns)
            self.assertIn("loop append simple transform", anti_patterns)
            self.assertIn("repeated if/elif dispatch chains", anti_patterns)
            self.assertIn("magic numbers", anti_patterns)

            self.assertIn("pure functions", good_patterns)
            self.assertIn("dataclass config pattern", good_patterns)
            self.assertIn("artifact return boundary", good_patterns)
            self.assertIn("registry pattern", good_patterns)
            self.assertIn("strategy or policy interface", good_patterns)
            self.assertIn("explicit boundary objects", good_patterns)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
