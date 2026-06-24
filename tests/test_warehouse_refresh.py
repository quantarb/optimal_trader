from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "data" / "warehouse.py"
    spec = importlib.util.spec_from_file_location("ot_warehouse", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_refresh_module():
    path = Path(__file__).resolve().parents[1] / "data" / "warehouse_refresh.py"
    spec = importlib.util.spec_from_file_location("ot_warehouse_refresh", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_use_warehouse_refresh_requires_optional_package(monkeypatch):
    monkeypatch.setenv("QW_REFRESH_ENABLED", "1")
    try:
        mod = _load_refresh_module()
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            pytest.skip("data package import path requires torch in this environment")
        raise
    monkeypatch.setattr(mod, "refresh_universe_prices", None)
    assert mod.use_warehouse_refresh() is False


def test_warehouse_sections_for_refresh_excludes_django_only_openbb_gaps():
    mod = _load_module()
    sections = mod.warehouse_sections_for_refresh(
        ("prices_div_adj", "key_metrics", "ratios", "earnings")
    )
    assert sections == ("metrics", "ratios", "earnings")


def test_django_only_sections_for_refresh_identifies_unmapped_sections():
    mod = _load_module()
    skipped = mod.django_only_sections_for_refresh(
        ("key_metrics", "earnings", "financial_growth", "income_statement_ttm")
    )
    assert skipped == ("income_statement_ttm",)
