import importlib.util
from pathlib import Path

import pandas as pd


def _load_warehouse_module():
    path = Path(__file__).resolve().parents[1] / "data" / "warehouse.py"
    spec = importlib.util.spec_from_file_location("ot_warehouse", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_warehouse_section_to_indexed_frame_prefixes_columns():
    wh_mod = _load_warehouse_module()

    class _FakeWarehouse:
        def read_fundamentals(self, symbol, *, section, provider, start=None, end=None):
            assert symbol == "AAPL"
            assert section == "metrics"
            assert provider == "fmp"
            return pd.DataFrame(
                {"pe_ratio": [20.0], "market_cap": [1e12]},
                index=pd.to_datetime(["2024-12-31"]),
            )

    original = wh_mod.get_warehouse
    wh_mod.get_warehouse = lambda: _FakeWarehouse()  # type: ignore[assignment]
    try:
        out = wh_mod.warehouse_section_to_indexed_frame(
            "AAPL",
            "key_metrics",
            prefix="km__",
        )
    finally:
        wh_mod.get_warehouse = original

    assert not out.empty
    assert "km__pe_ratio" in out.columns
    assert out.index.names == ["date", "symbol"]


def test_read_flags_default_true(monkeypatch):
    monkeypatch.delenv("QW_READ_PRICES", raising=False)
    monkeypatch.delenv("QW_READ_FUNDAMENTALS", raising=False)
    wh_mod = _load_warehouse_module()
    assert wh_mod.read_prices_from_warehouse() is True
    assert wh_mod.read_fundamentals_from_warehouse() is True