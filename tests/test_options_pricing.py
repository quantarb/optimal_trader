from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pandas as pd

_OPTIONS_PRICING_PATH = Path(__file__).resolve().parents[1] / "workflows" / "options_pricing.py"
_SPEC = importlib.util.spec_from_file_location("options_pricing_under_test", _OPTIONS_PRICING_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_OPTIONS_PRICING = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_OPTIONS_PRICING)
build_realized_vol_panel = _OPTIONS_PRICING.build_realized_vol_panel


def test_build_realized_vol_panel_preserves_shape_and_applies_bounds():
    close = pd.DataFrame(
        {
            "AAPL": [100.0, 110.0, 121.0, 133.1],
            "MSFT": [100.0, 100.0, 200.0, 100.0],
        },
        index=pd.date_range("2024-01-01", periods=4),
    )

    realized = build_realized_vol_panel(close, window=2, vol_floor=0.15, vol_cap=0.80)

    assert realized.index.equals(close.index)
    assert list(realized.columns) == list(close.columns)
    assert float(realized.min().min()) >= 0.15
    assert float(realized.max().max()) <= 0.80
    assert math.isclose(float(realized.iloc[0]["AAPL"]), 0.15)
