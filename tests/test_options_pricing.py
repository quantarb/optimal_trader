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
build_synthetic_option_return_panels = _OPTIONS_PRICING.build_synthetic_option_return_panels


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


def test_build_synthetic_option_return_panels_returns_equity_and_option_buckets():
    close = pd.DataFrame(
        {
            "AAPL": [100.0, 110.0, 121.0, 115.0],
            "MSFT": [100.0, 95.0, 98.0, 104.0],
        },
        index=pd.date_range("2024-01-01", periods=4),
    )
    realized = pd.DataFrame(0.20, index=close.index, columns=close.columns)

    return_panels, price_panels = build_synthetic_option_return_panels(
        close,
        realized_vol_df=realized,
        option_buckets={
            "atm_option": {
                "long_strike_multiplier": 1.0,
                "short_strike_multiplier": 1.0,
            },
        },
        tenor_days=30,
        premium_floor=0.0,
    )

    assert set(return_panels) == {"equity", "atm_option"}
    assert set(return_panels["atm_option"]) == {"long", "short"}
    assert return_panels["atm_option"]["long"].shape == close.shape
    assert return_panels["atm_option"]["short"].shape == close.shape
    assert set(price_panels["atm_option"]) == {
        "call",
        "put",
        "long_strike_multiplier",
        "short_strike_multiplier",
    }
    assert price_panels["atm_option"]["call"].notna().all().all()
    assert price_panels["atm_option"]["put"].notna().all().all()
