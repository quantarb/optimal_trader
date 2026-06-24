import importlib.util
from pathlib import Path

import pandas as pd


def test_to_adjusted_price_frame_mapping():
    path = Path(__file__).resolve().parents[1] / "data" / "warehouse.py"
    spec = importlib.util.spec_from_file_location("ot_warehouse", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    _to_adjusted_price_frame = module._to_adjusted_price_frame

    raw = pd.DataFrame(
        {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    out = _to_adjusted_price_frame(raw)
    assert out.loc["2024-01-02", "adj_close"] == 1.5
    assert out.loc["2024-01-02", "close"] == 1.5