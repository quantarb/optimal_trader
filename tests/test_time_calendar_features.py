from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from features.time_features import build_time_calendar_features


def test_build_time_calendar_features_uses_cached_events_and_symbol_ipo_date():
    symbol = SimpleNamespace(symbol="AAPL", payload={"ipoDate": "1980-12-12"})
    section_dates = {
        "earnings": ["2024-01-03", "2024-01-08"],
        "dividends": ["2024-01-02"],
        "splits": ["2024-01-04"],
    }

    def fake_section_event_frame(_symbol_obj, section_key, _target_index):
        dates = section_dates.get(str(section_key), [])
        if not dates:
            return pd.DataFrame(columns=["date", "symbol"])
        return pd.DataFrame(
            {
                "date": pd.DatetimeIndex(pd.to_datetime(dates)).normalize(),
                "symbol": ["AAPL"] * len(dates),
            }
        )

    target_index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=6), ["AAPL"]],
        names=["date", "symbol"],
    )

    with patch("features.time_features._section_event_frame", fake_section_event_frame):
        built = build_time_calendar_features(symbol, target_index)

    df = built.df

    assert "time__day_of_week" in built.feature_cols
    assert "time__week_of_year" in built.feature_cols
    assert "time__is_month_start" in built.feature_cols
    assert "time__is_month_end" in built.feature_cols
    assert "time__is_quarter_start" in built.feature_cols
    assert "time__is_quarter_end" in built.feature_cols
    assert "time__year" not in built.feature_cols
    assert "time__days_since_earnings" in built.feature_cols
    assert "time__days_until_earnings" in built.feature_cols
    assert "time__days_since_dividend" in built.feature_cols
    assert "time__days_since_stock_split" in built.feature_cols
    assert "time__days_after_ipo" in built.feature_cols

    jan_05 = (pd.Timestamp("2024-01-05"), "AAPL")
    jan_01 = (pd.Timestamp("2024-01-01"), "AAPL")
    assert int(df.loc[jan_01, "time__is_month_start"]) == 1
    assert int(df.loc[jan_01, "time__is_quarter_start"]) == 1
    assert float(df.loc[jan_05, "time__days_since_earnings"]) == 2.0
    assert float(df.loc[jan_05, "time__days_until_earnings"]) == 3.0
    assert float(df.loc[jan_05, "time__days_since_dividend"]) == 3.0
    assert float(df.loc[jan_05, "time__days_since_stock_split"]) == 1.0
    assert float(df.loc[jan_05, "time__days_after_ipo"]) > 15_000.0
