from app.trading_app_v2_runtime import _optional_date_string


def test_optional_date_string_normalizes_empty_values():
    assert _optional_date_string(None) is None
    assert _optional_date_string("") is None
    assert _optional_date_string("None") is None
    assert _optional_date_string(" NaT ") is None


def test_optional_date_string_preserves_dates():
    assert _optional_date_string("2026-07-06") == "2026-07-06"
