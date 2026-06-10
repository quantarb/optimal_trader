from __future__ import annotations

import os
import logging

import numpy as np
import pandas as pd
import django
from django.apps import apps
import pandas_ta_classic as ta

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")
if not apps.ready:
    django.setup()

from domain.features.ta_classic_technical import (
    TA_CLASSIC_FAMILY_PREFIXES,
    _indicator_specs,
    build_price_ta_classic_feature_families,
)
from domain.models.feature_families import infer_feature_family_columns


def test_build_price_ta_classic_feature_families_splits_categories():
    dates = pd.date_range("2024-01-01", periods=90, freq="B")
    close = pd.Series(np.linspace(100.0, 120.0, len(dates)), index=dates)
    df_prices = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1_000_000, 2_000_000, len(dates)),
        },
        index=dates,
    )

    built_by_family = build_price_ta_classic_feature_families("AAPL", df_prices)
    specs = _indicator_specs(ta)

    assert set(built_by_family) == set(TA_CLASSIC_FAMILY_PREFIXES)
    assert {spec.fn_name for family in specs.values() for spec in family} == {fn for fns in ta.Category.values() for fn in fns}
    min_feature_counts = {
        "technical_candles": 50,
        "technical_cycles": 5,
        "technical_math": 10,
        "technical_momentum": 20,
        "technical_overlap": 15,
        "technical_performance": 6,
    }
    for family_name, prefix in TA_CLASSIC_FAMILY_PREFIXES.items():
        built = built_by_family[family_name]
        assert built.feature_cols
        assert len(built.feature_cols) >= min_feature_counts[family_name]
        assert all(col.startswith(prefix) for col in built.feature_cols)
        assert built.df.index.names == ["date", "symbol"]

    grouped = infer_feature_family_columns(
        [col for built in built_by_family.values() for col in built.feature_cols]
    )
    for family_name in TA_CLASSIC_FAMILY_PREFIXES:
        assert grouped.get(family_name)


def test_short_price_histories_skip_long_window_indicators_without_warnings(caplog):
    dates = pd.date_range("2024-01-01", periods=13, freq="B")
    close = pd.Series(np.linspace(100.0, 103.0, len(dates)), index=dates)
    df_prices = pd.DataFrame(
        {
            "open": close - 0.25,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.linspace(1_000_000, 1_300_000, len(dates)),
        },
        index=dates,
    )

    with caplog.at_level(logging.WARNING):
        built_by_family = build_price_ta_classic_feature_families("NEW", df_prices)

    assert any(built.feature_cols for built in built_by_family.values())
    assert "indicator requires at least" not in caplog.text
