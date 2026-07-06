from __future__ import annotations

from platforms.brokers.option_pricing import normalize_option_limit_price, option_limit_tick_size


def test_option_tick_size_uses_penny_below_three_and_nickel_at_three_or_above():
    assert option_limit_tick_size(2.99) == 0.01
    assert option_limit_tick_size(3.00) == 0.05
    assert option_limit_tick_size(70.17) == 0.05


def test_option_limit_normalization_uses_executable_side_ticks():
    assert normalize_option_limit_price(70.17, side="buy") == 70.15
    assert normalize_option_limit_price(70.17, side="sell") == 70.20
    assert normalize_option_limit_price(1.999999999, side="buy") == 1.99

