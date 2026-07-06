from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from typing import Any

import pandas as pd

OPTION_PENNY_TICK_THRESHOLD = 3.00
OPTION_PENNY_TICK_SIZE = 0.01
OPTION_NICKEL_TICK_SIZE = 0.05


def positive_float(value: Any) -> float | None:
    try:
        number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    except Exception:
        return None
    if pd.isna(number):
        return None
    value_float = float(number)
    if value_float <= 0.0:
        return None
    return value_float


def option_limit_tick_size(price: float | None) -> float | None:
    number = positive_float(price)
    if number is None:
        return None
    if number < OPTION_PENNY_TICK_THRESHOLD:
        return OPTION_PENNY_TICK_SIZE
    return OPTION_NICKEL_TICK_SIZE


def _decimal_from_float(value: float) -> Decimal:
    return Decimal(format(float(value), ".12g"))


def normalize_option_limit_price(price: float | None, *, side: str = "nearest") -> float | None:
    number = positive_float(price)
    if number is None:
        return None
    tick = option_limit_tick_size(number)
    if tick is None:
        return None

    side_name = str(side or "nearest").strip().lower()
    if side_name in {"buy", "bid", "floor"}:
        rounding = ROUND_FLOOR
    elif side_name in {"sell", "ask", "ceil", "ceiling"}:
        rounding = ROUND_CEILING
    else:
        rounding = ROUND_HALF_UP

    decimal_number = _decimal_from_float(number)
    decimal_tick = _decimal_from_float(tick)
    normalized = (decimal_number / decimal_tick).to_integral_value(rounding=rounding) * decimal_tick
    if normalized <= 0:
        return None
    return float(normalized.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
