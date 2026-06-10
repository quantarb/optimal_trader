from __future__ import annotations

from typing import Any

import pandas as pd


def expected_latest_price_date_from_market_clock() -> Any:
    """Return the latest date for which we expect complete market prices.

    After 5pm ET on a weekday we treat "today" as complete; otherwise the
    previous business day.
    """
    now_et = pd.Timestamp.now(tz="America/New_York")
    if now_et.weekday() < 5 and now_et.hour >= 17:
        return now_et.date()
    return (now_et.normalize() - pd.offsets.BDay(1)).date()
