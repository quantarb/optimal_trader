from __future__ import annotations

import pandas as pd

from data.warehouse import load_warehouse_price_frames


def load_adjusted_price_frames(
    symbols: list[str],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    normalized = [str(symbol).strip().upper() for symbol in list(symbols or []) if str(symbol).strip()]
    if not normalized:
        return {}

    return load_warehouse_price_frames(normalized, start_date=start_date, end_date=end_date)


__all__ = ["load_adjusted_price_frames"]