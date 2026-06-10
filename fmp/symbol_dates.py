from __future__ import annotations

from datetime import date
from typing import Any


LISTING_DATE_KEYS = (
    "ipodate",
    "ipo",
    "firsttradedate",
    "listingdate",
    "listeddate",
)


def parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    try:
        from datetime import datetime

        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except (TypeError, ValueError):
        try:
            from datetime import datetime

            return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
        except (TypeError, ValueError):
            return None


def payload_listing_date(payload: Any) -> date | None:
    if not isinstance(payload, dict):
        return None
    normalized = {str(key).lower().replace("_", ""): value for key, value in payload.items()}
    for key in LISTING_DATE_KEYS:
        parsed = parse_date(normalized.get(key))
        if parsed is not None:
            return parsed
    return None


def symbol_listing_date(symbol) -> date | None:
    return payload_listing_date(getattr(symbol, "payload", None))


def effective_symbol_history_start(
    symbol,
    configured_start: date,
) -> date:
    listing_date = symbol_listing_date(symbol)
    if listing_date is None:
        return configured_start
    return max(configured_start, listing_date)


__all__ = [
    "effective_symbol_history_start",
    "parse_date",
    "payload_listing_date",
    "symbol_listing_date",
]
