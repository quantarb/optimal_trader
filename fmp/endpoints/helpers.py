from __future__ import annotations

from datetime import date

from django.utils import timezone


HISTORICAL_TARGET_MIN_DATE = timezone.datetime(1900, 1, 1).date()
GRANULARITY_PREFERENCE = ("day", "quarter", "annual")
DEFAULT_LIMIT = 10_000
DEFAULT_MAX_PAGES = 1_000


def recent_year_quarters(count: int = 8) -> list[tuple[int, int]]:
    now = timezone.now()
    year = now.year
    quarter = ((now.month - 1) // 3) + 1
    out: list[tuple[int, int]] = []
    for _ in range(max(1, count)):
        out.append((year, quarter))
        quarter -= 1
        if quarter == 0:
            quarter = 4
            year -= 1
    return out


def today() -> date:
    return timezone.now().date()


def preferred_period(*supported_periods: str) -> str:
    supported = {str(value).strip().lower() for value in supported_periods if str(value).strip()}
    for candidate in GRANULARITY_PREFERENCE:
        if candidate in supported:
            return candidate
    raise ValueError("No supported periods provided.")


def period_limit_params(limit: int, *supported_periods: str) -> dict[str, int | str]:
    if not supported_periods:
        raise ValueError("supported_periods must be provided explicitly.")
    return {
        "period": preferred_period(*supported_periods),
        "limit": DEFAULT_LIMIT,
    }


def limit_params() -> dict[str, int]:
    return {
        "limit": DEFAULT_LIMIT,
    }


def paginated_params(*, page: int = 0) -> dict[str, int]:
    return {
        "page": int(page),
        "limit": DEFAULT_LIMIT,
    }


def full_history_range_params() -> dict[str, str]:
    return {
        "from": HISTORICAL_TARGET_MIN_DATE.isoformat(),
        "to": today().isoformat(),
    }
