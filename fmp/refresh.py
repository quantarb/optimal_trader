from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

from django.utils import timezone

from data import FMPClient

from .endpoints.prices_div_adj import build as build_prices_div_adj_endpoint
from .models import Symbol
from .views import (
    _candidate_supports_date_window,
    _clone_candidates_with_date_window,
    _dedupe_records_by_record_date,
    _fetch_all_historical_records,
    _mark_section_fetched,
    _parse_iso_date,
    _run_with_retries,
    _save_historical_section,
    _sync_symbol_historical_ranges_from_db,
    _update_symbol_historical_range,
    _UNIVERSE_DOWNLOAD_RETRY_BASE_DELAY_S,
    _UNIVERSE_DOWNLOAD_RETRY_MAX_ATTEMPTS,
)


def refresh_symbol_price_history(
    symbol_obj: Symbol,
    client: FMPClient,
    *,
    target_start_date=None,
    target_end_date=None,
) -> dict[str, Any]:
    section = build_prices_div_adj_endpoint(symbol_obj)
    section_key = str(section.key)
    today = timezone.now().date()
    requested_end = min(target_end_date or today, today)
    requested_start = target_start_date
    if requested_start is None:
        history_years = int(section.min_history_years or 10)
        requested_start = requested_end - timedelta(days=365 * history_years)

    _sync_symbol_historical_ranges_from_db(symbol_obj, [section_key])
    symbol_obj.refresh_from_db(fields=["historical_date_ranges"])

    section_range = dict(symbol_obj.historical_date_ranges or {}).get(section_key) or {}
    min_date = _parse_iso_date(section_range.get("min_date"))
    max_date = _parse_iso_date(section_range.get("max_date"))
    has_date_window = _candidate_supports_date_window(section.candidates)

    fetch_ranges: list[tuple[Any, Any]] = []
    fetch_mode = "skip"
    if min_date is None or max_date is None:
        fetch_mode = "full"
    elif not has_date_window:
        if min_date > requested_start or max_date < requested_end:
            fetch_mode = "full"
    else:
        if min_date > requested_start:
            fetch_ranges.append((requested_start, min_date - timedelta(days=1)))
        if max_date < requested_end:
            fetch_ranges.append((max_date + timedelta(days=1), requested_end))
        if fetch_ranges:
            fetch_mode = "partial"

    fetched_records: list[dict[str, Any]] = []
    retries_used = 0
    if fetch_mode == "full":
        fetched, retries_used = _run_with_retries(
            lambda: _fetch_all_historical_records(client, section.candidates),
            max_attempts=_UNIVERSE_DOWNLOAD_RETRY_MAX_ATTEMPTS,
            base_delay_s=_UNIVERSE_DOWNLOAD_RETRY_BASE_DELAY_S,
        )
        fetched_records = list(fetched or [])
    elif fetch_mode == "partial":
        for from_date, to_date in fetch_ranges:
            if to_date < from_date:
                continue
            partial_candidates = _clone_candidates_with_date_window(
                section.candidates,
                from_date=from_date,
                to_date=to_date,
            )
            fetched, retries = _run_with_retries(
                lambda: _fetch_all_historical_records(client, partial_candidates),
                max_attempts=_UNIVERSE_DOWNLOAD_RETRY_MAX_ATTEMPTS,
                base_delay_s=_UNIVERSE_DOWNLOAD_RETRY_BASE_DELAY_S,
            )
            retries_used += retries
            fetched_records.extend(list(fetched or []))

    records = _dedupe_records_by_record_date(fetched_records)
    if records:
        _save_historical_section(symbol_obj, section_key, records)
        _update_symbol_historical_range(symbol_obj, section_key)
        _sync_symbol_historical_ranges_from_db(symbol_obj, [section_key])
    _mark_section_fetched(symbol_obj, section_key, section.kind)
    symbol_obj.refresh_from_db(fields=["historical_date_ranges"])

    final_range = dict(symbol_obj.historical_date_ranges or {}).get(section_key) or {}
    return {
        "symbol": str(symbol_obj.symbol).strip().upper(),
        "section_key": section_key,
        "fetch_mode": fetch_mode,
        "requested_start_date": requested_start.isoformat() if requested_start else None,
        "requested_end_date": requested_end.isoformat() if requested_end else None,
        "records_fetched": int(len(records)),
        "retries_used": int(retries_used),
        "range_after": final_range,
    }


def ensure_symbol_price_history(
    symbol_obj: Symbol,
    *,
    api_key: str | None = None,
    target_start_date=None,
    target_end_date=None,
) -> dict[str, Any]:
    resolved_api_key = str(api_key or os.getenv("FMP_API_KEY") or "").strip()
    if not resolved_api_key:
        raise ValueError("Missing FMP_API_KEY in environment/.env.")
    client = FMPClient(api_key=resolved_api_key, timeout_s=30.0, max_retries=2)
    return refresh_symbol_price_history(
        symbol_obj,
        client,
        target_start_date=target_start_date,
        target_end_date=target_end_date,
    )


__all__ = [
    "ensure_symbol_price_history",
    "refresh_symbol_price_history",
]
