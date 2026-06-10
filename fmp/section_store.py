from __future__ import annotations

import math
from typing import Any, Iterable

from django.db.models import Count, Max, Min
from django.utils import timezone

from fmp.models import Symbol, SymbolSectionHistorical, SymbolSectionSnapshot, SymbolSectionState
from fmp.records import extract_record_date, stable_record_key


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def mark_section_fetched(symbol: Symbol, section_key: str, kind: str) -> None:
    SymbolSectionState.objects.update_or_create(
        symbol=symbol,
        section_key=str(section_key),
        defaults={"kind": str(kind), "last_fetched_at": timezone.now()},
    )


def save_snapshot_section(symbol: Symbol, section_key: str, payload: Any) -> None:
    SymbolSectionSnapshot.objects.update_or_create(
        symbol=symbol,
        section_key=str(section_key),
        defaults={"payload": json_safe(payload)},
    )


def _record_key(record: Any, *, dedupe_by_date: bool) -> str:
    record_date = extract_record_date(record)
    if dedupe_by_date and record_date is not None:
        return stable_record_key({"date": record_date.isoformat()})
    return stable_record_key(record, prepare=json_safe)


def save_historical_section(
    symbol: Symbol,
    section_key: str,
    records: Iterable[Any],
    *,
    dedupe_by_date: bool = False,
) -> None:
    for record in records:
        payload = json_safe(record)
        record_date = extract_record_date(record)
        record_key = _record_key(record, dedupe_by_date=dedupe_by_date)
        if dedupe_by_date and record_date is not None:
            SymbolSectionHistorical.objects.filter(
                symbol=symbol,
                section_key=str(section_key),
                record_date=record_date,
            ).exclude(record_key=record_key).delete()
        SymbolSectionHistorical.objects.update_or_create(
            symbol=symbol,
            section_key=str(section_key),
            record_key=record_key,
            defaults={"record_date": record_date, "payload": payload},
        )


def repair_missing_record_dates(symbol: Symbol, section_key: str) -> None:
    queryset = SymbolSectionHistorical.objects.filter(
        symbol=symbol,
        section_key=str(section_key),
        record_date__isnull=True,
    )
    for item in queryset.iterator():
        parsed = extract_record_date(item.payload)
        if parsed is not None:
            item.record_date = parsed
            item.save(update_fields=["record_date"])


def update_symbol_historical_range(symbol: Symbol, section_key: str) -> None:
    aggregate = SymbolSectionHistorical.objects.filter(symbol=symbol, section_key=str(section_key)).aggregate(
        min_date=Min("record_date"), max_date=Max("record_date"), count=Count("id")
    )
    ranges = dict(symbol.historical_date_ranges or {})
    ranges[str(section_key)] = {
        "min_date": aggregate["min_date"].isoformat() if aggregate["min_date"] else None,
        "max_date": aggregate["max_date"].isoformat() if aggregate["max_date"] else None,
        "count": int(aggregate["count"] or 0),
    }
    symbol.historical_date_ranges = ranges
    symbol.save(update_fields=["historical_date_ranges"])


def sync_symbol_historical_ranges(symbol: Symbol, section_keys: Iterable[str]) -> None:
    keys = [str(key) for key in section_keys if str(key)]
    if not keys:
        return
    for key in keys:
        repair_missing_record_dates(symbol, key)
    rows = (
        SymbolSectionHistorical.objects.filter(symbol=symbol, section_key__in=keys)
        .values("section_key")
        .annotate(min_date=Min("record_date"), max_date=Max("record_date"), count=Count("id"))
    )
    ranges = dict(symbol.historical_date_ranges or {})
    changed = False
    for row in rows:
        payload = {
            "min_date": row["min_date"].isoformat() if row["min_date"] else None,
            "max_date": row["max_date"].isoformat() if row["max_date"] else None,
            "count": int(row["count"] or 0),
        }
        if ranges.get(row["section_key"]) != payload:
            ranges[row["section_key"]] = payload
            changed = True
    if changed:
        symbol.historical_date_ranges = ranges
        symbol.save(update_fields=["historical_date_ranges"])
