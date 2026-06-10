from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Any, Callable, Iterable


DATE_KEYS = (
    "date",
    "publishedDate",
    "publishedAt",
    "published",
    "filingDate",
    "acceptedDate",
    "recordDate",
    "periodOfReport",
    "calendarYear",
)


def extract_record_date(record: Any) -> date | None:
    if not isinstance(record, dict):
        return None
    for key in DATE_KEYS:
        value = record.get(key)
        if value in (None, ""):
            continue
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
        except (TypeError, ValueError):
            try:
                return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
            except (TypeError, ValueError):
                continue
    return None


def stable_record_key(record: Any, *, prepare: Callable[[Any], Any] | None = None) -> str:
    payload = prepare(record) if prepare is not None else record
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def dedupe_historical_records(
    records: Iterable[Any],
    *,
    by_date: bool = False,
    prepare: Callable[[Any], Any] | None = None,
) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for raw_record in records:
        record = raw_record if isinstance(raw_record, dict) else {"value": raw_record}
        record_date = extract_record_date(record)
        if by_date and record_date is not None:
            identity = f"date:{record_date.isoformat()}"
        else:
            identity = f"record:{stable_record_key(record, prepare=prepare)}"
        unique[identity] = record
    return sorted(
        unique.values(),
        key=lambda item: (extract_record_date(item) or date.min, stable_record_key(item, prepare=prepare)),
    )
