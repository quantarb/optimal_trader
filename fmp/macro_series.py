from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Iterable

from django.db import transaction
from django.db.models import Max, Min
from django.utils import timezone

from data import FMPClient
from fmp.models import MacroObservation, MacroSeries

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


def resolve_fmp_client(client: FMPClient | None = None) -> FMPClient:
    if client is not None:
        return client
    if load_dotenv is not None:
        load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
    api_key = str(os.getenv("FMP_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("Missing FMP_API_KEY in environment/.env.")
    return FMPClient(api_key=api_key, timeout_s=30.0, max_retries=2)


@transaction.atomic
def save_dated_macro_series(
    *,
    code: str,
    display_name: str,
    category: str,
    rows: Iterable[dict[str, Any]],
    value_key: str,
    payload_builder: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, int | str | None]:
    series, _ = MacroSeries.objects.update_or_create(
        code=str(code),
        defaults={
            "display_name": str(display_name),
            "category": str(category),
            "last_fetched_at": timezone.now(),
        },
    )
    saved = 0
    for row in rows:
        observation_date = row.get("date")
        value = row.get(value_key)
        if observation_date is None or value is None:
            continue
        MacroObservation.objects.update_or_create(
            series=series,
            observation_date=observation_date,
            defaults={"value": float(value), "payload": payload_builder(row)},
        )
        saved += 1
    bounds = series.observations.aggregate(min_date=Min("observation_date"), max_date=Max("observation_date"))
    series.min_date = bounds["min_date"]
    series.max_date = bounds["max_date"]
    series.save(update_fields=["min_date", "max_date", "last_updated"])
    return {
        "code": series.code,
        "category": series.category,
        "observations_saved": saved,
        "min_date": series.min_date.isoformat() if series.min_date else None,
        "max_date": series.max_date.isoformat() if series.max_date else None,
    }
