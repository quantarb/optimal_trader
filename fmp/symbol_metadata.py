from __future__ import annotations

import os
import math
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from django.utils import timezone

from fmp.models import Symbol, SymbolSectionState
from fmp.section_store import json_safe, mark_section_fetched, save_snapshot_section
from fmp.symbol_dates import payload_listing_date


PROFILE_SECTION_KEY = "profile"
PROFILE_REFRESH_DAYS = 30
_REPO_DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"
PROFILE_FIELD_ALIASES = {
    "company_name": ("companyName", "name"),
    "exchange": ("exchangeShortName", "exchange"),
    "country": ("country",),
    "sector": ("sector",),
    "industry": ("industry",),
    "market_cap": ("marketCap",),
    "price": ("price",),
    "beta": ("beta",),
    "volume": ("volume",),
    "dividend": ("lastDividend", "dividend"),
    "dividend_yield": ("dividendYield",),
}


def _first_value(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
    return parsed if parsed is not None and math.isfinite(parsed) else None


def profile_record(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        return next((dict(item) for item in payload if isinstance(item, dict)), {})
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                record = next((dict(item) for item in value if isinstance(item, dict)), None)
                if record is not None:
                    return record
        return dict(payload)
    return {}


def symbol_metadata_missing(symbol: Symbol) -> list[str]:
    missing = [
        field
        for field in ("company_name", "exchange", "country", "sector", "industry")
        if not str(getattr(symbol, field, "") or "").strip()
    ]
    if payload_listing_date(symbol.payload) is None:
        missing.append("ipo_date")
    if not symbol.section_snapshots.filter(section_key=PROFILE_SECTION_KEY).exists():
        missing.append("profile_snapshot")
    return missing


def incomplete_symbol_metadata(symbols: Iterable[str]) -> dict[str, list[str]]:
    normalized = [str(value).strip().upper() for value in symbols if str(value).strip()]
    found = {symbol.symbol: symbol for symbol in Symbol.objects.filter(symbol__in=normalized)}
    return {
        code: (["symbol_record"] if code not in found else symbol_metadata_missing(found[code]))
        for code in normalized
        if code not in found or symbol_metadata_missing(found[code])
    }


def sync_symbol_metadata_from_fmp(
    *,
    symbols: Iterable[str],
    client=None,
    force: bool = False,
    progress_logger=None,
) -> pd.DataFrame:
    normalized = [str(value).strip().upper() for value in symbols if str(value).strip()]
    sync_results = refresh_symbol_metadata_from_fmp(
        symbols=normalized,
        client=client,
        force=force,
        progress_logger=progress_logger,
    )
    incomplete = incomplete_symbol_metadata(normalized)
    if incomplete:
        preview = ", ".join(
            f"{symbol}({','.join(fields)})"
            for symbol, fields in list(incomplete.items())[:20]
        )
        hidden = max(0, len(incomplete) - 20)
        suffix = f" and {hidden} more" if hidden else ""
        raise RuntimeError(
            "Required symbol metadata is unavailable after FMP profile repair: "
            f"{preview}{suffix}."
        )
    return sync_results


def profile_fetch_is_recent(symbol: Symbol, *, now=None, threshold_days: int = PROFILE_REFRESH_DAYS) -> bool:
    now = now or timezone.now()
    state = SymbolSectionState.objects.filter(symbol=symbol, section_key=PROFILE_SECTION_KEY).first()
    return bool(
        state
        and state.last_fetched_at
        and state.last_fetched_at >= now - timedelta(days=max(1, int(threshold_days)))
    )


def apply_profile_metadata(symbol: Symbol, payload: Any) -> list[str]:
    record = json_safe(profile_record(payload))
    if not isinstance(record, dict):
        return []
    if not record:
        return []

    # Retain non-profile metadata collected by other endpoints, while storing
    # every key returned by the authoritative profile response, including
    # explicit nulls.
    merged_payload = json_safe(dict(symbol.payload or {}))
    merged_payload.update({str(key): value for key, value in record.items()})
    changed_fields: list[str] = []
    for model_field, aliases in PROFILE_FIELD_ALIASES.items():
        raw_value = _first_value(record, aliases)
        if raw_value in (None, ""):
            continue
        value = (
            _float_or_none(raw_value)
            if model_field in {"market_cap", "price", "beta", "volume", "dividend", "dividend_yield"}
            else str(raw_value).strip()
        )
        if value is None or getattr(symbol, model_field) == value:
            continue
        setattr(symbol, model_field, value)
        changed_fields.append(model_field)
    if merged_payload != dict(symbol.payload or {}):
        symbol.payload = merged_payload
        changed_fields.append("payload")
    if changed_fields:
        symbol.save(update_fields=list(dict.fromkeys(changed_fields)))
    return list(dict.fromkeys(changed_fields))


def refresh_symbol_metadata_from_fmp(
    *,
    symbols: Iterable[str] | None = None,
    client=None,
    force: bool = False,
    max_symbols: int | None = None,
    progress_logger=None,
) -> pd.DataFrame:
    normalized = [str(value).strip().upper() for value in list(symbols or []) if str(value).strip()]
    if normalized:
        existing = set(Symbol.objects.filter(symbol__in=normalized).values_list("symbol", flat=True))
        Symbol.objects.bulk_create(
            [Symbol(symbol=symbol) for symbol in normalized if symbol not in existing],
            ignore_conflicts=True,
        )
    queryset = Symbol.objects.all().order_by("symbol")
    if normalized:
        queryset = queryset.filter(symbol__in=normalized)

    # Validate the complete selected universe, but only incomplete symbols
    # require a profile request. A fetched profile is still persisted in full.
    targets = list(queryset.iterator())
    if max_symbols is not None:
        targets = targets[: max(0, int(max_symbols))]
    if client is None and targets:
        from infra.fmp.client import FMPClient

        api_key = str(os.getenv("FMP_API_KEY") or "").strip()
        if not api_key and _REPO_DOTENV_PATH.exists():
            for raw_line in _REPO_DOTENV_PATH.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "FMP_API_KEY":
                    api_key = value.strip().strip('"').strip("'")
                    break
        if not api_key:
            raise ValueError("Missing FMP_API_KEY in environment/.env.")
        client = FMPClient(api_key=api_key, timeout_s=30.0, max_retries=2)

    rows: list[dict[str, Any]] = []
    total = len(targets)
    for index, symbol in enumerate(targets, start=1):
        missing_before = symbol_metadata_missing(symbol)
        if not missing_before:
            rows.append({
                "symbol": symbol.symbol,
                "status": "skipped_complete",
                "missing_before": [],
                "missing_after": [],
                "updated_fields": [],
            })
            continue
        if not force and profile_fetch_is_recent(symbol):
            rows.append({
                "symbol": symbol.symbol,
                "status": "skipped_recent",
                "missing_before": missing_before,
                "missing_after": missing_before,
                "updated_fields": [],
            })
            continue
        try:
            payload = client.get_json("/stable/profile", params={"symbol": symbol.symbol})
            record = profile_record(payload)
            updated_fields = apply_profile_metadata(symbol, record)
            save_snapshot_section(symbol, PROFILE_SECTION_KEY, payload)
            mark_section_fetched(symbol, PROFILE_SECTION_KEY, "snapshot")
            symbol.refresh_from_db()
            missing_after = symbol_metadata_missing(symbol)
            status = "updated" if updated_fields else ("empty_profile" if not record else "unchanged")
            rows.append({
                "symbol": symbol.symbol,
                "status": status,
                "missing_before": missing_before,
                "missing_after": missing_after,
                "updated_fields": updated_fields,
            })
        except Exception as exc:
            rows.append({
                "symbol": symbol.symbol,
                "status": "error",
                "missing_before": missing_before,
                "missing_after": missing_before,
                "updated_fields": [],
                "error": str(exc),
            })
        if callable(progress_logger) and (index == 1 or index % 25 == 0 or index == total):
            progress_logger(f"FMP symbol metadata sync progress: {index:,}/{total:,} symbols processed")
    return pd.DataFrame(rows)


__all__ = [
    "apply_profile_metadata",
    "sync_symbol_metadata_from_fmp",
    "incomplete_symbol_metadata",
    "profile_fetch_is_recent",
    "profile_record",
    "refresh_symbol_metadata_from_fmp",
    "symbol_metadata_missing",
]
