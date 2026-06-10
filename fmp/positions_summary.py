from __future__ import annotations

import math
from datetime import date
from typing import Any

import pandas as pd
from django.db import DatabaseError
from django.db.models import Count, Max, Min
from django.utils import timezone

# Note: Django models (PositionSummarySeries etc.) are imported locally inside
# the functions that touch the DB. This (plus from __future__ annotations)
# keeps importing this module safe before django.setup(). The tables are
# optional; callers fall back to raw section payloads when data/tables absent.


POSITIONS_SUMMARY_SECTION_KEY = "positions_summary"
POSITIONS_SUMMARY_LOOKBACK_QUARTERS = 8

_INT_CANDIDATES = {
    "investor_count": (
        "investorcount",
        "investorscount",
        "holdercount",
        "holderscount",
        "institutioncount",
        "institutioncount",
        "institutionalholders",
        "numberofinvestors",
        "numberofholders",
    ),
    "call_count": ("callcount", "calls", "callscount", "callpositions"),
    "put_count": ("putcount", "puts", "putscount", "putpositions"),
}

_FLOAT_CANDIDATES = {
    "shares_held": (
        "sharesheld",
        "totalshares",
        "sharecount",
        "shares",
        "institutionshares",
        "institutionalshares",
        "positionshares",
    ),
    "investment_value": (
        "investmentvalue",
        "totalinvestmentvalue",
        "marketvalue",
        "value",
        "positionvalue",
        "portfolio_value",
    ),
    "ownership_pct": (
        "ownershippct",
        "ownershippercentage",
        "ownershippercent",
        "ownership",
        "institutionownershippct",
        "institutionalownershippct",
    ),
    "shares_change": (
        "shareschange",
        "changeinshares",
        "sharechange",
        "delta_shares",
    ),
    "investment_change": (
        "investmentchange",
        "changeininvestment",
        "valuechange",
        "delta_value",
    ),
    "ownership_pct_change": (
        "ownershippctchange",
        "ownershippercentagechange",
        "changeinownership",
        "ownershipchange",
    ),
    "put_call_ratio": (
        "putcallratio",
        "put_call_ratio",
        "putcall",
    ),
}

_YEAR_KEYS = ("year", "reportyear", "calendaryear", "fiscalyear")
_QUARTER_KEYS = ("quarter", "reportquarter", "calendarquarter", "fiscalquarter")
_DATE_KEYS = ("date", "period", "reportdate", "asofdate", "calendar_date")


def _normalize_payload_keys(payload: dict[str, Any]) -> dict[str, Any]:
    return {str(key).strip().lower().replace("-", "").replace("_", ""): value for key, value in dict(payload or {}).items()}


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        out = float(value)
        return out if pd.notna(out) and math.isfinite(out) else None
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        out = int(float(value))
        return out
    except Exception:
        return None


def _quarter_end_date(year: int | None, quarter: int | None) -> date | None:
    if year is None or quarter is None:
        return None
    year = int(year)
    quarter = max(1, min(4, int(quarter)))
    month_day = {
        1: (3, 31),
        2: (6, 30),
        3: (9, 30),
        4: (12, 31),
    }[quarter]
    return date(year, month_day[0], month_day[1])


def _first_key_value(payload: dict[str, Any], candidates: tuple[str, ...]) -> Any:
    for key in candidates:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def extract_positions_summary_period(payload: dict[str, Any]) -> tuple[int | None, int | None, date | None]:
    normalized = _normalize_payload_keys(payload)
    year = _safe_int(_first_key_value(normalized, _YEAR_KEYS))
    quarter = _safe_int(_first_key_value(normalized, _QUARTER_KEYS))
    report_date = _quarter_end_date(year, quarter)
    if report_date is None:
        for key in _DATE_KEYS:
            raw = normalized.get(key)
            if raw in (None, ""):
                continue
            parsed = pd.to_datetime(raw, errors="coerce")
            if pd.notna(parsed):
                report_date = parsed.normalize().date()
                break
    return year, quarter, report_date


def extract_positions_summary_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_payload_keys(payload)
    out: dict[str, Any] = {}
    for field_name, candidates in _INT_CANDIDATES.items():
        out[field_name] = _safe_int(_first_key_value(normalized, candidates))
    for field_name, candidates in _FLOAT_CANDIDATES.items():
        out[field_name] = _safe_float(_first_key_value(normalized, candidates))
    return out


def normalize_positions_summary_record(symbol: Symbol, payload: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    year, quarter, report_date = extract_positions_summary_period(payload)
    if year is None or quarter is None:
        return None
    metrics = extract_positions_summary_metrics(payload)
    return {
        "symbol": str(symbol.symbol).strip().upper(),
        "report_year": int(year),
        "report_quarter": int(quarter),
        "report_date": report_date,
        "date": pd.Timestamp(report_date) if report_date is not None else None,
        "year": int(year),
        "quarter": int(quarter),
        **metrics,
        "payload": dict(payload),
    }


def save_positions_summary_records(symbol: Symbol, records: list[dict[str, Any]]) -> None:
    from .models import PositionSummaryObservation, PositionSummarySeries

    normalized_records = [normalize_positions_summary_record(symbol, record) for record in list(records or [])]
    normalized_records = [record for record in normalized_records if record is not None]
    try:
        series, _created = PositionSummarySeries.objects.get_or_create(symbol=symbol)
    except DatabaseError:
        # Table(s) do not exist (migrations not run, or using an older DB snapshot).
        # The raw section payload path can still provide data for features.
        return
    if not normalized_records:
        try:
            series.last_fetched_at = timezone.now()
            series.save(update_fields=["last_fetched_at", "last_updated"])
        except DatabaseError:
            pass
        return

    for record in normalized_records:
        try:
            PositionSummaryObservation.objects.update_or_create(
                series=series,
                report_year=int(record["report_year"]),
                report_quarter=int(record["report_quarter"]),
                defaults={
                    "report_date": record.get("report_date"),
                    "investor_count": record.get("investor_count"),
                    "shares_held": record.get("shares_held"),
                    "investment_value": record.get("investment_value"),
                    "ownership_pct": record.get("ownership_pct"),
                    "shares_change": record.get("shares_change"),
                    "investment_change": record.get("investment_change"),
                    "ownership_pct_change": record.get("ownership_pct_change"),
                    "put_call_ratio": record.get("put_call_ratio"),
                    "call_count": record.get("call_count"),
                    "put_count": record.get("put_count"),
                    "payload": record.get("payload") or {},
                },
            )
        except DatabaseError:
            return

    try:
        aggregate = PositionSummaryObservation.objects.filter(series=series).aggregate(
            min_date=Min("report_date"),
            max_date=Max("report_date"),
            count=Count("id"),
        )
        series.last_fetched_at = timezone.now()
        series.min_report_date = aggregate["min_date"]
        series.max_report_date = aggregate["max_date"]
        latest_row = max(normalized_records, key=lambda record: (int(record["report_year"]), int(record["report_quarter"])))
        series.last_year = int(latest_row["report_year"])
        series.last_quarter = int(latest_row["report_quarter"])
        series.report_count = int(aggregate["count"] or 0)
        series.save(
            update_fields=[
                "last_fetched_at",
                "min_report_date",
                "max_report_date",
                "last_year",
                "last_quarter",
                "report_count",
                "last_updated",
            ]
        )
    except DatabaseError:
        pass


def load_positions_summary_frame(symbol: Symbol) -> pd.DataFrame:
    from .models import PositionSummaryObservation, PositionSummarySeries

    try:
        series = PositionSummarySeries.objects.filter(symbol=symbol).first()
    except DatabaseError:
        # Table does not exist yet (migrations not applied for positions-summary models,
        # or the DB is an older snapshot). Fall back to raw section payload or no data.
        return pd.DataFrame()
    if series is None:
        return pd.DataFrame()
    try:
        rows = (
            PositionSummaryObservation.objects.filter(series=series)
            .order_by("report_date", "report_year", "report_quarter")
            .values(
                "report_date",
                "report_year",
                "report_quarter",
                "investor_count",
                "shares_held",
                "investment_value",
                "ownership_pct",
                "shares_change",
                "investment_change",
                "ownership_pct_change",
                "put_call_ratio",
                "call_count",
                "put_count",
            )
        )
        records = list(rows)
    except DatabaseError:
        return pd.DataFrame()
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    frame["date"] = pd.to_datetime(frame.pop("report_date"), errors="coerce")
    frame["symbol"] = str(symbol.symbol).strip().upper()
    frame = frame.dropna(subset=["date"]).sort_values(["date", "symbol"])
    if frame.empty:
        return pd.DataFrame()
    return frame.set_index(["date", "symbol"]).sort_index()


__all__ = [
    "POSITIONS_SUMMARY_LOOKBACK_QUARTERS",
    "POSITIONS_SUMMARY_SECTION_KEY",
    "extract_positions_summary_metrics",
    "extract_positions_summary_period",
    "load_positions_summary_frame",
    "normalize_positions_summary_record",
    "save_positions_summary_records",
]
