from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd
from data import FMPClient
from fmp.macro_series import resolve_fmp_client, save_dated_macro_series
from fmp.models import Symbol


SECTOR_PERFORMANCE_CATEGORY = "sector_performance"
INDUSTRY_PERFORMANCE_CATEGORY = "industry_performance"


@dataclass(frozen=True)
class ClassificationPerformanceKey:
    category: str
    classification: str
    exchange: str


def classification_performance_series_code(category: str, classification: str, exchange: str) -> str:
    identity = "|".join((str(category).strip(), str(classification).strip(), str(exchange).strip().upper()))
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:20]
    prefix = "sector_perf" if category == SECTOR_PERFORMANCE_CATEGORY else "industry_perf"
    return f"{prefix}__{digest}"


def symbol_classification_performance_keys(symbols: Iterable[Symbol]) -> list[ClassificationPerformanceKey]:
    keys: set[ClassificationPerformanceKey] = set()
    for symbol in symbols:
        exchange = str(symbol.exchange or "").strip().upper()
        sector = str(symbol.sector or "").strip()
        industry = str(symbol.industry or "").strip()
        if exchange and sector:
            keys.add(ClassificationPerformanceKey(SECTOR_PERFORMANCE_CATEGORY, sector, exchange))
        if exchange and industry:
            keys.add(ClassificationPerformanceKey(INDUSTRY_PERFORMANCE_CATEGORY, industry, exchange))
    return sorted(keys, key=lambda item: (item.category, item.exchange, item.classification))


def normalize_classification_performance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["date", "return_decimal", "average_change_pct"])
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized.get("date"), errors="coerce")
    normalized["average_change_pct"] = pd.to_numeric(normalized.get("averageChange"), errors="coerce")
    normalized = normalized.dropna(subset=["date", "average_change_pct"]).copy()
    normalized["date"] = normalized["date"].dt.date
    normalized["return_decimal"] = normalized["average_change_pct"] / 100.0
    return normalized.sort_values("date").drop_duplicates("date", keep="last")


def _fetch_frame(client: FMPClient, key: ClassificationPerformanceKey, start_date: date, end_date: date) -> pd.DataFrame:
    kwargs = {"from_date": start_date.isoformat(), "to_date": end_date.isoformat()}
    if key.category == SECTOR_PERFORMANCE_CATEGORY:
        return client.historical_sector_performance(key.classification, key.exchange, **kwargs)
    return client.historical_industry_performance(key.classification, key.exchange, **kwargs)


def save_classification_performance_frame(
    key: ClassificationPerformanceKey,
    frame: pd.DataFrame,
) -> dict[str, int | str | None]:
    normalized = normalize_classification_performance_frame(frame)
    code = classification_performance_series_code(key.category, key.classification, key.exchange)
    result = save_dated_macro_series(
        code=code,
        display_name=f"{key.classification} ({key.exchange})",
        category=key.category,
        rows=normalized.to_dict("records"),
        value_key="return_decimal",
        payload_builder=lambda row: {
            "date": row["date"].isoformat(),
            "averageChange": float(row["average_change_pct"]),
            "exchange": key.exchange,
            "classification": key.classification,
            "category": key.category,
        },
    )
    return {
        **result,
        "classification": key.classification,
        "exchange": key.exchange,
    }


def refresh_classification_performance(
    symbols: Iterable[Symbol],
    *,
    start_date: date,
    end_date: date,
    client: FMPClient | None = None,
) -> list[dict[str, int | str | None]]:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    client = resolve_fmp_client(client)
    results = []
    for key in symbol_classification_performance_keys(symbols):
        results.append(save_classification_performance_frame(key, _fetch_frame(client, key, start_date, end_date)))
    return results


__all__ = [
    "ClassificationPerformanceKey",
    "INDUSTRY_PERFORMANCE_CATEGORY",
    "SECTOR_PERFORMANCE_CATEGORY",
    "classification_performance_series_code",
    "normalize_classification_performance_frame",
    "refresh_classification_performance",
    "save_classification_performance_frame",
    "symbol_classification_performance_keys",
]
