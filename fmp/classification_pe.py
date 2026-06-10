from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd
from data import FMPClient
from fmp.macro_series import resolve_fmp_client, save_dated_macro_series
from fmp.models import Symbol


SECTOR_PE_CATEGORY = "sector_pe"
INDUSTRY_PE_CATEGORY = "industry_pe"


@dataclass(frozen=True)
class ClassificationPEKey:
    category: str
    classification: str
    exchange: str


def classification_pe_series_code(category: str, classification: str, exchange: str) -> str:
    identity = "|".join((str(category).strip(), str(classification).strip(), str(exchange).strip().upper()))
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:20]
    prefix = "sector_pe" if category == SECTOR_PE_CATEGORY else "industry_pe"
    return f"{prefix}__{digest}"


def symbol_classification_pe_keys(symbols: Iterable[Symbol]) -> list[ClassificationPEKey]:
    keys: set[ClassificationPEKey] = set()
    for symbol in symbols:
        exchange = str(symbol.exchange or "").strip().upper()
        sector = str(symbol.sector or "").strip()
        industry = str(symbol.industry or "").strip()
        if exchange and sector:
            keys.add(ClassificationPEKey(SECTOR_PE_CATEGORY, sector, exchange))
        if exchange and industry:
            keys.add(ClassificationPEKey(INDUSTRY_PE_CATEGORY, industry, exchange))
    return sorted(keys, key=lambda item: (item.category, item.exchange, item.classification))


def normalize_classification_pe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["date", "pe"])
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized.get("date"), errors="coerce")
    normalized["pe"] = pd.to_numeric(normalized.get("pe"), errors="coerce")
    normalized = normalized.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=["date", "pe"])
    normalized["date"] = normalized["date"].dt.date
    return normalized.sort_values("date").drop_duplicates("date", keep="last")


def _fetch_frame(client: FMPClient, key: ClassificationPEKey, start_date: date, end_date: date) -> pd.DataFrame:
    kwargs = {"from_date": start_date.isoformat(), "to_date": end_date.isoformat()}
    if key.category == SECTOR_PE_CATEGORY:
        return client.historical_sector_pe(key.classification, key.exchange, **kwargs)
    return client.historical_industry_pe(key.classification, key.exchange, **kwargs)


def save_classification_pe_frame(key: ClassificationPEKey, frame: pd.DataFrame) -> dict[str, int | str | None]:
    normalized = normalize_classification_pe_frame(frame)
    code = classification_pe_series_code(key.category, key.classification, key.exchange)
    result = save_dated_macro_series(
        code=code,
        display_name=f"{key.classification} P/E ({key.exchange})",
        category=key.category,
        rows=normalized.to_dict("records"),
        value_key="pe",
        payload_builder=lambda row: {
            "date": row["date"].isoformat(),
            "pe": float(row["pe"]),
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


def refresh_classification_pe(
    symbols: Iterable[Symbol],
    *,
    start_date: date,
    end_date: date,
    client: FMPClient | None = None,
) -> list[dict[str, int | str | None]]:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    client = resolve_fmp_client(client)
    return [
        save_classification_pe_frame(key, _fetch_frame(client, key, start_date, end_date))
        for key in symbol_classification_pe_keys(symbols)
    ]
