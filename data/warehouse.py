from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Iterable

import pandas as pd

if TYPE_CHECKING:
    from quant_warehouse import Warehouse

try:
    from quant_warehouse.feature_engineering.fundamentals import (
        django_only_sections_for_refresh as _qw_django_only_sections_for_refresh,
        warehouse_section_for_django as _qw_warehouse_section_for_django,
        warehouse_sections_for_django_keys as _qw_warehouse_sections_for_django_keys,
        warehouse_sections_for_refresh as _qw_warehouse_sections_for_refresh,
        warehouse_section_to_indexed_frame as _qw_warehouse_section_to_indexed_frame,
        warehouse_section_to_payload_rows as _qw_warehouse_section_to_payload_rows,
    )
except Exception:  # pragma: no cover - quant-warehouse optional at import time
    _qw_django_only_sections_for_refresh = None
    _qw_warehouse_section_for_django = None
    _qw_warehouse_sections_for_django_keys = None
    _qw_warehouse_sections_for_refresh = None
    _qw_warehouse_section_to_indexed_frame = None
    _qw_warehouse_section_to_payload_rows = None

DJANGO_HISTORICAL_SECTION_MAP = {
    "income_statement": "income",
    "balance_sheet": "balance",
    "cash_flow": "cash",
    "key_metrics": "metrics",
    "ratios": "ratios",
    "income_statement_growth": "income_growth",
    "balance_sheet_growth": "balance_growth",
    "cash_flow_growth": "cash_growth",
    "dividends": "dividends",
    "splits": "historical_splits",
    "earnings": "earnings",
    "financial_growth": "financial_growth",
    "senate_trading": "senate_trading",
    "income_statement_ttm": "income_ttm",
    "balance_sheet_ttm": "balance_ttm",
    "cash_flow_ttm": "cash_ttm",
    "key_metrics_ttm": "metrics_ttm",
    "ratios_ttm": "ratios_ttm",
}
DJANGO_ONLY_FUNDAMENTAL_SECTIONS = frozenset(
    {"income_ttm", "balance_ttm", "cash_ttm", "metrics_ttm", "ratios_ttm"}
)


def _setting(name: str, default: object) -> object:
    try:
        from django.conf import settings

        return getattr(settings, name, default)
    except Exception:
        return os.getenv(name, default)


def _setting_bool(name: str, *, default: bool) -> bool:
    raw = _setting(name, "1" if default else "0")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def read_prices_from_warehouse() -> bool:
    return _setting_bool("QW_READ_PRICES", default=True)


def read_fundamentals_from_warehouse() -> bool:
    return _setting_bool("QW_READ_FUNDAMENTALS", default=True)


def read_macro_from_warehouse() -> bool:
    return _setting_bool("QW_READ_MACRO", default=True)


def price_provider() -> str:
    return str(_setting("QW_PRICE_PROVIDER", "fmp")).strip().lower()


def price_providers_for_refresh() -> tuple[str, ...]:
    """Primary read provider plus yfinance fallback for warehouse gap-fill refresh."""
    primary = price_provider()
    providers: list[str] = [primary]
    if primary != "yfinance":
        providers.append("yfinance")
    return tuple(providers)


def fundamental_provider() -> str:
    return str(_setting("QW_FUNDAMENTAL_PROVIDER", "fmp")).strip().lower()


def macro_provider() -> str:
    return str(_setting("QW_MACRO_PROVIDER", "fmp")).strip().lower()


@lru_cache(maxsize=1)
def get_warehouse() -> Warehouse:
    from quant_warehouse import Warehouse

    return Warehouse()


def warehouse_section_for_django(section_key: str) -> str | None:
    if _qw_warehouse_section_for_django is not None:
        return _qw_warehouse_section_for_django(str(section_key).strip())
    section = DJANGO_HISTORICAL_SECTION_MAP.get(str(section_key).strip())
    if section in DJANGO_ONLY_FUNDAMENTAL_SECTIONS:
        return None
    return section


def warehouse_sections_for_django_keys(django_section_keys: Iterable[str]) -> tuple[str, ...]:
    if _qw_warehouse_sections_for_django_keys is not None:
        return _qw_warehouse_sections_for_django_keys(django_section_keys)
    mapped: list[str] = []
    seen: set[str] = set()
    for django_key in django_section_keys:
        key = str(django_key or "").strip()
        if not key or key == "prices_div_adj":
            continue
        warehouse_key = warehouse_section_for_django(key)
        if not warehouse_key or warehouse_key in seen:
            continue
        seen.add(warehouse_key)
        mapped.append(warehouse_key)
    return tuple(mapped)


def _django_only_warehouse_sections() -> frozenset[str]:
    try:
        from quant_warehouse.warehouse.sections import DJANGO_ONLY_FUNDAMENTAL_SECTIONS

        return frozenset(DJANGO_ONLY_FUNDAMENTAL_SECTIONS)
    except Exception:
        return DJANGO_ONLY_FUNDAMENTAL_SECTIONS


def warehouse_sections_for_refresh(django_section_keys: Iterable[str]) -> tuple[str, ...]:
    if _qw_warehouse_sections_for_refresh is not None:
        return _qw_warehouse_sections_for_refresh(django_section_keys)
    blocked = _django_only_warehouse_sections()
    return tuple(
        section
        for section in warehouse_sections_for_django_keys(django_section_keys)
        if section not in blocked
    )


def django_only_sections_for_refresh(django_section_keys: Iterable[str]) -> tuple[str, ...]:
    if _qw_django_only_sections_for_refresh is not None:
        return _qw_django_only_sections_for_refresh(django_section_keys)
    blocked = _django_only_warehouse_sections()
    return tuple(
        str(key).strip()
        for key in django_section_keys
        if str(key).strip()
        and str(key).strip() != "prices_div_adj"
        and (
            warehouse_section_for_django(str(key).strip()) is None
            or warehouse_section_for_django(str(key).strip()) in blocked
        )
    )


def _symbol_is_etf(symbol_obj: Any) -> bool:
    payload = getattr(symbol_obj, "payload", None) or {}
    if not isinstance(payload, dict):
        return False
    value = payload.get("isEtf", payload.get("isETF", payload.get("is_etf")))
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _to_adjusted_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    for src, dst in (
        ("open", "adj_open"),
        ("high", "adj_high"),
        ("low", "adj_low"),
        ("close", "adj_close"),
    ):
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
        elif dst in out.columns and src not in out.columns:
            out[src] = out[dst]

    keep = [
        c
        for c in ("open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close", "volume")
        if c in out.columns
    ]
    out = out[keep]
    for col in keep:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_index()


def _price_frame_covers_end_date(frame: pd.DataFrame, end_date: str | None) -> bool:
    if frame is None or frame.empty or end_date in (None, ""):
        return bool(frame is not None and not frame.empty)
    try:
        return pd.Timestamp(frame.index.max()).normalize() >= pd.Timestamp(end_date).normalize()
    except (TypeError, ValueError):
        return False


def _read_warehouse_price_raw(
    wh: Warehouse,
    symbol: str,
    *,
    provider_name: str,
    start_date: str | None,
    end_date: str | None,
    is_etf: bool,
) -> pd.DataFrame:
    if is_etf:
        return wh.etf.read_prices(symbol, provider=provider_name, start=start_date, end=end_date)
    return wh.read_prices(symbol, provider=provider_name, start=start_date, end=end_date)


def load_warehouse_price_frame(
    symbol: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    provider: str | None = None,
    is_etf: bool = False,
) -> pd.DataFrame:
    symbol = str(symbol).strip().upper()
    if not symbol:
        return pd.DataFrame()

    wh = get_warehouse()
    provider_name = (provider or price_provider()).strip().lower()
    raw = _read_warehouse_price_raw(
        wh,
        symbol,
        provider_name=provider_name,
        start_date=start_date,
        end_date=end_date,
        is_etf=is_etf,
    )
    frame = _to_adjusted_price_frame(raw)
    if _price_frame_covers_end_date(frame, end_date):
        return frame

    fallback_providers = ("yfinance", "fmp")
    for fallback in fallback_providers:
        if fallback == provider_name:
            continue
        alt_raw = _read_warehouse_price_raw(
            wh,
            symbol,
            provider_name=fallback,
            start_date=start_date,
            end_date=end_date,
            is_etf=is_etf,
        )
        alt_frame = _to_adjusted_price_frame(alt_raw)
        if alt_frame.empty:
            continue
        if not frame.empty and alt_frame.index.max() <= frame.index.max():
            continue
        if _price_frame_covers_end_date(alt_frame, end_date) or alt_frame.index.max() > frame.index.max():
            return alt_frame
    return frame


def load_warehouse_price_frames(
    symbols: list[str],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    provider: str | None = None,
    etf_symbols: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    normalized = [str(symbol).strip().upper() for symbol in list(symbols or []) if str(symbol).strip()]
    if not normalized:
        return {}

    etf_set = {str(symbol).strip().upper() for symbol in (etf_symbols or set())}
    frames: dict[str, pd.DataFrame] = {}
    for symbol in normalized:
        frames[symbol] = load_warehouse_price_frame(
            symbol,
            start_date=start_date,
            end_date=end_date,
            provider=provider,
            is_etf=symbol in etf_set,
        )
    return frames


def load_warehouse_fundamental_frame(
    symbol: str,
    django_section_key: str,
    *,
    provider: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    section = warehouse_section_for_django(django_section_key)
    if section is None:
        return pd.DataFrame()

    wh = get_warehouse()
    provider_name = (provider or fundamental_provider()).strip().lower()
    return wh.read_fundamentals(
        str(symbol).strip().upper(),
        section=section,
        provider=provider_name,
        start=start_date,
        end=end_date,
    )


def warehouse_section_to_payload_rows(
    symbol: str,
    django_section_key: str,
    *,
    prefix: str,
    keep_fields: Iterable[str] | None = None,
    filing_lag_days: int = 0,
    start_date: str | None = None,
    end_date: str | None = None,
    provider: str | None = None,
) -> list[dict[str, Any]]:
    if _qw_warehouse_section_to_payload_rows is not None:
        return _qw_warehouse_section_to_payload_rows(
            symbol,
            django_section_key,
            prefix=prefix,
            keep_fields=keep_fields,
            filing_lag_days=filing_lag_days,
            start_date=start_date,
            end_date=end_date,
            provider=provider or fundamental_provider(),
            warehouse=get_warehouse(),
        )
    frame = load_warehouse_fundamental_frame(
        symbol,
        django_section_key,
        provider=provider,
        start_date=start_date,
        end_date=end_date,
    )
    if frame is None or frame.empty:
        return []

    keep = {str(value).lower().strip() for value in (keep_fields or [])}
    rows: list[dict[str, Any]] = []
    working = frame.reset_index()
    date_col = working.columns[0]
    working = working.rename(columns={date_col: "date"})
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date"])
    if filing_lag_days:
        working["date"] = working["date"] + pd.Timedelta(days=int(filing_lag_days))

    for _, series in working.iterrows():
        ts = pd.Timestamp(series["date"]).normalize()
        if pd.isna(ts):
            continue
        row: dict[str, Any] = {
            "date": ts,
            "symbol": str(symbol).strip().upper(),
        }
        for col, value in series.items():
            if col in {"date", "symbol"}:
                continue
            key = str(col).lower().strip()
            if keep and key not in keep:
                continue
            row[f"{prefix}{key}"] = value
        rows.append(row)
    return rows


def load_warehouse_macro_panel(
    series_codes: Iterable[str],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    provider: str | None = None,
) -> pd.DataFrame:
    codes = [str(code).strip() for code in series_codes if str(code).strip()]
    if not codes:
        return pd.DataFrame()
    wh = get_warehouse()
    panel = wh.read_macro_panel(
        codes,
        provider=(provider or macro_provider()).strip().lower(),
        start=start_date,
        end=end_date,
    )
    if panel is None or panel.empty:
        return pd.DataFrame()
    return panel.sort_index()


def list_warehouse_treasury_series_codes(*, provider: str | None = None) -> tuple[str, ...]:
    wh = get_warehouse()
    return tuple(wh.macro.list_treasury_series_codes(provider=(provider or macro_provider()).strip().lower()))


def warehouse_section_to_indexed_frame(
    symbol: str,
    django_section_key: str,
    *,
    prefix: str,
    keep_fields: Iterable[str] | None = None,
    filing_lag_days: int = 0,
    start_date: str | None = None,
    end_date: str | None = None,
    provider: str | None = None,
) -> pd.DataFrame:
    if _qw_warehouse_section_to_indexed_frame is not None:
        return _qw_warehouse_section_to_indexed_frame(
            symbol,
            django_section_key,
            prefix=prefix,
            keep_fields=keep_fields,
            filing_lag_days=filing_lag_days,
            start_date=start_date,
            end_date=end_date,
            provider=provider or fundamental_provider(),
            warehouse=get_warehouse(),
        )
    rows = warehouse_section_to_payload_rows(
        symbol,
        django_section_key,
        prefix=prefix,
        keep_fields=keep_fields,
        filing_lag_days=filing_lag_days,
        start_date=start_date,
        end_date=end_date,
        provider=provider,
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col in {"date", "symbol"}:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            df[col] = converted
    return (
        df.sort_values(["date", "symbol"])
        .drop_duplicates(subset=["date", "symbol"], keep="last")
        .set_index(["date", "symbol"])
        .sort_index()
    )


__all__ = [
    "django_only_sections_for_refresh",
    "fundamental_provider",
    "get_warehouse",
    "list_warehouse_treasury_series_codes",
    "load_warehouse_fundamental_frame",
    "load_warehouse_macro_panel",
    "load_warehouse_price_frame",
    "load_warehouse_price_frames",
    "macro_provider",
    "price_provider",
    "price_providers_for_refresh",
    "read_fundamentals_from_warehouse",
    "read_macro_from_warehouse",
    "read_prices_from_warehouse",
    "warehouse_section_for_django",
    "warehouse_sections_for_django_keys",
    "warehouse_sections_for_refresh",
    "warehouse_section_to_indexed_frame",
    "warehouse_section_to_payload_rows",
    "_symbol_is_etf",
]
