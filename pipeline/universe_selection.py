from __future__ import annotations

from typing import Any, Callable, Iterable

import pandas as pd

from data.historical_prices import load_adjusted_price_frames
from fmp.models import Symbol


DEFAULT_US_EXCHANGES: tuple[str, ...] = ("NASDAQ", "NYSE", "AMEX")

MARKET_CAP_TIERS: dict[str, float] = {
    "1t": 1_000_000_000_000.0,
    "100b": 100_000_000_000.0,
    "10b": 10_000_000_000.0,
}


def parse_exchange_values(raw_values: object) -> list[str]:
    if raw_values is None:
        return []
    if isinstance(raw_values, str):
        tokens = raw_values.split(",")
    elif isinstance(raw_values, Iterable):
        tokens = list(raw_values)
    else:
        tokens = [raw_values]
    values: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        value = str(token or "").strip().upper()
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def parse_symbol_prefix_values(raw_values: object) -> list[str]:
    values = parse_exchange_values(raw_values)
    return [value for value in values if value]


def _is_pooled_vehicle_name(name: str) -> bool:
    text = str(name or "").strip().lower()
    if not text:
        return False
    return (
        " etf" in f" {text}"
        or " fund" in f" {text}"
        or "portfolio" in text
        or "index fund" in text
    )


def resolve_symbol_universe(
    *,
    min_market_cap: float | None = None,
    max_market_cap: float | None = None,
    country: str | None = None,
    exchanges: object = None,
    limit: int | None = None,
    exclude_pooled_vehicles: bool = False,
    exclude_symbol_prefixes: object = None,
) -> list[str]:
    qs = Symbol.objects.exclude(symbol="")
    if min_market_cap is not None:
        qs = qs.filter(market_cap__gte=float(min_market_cap))
    if max_market_cap is not None:
        qs = qs.filter(market_cap__lte=float(max_market_cap))
    if country:
        qs = qs.filter(country__iexact=str(country).strip())
    exchange_values = parse_exchange_values(exchanges)
    if exchange_values:
        qs = qs.filter(exchange__in=exchange_values)
    excluded_prefixes = tuple(parse_symbol_prefix_values(exclude_symbol_prefixes))
    qs = qs.order_by("-market_cap", "symbol")
    rows: list[tuple[str, str, dict]] = list(qs.values_list("symbol", "company_name", "payload"))
    symbols: list[str] = []
    for symbol, company_name, payload in rows:
        normalized = str(symbol or "").strip().upper()
        if not normalized:
            continue
        if excluded_prefixes and normalized.startswith(excluded_prefixes):
            continue
        payload_dict = payload if isinstance(payload, dict) else {}
        if exclude_pooled_vehicles and (
            bool(payload_dict.get("isEtf")) or bool(payload_dict.get("isFund")) or _is_pooled_vehicle_name(company_name)
        ):
            continue
        symbols.append(normalized)
        if limit is not None and int(limit) > 0 and len(symbols) >= int(limit):
            break
    return symbols


def resolve_market_cap_tier_symbols(
    *,
    tier_key: str,
    country: str = "US",
    exchanges: object = DEFAULT_US_EXCHANGES,
    limit: int | None = None,
    exclude_pooled_vehicles: bool = True,
) -> list[str]:
    key = str(tier_key or "").strip().lower()
    if key not in MARKET_CAP_TIERS:
        raise ValueError(f"Unknown market cap tier: {tier_key!r}")
    return resolve_symbol_universe(
        min_market_cap=MARKET_CAP_TIERS[key],
        country=country,
        exchanges=exchanges,
        limit=limit,
        exclude_pooled_vehicles=exclude_pooled_vehicles,
    )


def summarize_symbol_price_history(
    symbols: Iterable[str],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    required_start_date: str | None = None,
    required_end_date: str | None = None,
    min_history_days: int = 0,
    history_loader: Callable[..., dict[str, pd.DataFrame]] | None = None,
) -> list[dict[str, Any]]:
    normalized = [str(symbol).strip().upper() for symbol in list(symbols or []) if str(symbol).strip()]
    if not normalized:
        return []
    loader = history_loader or load_adjusted_price_frames
    frames = loader(normalized, start_date=start_date, end_date=end_date)
    required_start_ts = pd.Timestamp(str(required_start_date)) if required_start_date else None
    required_end_ts = pd.Timestamp(str(required_end_date)) if required_end_date else None
    min_days = max(int(min_history_days), 0)

    rows: list[dict[str, Any]] = []
    for symbol in normalized:
        frame = frames.get(symbol)
        if frame is None or frame.empty:
            rows.append(
                {
                    "symbol": symbol,
                    "history_days": 0,
                    "history_start_date": "",
                    "history_end_date": "",
                    "passes_history_filter": False,
                }
            )
            continue
        history_start = pd.Timestamp(frame.index.min())
        history_end = pd.Timestamp(frame.index.max())
        history_days = int(len(frame))
        passes = history_days >= min_days
        if required_start_ts is not None and history_start > required_start_ts:
            passes = False
        if required_end_ts is not None and history_end < required_end_ts:
            passes = False
        rows.append(
            {
                "symbol": symbol,
                "history_days": history_days,
                "history_start_date": history_start.strftime("%Y-%m-%d"),
                "history_end_date": history_end.strftime("%Y-%m-%d"),
                "passes_history_filter": bool(passes),
            }
        )
    return rows


def filter_symbols_by_price_history(
    symbols: Iterable[str],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    required_start_date: str | None = None,
    required_end_date: str | None = None,
    min_history_days: int = 0,
    history_loader: Callable[..., dict[str, pd.DataFrame]] | None = None,
) -> list[str]:
    rows = summarize_symbol_price_history(
        symbols,
        start_date=start_date,
        end_date=end_date,
        required_start_date=required_start_date,
        required_end_date=required_end_date,
        min_history_days=min_history_days,
        history_loader=history_loader,
    )
    return [str(row["symbol"]) for row in rows if bool(row.get("passes_history_filter"))]
