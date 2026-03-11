from __future__ import annotations

from typing import Iterable

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
    qs = qs.order_by("-market_cap", "symbol")
    rows: list[tuple[str, str, dict]] = list(qs.values_list("symbol", "company_name", "payload"))
    symbols: list[str] = []
    for symbol, company_name, payload in rows:
        normalized = str(symbol or "").strip().upper()
        if not normalized:
            continue
        payload_dict = payload if isinstance(payload, dict) else {}
        if exclude_pooled_vehicles and (
            bool(payload_dict.get("isEtf")) or _is_pooled_vehicle_name(company_name)
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
