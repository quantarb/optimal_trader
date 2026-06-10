from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ResolvedNotebookUniverse:
    symbols: tuple[str, ...]
    source: str


def normalize_symbols(values: Iterable[Any] | str | None) -> tuple[str, ...]:
    if values is None:
        return ()
    tokens = values.replace("\n", ",").split(",") if isinstance(values, str) else list(values)
    symbols: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        symbol = str(token or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    return tuple(symbols)


def ensure_symbol_records(symbols: Iterable[str]) -> tuple[str, ...]:
    from fmp.models import Symbol

    normalized = normalize_symbols(symbols)
    existing = set(Symbol.objects.filter(symbol__in=normalized).values_list("symbol", flat=True))
    Symbol.objects.bulk_create(
        [Symbol(symbol=symbol) for symbol in normalized if symbol not in existing],
        ignore_conflicts=True,
    )
    return normalized


def _save_screener_records(records: Iterable[Mapping[str, Any]]) -> None:
    from fmp.models import Symbol
    from fmp.section_store import json_safe

    numeric_fields = {
        "market_cap": "marketCap",
        "price": "price",
        "beta": "beta",
        "volume": "volume",
        "dividend": "lastDividend",
        "dividend_yield": "dividendYield",
    }
    for raw_record in records:
        record = json_safe(dict(raw_record or {}))
        if not isinstance(record, dict):
            continue
        symbol = str(record.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        symbol_obj, _created = Symbol.objects.get_or_create(symbol=symbol)
        changed_fields: list[str] = []
        text_fields = {
            "company_name": record.get("companyName") or record.get("name"),
            "exchange": record.get("exchangeShortName") or record.get("exchange"),
            "country": record.get("country"),
            "sector": record.get("sector"),
            "industry": record.get("industry"),
        }
        for field_name, raw_value in text_fields.items():
            value = str(raw_value or "").strip()
            if value and getattr(symbol_obj, field_name) != value:
                setattr(symbol_obj, field_name, value)
                changed_fields.append(field_name)
        for field_name, source_key in numeric_fields.items():
            try:
                value = float(record[source_key]) if record.get(source_key) not in (None, "") else None
            except (TypeError, ValueError):
                value = None
            if value is not None and getattr(symbol_obj, field_name) != value:
                setattr(symbol_obj, field_name, value)
                changed_fields.append(field_name)
        merged_payload = json_safe(dict(symbol_obj.payload or {}))
        merged_payload.update({str(key): value for key, value in record.items() if value not in (None, "")})
        if merged_payload != dict(symbol_obj.payload or {}):
            symbol_obj.payload = merged_payload
            changed_fields.append("payload")
        if changed_fields:
            symbol_obj.save(update_fields=list(dict.fromkeys(changed_fields)))


def resolve_notebook_universe(
    config: Mapping[str, Any],
    *,
    api_key: str = "",
    metadata_client=None,
    progress_logger=None,
) -> ResolvedNotebookUniverse:
    from data.universe_fmp import screen_companies_fmp
    from pipeline.universe_selection import resolve_symbol_universe

    cfg = dict(config or {})
    explicit = normalize_symbols(cfg.get("symbols"))
    source = str(cfg.get("source") or "auto").strip().lower()
    def finalize(symbols, source: str) -> ResolvedNotebookUniverse:
        from fmp.symbol_metadata import sync_symbol_metadata_from_fmp

        normalized = ensure_symbol_records(symbols)
        sync_symbol_metadata_from_fmp(
            symbols=normalized,
            client=metadata_client,
            progress_logger=progress_logger,
        )
        return ResolvedNotebookUniverse(normalized, source)

    if explicit:
        return finalize(explicit, "explicit symbols")

    if source not in {"auto", "screener", "local"}:
        raise ValueError("universe.source must be one of: auto, screener, local")
    use_screener = source == "screener" or (source == "auto" and bool(str(api_key or "").strip()))
    if use_screener:
        if not str(api_key or "").strip():
            raise ValueError("FMP_API_KEY is required when universe.source='screener'.")
        symbols, records = screen_companies_fmp(
            api_key=str(api_key),
            marketCapMoreThan=cfg.get("min_market_cap"),
            marketCapLowerThan=cfg.get("max_market_cap"),
            country=str(cfg.get("country") or "") or None,
            exchange=",".join(normalize_symbols(cfg.get("exchanges"))) or None,
            isEtf=False if bool(cfg.get("exclude_pooled_vehicles", False)) else None,
            isFund=False if bool(cfg.get("exclude_pooled_vehicles", False)) else None,
            isActivelyTrading=cfg.get("is_actively_trading"),
            limit=int(cfg.get("size") or 10_000),
        )
        _save_screener_records(records)
        normalized = ensure_symbol_records(symbols)
        if not normalized:
            raise RuntimeError("The configured FMP screener returned no symbols.")
        return finalize(normalized, "FMP screener")

    symbols = resolve_symbol_universe(
        min_market_cap=cfg.get("min_market_cap"),
        max_market_cap=cfg.get("max_market_cap"),
        country=str(cfg.get("country") or "") or None,
        exchanges=list(normalize_symbols(cfg.get("exchanges"))),
        exclude_pooled_vehicles=bool(cfg.get("exclude_pooled_vehicles", False)),
        limit=cfg.get("size"),
    )
    normalized = ensure_symbol_records(symbols)
    if not normalized:
        raise RuntimeError("No symbols resolved from the configured local universe.")
    return finalize(normalized, "local DB")


__all__ = [
    "ResolvedNotebookUniverse",
    "ensure_symbol_records",
    "normalize_symbols",
    "resolve_notebook_universe",
]
