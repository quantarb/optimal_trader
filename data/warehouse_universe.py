from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from pipeline.notebook_universe import normalize_symbols

try:
    from quant_warehouse.ingest.screener_fetch import ScreenerQuery
    from quant_warehouse.refresh.screener import resolve_universe_from_catalog, screen_universe_to_catalog
except Exception:  # pragma: no cover - quant-warehouse optional at import time
    ScreenerQuery = None  # type: ignore[assignment,misc]
    resolve_universe_from_catalog = None  # type: ignore[assignment]
    screen_universe_to_catalog = None  # type: ignore[assignment]


def _setting_bool(name: str, *, default: bool) -> bool:
    try:
        from django.conf import settings

        raw = getattr(settings, name, None)
        if raw is not None:
            return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        pass
    raw = os.getenv(name, "1" if default else "0")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def use_warehouse_screener() -> bool:
    try:
        from data.warehouse_refresh import use_warehouse_refresh

        return bool(use_warehouse_refresh() and _setting_bool("QW_SCREENER_ENABLED", default=True))
    except Exception:
        return _setting_bool("QW_SCREENER_ENABLED", default=True)


def screener_provider() -> str:
    try:
        from django.conf import settings

        return str(getattr(settings, "QW_SCREENER_PROVIDER", "fmp")).strip().lower()
    except Exception:
        return str(os.getenv("QW_SCREENER_PROVIDER", "fmp")).strip().lower()


def _screener_query_from_config(config: Mapping[str, Any]) -> "ScreenerQuery":
    if ScreenerQuery is None:
        raise RuntimeError("quant-warehouse is not installed; cannot run warehouse screener.")
    cfg = dict(config or {})
    exchanges = normalize_symbols(cfg.get("exchanges"))
    mktcap_min = cfg.get("min_market_cap")
    mktcap_max = cfg.get("max_market_cap")
    exclude_pooled = bool(cfg.get("exclude_pooled_vehicles", False))
    return ScreenerQuery(
        provider=screener_provider(),
        mktcap_min=int(mktcap_min) if mktcap_min not in (None, "") else None,
        mktcap_max=int(mktcap_max) if mktcap_max not in (None, "") else None,
        country=str(cfg.get("country") or "").strip().upper() or None,
        exchanges=exchanges,
        sector=str(cfg.get("sector") or "").strip() or None,
        industry=str(cfg.get("industry") or "").strip() or None,
        is_etf=False if exclude_pooled else None,
        is_fund=False if exclude_pooled else None,
        is_active=cfg.get("is_actively_trading"),
        limit=int(cfg.get("size") or 10_000),
    )


def screen_universe_from_warehouse(
    config: Mapping[str, Any],
    *,
    progress_logger=None,
) -> tuple[tuple[str, ...], str]:
    if screen_universe_to_catalog is None:
        raise RuntimeError("quant-warehouse is not installed; cannot run warehouse screener.")
    from data.warehouse import get_warehouse

    query = _screener_query_from_config(config)
    symbols, source = screen_universe_to_catalog(
        get_warehouse(),
        query,
        progress_logger=progress_logger,
    )
    if not symbols:
        raise RuntimeError("The configured warehouse screener returned no symbols.")
    return symbols, source


def resolve_local_universe_from_warehouse(
    config: Mapping[str, Any],
) -> tuple[str, ...]:
    if resolve_universe_from_catalog is None:
        raise RuntimeError("quant-warehouse is not installed; cannot query warehouse catalog.")
    from data.warehouse import get_warehouse

    cfg = dict(config or {})
    symbols = resolve_universe_from_catalog(
        get_warehouse(),
        provider=screener_provider(),
        min_market_cap=cfg.get("min_market_cap"),
        max_market_cap=cfg.get("max_market_cap"),
        country=str(cfg.get("country") or "").strip().upper() or None,
        exchanges=normalize_symbols(cfg.get("exchanges")),
        exclude_pooled_vehicles=bool(cfg.get("exclude_pooled_vehicles", False)),
        limit=cfg.get("size"),
    )
    return symbols


def sync_warehouse_catalog_profiles_to_django_symbols(
    symbols: Sequence[str],
    *,
    providers: Sequence[str] | None = None,
) -> None:
    from fmp.models import Symbol
    from fmp.section_store import json_safe

    from data.warehouse import get_warehouse

    wh = get_warehouse()
    provider_list = [str(value).strip().lower() for value in (providers or (screener_provider(),)) if str(value).strip()]
    normalized = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]

    for symbol_code in normalized:
        profiles = wh.catalog.list_profiles(symbol_code)
        if not profiles:
            continue
        profile = None
        for provider in provider_list:
            profile = next((row for row in profiles if row.provider == provider), None)
            if profile is not None:
                break
        if profile is None:
            profile = max(
                profiles,
                key=lambda row: (
                    1 if row.ipo_date else 0,
                    1 if row.market_cap else 0,
                    1 if row.sector else 0,
                ),
            )

        symbol_obj, _created = Symbol.objects.get_or_create(symbol=symbol_code)
        changed_fields: list[str] = []
        merged_payload = json_safe(dict(symbol_obj.payload or {}))
        payload = json_safe(dict(profile.payload or {}))
        merged_payload.update({str(key): value for key, value in payload.items() if value not in (None, "")})
        if profile.ipo_date:
            merged_payload["ipoDate"] = profile.ipo_date

        text_fields = {
            "company_name": profile.company_name,
            "exchange": profile.exchange,
            "country": profile.country,
            "sector": profile.sector,
            "industry": profile.industry,
        }
        for field_name, raw_value in text_fields.items():
            value = str(raw_value or "").strip()
            if value and getattr(symbol_obj, field_name) != value:
                setattr(symbol_obj, field_name, value)
                changed_fields.append(field_name)
        for field_name, raw_value in (
            ("market_cap", profile.market_cap),
            ("beta", profile.beta),
        ):
            if raw_value is not None and getattr(symbol_obj, field_name) != raw_value:
                setattr(symbol_obj, field_name, raw_value)
                changed_fields.append(field_name)
        if merged_payload != dict(symbol_obj.payload or {}):
            symbol_obj.payload = merged_payload
            changed_fields.append("payload")
        if changed_fields:
            symbol_obj.save(update_fields=list(dict.fromkeys(changed_fields)))


__all__ = [
    "resolve_local_universe_from_warehouse",
    "screener_provider",
    "screen_universe_from_warehouse",
    "sync_warehouse_catalog_profiles_to_django_symbols",
    "use_warehouse_screener",
]