from __future__ import annotations

from datetime import date
from typing import Any, Sequence

import pandas as pd

from data.warehouse import (
    _symbol_is_etf,
    django_only_sections_for_refresh,
    fundamental_provider,
    get_warehouse,
    macro_provider,
    price_providers_for_refresh,
    warehouse_sections_for_refresh,
)
from fmp.market_clock import expected_latest_price_date_from_market_clock
from fmp.sections import REQUIRED_FUNDAMENTAL_SECTION_KEYS, REQUIRED_SCORING_HISTORICAL_SECTIONS

try:
    from quant_warehouse.refresh import (
        expected_latest_price_date,
        refresh_universe_fundamentals,
        refresh_universe_macro,
        refresh_universe_prices,
        refresh_universe_profiles,
    )
except Exception:  # pragma: no cover - quant-warehouse optional at import time
    expected_latest_price_date = None  # type: ignore[assignment]
    refresh_universe_fundamentals = None  # type: ignore[assignment]
    refresh_universe_prices = None  # type: ignore[assignment]
    refresh_universe_profiles = None  # type: ignore[assignment]
    refresh_universe_macro = None  # type: ignore[assignment]

try:
    from quant_warehouse.warehouse.sections import DJANGO_ONLY_FUNDAMENTAL_SECTIONS
except Exception:  # pragma: no cover - older quant-warehouse versions do not expose this
    DJANGO_ONLY_FUNDAMENTAL_SECTIONS = frozenset()  # type: ignore[assignment]

PROFILE_PROVIDER_DEFAULT = "yfinance"
FUNDAMENTAL_PERIOD_DEFAULT = "quarter"


def _setting_bool(name: str, *, default: bool) -> bool:
    import os

    try:
        from django.conf import settings

        raw = getattr(settings, name, None)
        if raw is not None:
            return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        pass
    raw = os.getenv(name, "1" if default else "0")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def use_warehouse_refresh() -> bool:
    return _setting_bool("QW_REFRESH_ENABLED", default=True) and refresh_universe_prices is not None


def profile_provider() -> str:
    import os

    try:
        from django.conf import settings

        return str(getattr(settings, "QW_PROFILE_PROVIDER", "yfinance")).strip().lower()
    except Exception:
        return str(os.getenv("QW_PROFILE_PROVIDER", PROFILE_PROVIDER_DEFAULT)).strip().lower()


def _symbols_for_price_refresh(
    symbols: Sequence[str],
    *,
    skip_cached_inactive_symbols: bool,
) -> list[str]:
    normalized = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    if not normalized or not skip_cached_inactive_symbols:
        return normalized

    from fmp.models import Symbol
    from fmp.refresh import _symbol_has_price_history, _symbol_is_explicitly_inactive

    symbol_rows = {
        row.symbol: row
        for row in Symbol.objects.filter(symbol__in=normalized).only("symbol", "payload", "company_name", "historical_date_ranges")
    }
    active: list[str] = []
    for symbol_code in normalized:
        symbol_obj = symbol_rows.get(symbol_code)
        if (
            symbol_obj is not None
            and _symbol_is_explicitly_inactive(symbol_obj)
            and _symbol_has_price_history(symbol_obj)
        ):
            continue
        active.append(symbol_code)
    return active


def _etf_symbol_set(symbols: Sequence[str]) -> set[str]:
    from fmp.models import Symbol

    normalized = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    if not normalized:
        return set()
    found = Symbol.objects.filter(symbol__in=normalized).only("symbol", "payload")
    return {symbol.symbol for symbol in found if _symbol_is_etf(symbol)}


def sync_warehouse_profiles_to_django_symbols(
    symbols: Sequence[str],
    *,
    providers: Sequence[str] | None = None,
    progress_logger=None,
) -> pd.DataFrame:
    from fmp.symbol_dates import payload_listing_date
    from fmp.symbol_metadata import blocking_symbol_metadata_missing
    from fmp.models import Symbol

    wh = get_warehouse()
    provider_list = list(providers or [profile_provider()])
    etf_symbols = _etf_symbol_set(symbols)
    normalized = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]

    refresh_universe_profiles(
        wh,
        normalized,
        providers=provider_list,
        etf_symbols=etf_symbols,
        progress_logger=progress_logger,
    )

    rows: list[dict[str, Any]] = []
    for symbol_code in normalized:
        is_etf = symbol_code in etf_symbols
        symbol_obj, _created = Symbol.objects.get_or_create(symbol=symbol_code)
        updated_fields: list[str] = []
        merged_payload = dict(symbol_obj.payload or {})
        profiles = (
            wh.catalog.list_etf_profiles(symbol=symbol_code)
            if is_etf
            else wh.catalog.list_profiles(symbol=symbol_code)
        )
        profile = None
        for provider in provider_list:
            profile = next((row for row in profiles if row.provider == provider), None)
            if profile is not None:
                break
        if profile is None and profiles:
            profile = max(
                profiles,
                key=lambda row: (
                    1 if row.ipo_date else 0,
                    1 if row.market_cap else 0,
                    1 if row.sector else 0,
                ),
            )
        if profile is None:
            rows.append({"symbol": symbol_code, "status": "missing_profile"})
            continue

        payload = dict(profile.payload or {})
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
                updated_fields.append(field_name)
        for field_name, raw_value in (
            ("market_cap", profile.market_cap),
            ("beta", profile.beta),
        ):
            if raw_value is not None and getattr(symbol_obj, field_name) != raw_value:
                setattr(symbol_obj, field_name, raw_value)
                updated_fields.append(field_name)
        if merged_payload != dict(symbol_obj.payload or {}):
            symbol_obj.payload = merged_payload
            updated_fields.append("payload")
        if updated_fields:
            symbol_obj.save(update_fields=list(dict.fromkeys(updated_fields)))

        missing = blocking_symbol_metadata_missing(symbol_obj)
        rows.append(
            {
                "symbol": symbol_code,
                "status": "updated" if updated_fields else "unchanged",
                "missing_blocking": missing,
                "has_ipo_date": payload_listing_date(symbol_obj.payload) is not None,
            }
        )

    incomplete_blocking = {
        row["symbol"]: row["missing_blocking"]
        for row in rows
        if row.get("missing_blocking")
    }
    if incomplete_blocking:
        preview = ", ".join(
            f"{symbol}({','.join(fields)})"
            for symbol, fields in list(incomplete_blocking.items())[:20]
        )
        raise RuntimeError(
            "Required symbol metadata is unavailable after warehouse profile repair: "
            f"{preview}."
        )
    return pd.DataFrame(rows)


def run_scoring_data_refresh_from_warehouse(
    *,
    symbols: Sequence[str],
    target_start_date=None,
    target_end_date=None,
    refresh_mode: str = "scoring_ready",
    refresh_symbol_sections_before_build: bool = True,
    repair_symbol_metadata_before_build: bool = False,
    refresh_macro_before_build: bool = False,
    skip_cached_inactive_symbols: bool = True,
    skip_recent_price_attempts: bool = True,
    max_symbols=None,
    existing_historical_sections_only: bool = True,
    required_historical_sections: Sequence[str] | None = None,
    macro_config=None,
    verbose: bool = False,
    progress_logger=None,
) -> dict[str, Any]:
    del target_start_date, existing_historical_sections_only, verbose
    if refresh_universe_prices is None:
        raise RuntimeError("quant-warehouse is not installed; cannot run warehouse refresh.")

    log = progress_logger if callable(progress_logger) else None
    refresh_mode = str(refresh_mode or "scoring_ready").strip().lower()
    results: dict[str, Any] = {
        "refresh_mode": refresh_mode,
        "refresh_backend": "quant-warehouse",
        "price_plan": pd.DataFrame(),
        "price_refresh_results": pd.DataFrame(),
        "fundamental_plan": pd.DataFrame(),
        "fundamental_refresh_results": pd.DataFrame(),
        "symbol_refresh_plan": pd.DataFrame(),
        "symbol_refresh_results": pd.DataFrame(),
        "macro_refresh_results": pd.DataFrame(),
        "symbol_metadata_repair_results": pd.DataFrame(),
        "django_only_sections_skipped": (),
    }

    normalized = _symbols_for_price_refresh(
        symbols,
        skip_cached_inactive_symbols=bool(skip_cached_inactive_symbols),
    )
    if max_symbols is not None:
        normalized = normalized[: max(0, int(max_symbols))]
    if not normalized:
        return results

    if repair_symbol_metadata_before_build:
        if log is not None:
            log("Refreshing symbol profiles via quant-warehouse OpenBB and syncing Django catalog metadata")
        results["symbol_metadata_repair_results"] = sync_warehouse_profiles_to_django_symbols(
            normalized,
            progress_logger=progress_logger,
        )

    if not refresh_symbol_sections_before_build and not refresh_macro_before_build:
        return results

    wh = get_warehouse()
    etf_symbols = _etf_symbol_set(normalized)
    target_end = pd.Timestamp(target_end_date).date() if target_end_date is not None else expected_latest_price_date_from_market_clock()
    price_providers = list(price_providers_for_refresh())
    fundamental_providers = [fundamental_provider()]
    skip_recent_hours = 24.0 if bool(skip_recent_price_attempts) else 0.0

    scoring_sections = tuple(
        str(section).strip()
        for section in (required_historical_sections or REQUIRED_SCORING_HISTORICAL_SECTIONS)
        if str(section).strip()
    )
    django_only = django_only_sections_for_refresh(scoring_sections)
    results["django_only_sections_skipped"] = django_only
    if django_only and log is not None:
        preview = ", ".join(django_only[:12])
        hidden = max(0, len(django_only) - 12)
        suffix = f" and {hidden} more" if hidden else ""
        log(
            "Warehouse refresh skips Django-only FMP sections with no OpenBB route: "
            f"{preview}{suffix}"
        )

    warehouse_fundamental_sections = warehouse_sections_for_refresh(scoring_sections)

    if refresh_symbol_sections_before_build:
        if refresh_mode in {"prices_only", "scoring_ready"}:
            if log is not None:
                log(
                    "Refreshing stale warehouse prices via OpenBB gap-fill"
                    f" | providers={','.join(price_providers)}"
                    f" | target_end={target_end.isoformat()}"
                )
            price_rows = refresh_universe_prices(
                wh,
                normalized,
                providers=price_providers,
                target_end_date=target_end,
                etf_symbols=etf_symbols,
                skip_recent_hours=skip_recent_hours,
                progress_logger=progress_logger,
            )
            price_df = pd.DataFrame(price_rows)
            results["price_refresh_results"] = price_df
            if log is not None and not price_df.empty:
                updated = int((price_df["status"] == "updated").sum())
                skipped = int((price_df["status"] == "skipped_fresh").sum())
                still_stale = int((price_df["status"] == "still_stale").sum())
                errors = int((price_df["status"] == "error").sum())
                log(
                    "Warehouse price refresh complete"
                    f" | updated {updated:,} | skipped {skipped:,}"
                    f" | still_stale {still_stale:,} | errors {errors:,}"
                )

        if refresh_mode == "scoring_ready" and warehouse_fundamental_sections:
            if log is not None:
                log(
                    "Refreshing stale warehouse fundamentals via OpenBB"
                    f" | sections={','.join(warehouse_fundamental_sections)}"
                    f" | providers={','.join(fundamental_providers)}"
                )
            fundamental_rows = refresh_universe_fundamentals(
                wh,
                normalized,
                sections=warehouse_fundamental_sections,
                providers=fundamental_providers,
                period=FUNDAMENTAL_PERIOD_DEFAULT,
                etf_symbols=etf_symbols,
                skip_recent_hours=skip_recent_hours,
                progress_logger=progress_logger,
            )
            fundamental_df = pd.DataFrame(fundamental_rows)
            results["fundamental_refresh_results"] = fundamental_df
            if log is not None and not fundamental_df.empty:
                updated = int((fundamental_df["status"] == "updated").sum())
                skipped = int((fundamental_df["status"] == "skipped_fresh").sum())
                errors = int((fundamental_df["status"] == "error").sum())
                log(
                    "Warehouse fundamental refresh complete"
                    f" | updated {updated:,} | skipped {skipped:,} | errors {errors:,}"
                )

    if refresh_macro_before_build:
        if refresh_universe_macro is None:
            raise RuntimeError("quant-warehouse is not installed; cannot refresh macro data.")
        cfg = macro_config
        economic_series = tuple(
            str(raw).strip()
            for raw in tuple(getattr(cfg, "economic_indicator_series", ()) or ())
            if str(raw).strip()
        )
        include_treasury = bool(getattr(cfg, "include_treasury_rates", True))
        if log is not None:
            log(
                "Refreshing warehouse macro series via OpenBB/FMP"
                f" | economic={','.join(economic_series) or '<default>'}"
                f" | treasury={include_treasury}"
                f" | provider={macro_provider()}"
            )
        macro_rows = refresh_universe_macro(
            wh,
            economic_series=economic_series or None,
            include_treasury_rates=include_treasury,
            provider=macro_provider(),
            target_end_date=target_end,
            skip_recent_hours=skip_recent_hours,
            progress_logger=progress_logger,
        )
        results["macro_refresh_results"] = pd.DataFrame(macro_rows)

    return results


__all__ = [
    "django_only_sections_for_refresh",
    "profile_provider",
    "run_scoring_data_refresh_from_warehouse",
    "sync_warehouse_profiles_to_django_symbols",
    "use_warehouse_refresh",
    "warehouse_sections_for_django_keys",
]
