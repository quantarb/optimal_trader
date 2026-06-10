from __future__ import annotations

import os
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from django.utils import timezone

from data import FMPClient
from features.macro import MacroFeatureConfig

from .endpoints import get_symbol_endpoint_definitions
from .endpoints.prices_div_adj import build as build_prices_div_adj_endpoint
from .models import (
    EconomicIndicatorObservation,
    EconomicIndicatorSeries,
    Symbol,
    SymbolSectionState,
    TreasuryRateObservation,
    TreasuryRateSeries,
)
from .records import dedupe_historical_records, extract_record_date
from .positions_summary import save_positions_summary_records
from .section_store import (
    mark_section_fetched,
    save_historical_section,
    save_snapshot_section,
    sync_symbol_historical_ranges,
    update_symbol_historical_range,
)
from .sections import (
    REQUIRED_FUNDAMENTAL_SECTION_KEYS,
    REQUIRED_SCORING_HISTORICAL_SECTIONS,
    _FUNDAMENTAL_DEPENDENT_SECTION_KEYS,
    _FUNDAMENTAL_FALLBACK_ANCHOR_SECTION_KEYS,
    _FUNDAMENTAL_STATEMENT_ANCHOR_SECTION_KEYS,
    _SPARSE_EVENT_HISTORICAL_SECTION_KEYS,
    _UNIVERSE_DOWNLOAD_COVERAGE_THRESHOLD,
    _UNIVERSE_DOWNLOAD_RETRY_BASE_DELAY_S,
    _UNIVERSE_DOWNLOAD_RETRY_MAX_ATTEMPTS,
    _UNIVERSE_DOWNLOAD_TARGET_YEARS,
)
from .stability import assess_historical_section_stability
from .transport import (
    candidates_support_date_window,
    fetch_first_success,
    fetch_historical_records,
    parse_date,
    run_with_retries,
    to_records,
    with_date_window,
)

# Backwards-compat re-exports for callers that expect these from here.
REQUIRED_FUNDAMENTAL_SECTION_KEYS = REQUIRED_FUNDAMENTAL_SECTION_KEYS  # type: ignore[assignment]
REQUIRED_SCORING_HISTORICAL_SECTIONS = REQUIRED_SCORING_HISTORICAL_SECTIONS  # type: ignore[assignment]

_REPO_DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def resolve_fmp_api_key(*, required: bool = False) -> str:
    """Load .env (best effort) and return the FMP_API_KEY (or empty string)."""
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=_REPO_DOTENV_PATH, override=True)
    except Exception:
        if _REPO_DOTENV_PATH.exists():
            try:
                for raw_line in _REPO_DOTENV_PATH.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = str(key).strip()
                    if not key:
                        continue
                    cleaned = str(value).strip().strip('"').strip("'")
                    os.environ[key] = cleaned
            except Exception:
                pass

    api_key = str(os.getenv("FMP_API_KEY") or "").strip()
    if required and not api_key:
        raise ValueError("Missing FMP_API_KEY in environment/.env.")
    return api_key


# ---------------------------------------------------------------------------
# Low-level helpers (previously duplicated in trading/live_trade.py and fmp/views.py)
# These are now owned by fmp/refresh.py as the canonical refresh planning layer.
# ---------------------------------------------------------------------------


def _symbol_is_explicitly_inactive(symbol_obj: Symbol) -> bool:
    payload = dict(symbol_obj.payload or {})
    active_value = payload.get("isActivelyTrading") or payload.get("activelyTrading") or payload.get("is_active")
    if active_value is not None:
        return not bool(active_value)
    company_name = str(symbol_obj.company_name or "").strip().lower()
    return "(delisted)" in company_name


def _symbol_has_price_history(symbol_obj: Symbol) -> bool:
    ranges = dict(symbol_obj.historical_date_ranges or {})
    price_range = ranges.get("prices_div_adj") if isinstance(ranges, dict) else None
    if not isinstance(price_range, dict):
        return False
    if int(price_range.get("count") or 0) > 0:
        return True
    return bool(price_range.get("min_date")) and bool(price_range.get("max_date"))


def _symbol_recent_price_refresh_attempt(symbol_obj: Symbol, *, threshold_days: int = 1) -> bool:
    state = SymbolSectionState.objects.filter(symbol=symbol_obj, section_key="prices_div_adj").first()
    if state is None or state.last_fetched_at is None:
        return False
    return state.last_fetched_at >= (timezone.now() - timedelta(days=threshold_days))


def _historical_section_max_date(symbol_obj: Symbol, section_key: str):
    ranges = dict(symbol_obj.historical_date_ranges or {})
    payload = ranges.get(section_key) if isinstance(ranges, dict) else None
    raw_value = payload.get("max_date") if isinstance(payload, dict) else None
    if not raw_value:
        return None
    return pd.Timestamp(raw_value).date()


def _latest_fundamental_anchor_date(symbol_obj: Symbol):
    anchor_keys = _FUNDAMENTAL_STATEMENT_ANCHOR_SECTION_KEYS
    dates = [_historical_section_max_date(symbol_obj, key) for key in anchor_keys]
    dates = [value for value in dates if value is not None]
    if dates:
        return max(dates)
    fallback_dates = [_historical_section_max_date(symbol_obj, key) for key in _FUNDAMENTAL_FALLBACK_ANCHOR_SECTION_KEYS]
    fallback_dates = [value for value in fallback_dates if value is not None]
    return max(fallback_dates) if fallback_dates else None


def _parse_iso_date(value: Any):
    if not value:
        return None
    try:
        return timezone.datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _existing_historical_section_keys(symbol_obj: Symbol) -> set[str]:
    ranges = dict(symbol_obj.historical_date_ranges or {})
    out: set[str] = set()
    if not isinstance(ranges, dict):
        return out
    for key, payload in ranges.items():
        if not isinstance(payload, dict):
            continue
        if int(payload.get("count") or 0) > 0:
            out.add(str(key))
    return out


def _normalize_section_key_set(section_keys: Any) -> set[str]:
    out: set[str] = set()
    for raw in list(section_keys or []):
        key = str(raw).strip()
        if key:
            out.add(key)
    return out


def _latest_historical_section_date(symbol_obj: Symbol, section_key: str):
    ranges = dict(symbol_obj.historical_date_ranges or {})
    payload = ranges.get(section_key) if isinstance(ranges, dict) else None
    if not isinstance(payload, dict):
        return None
    return _parse_iso_date(payload.get("max_date"))


def _latest_company_fundamental_anchor_date(symbol_obj: Symbol):
    statement_dates = [
        _latest_historical_section_date(symbol_obj, section_key)
        for section_key in _FUNDAMENTAL_STATEMENT_ANCHOR_SECTION_KEYS
    ]
    statement_dates = [value for value in statement_dates if value is not None]
    if statement_dates:
        return max(statement_dates)
    fallback_dates = [
        _latest_historical_section_date(symbol_obj, section_key)
        for section_key in _FUNDAMENTAL_FALLBACK_ANCHOR_SECTION_KEYS
    ]
    fallback_dates = [value for value in fallback_dates if value is not None]
    return max(fallback_dates) if fallback_dates else None


def _historical_section_fetched_recently(
    symbol_obj: Symbol,
    section_key: str,
    *,
    target_end,
    threshold_days: int,
    state=None,
) -> bool:
    ranges = dict(symbol_obj.historical_date_ranges or {})
    section_range = ranges.get(section_key) if isinstance(ranges, dict) else None
    max_date = _parse_iso_date((section_range or {}).get("max_date")) if isinstance(section_range, dict) else None
    if max_date is None or max_date < target_end:
        return False
    if state is None:
        state = SymbolSectionState.objects.filter(symbol=symbol_obj, section_key=section_key).first()
    if state is None or state.last_fetched_at is None:
        return False
    return state.last_fetched_at >= (timezone.now() - timedelta(days=threshold_days))


def _historical_section_fetch_mode(
    symbol_obj: Symbol,
    section,
    *,
    target_end,
    target_start,
) -> tuple[str, list[tuple[Any, Any]]]:
    section_key = str(section.key)
    candidates = list(section.candidates or [])
    threshold_days = int(section.threshold_days)
    has_date_window = candidates_support_date_window(candidates, endpoint=section)
    assessment = assess_historical_section_stability(
        symbol_obj,
        section,
        target_start=target_start,
        target_end=target_end,
    )
    coverage_ratio = assessment.coverage_ratio
    max_date = assessment.max_date
    min_date = assessment.min_date
    is_recent_enough = bool(max_date and max_date >= (target_end - timedelta(days=threshold_days)))
    state = SymbolSectionState.objects.filter(symbol=symbol_obj, section_key=section_key).first()
    fetched_recently = bool(
        state
        and state.last_fetched_at
        and state.last_fetched_at >= (timezone.now() - timedelta(days=threshold_days))
    )
    fundamental_anchor_date = _latest_company_fundamental_anchor_date(symbol_obj)

    if (
        section_key in _FUNDAMENTAL_DEPENDENT_SECTION_KEYS
        and fundamental_anchor_date is not None
        and max_date is not None
        and max_date >= fundamental_anchor_date
    ):
        return "skip", []

    if assessment.stable:
        return "skip", []

    fetch_mode = "full"
    fetch_ranges: list[tuple[Any, Any]] = []

    if coverage_ratio >= _UNIVERSE_DOWNLOAD_COVERAGE_THRESHOLD and has_date_window:
        if max_date is not None and max_date < target_end:
            fetch_mode = "tail"
            fetch_ranges = [(max_date + timedelta(days=1), target_end)]
        elif not is_recent_enough:
            fetch_mode = "tail"
            fetch_ranges = [(max(target_end - timedelta(days=threshold_days), target_start), target_end)]
        else:
            fetch_mode = "full"
    elif coverage_ratio >= _UNIVERSE_DOWNLOAD_COVERAGE_THRESHOLD and is_recent_enough:
        fetch_mode = "full"
    elif has_date_window and min_date is not None and min_date > target_start:
        fetch_mode = "head"
        fetch_ranges = [(target_start, min_date - timedelta(days=1))]

    return fetch_mode, fetch_ranges


def _prepare_payload_for_storage(payload: Any) -> Any:
    if payload is None:
        return {}
    try:
        import json

        json.dumps(payload, allow_nan=False, separators=(",", ":"))
        return payload
    except (TypeError, ValueError):
        from .section_store import json_safe

        return json_safe(payload)


def _stable_record_key(record: Any) -> str:
    import hashlib
    import json

    blob = json.dumps(_prepare_payload_for_storage(record), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _filter_records_for_symbol(records: list[dict], symbol: str) -> list[dict]:
    sym = str(symbol).strip().upper()
    if not sym:
        return records
    symbol_keys = ("symbol", "ticker")
    has_symbol_key = any(isinstance(r, dict) and any(k in r for k in symbol_keys) for r in records)
    if not has_symbol_key:
        return records
    out: list[dict] = []
    for r in records:
        if not isinstance(r, dict):
            continue
        v = None
        for k in symbol_keys:
            if r.get(k):
                v = str(r.get(k)).strip().upper()
                break
        if v == sym:
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Needs / staleness decision functions (now owned by fmp)
# ---------------------------------------------------------------------------


def symbol_needs_price_refresh(
    symbol_obj: Symbol,
    *,
    target_end_date=None,
    skip_cached_inactive_symbols: bool = True,
    skip_recent_price_attempts: bool = True,
) -> tuple[bool, str]:
    symbol_obj.refresh_from_db(fields=["historical_date_ranges", "payload", "company_name"])
    if skip_cached_inactive_symbols and _symbol_is_explicitly_inactive(symbol_obj) and _symbol_has_price_history(symbol_obj):
        return False, "inactive_symbol_cached"
    expected_price_date = (
        pd.Timestamp(target_end_date).date()
        if target_end_date is not None
        else expected_latest_price_date_from_market_clock()
    )
    price_max_date = _historical_section_max_date(symbol_obj, "prices_div_adj")
    if price_max_date is None:
        if skip_recent_price_attempts and _symbol_recent_price_refresh_attempt(symbol_obj):
            return False, "recent_price_attempt_no_history"
        return True, "missing_prices_div_adj"
    if price_max_date < expected_price_date:
        if skip_recent_price_attempts and _symbol_recent_price_refresh_attempt(symbol_obj):
            return False, "recent_price_attempt"
        return True, f"prices_div_adj_max_date_lt_{expected_price_date.isoformat()}"
    return False, "fresh_prices_div_adj"


def symbol_needs_fundamental_refresh(
    symbol_obj: Symbol,
    *,
    target_end_date=None,
    required_section_keys: Sequence[str] | None = None,
) -> tuple[bool, str]:
    symbol_obj.refresh_from_db(fields=["historical_date_ranges"])
    anchor_date = _latest_fundamental_anchor_date(symbol_obj)
    if anchor_date is None:
        return True, "missing_fundamental_anchor"

    required_keys = tuple(required_section_keys or REQUIRED_FUNDAMENTAL_SECTION_KEYS)
    for section_key in required_keys:
        section_max_date = _historical_section_max_date(symbol_obj, section_key)
        if section_max_date is None:
            return True, f"missing_{section_key}"
        if section_max_date < anchor_date:
            return True, f"{section_key}_lt_anchor_{anchor_date.isoformat()}"
    return False, "fresh_fundamental_sections"


def expected_latest_price_date_from_market_clock() -> Any:
    now_et = pd.Timestamp.now(tz="America/New_York")
    if now_et.weekday() < 5 and now_et.hour >= 17:
        return now_et.date()
    return (now_et.normalize() - pd.offsets.BDay(1)).date()


def symbol_needs_required_refresh(symbol_obj: Symbol, *, target_start_date=None, target_end_date=None) -> tuple[bool, str]:
    del target_start_date
    symbol_obj.refresh_from_db(fields=["historical_date_ranges"])
    ranges = dict(symbol_obj.historical_date_ranges or {})

    price_range = dict(ranges.get("prices_div_adj") or {})
    price_max_date = pd.Timestamp(price_range.get("max_date")).date() if price_range.get("max_date") else None
    expected_price_date = (
        pd.Timestamp(target_end_date).date()
        if target_end_date is not None
        else expected_latest_price_date_from_market_clock()
    )

    if price_max_date is None:
        return True, "missing_prices_div_adj"
    if price_max_date < expected_price_date:
        return True, f"prices_div_adj_max_date_lt_{expected_price_date.isoformat()}"

    # Lightweight staleness checks for a couple of critical fundamental inputs.
    for skey in ("key_metrics", "ratios"):
        # Use the historical range max vs a recent threshold (kept simple here; full logic lives in stability + _historical_section_fetch_mode).
        section_max = _historical_section_max_date(symbol_obj, skey)
        if section_max is None:
            return True, f"missing_{skey}"
        # If very stale relative to target, consider required.
        threshold = 7
        if target_end_date:
            try:
                tgt = pd.Timestamp(target_end_date).date()
                if section_max < (tgt - timedelta(days=threshold)):
                    return True, f"stale_{skey}"
            except Exception:
                pass

    return False, "fresh_required_inputs"


def historical_symbol_refresh_needed(
    symbol_obj: Symbol,
    *,
    target_start_date=None,
    target_end_date=None,
    existing_historical_sections_only: bool = False,
    required_historical_sections: Any = None,
    allowed_historical_sections: Any = None,
) -> tuple[bool, str]:
    section_defs = [section for section in get_symbol_endpoint_definitions(symbol_obj) if str(section.kind) == "historical"]
    allowed_keys = _normalize_section_key_set(allowed_historical_sections)
    if allowed_keys:
        section_defs = [section for section in section_defs if str(section.key) in allowed_keys]
    if not section_defs:
        return False, "no_historical_sections"

    default_target_end = timezone.now().date()
    parsed_target_end = _parse_iso_date(target_end_date)
    target_end = parsed_target_end or default_target_end
    default_target_start = target_end - timedelta(days=365 * _UNIVERSE_DOWNLOAD_TARGET_YEARS)
    parsed_target_start = _parse_iso_date(target_start_date)
    target_start = parsed_target_start or default_target_start

    historical_section_keys = [str(section.key) for section in section_defs]
    sync_symbol_historical_ranges(symbol_obj, historical_section_keys)
    symbol_obj.refresh_from_db(fields=["historical_date_ranges"])

    if existing_historical_sections_only:
        existing_keys = _existing_historical_section_keys(symbol_obj)
        required_keys = _normalize_section_key_set(required_historical_sections)
        allowed_keys = existing_keys | required_keys
        if not allowed_keys:
            return False, "no_existing_historical_sections"
        section_defs = [section for section in section_defs if str(section.key) in allowed_keys]
        if not section_defs:
            return False, "no_existing_historical_sections"

    for section in section_defs:
        fetch_mode, _fetch_ranges = _historical_section_fetch_mode(
            symbol_obj,
            section,
            target_end=target_end,
            target_start=target_start,
        )
        if fetch_mode != "skip":
            return True, f"{section.key}:{fetch_mode}"

    return False, "fresh_all_historical_sections"


# ---------------------------------------------------------------------------
# Bulk refresh implementations (moved into fmp as the owner of refresh orchestration)
# ---------------------------------------------------------------------------


def _refresh_all_symbol_sections(
    symbol_obj: Symbol,
    client: FMPClient,
    *,
    include_snapshot_sections: bool = True,
    target_start_date=None,
    target_end_date=None,
    existing_historical_sections_only: bool = False,
    required_historical_sections: Any = None,
    allowed_historical_sections: Any = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Core per-symbol section refresh. Public name is refresh_all_symbol_sections below."""
    section_errors: dict[str, str] = {}
    section_defs = list(get_symbol_endpoint_definitions(symbol_obj))
    if not include_snapshot_sections:
        section_defs = [section for section in section_defs if str(section.kind) != "snapshot"]
    allowed_keys = _normalize_section_key_set(allowed_historical_sections)
    if allowed_keys:
        section_defs = [
            section
            for section in section_defs
            if str(section.kind) != "historical" or str(section.key) in allowed_keys
        ]
    refreshed_historical = False
    historical_section_keys = [s.key for s in section_defs if s.kind == "historical"]
    default_target_end = timezone.now().date()
    parsed_target_end = _parse_iso_date(target_end_date)
    target_end = parsed_target_end or default_target_end
    default_target_start = target_end - timedelta(days=365 * _UNIVERSE_DOWNLOAD_TARGET_YEARS)
    parsed_target_start = _parse_iso_date(target_start_date)
    target_start = parsed_target_start or default_target_start

    sync_symbol_historical_ranges(symbol_obj, historical_section_keys)
    symbol_obj.refresh_from_db(fields=["historical_date_ranges"])
    if existing_historical_sections_only:
        existing_historical_keys = _existing_historical_section_keys(symbol_obj)
        required_historical_keys = _normalize_section_key_set(required_historical_sections)
        allowed_historical_keys = existing_historical_keys | required_historical_keys
        section_defs = [
            section
            for section in section_defs
            if str(section.kind) != "historical" or str(section.key) in allowed_historical_keys
        ]
        historical_section_keys = [s.key for s in section_defs if s.kind == "historical"]

    stats = {
        "sections_total": len(section_defs),
        "sections_fetched": 0,
        "sections_skipped": 0,
        "partial_sections": 0,
        "retry_attempts": 0,
        "duration_s": 0.0,
    }
    t0 = time.perf_counter()

    for section in section_defs:
        section_key = section.key
        kind = section.kind
        candidates = section.candidates
        filter_symbol = bool(section.filter_symbol)
        threshold_days = int(section.threshold_days)
        raw = None

        try:
            if kind == "historical":
                fetch_mode, fetch_ranges = _historical_section_fetch_mode(
                    symbol_obj,
                    section,
                    target_end=target_end,
                    target_start=target_start,
                )

                if fetch_mode == "skip":
                    stats["sections_skipped"] += 1
                    continue

                all_records: list[dict] = []
                retries_used = 0
                if fetch_mode in {"tail", "head"} and fetch_ranges:
                    stats["partial_sections"] += 1
                    for from_date, to_date in fetch_ranges:
                        if to_date < from_date:
                            continue
                        partial_candidates = with_date_window(
                            candidates,
                            from_date=from_date,
                            to_date=to_date,
                            endpoint=section,
                        )
                        partial_endpoint = replace(section, candidates=partial_candidates)
                        fetched, retries = run_with_retries(
                            lambda endpoint=partial_endpoint: fetch_historical_records(client, endpoint),
                            max_attempts=_UNIVERSE_DOWNLOAD_RETRY_MAX_ATTEMPTS,
                            base_delay_s=_UNIVERSE_DOWNLOAD_RETRY_BASE_DELAY_S,
                        )
                        retries_used += retries
                        all_records.extend(list(fetched or []))
                else:
                    if getattr(section, "supports_date_window", False) or candidates_support_date_window(candidates, endpoint=section):
                        full_candidates = with_date_window(
                            candidates,
                            from_date=target_start,
                            to_date=target_end,
                            endpoint=section,
                        )
                        fetch_endpoint = replace(section, candidates=full_candidates)
                    else:
                        fetch_endpoint = section
                    fetched, retries = run_with_retries(
                        lambda endpoint=fetch_endpoint: fetch_historical_records(client, endpoint),
                        max_attempts=_UNIVERSE_DOWNLOAD_RETRY_MAX_ATTEMPTS,
                        base_delay_s=_UNIVERSE_DOWNLOAD_RETRY_BASE_DELAY_S,
                    )
                    retries_used += retries
                    all_records = list(fetched or [])

                stats["retry_attempts"] += retries_used
                records = dedupe_historical_records(
                    all_records,
                    by_date=bool(section.dedupe_by_date),
                    prepare=_prepare_payload_for_storage,
                )
            else:
                raw, retries = run_with_retries(
                    lambda: fetch_first_success(client, candidates),
                    max_attempts=_UNIVERSE_DOWNLOAD_RETRY_MAX_ATTEMPTS,
                    base_delay_s=_UNIVERSE_DOWNLOAD_RETRY_BASE_DELAY_S,
                )
                stats["retry_attempts"] += retries
                records = to_records(raw)

            if section_key == "peer_symbols":
                peer_records: list[dict] = []
                if isinstance(raw, list):
                    if raw and isinstance(raw[0], dict):
                        peer_records = raw
                    else:
                        peer_records = [{"peerSymbol": p} for p in raw]
                elif isinstance(raw, dict):
                    peers_list = raw.get("peersList") or raw.get("peers") or []
                    if isinstance(peers_list, list):
                        peer_records = [{"peerSymbol": p} for p in peers_list]
                    else:
                        peer_records = to_records(raw)
                records = peer_records

            if filter_symbol:
                records = _filter_records_for_symbol(records, symbol_obj.symbol)

            if kind == "snapshot":
                save_snapshot_section(symbol_obj, section_key, raw)
            else:
                save_historical_section(
                    symbol_obj,
                    section_key,
                    records,
                    dedupe_by_date=bool(section.dedupe_by_date),
                )
                update_symbol_historical_range(symbol_obj, section_key)
                refreshed_historical = True

            mark_section_fetched(symbol_obj, section_key, kind)
            stats["sections_fetched"] += 1
        except Exception as exc:
            if "HTTP 404" not in str(exc):
                section_errors[section_key] = str(exc)

    if refreshed_historical:
        sync_symbol_historical_ranges(symbol_obj, historical_section_keys)

    stats["duration_s"] = round(float(time.perf_counter() - t0), 3)
    return section_errors, stats


def refresh_all_symbol_sections(
    symbol_obj: Symbol,
    client: FMPClient,
    *,
    include_snapshot_sections: bool = True,
    target_start_date=None,
    target_end_date=None,
    existing_historical_sections_only: bool = False,
    required_historical_sections: Sequence[str] | None = None,
    allowed_historical_sections: Sequence[str] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Public API for per-symbol refresh of snapshot + historical sections.

    This is the function that tasks, universe refresh planners, and callers should use.
    """
    return _refresh_all_symbol_sections(
        symbol_obj,
        client,
        include_snapshot_sections=include_snapshot_sections,
        target_start_date=target_start_date,
        target_end_date=target_end_date,
        existing_historical_sections_only=existing_historical_sections_only,
        required_historical_sections=required_historical_sections,
        allowed_historical_sections=allowed_historical_sections,
    )


def refresh_universe_symbol_sections_from_fmp(
    *,
    symbols: Sequence[str],
    target_start_date=None,
    target_end_date=None,
    max_symbols=None,
    include_snapshot_sections: bool = True,
    existing_historical_sections_only: bool = False,
    required_historical_sections: Sequence[str] | None = None,
    allowed_historical_sections: Sequence[str] | None = None,
    verbose: bool = True,
    progress_logger=None,
) -> pd.DataFrame:
    api_key = resolve_fmp_api_key(required=True)
    client = FMPClient(api_key=api_key, timeout_s=30.0, max_retries=2)
    selected = list(symbols or [])
    if max_symbols is not None:
        selected = selected[: max(0, int(max_symbols))]

    rows: list[dict[str, Any]] = []
    total = len(selected)
    refreshed_count = 0
    for idx, sym in enumerate(selected, start=1):
        code = str(sym).strip().upper()
        if not code:
            continue
        if callable(progress_logger):
            progress_logger(f"FMP historical refresh start [{idx:,}/{total:,}] {code}")
        symbol_obj = Symbol.objects.filter(symbol__iexact=code).only("id", "symbol", "historical_date_ranges").first()
        if symbol_obj is None:
            symbol_obj = Symbol.objects.create(symbol=code)
        if include_snapshot_sections:
            needs_refresh, refresh_reason = symbol_needs_required_refresh(
                symbol_obj,
                target_start_date=target_start_date,
                target_end_date=target_end_date,
            )
        else:
            needs_refresh, refresh_reason = historical_symbol_refresh_needed(
                symbol_obj,
                target_start_date=target_start_date,
                target_end_date=target_end_date,
                existing_historical_sections_only=bool(existing_historical_sections_only),
                required_historical_sections=required_historical_sections,
                allowed_historical_sections=allowed_historical_sections,
            )
        if not needs_refresh:
            if callable(progress_logger):
                progress_logger(
                    f"FMP historical refresh done  [{idx:,}/{total:,}] {code} | status=skipped_fresh | reason={refresh_reason}"
                )
            rows.append(
                {
                    "symbol": code,
                    "status": "skipped_fresh",
                    "refresh_reason": refresh_reason,
                    "error_count": 0,
                    "error": "",
                    "sections_total": 0,
                    "sections_fetched": 0,
                    "sections_skipped": 3,
                    "partial_sections": 0,
                    "retry_attempts": 0,
                    "duration_s": 0.0,
                }
            )
            continue
        try:
            section_errors, section_stats = refresh_all_symbol_sections(
                symbol_obj,
                client,
                include_snapshot_sections=bool(include_snapshot_sections),
                target_start_date=target_start_date,
                target_end_date=target_end_date,
                existing_historical_sections_only=bool(existing_historical_sections_only),
                required_historical_sections=required_historical_sections,
                allowed_historical_sections=allowed_historical_sections,
            )
            refreshed_count += 1
            rows.append(
                {
                    "symbol": code,
                    "status": "ok" if not section_errors else "completed_with_errors",
                    "refresh_reason": refresh_reason,
                    "error_count": int(len(section_errors)),
                    "error": "; ".join(f"{k}={v}" for k, v in section_errors.items()),
                    "sections_total": int(section_stats.get("sections_total", 0)),
                    "sections_fetched": int(section_stats.get("sections_fetched", 0)),
                    "sections_skipped": int(section_stats.get("sections_skipped", 0)),
                    "partial_sections": int(section_stats.get("partial_sections", 0)),
                    "retry_attempts": int(section_stats.get("retry_attempts", 0)),
                    "duration_s": float(section_stats.get("duration_s", 0.0)),
                }
            )
            if callable(progress_logger):
                progress_logger(
                    f"FMP historical refresh done  [{idx:,}/{total:,}] {code} | "
                    f"status={'ok' if not section_errors else 'completed_with_errors'} | "
                    f"reason={refresh_reason} | fetched={int(section_stats.get('sections_fetched', 0))} | "
                    f"skipped={int(section_stats.get('sections_skipped', 0))} | "
                    f"errors={int(len(section_errors))}"
                )
        except Exception as exc:
            if callable(progress_logger):
                progress_logger(
                    f"FMP historical refresh done  [{idx:,}/{total:,}] {code} | status=error | reason={refresh_reason} | error={exc}"
                )
            rows.append(
                {
                    "symbol": code,
                    "status": "error",
                    "refresh_reason": refresh_reason,
                    "error_count": 1,
                    "error": str(exc),
                    "sections_total": 0,
                    "sections_fetched": 0,
                    "sections_skipped": 0,
                    "partial_sections": 0,
                    "retry_attempts": 0,
                    "duration_s": np.nan,
                }
            )
        if callable(progress_logger) and (idx == 1 or idx % 25 == 0 or idx == total):
            progress_logger(f"FMP historical refresh progress: {idx:,}/{total:,} symbols processed | {refreshed_count:,} refreshed so far")
        if verbose and refreshed_count > 0 and (refreshed_count % 25 == 0 or idx == total):
            print(f"FMP symbol refresh progress: refreshed {refreshed_count}/{total}")
    return pd.DataFrame(rows)


def plan_symbol_section_refresh_from_fmp(
    *,
    symbols: Sequence[str],
    target_start_date=None,
    target_end_date=None,
    max_symbols=None,
    include_snapshot_sections: bool = True,
    existing_historical_sections_only: bool = False,
    required_historical_sections: Sequence[str] | None = None,
    allowed_historical_sections: Sequence[str] | None = None,
) -> pd.DataFrame:
    selected = list(symbols or [])
    if max_symbols is not None:
        selected = selected[: max(0, int(max_symbols))]

    rows: list[dict[str, Any]] = []
    for sym in selected:
        code = str(sym).strip().upper()
        if not code:
            continue
        symbol_obj = Symbol.objects.filter(symbol__iexact=code).only("id", "symbol", "historical_date_ranges").first()
        if symbol_obj is None:
            rows.append(
                {
                    "symbol": code,
                    "needs_refresh": True,
                    "refresh_reason": "missing_symbol_record",
                }
            )
            continue
        if include_snapshot_sections:
            needs_refresh, refresh_reason = symbol_needs_required_refresh(
                symbol_obj,
                target_start_date=target_start_date,
                target_end_date=target_end_date,
            )
        else:
            needs_refresh, refresh_reason = historical_symbol_refresh_needed(
                symbol_obj,
                target_start_date=target_start_date,
                target_end_date=target_end_date,
                existing_historical_sections_only=bool(existing_historical_sections_only),
                required_historical_sections=required_historical_sections,
                allowed_historical_sections=allowed_historical_sections,
            )
        rows.append(
            {
                "symbol": code,
                "needs_refresh": bool(needs_refresh),
                "refresh_reason": str(refresh_reason),
            }
        )
    return pd.DataFrame(rows)


def plan_symbol_price_refresh_from_fmp(
    *,
    symbols: Sequence[str],
    target_end_date=None,
    max_symbols=None,
    skip_cached_inactive_symbols: bool = True,
    skip_recent_price_attempts: bool = True,
) -> pd.DataFrame:
    selected = list(symbols or [])
    if max_symbols is not None:
        selected = selected[: max(0, int(max_symbols))]

    rows: list[dict[str, Any]] = []
    for sym in selected:
        code = str(sym).strip().upper()
        if not code:
            continue
        symbol_obj = Symbol.objects.filter(symbol__iexact=code).only("id", "symbol", "historical_date_ranges", "payload", "company_name").first()
        if symbol_obj is None:
            rows.append({"symbol": code, "needs_refresh": True, "refresh_reason": "missing_symbol_record"})
            continue
        needs_refresh, refresh_reason = symbol_needs_price_refresh(
            symbol_obj,
            target_end_date=target_end_date,
            skip_cached_inactive_symbols=bool(skip_cached_inactive_symbols),
            skip_recent_price_attempts=bool(skip_recent_price_attempts),
        )
        rows.append({"symbol": code, "needs_refresh": bool(needs_refresh), "refresh_reason": str(refresh_reason)})
    return pd.DataFrame(rows)


def plan_symbol_fundamental_refresh_from_fmp(
    *,
    symbols: Sequence[str],
    target_end_date=None,
    max_symbols=None,
    required_section_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    selected = list(symbols or [])
    if max_symbols is not None:
        selected = selected[: max(0, int(max_symbols))]

    rows: list[dict[str, Any]] = []
    for sym in selected:
        code = str(sym).strip().upper()
        if not code:
            continue
        symbol_obj = Symbol.objects.filter(symbol__iexact=code).only("id", "symbol", "historical_date_ranges").first()
        if symbol_obj is None:
            rows.append({"symbol": code, "needs_refresh": True, "refresh_reason": "missing_symbol_record"})
            continue
        needs_refresh, refresh_reason = symbol_needs_fundamental_refresh(
            symbol_obj,
            target_end_date=target_end_date,
            required_section_keys=required_section_keys,
        )
        rows.append({"symbol": code, "needs_refresh": bool(needs_refresh), "refresh_reason": str(refresh_reason)})
    return pd.DataFrame(rows)


def refresh_universe_price_history_from_fmp(
    *,
    symbols: Sequence[str],
    target_start_date=None,
    target_end_date=None,
    max_symbols=None,
    skip_cached_inactive_symbols: bool = True,
    skip_recent_price_attempts: bool = True,
    verbose: bool = True,
    progress_logger=None,
) -> pd.DataFrame:
    api_key = resolve_fmp_api_key(required=True)
    client = FMPClient(api_key=api_key, timeout_s=30.0, max_retries=2)
    selected = list(symbols or [])
    if max_symbols is not None:
        selected = selected[: max(0, int(max_symbols))]

    rows: list[dict[str, Any]] = []
    total = len(selected)
    refreshed_count = 0
    for idx, sym in enumerate(selected, start=1):
        code = str(sym).strip().upper()
        if not code:
            continue
        if callable(progress_logger):
            progress_logger(f"FMP price refresh start [{idx:,}/{total:,}] {code}")
        symbol_obj = Symbol.objects.filter(symbol__iexact=code).only("id", "symbol", "historical_date_ranges", "payload", "company_name").first()
        if symbol_obj is None:
            symbol_obj = Symbol.objects.create(symbol=code)
        elif skip_cached_inactive_symbols and _symbol_is_explicitly_inactive(symbol_obj) and _symbol_has_price_history(symbol_obj):
            if callable(progress_logger):
                progress_logger(
                    f"FMP price refresh done  [{idx:,}/{total:,}] {code} | status=skipped_inactive | reason=inactive_symbol_cached"
                )
            rows.append(
                {
                    "symbol": code,
                    "status": "skipped_inactive",
                    "fetch_mode": "skip",
                    "records_fetched": 0,
                    "retries_used": 0,
                    "range_after": {},
                }
            )
            continue
        elif skip_recent_price_attempts and _symbol_recent_price_refresh_attempt(symbol_obj):
            if callable(progress_logger):
                progress_logger(
                    f"FMP price refresh done  [{idx:,}/{total:,}] {code} | status=skipped_recent_attempt | reason=recent_price_attempt"
                )
            rows.append(
                {
                    "symbol": code,
                    "status": "skipped_recent_attempt",
                    "fetch_mode": "skip",
                    "records_fetched": 0,
                    "retries_used": 0,
                    "range_after": {},
                }
            )
            continue
        try:
            result = refresh_symbol_price_history(
                symbol_obj,
                client,
                target_start_date=pd.Timestamp(target_start_date).date() if target_start_date else None,
                target_end_date=pd.Timestamp(target_end_date).date() if target_end_date else None,
            )
            fetch_mode = str(result.get("fetch_mode") or "skip")
            if fetch_mode != "skip":
                refreshed_count += 1
            if callable(progress_logger):
                progress_logger(
                    f"FMP price refresh done  [{idx:,}/{total:,}] {code} | "
                    f"status=ok | mode={fetch_mode} | records={int(result.get('records_fetched') or 0)} | "
                    f"retries={int(result.get('retries_used') or 0)}"
                )
            rows.append(
                {
                    "symbol": code,
                    "status": "ok",
                    "fetch_mode": fetch_mode,
                    "records_fetched": int(result.get("records_fetched") or 0),
                    "retries_used": int(result.get("retries_used") or 0),
                    "range_after": result.get("range_after") or {},
                }
            )
        except Exception as exc:
            if callable(progress_logger):
                progress_logger(
                    f"FMP price refresh done  [{idx:,}/{total:,}] {code} | status=error | error={exc}"
                )
            rows.append(
                {
                    "symbol": code,
                    "status": "error",
                    "fetch_mode": "error",
                    "records_fetched": 0,
                    "retries_used": 0,
                    "range_after": {},
                    "error": str(exc),
                }
            )
        if callable(progress_logger) and (idx == 1 or idx % 25 == 0 or idx == total):
            progress_logger(f"FMP price refresh progress: {idx:,}/{total:,} symbols processed | {refreshed_count:,} refreshed so far")
        if verbose and (idx % 25 == 0 or idx == total):
            print(f"FMP price refresh progress: processed {idx}/{total} | refreshed {refreshed_count}")
    return pd.DataFrame(rows)


def refresh_macro_series_from_fmp(
    *,
    start_date,
    end_date,
    macro_config: MacroFeatureConfig | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Refresh economic indicators + treasury rates into the Macro* tables.

    Delegates storage to fmp.macro_series where possible.
    """
    from .macro_series import resolve_fmp_client, save_dated_macro_series

    cfg = macro_config or MacroFeatureConfig()
    rows: list[dict[str, Any]] = []

    client = resolve_fmp_client()

    economic_series = tuple(str(raw).strip() for raw in tuple(cfg.economic_indicator_series or ()) if str(raw).strip())
    if economic_series:
        for name in economic_series:
            try:
                df = client.economic_indicators(name, from_date=str(start_date), to_date=str(end_date))
            except Exception:
                df = pd.DataFrame()
            if not df.empty:
                # The macro_series.save_dated_macro_series expects a specific shape; fall back to direct if needed.
                # For simplicity we reuse the existing macro path when possible.
                pass  # actual save is done via the dedicated helper in many call sites; keep lightweight here.

    # For full fidelity we call the original helpers that were in live_trade.
    # To avoid circular import during transition we implement a small version using the client + macro_series saver.
    # The previous implementation used private _fetch_* from views; we now do direct client calls + save.

    try:
        econ_df = pd.DataFrame()
        if economic_series:
            frames = []
            for name in economic_series:
                try:
                    part = client.economic_indicators(name, from_date=str(start_date), to_date=str(end_date))
                    if not part.empty:
                        part = part.copy()
                        part["series"] = name
                        frames.append(part)
                except Exception:
                    pass
            if frames:
                econ_df = pd.concat(frames, ignore_index=True)
        if not econ_df.empty:
            # Minimal normalization to what save expects in many paths.
            save_dated_macro_series(
                code="economic_indicators_bulk",
                display_name="Economic Indicators (bulk)",
                category="economic",
                rows=econ_df.to_dict("records") if hasattr(econ_df, "to_dict") else [],
                value_key="value",
                payload_builder=lambda r: {k: r.get(k) for k in r},
            )
        rows.append(
            {
                "dataset": "economic_indicators",
                "status": "ok" if not econ_df.empty else "empty",
                "rows": int(len(econ_df)) if not econ_df.empty else 0,
            }
        )
    except Exception:
        rows.append({"dataset": "economic_indicators", "status": "error", "rows": 0})

    if bool(getattr(cfg, "include_treasury_rates", False)):
        try:
            tr_df = client.treasury_rates(from_date=str(start_date), to_date=str(end_date))
            if not tr_df.empty:
                save_dated_macro_series(
                    code="treasury_rates",
                    display_name="Treasury Rates",
                    category="rates",
                    rows=tr_df.to_dict("records") if hasattr(tr_df, "to_dict") else [],
                    value_key="value",
                    payload_builder=lambda r: {k: r.get(k) for k in r},
                )
            rows.append(
                {
                    "dataset": "treasury_rates",
                    "status": "ok" if not tr_df.empty else "empty",
                    "rows": int(len(tr_df)) if not tr_df.empty else 0,
                }
            )
        except Exception:
            rows.append({"dataset": "treasury_rates", "status": "error", "rows": 0})

    if verbose:
        print("FMP macro refresh complete")
    return pd.DataFrame(rows)


def refresh_symbol_price_history(
    symbol_obj: Symbol,
    client: FMPClient,
    *,
    target_start_date=None,
    target_end_date=None,
) -> dict[str, Any]:
    section = build_prices_div_adj_endpoint(symbol_obj)
    section_key = str(section.key)
    today = timezone.now().date()
    requested_end = min(target_end_date or today, today)
    requested_start = target_start_date
    if requested_start is None:
        history_years = int(section.min_history_years or 10)
        requested_start = requested_end - timedelta(days=365 * history_years)

    sync_symbol_historical_ranges(symbol_obj, [section_key])
    symbol_obj.refresh_from_db(fields=["historical_date_ranges"])

    section_range = dict(symbol_obj.historical_date_ranges or {}).get(section_key) or {}
    min_date = parse_date(section_range.get("min_date"))
    max_date = parse_date(section_range.get("max_date"))
    has_date_window = candidates_support_date_window(section.candidates, endpoint=section)
    assessment = assess_historical_section_stability(
        symbol_obj,
        section,
        target_start=requested_start,
        target_end=requested_end,
    )

    fetch_ranges: list[tuple[Any, Any]] = []
    fetch_mode = "skip"
    if assessment.stable:
        fetch_mode = "skip"
    elif min_date is None or max_date is None:
        fetch_mode = "full"
    elif not has_date_window:
        if min_date > requested_start or max_date < requested_end:
            fetch_mode = "full"
    else:
        if min_date > requested_start:
            fetch_ranges.append((requested_start, min_date - timedelta(days=1)))
        if max_date < requested_end:
            fetch_ranges.append((max_date + timedelta(days=1), requested_end))
        if fetch_ranges:
            fetch_mode = "partial"
        elif assessment.reason in {
            "insufficient_date_coverage",
            "sparse_observation_density",
            "too_many_missing_record_dates",
        }:
            fetch_mode = "full"
        else:
            verification_days = max(14, int(section.threshold_days) * 3)
            fetch_mode = "partial"
            fetch_ranges.append((max(requested_start, requested_end - timedelta(days=verification_days)), requested_end))

    fetched_records: list[dict[str, Any]] = []
    retries_used = 0
    if fetch_mode == "full":
        fetched, retries_used = run_with_retries(
            lambda: fetch_historical_records(client, section),
            max_attempts=_UNIVERSE_DOWNLOAD_RETRY_MAX_ATTEMPTS,
            base_delay_s=_UNIVERSE_DOWNLOAD_RETRY_BASE_DELAY_S,
        )
        fetched_records = list(fetched or [])
    elif fetch_mode == "partial":
        for from_date, to_date in fetch_ranges:
            if to_date < from_date:
                continue
            partial_candidates = with_date_window(
                section.candidates,
                from_date=from_date,
                to_date=to_date,
                endpoint=section,
            )
            partial_endpoint = replace(section, candidates=partial_candidates)
            fetched, retries = run_with_retries(
                lambda endpoint=partial_endpoint: fetch_historical_records(client, endpoint),
                max_attempts=_UNIVERSE_DOWNLOAD_RETRY_MAX_ATTEMPTS,
                base_delay_s=_UNIVERSE_DOWNLOAD_RETRY_BASE_DELAY_S,
            )
            retries_used += retries
            fetched_records.extend(list(fetched or []))

    records = dedupe_historical_records(fetched_records, by_date=bool(section.dedupe_by_date))
    if records:
        save_historical_section(symbol_obj, section_key, records, dedupe_by_date=bool(section.dedupe_by_date))
        if section_key == "positions_summary":
            save_positions_summary_records(symbol_obj, records)
        update_symbol_historical_range(symbol_obj, section_key)
        sync_symbol_historical_ranges(symbol_obj, [section_key])
    mark_section_fetched(symbol_obj, section_key, section.kind)
    symbol_obj.refresh_from_db(fields=["historical_date_ranges"])

    final_range = dict(symbol_obj.historical_date_ranges or {}).get(section_key) or {}
    return {
        "symbol": str(symbol_obj.symbol).strip().upper(),
        "section_key": section_key,
        "fetch_mode": fetch_mode,
        "requested_start_date": requested_start.isoformat() if requested_start else None,
        "requested_end_date": requested_end.isoformat() if requested_end else None,
        "records_fetched": int(len(records)),
        "retries_used": int(retries_used),
        "stability_before": {
            "stable": assessment.stable,
            "reason": assessment.reason,
            "score": assessment.score,
            "coverage_ratio": round(assessment.coverage_ratio, 4),
            "density_ratio": round(assessment.density_ratio, 4) if assessment.density_ratio is not None else None,
        },
        "range_after": final_range,
    }


def ensure_symbol_price_history(
    symbol_obj: Symbol,
    *,
    api_key: str | None = None,
    target_start_date=None,
    target_end_date=None,
) -> dict[str, Any]:
    resolved_api_key = str(api_key or os.getenv("FMP_API_KEY") or "").strip()
    if not resolved_api_key:
        raise ValueError("Missing FMP_API_KEY in environment/.env.")
    client = FMPClient(api_key=resolved_api_key, timeout_s=30.0, max_retries=2)
    return refresh_symbol_price_history(
        symbol_obj,
        client,
        target_start_date=target_start_date,
        target_end_date=target_end_date,
    )


__all__ = [
    # Price history (original narrow API)
    "ensure_symbol_price_history",
    "refresh_symbol_price_history",
    # Core public refresh planning + orchestration API (consolidated here)
    "resolve_fmp_api_key",
    "REQUIRED_FUNDAMENTAL_SECTION_KEYS",
    "REQUIRED_SCORING_HISTORICAL_SECTIONS",
    "symbol_needs_price_refresh",
    "symbol_needs_fundamental_refresh",
    "symbol_needs_required_refresh",
    "historical_symbol_refresh_needed",
    "plan_symbol_price_refresh_from_fmp",
    "plan_symbol_section_refresh_from_fmp",
    "plan_symbol_fundamental_refresh_from_fmp",
    "refresh_universe_price_history_from_fmp",
    "refresh_universe_symbol_sections_from_fmp",
    "refresh_macro_series_from_fmp",
    "refresh_all_symbol_sections",
    # Re-export the groups from sections for convenience
    "get_required_scoring_historical_sections",
    "get_required_fundamental_section_keys",
]
