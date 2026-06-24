import os
from pathlib import Path
import math
import re
import json
import time
import hashlib
from dataclasses import replace
from datetime import timedelta
from typing import Any
import pandas as pd

from django.http import Http404, HttpRequest, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.clickjacking import xframe_options_exempt
from django.utils import timezone
from django.db.models import Min, Max, Count

from .endpoints import get_symbol_endpoint_definitions
from .endpoints.base import EndpointDefinition
from .records import dedupe_historical_records
from .section_store import (
    mark_section_fetched,
    save_historical_section,
    save_snapshot_section,
    sync_symbol_historical_ranges,
    update_symbol_historical_range,
)
from .stability import assess_historical_section_stability
from .transport import (
    candidates_support_date_window,
    fetch_first_success,
    fetch_historical_records,
    run_with_retries,
    to_records,
    with_date_window,
)
from .forms import (
    COUNTRY_CHOICES,
    EXCHANGE_CHOICES,
    INDUSTRY_CHOICES,
    SECTOR_CHOICES,
    EconomicIndicatorsForm,
    TreasuryRatesForm,
    UniverseScreenerForm,
)
from .models import (
    Country,
    EconomicIndicatorObservation,
    EconomicIndicatorSeries,
    Exchange,
    Industry,
    Sector,
    Symbol,
    SymbolSectionHistorical,
    SymbolSectionSnapshot,
    SymbolSectionState,
    TreasuryRateObservation,
    TreasuryRateSeries,
    UniverseDownloadJob,
    WorkflowState,
)
from features.naming import feature_display_name
from data import FMPClient, screen_companies_fmp
from utils.workflow import default_feature_symbol

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass


def form_tabs_view(request: HttpRequest):
    tabs = [
        {"key": "pipeline", "label": "Pipeline", "url": "/pipeline/ui/"},
        {"key": "universe", "label": "Universe Screener", "url": "/fmp/universe-screener/form/"},
        {"key": "labels", "label": "Labels", "url": "/labels/form/"},
        {"key": "feature", "label": "Feature", "url": "/features/form/"},
        {"key": "models", "label": "Models", "url": "/ml/form/"},
    ]
    return render(
        request,
        "forms/tabs.html",
        {
            "tabs": tabs,
            "feature_symbol": default_feature_symbol(request),
        },
    )


def _parse_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _parse_float(value: str | None, field: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {field}: {value}") from exc


def _parse_int(value: str | None, field: str, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {field}: {value}") from exc


def _format_cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return f"{int(value):,}"
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        if abs(value) >= 1:
            return f"{value:,.4f}".rstrip("0").rstrip(".")
        return f"{value:,.6f}".rstrip("0").rstrip(".")
    return str(value)


def _abbrev_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}".rstrip("0").rstrip(".") + "T"
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}".rstrip("0").rstrip(".") + "B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}".rstrip("0").rstrip(".") + "M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}".rstrip("0").rstrip(".") + "K"
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}".rstrip("0").rstrip(".")


def _format_cell_for_column(column: str, value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""

    col = column.lower()
    numeric_value = value if isinstance(value, (int, float)) else None

    if "marketcap" in col and isinstance(numeric_value, (int, float)):
        return _abbrev_number(float(numeric_value))

    if "price" in col and isinstance(numeric_value, (int, float)):
        return "$" + f"{float(numeric_value):,.2f}"

    if "dividend" in col and isinstance(numeric_value, (int, float)):
        if "yield" in col or "percent" in col or col.endswith("pct"):
            pct_value = float(numeric_value) * 100 if abs(float(numeric_value)) <= 1 else float(numeric_value)
            return f"{pct_value:.2f}".rstrip("0").rstrip(".") + "%"
        return "$" + f"{float(numeric_value):,.2f}"

    if ("yield" in col or "percent" in col or col.endswith("pct")) and isinstance(numeric_value, (int, float)):
        pct_value = float(numeric_value) * 100 if abs(float(numeric_value)) <= 1 else float(numeric_value)
        return f"{pct_value:.2f}".rstrip("0").rstrip(".") + "%"

    return _format_cell(value)


def _prettify_header(name: str) -> str:
    return feature_display_name(name)


def _collect_columns(records: list[dict]) -> list[str]:
    cols: list[str] = []
    seen: set[str] = set()
    for row in records:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                cols.append(key)
    return cols


def _market_cap_millions_to_dollars(value):
    if value is None:
        return None
    return float(value) * 1_000_000


def _volume_millions_to_units(value):
    if value is None:
        return None
    return float(value) * 1_000_000


def _default_country_value(country_choices: list[tuple[str, str]] | None) -> str:
    if not country_choices:
        return "US"
    for value, _label in country_choices:
        if str(value).strip().upper() == "US":
            return str(value)
    for value, label in country_choices:
        v = str(value).strip().lower()
        l = str(label).strip().lower()
        if v in {"united states", "usa"} or "united states" in l:
            return str(value)
    return "US"


def _default_us_exchange_values(exchange_choices: list[tuple[str, str]] | None) -> list[str]:
    if not exchange_choices:
        return []
    # Keep defaults strict to avoid accidental non-US exchanges (e.g., NASDAQ Helsinki).
    known_us_codes = {"NASDAQ", "NYSE", "AMEX", "CBOE", "OTC", "PNK", "IEX", "ARCA", "BATS"}
    selected: list[str] = []
    seen: set[str] = set()
    for value, _label in exchange_choices:
        v = str(value).strip()
        if not v:
            continue
        if v.upper() in known_us_codes:
            if v not in seen:
                seen.add(v)
                selected.append(v)
    return selected


def _safe_float(value):
    if value is None or value == "":
        return None
    try:
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _fetch_economic_indicators_from_api(
    api_key: str,
    start_date: str,
    end_date: str,
    series: tuple[str, ...],
) -> pd.DataFrame:
    client = FMPClient(api_key=api_key, timeout_s=30.0, max_retries=2)
    frames: list[pd.DataFrame] = []

    for raw_series in series:
        series_name = str(raw_series).strip()
        if not series_name:
            continue
        try:
            df = client.economic_indicators(series_name, from_date=start_date, to_date=end_date)
        except Exception:
            continue
        if df is None or df.empty or "Error Message" in df.columns or "date" not in df.columns:
            continue
        chosen_df = df.copy()
        chosen_df["date"] = pd.to_datetime(chosen_df["date"], errors="coerce")
        chosen_df = chosen_df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
        value_col = "value"
        if value_col not in chosen_df.columns:
            numeric_cols = [c for c in chosen_df.columns if c != "date" and pd.api.types.is_numeric_dtype(chosen_df[c])]
            if not numeric_cols:
                continue
            value_col = numeric_cols[0]
        clean = chosen_df[["date", value_col]].rename(columns={value_col: series_name}).set_index("date")
        frames.append(clean)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1).sort_index()
    df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
    return df


def _fetch_treasury_rates_from_api(
    api_key: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    client = FMPClient(api_key=api_key, timeout_s=30.0, max_retries=2)
    try:
        tr_df = client.treasury_rates(from_date=start_date, to_date=end_date)
    except Exception:
        return pd.DataFrame()
    if tr_df is None or tr_df.empty or "date" not in tr_df.columns:
        return pd.DataFrame()
    tr_df["date"] = pd.to_datetime(tr_df["date"], errors="coerce")
    tr_df = tr_df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    rename_map = {c: f"macro__ust_{c}" for c in tr_df.columns if c != "date"}
    tr_df = tr_df.rename(columns=rename_map).set_index("date")
    return tr_df


def _series_is_stale(series_obj, start_date, end_date, *, threshold_days: int) -> bool:
    if not series_obj.last_fetched_at:
        return True
    age = timezone.now() - series_obj.last_fetched_at
    if age > timedelta(days=threshold_days):
        return True
    if series_obj.min_date and series_obj.min_date > start_date:
        return True
    if series_obj.max_date and series_obj.max_date < end_date:
        return True
    return False


def _store_series_dataframe(
    df: pd.DataFrame,
    *,
    series_model,
    observation_model,
) -> None:
    if df.empty:
        return
    safe_df = df.sort_index()
    fetched_at = timezone.now()
    for col in safe_df.columns:
        series_obj, _ = series_model.objects.get_or_create(
            code=str(col),
            defaults={
                "display_name": _prettify_header(str(col)),
            },
        )
        values_by_date = []
        for idx, value in safe_df[col].items():
            obs_date = pd.Timestamp(idx).date()
            if pd.isna(value):
                continue
            val = float(value)
            observation_model.objects.update_or_create(
                series=series_obj,
                observation_date=obs_date,
                defaults={
                    "value": val,
                    "payload": {"value": val},
                },
            )
            values_by_date.append(obs_date)
        if values_by_date:
            existing_bounds = observation_model.objects.filter(series=series_obj).aggregate(
                min_date=Min("observation_date"),
                max_date=Max("observation_date"),
            )
            series_obj.display_name = _prettify_header(str(col))
            series_obj.last_fetched_at = fetched_at
            series_obj.min_date = existing_bounds.get("min_date") or min(values_by_date)
            series_obj.max_date = existing_bounds.get("max_date") or max(values_by_date)
            series_obj.save(
                update_fields=[
                    "display_name",
                    "last_fetched_at",
                    "min_date",
                    "max_date",
                    "last_updated",
                ]
            )


def _load_series_dataframe_from_db(
    series_codes: list[str],
    start_date,
    end_date,
    *,
    series_model,
    observation_model,
    threshold_days: int,
) -> pd.DataFrame:
    if not series_codes:
        return pd.DataFrame()
    series_qs = series_model.objects.filter(code__in=series_codes)
    series_map = {row.code: row for row in series_qs}
    if len(series_map) != len(set(series_codes)):
        return pd.DataFrame()
    for code in series_codes:
        series_obj = series_map[code]
        if _series_is_stale(series_obj, start_date, end_date, threshold_days=threshold_days):
            return pd.DataFrame()
    obs_qs = (
        observation_model.objects.filter(
            series__code__in=series_codes,
            observation_date__gte=start_date,
            observation_date__lte=end_date,
        )
        .select_related("series")
        .order_by("observation_date")
    )
    rows: dict[str, dict[str, float | None]] = {}
    for obs in obs_qs.iterator():
        date_key = obs.observation_date.isoformat()
        row = rows.setdefault(date_key, {})
        row[obs.series.code] = obs.value
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    out.index = pd.to_datetime(out.index)
    for code in series_codes:
        if code not in out.columns:
            out[code] = None
    return out[series_codes]


def _build_series_summaries(series_codes: list[str], *, series_model) -> list[dict]:
    if not series_codes:
        return []
    rows = (
        series_model.objects.filter(code__in=series_codes)
        .annotate(observation_count=Count("observations"))
        .order_by("code")
    )
    summaries = []
    for row in rows:
        summaries.append(
            {
                "code": row.code,
                "display_name": row.display_name or _prettify_header(row.code),
                "start_date": row.min_date.isoformat() if row.min_date else "",
                "end_date": row.max_date.isoformat() if row.max_date else "",
                "count": int(row.observation_count or 0),
                "detail_href": reverse("macro-series-detail", args=[row.code]),
            }
        )
    return summaries


def _save_symbols(records: list[dict]) -> int:
    saved = 0
    for row in records:
        symbol = str(row.get("symbol") or "").strip()
        if not symbol:
            continue

        Symbol.objects.update_or_create(
            symbol=symbol,
            defaults={
                "company_name": str(row.get("companyName") or row.get("name") or ""),
                "exchange": str(row.get("exchangeShortName") or row.get("exchange") or ""),
                "country": str(row.get("country") or ""),
                "sector": str(row.get("sector") or ""),
                "industry": str(row.get("industry") or ""),
                "market_cap": _safe_float(row.get("marketCap")),
                "price": _safe_float(row.get("price")),
                "beta": _safe_float(row.get("beta")),
                "volume": _safe_float(row.get("volume")),
                "dividend": _safe_float(row.get("lastDividend") or row.get("dividend")),
                "dividend_yield": _safe_float(row.get("dividendYield")),
                "payload": _json_safe(row),
            },
        )
        saved += 1
    return saved


def _payload_bool(record: dict[str, Any], *keys: str) -> bool | None:
    for key in keys:
        value = record.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "t", "yes", "y"}:
            return True
        if text in {"0", "false", "f", "no", "n"}:
            return False
    return None


def _bool_match(desired: bool | None, actual: bool | None) -> bool:
    if desired is None:
        return True
    if desired is True:
        return actual is True
    # For "false", include explicit False and unknown values.
    return actual is not True


def _build_symbol_record_from_db(symbol_obj: Symbol) -> dict[str, Any]:
    payload = symbol_obj.payload if isinstance(symbol_obj.payload, dict) else {}
    record = dict(payload)
    record["symbol"] = record.get("symbol") or symbol_obj.symbol
    company_name = (
        record.get("companyName")
        or record.get("name")
        or symbol_obj.company_name
        or ""
    )
    exchange_value = record.get("exchangeShortName") or record.get("exchange") or symbol_obj.exchange or ""
    record["companyName"] = company_name
    record["name"] = record.get("name") or company_name
    record["exchangeShortName"] = exchange_value
    record["exchange"] = record.get("exchange") or exchange_value
    record["country"] = record.get("country") or symbol_obj.country or ""
    record["sector"] = record.get("sector") or symbol_obj.sector or ""
    record["industry"] = record.get("industry") or symbol_obj.industry or ""
    record["marketCap"] = record.get("marketCap")
    if record["marketCap"] in (None, ""):
        record["marketCap"] = symbol_obj.market_cap
    record["price"] = record.get("price")
    if record["price"] in (None, ""):
        record["price"] = symbol_obj.price
    record["beta"] = record.get("beta")
    if record["beta"] in (None, ""):
        record["beta"] = symbol_obj.beta
    record["volume"] = record.get("volume")
    if record["volume"] in (None, ""):
        record["volume"] = symbol_obj.volume
    record["lastDividend"] = record.get("lastDividend")
    if record["lastDividend"] in (None, ""):
        record["lastDividend"] = symbol_obj.dividend
    record["dividendYield"] = record.get("dividendYield")
    if record["dividendYield"] in (None, ""):
        record["dividendYield"] = symbol_obj.dividend_yield
    return record


def _screen_companies_db(
    *,
    limit: int = 10_000,
    marketCapMoreThan: float | None = None,
    marketCapLowerThan: float | None = None,
    sector: str | None = None,
    industry: str | None = None,
    betaMoreThan: float | None = None,
    betaLowerThan: float | None = None,
    priceMoreThan: float | None = None,
    priceLowerThan: float | None = None,
    dividendMoreThan: float | None = None,
    dividendLowerThan: float | None = None,
    volumeMoreThan: float | None = None,
    volumeLowerThan: float | None = None,
    exchange_values: list[str] | None = None,
    country: str | None = None,
    isEtf: bool | None = None,
    isFund: bool | None = None,
    isActivelyTrading: bool | None = None,
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    qs = Symbol.objects.all()

    if marketCapMoreThan is not None:
        qs = qs.filter(market_cap__gte=float(marketCapMoreThan))
    if marketCapLowerThan is not None:
        qs = qs.filter(market_cap__lte=float(marketCapLowerThan))
    if betaMoreThan is not None:
        qs = qs.filter(beta__gte=float(betaMoreThan))
    if betaLowerThan is not None:
        qs = qs.filter(beta__lte=float(betaLowerThan))
    if priceMoreThan is not None:
        qs = qs.filter(price__gte=float(priceMoreThan))
    if priceLowerThan is not None:
        qs = qs.filter(price__lte=float(priceLowerThan))
    if dividendMoreThan is not None:
        qs = qs.filter(dividend__gte=float(dividendMoreThan))
    if dividendLowerThan is not None:
        qs = qs.filter(dividend__lte=float(dividendLowerThan))
    if volumeMoreThan is not None:
        qs = qs.filter(volume__gte=float(volumeMoreThan))
    if volumeLowerThan is not None:
        qs = qs.filter(volume__lte=float(volumeLowerThan))
    if sector:
        qs = qs.filter(sector__iexact=str(sector).strip())
    if industry:
        qs = qs.filter(industry__iexact=str(industry).strip())
    if country:
        qs = qs.filter(country__iexact=str(country).strip())
    if exchange_values:
        normalized = [str(value).strip() for value in exchange_values if str(value).strip()]
        if normalized:
            qs = qs.filter(exchange__in=normalized)

    records: list[dict[str, Any]] = []
    for symbol_obj in qs.order_by("symbol").iterator():
        record = _build_symbol_record_from_db(symbol_obj)
        etf_value = _payload_bool(record, "isEtf", "isETF", "etf", "is_etf")
        fund_value = _payload_bool(record, "isFund", "isMutualFund", "fund", "is_fund")
        active_value = _payload_bool(
            record,
            "isActivelyTrading",
            "activelyTrading",
            "is_active",
        )
        if not _bool_match(isEtf, etf_value):
            continue
        if not _bool_match(isFund, fund_value):
            continue
        if not _bool_match(isActivelyTrading, active_value):
            continue
        records.append(record)
        if len(records) >= int(limit):
            break

    symbols = tuple(sorted({str(r.get("symbol", "")).strip() for r in records if str(r.get("symbol", "")).strip()}))
    return symbols, records


def _normalize_cell(value):
    try:
        value = value.item()  # numpy scalar -> python scalar
    except Exception:
        pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _df_to_table(df, *, max_rows: int = 20) -> tuple[list[str], list[list]]:
    if df is None or getattr(df, "empty", True):
        return [], []
    if "date" in df.columns:
        try:
            df = df.sort_values("date", ascending=False)
        except Exception:
            pass
    df = df.head(max_rows)
    columns = [str(c) for c in list(df.columns)]
    rows: list[list] = []
    for row in df.to_dict(orient="records"):
        rows.append([_format_cell_for_column(col, _normalize_cell(row.get(col))) for col in columns])
    return columns, rows


def _records_to_table(records: list[dict], *, max_rows: int = 20) -> tuple[list[str], list[str], list[list]]:
    if not records:
        return [], [], []
    records = records[:max_rows]
    columns = _collect_columns(records)
    header_labels = [_prettify_header(c) for c in columns]
    rows: list[list] = []
    for row in records:
        rows.append([_format_cell_for_column(col, _normalize_cell(row.get(col))) for col in columns])
    return columns, header_labels, rows


def _to_records(data: Any) -> list[dict]:
    return to_records(data)


def _fetch_first_success(client: FMPClient, candidates: list[tuple[str, dict]]) -> Any:
    return fetch_first_success(client, candidates)


def _fetch_all_historical_records(client: FMPClient, candidates: list[tuple[str, dict]]) -> list[dict]:
    """
    Fetch as much historical data as an endpoint exposes.
    - If endpoint supports page+limit: paginate until empty/short page.
    - Otherwise: single request (or with increased limit if 'limit' is present).
    """
    first_params = dict(candidates[0][1] or {}) if candidates else {}
    endpoint = EndpointDefinition(
        key="legacy",
        title="Legacy",
        kind="historical",
        threshold_days=1,
        max_rows=100,
        candidates=candidates,
        pagination="page" if "page" in first_params else "none",
        supports_date_window="from" in first_params and "to" in first_params,
        chunk_years=int(first_params.get("__chunk_years", 0) or 0) or None,
    )
    return fetch_historical_records(client, endpoint)


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


def _prepare_payload_for_storage(payload: Any) -> Any:
    if payload is None:
        return {}
    try:
        json.dumps(payload, allow_nan=False, separators=(",", ":"))
        return payload
    except (TypeError, ValueError):
        return _json_safe(payload)


def _stable_record_key(record: Any) -> str:
    blob = json.dumps(_prepare_payload_for_storage(record), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _extract_record_date(record: Any):
    if not isinstance(record, dict):
        return None
    for key in (
        "date",
        "publishedDate",
        "publishedAt",
        "published",
        "filingDate",
        "acceptedDate",
        "recordDate",
        "periodOfReport",
        "calendarYear",
    ):
        v = record.get(key)
        if v in (None, ""):
            continue
        try:
            dt = timezone.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
            return dt.date()
        except Exception:
            try:
                return timezone.datetime.strptime(str(v)[:10], "%Y-%m-%d").date()
            except Exception:
                pass
    return None


def _dedupe_records_by_record_date(records: list[dict]) -> list[dict]:
    return dedupe_historical_records(records, by_date=True, prepare=_prepare_payload_for_storage)


def _repair_missing_record_dates(symbol_obj: Symbol, section_key: str):
    qs = SymbolSectionHistorical.objects.filter(
        symbol=symbol_obj,
        section_key=section_key,
        record_date__isnull=True,
    )
    for obj in qs.iterator():
        payload = obj.payload if isinstance(obj.payload, dict) else {}
        parsed = _extract_record_date(payload)
        if parsed is not None:
            obj.record_date = parsed
            obj.save(update_fields=["record_date"])


def _is_section_stale(
    symbol_obj: Symbol,
    section_key: str,
    threshold_days: int,
    *,
    state: SymbolSectionState | None = None,
) -> bool:
    if state is None:
        state = SymbolSectionState.objects.filter(symbol=symbol_obj, section_key=section_key).first()
    if not state or not state.last_fetched_at:
        return True
    return state.last_fetched_at < (timezone.now() - timedelta(days=threshold_days))


def _is_historical_section_stale_from_coverage(
    symbol_obj: Symbol,
    section_key: str,
    threshold_days: int,
    target_min_date=None,
    *,
    state: SymbolSectionState | None = None,
) -> bool:
    ranges = symbol_obj.historical_date_ranges or {}
    section_range = ranges.get(section_key) if isinstance(ranges, dict) else None
    if not section_range or not section_range.get("max_date"):
        return True

    now = timezone.now()
    today = now.date()
    threshold_cutoff = today - timedelta(days=threshold_days)

    try:
        max_date = timezone.datetime.strptime(str(section_range.get("max_date"))[:10], "%Y-%m-%d").date()
    except Exception:
        return True
    if max_date < threshold_cutoff:
        return True

    # If the oldest stored point is too recent, request a deeper backfill.
    # Guard with last_fetched_at so we don't refetch on every page load.
    if target_min_date is not None:
        min_date_raw = section_range.get("min_date")
        try:
            min_date = (
                timezone.datetime.strptime(str(min_date_raw)[:10], "%Y-%m-%d").date()
                if min_date_raw
                else None
            )
        except Exception:
            min_date = None

        if min_date is None or min_date > target_min_date:
            if state is None:
                state = SymbolSectionState.objects.filter(symbol=symbol_obj, section_key=section_key).first()
            if not state or not state.last_fetched_at:
                return True
            return state.last_fetched_at < (now - timedelta(days=threshold_days))

    return False


def _mark_section_fetched(symbol_obj: Symbol, section_key: str, kind: str):
    mark_section_fetched(symbol_obj, section_key, kind)


def _save_snapshot_section(symbol_obj: Symbol, section_key: str, payload: Any):
    save_snapshot_section(symbol_obj, section_key, payload)


def _save_historical_section(
    symbol_obj: Symbol,
    section_key: str,
    records: list[Any],
    *,
    dedupe_by_date: bool = False,
):
    save_historical_section(symbol_obj, section_key, records, dedupe_by_date=dedupe_by_date)


def _update_symbol_historical_range(symbol_obj: Symbol, section_key: str):
    update_symbol_historical_range(symbol_obj, section_key)


def _sync_symbol_historical_ranges_from_db(symbol_obj: Symbol, section_keys: list[str]):
    sync_symbol_historical_ranges(symbol_obj, section_keys)


def _load_snapshot_section(symbol_obj: Symbol, section_key: str, *, max_rows: int = 1):
    obj = SymbolSectionSnapshot.objects.filter(symbol=symbol_obj, section_key=section_key).first()
    records = _to_records(obj.payload) if obj else []
    _, header_labels, rows = _records_to_table(records, max_rows=max_rows)
    return header_labels, rows


def _load_snapshot_pairs(symbol_obj: Symbol, section_key: str) -> list[dict[str, str]]:
    obj = SymbolSectionSnapshot.objects.filter(symbol=symbol_obj, section_key=section_key).first()
    records = _to_records(obj.payload) if obj else []
    payload = records[0] if records and isinstance(records[0], dict) else {}
    items: list[dict[str, str]] = []
    for key, value in payload.items():
        items.append(
            {
                "label": _prettify_header(str(key)),
                "value": _format_cell_for_column(str(key), _normalize_cell(value)),
            }
        )
    return items


def _load_historical_section(symbol_obj: Symbol, section_key: str, *, max_rows: int = 50):
    qs = SymbolSectionHistorical.objects.filter(symbol=symbol_obj, section_key=section_key).order_by("-record_date", "-updated_at")[:max_rows]
    records = [obj.payload for obj in qs]
    _, header_labels, rows = _records_to_table(records, max_rows=max_rows)
    return header_labels, rows


def _load_cached_section_states(symbol_obj: Symbol) -> dict[str, SymbolSectionState]:
    rows = SymbolSectionState.objects.filter(symbol=symbol_obj)
    return {row.section_key: row for row in rows}


def _load_cached_snapshots(symbol_obj: Symbol) -> dict[str, Any]:
    rows = SymbolSectionSnapshot.objects.filter(symbol=symbol_obj).only("section_key", "payload")
    out: dict[str, Any] = {}
    for row in rows:
        out[row.section_key] = row.payload
    return out


def _load_cached_history_rows(symbol_obj: Symbol, section_limits: dict[str, int]) -> dict[str, list[dict]]:
    if not section_limits:
        return {}
    rows = (
        SymbolSectionHistorical.objects.filter(symbol=symbol_obj, section_key__in=list(section_limits.keys()))
        .only("section_key", "payload", "record_date", "updated_at")
        .order_by("section_key", "-record_date", "-updated_at")
    )
    out: dict[str, list[dict]] = {key: [] for key in section_limits}
    for row in rows.iterator():
        section_key = row.section_key
        bucket = out.get(section_key)
        if bucket is None or len(bucket) >= int(section_limits.get(section_key, 0) or 0):
            continue
        records = _to_records(row.payload)
        if records:
            bucket.append(records[0])
    return out


_FILTER_CHOICES_CACHE: dict[str, Any] = {"ts": 0.0, "data": None}
_FILTER_TTL_SECONDS = 21600
_LOOKUP_MAX_AGE_DAYS = 30

# Canonical values now live in fmp.sections (single source of truth for refresh planning).
from .sections import (
    _UNIVERSE_DOWNLOAD_COVERAGE_THRESHOLD,
    _UNIVERSE_DOWNLOAD_TARGET_YEARS,
    _UNIVERSE_DOWNLOAD_RETRY_MAX_ATTEMPTS,
    _UNIVERSE_DOWNLOAD_RETRY_BASE_DELAY_S,
    _FUNDAMENTAL_STATEMENT_ANCHOR_SECTION_KEYS,
    _FUNDAMENTAL_FALLBACK_ANCHOR_SECTION_KEYS,
    _SPARSE_EVENT_HISTORICAL_SECTION_KEYS,
    _FUNDAMENTAL_DEPENDENT_SECTION_KEYS,
)


def _dedupe_choice_tuples(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value, label in items:
        v = str(value).strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append((v, str(label).strip() or v))
    return sorted(out, key=lambda x: x[1].lower())


def _extract_choices(
    payload: Any,
    *,
    value_keys: tuple[str, ...],
    label_keys: tuple[str, ...],
) -> list[tuple[str, str]]:
    records = _to_records(payload)
    choices: list[tuple[str, str]] = []
    for rec in records:
        if not isinstance(rec, dict):
            v = str(rec).strip()
            if v:
                choices.append((v, v))
            continue
        value = None
        for k in value_keys:
            if rec.get(k) not in (None, ""):
                value = rec.get(k)
                break
        if value is None and len(rec) == 1:
            value = list(rec.values())[0]
        if value is None:
            continue
        label = None
        for k in label_keys:
            if rec.get(k) not in (None, ""):
                label = rec.get(k)
                break
        if label is None:
            label = value
        choices.append((str(value), str(label)))
    return _dedupe_choice_tuples(choices)


def _fetch_filter_choices(api_key: str | None) -> dict[str, list[tuple[str, str]]]:
    # 6h in-process cache to avoid repeated calls on every page load.
    now = time.time()
    cached = _FILTER_CHOICES_CACHE.get("data")
    if cached and (now - float(_FILTER_CHOICES_CACHE.get("ts", 0.0)) < _FILTER_TTL_SECONDS):
        return cached

    fallback = {
        "industry_choices": list(INDUSTRY_CHOICES),
        "sector_choices": list(SECTOR_CHOICES),
        "exchange_choices": list(EXCHANGE_CHOICES[1:]),
        "country_choices": list(COUNTRY_CHOICES),
    }

    def _db_choices() -> dict[str, list[tuple[str, str]]]:
        industries_db = [(obj.name, obj.name) for obj in Industry.objects.all()]
        sectors_db = [(obj.name, obj.name) for obj in Sector.objects.all()]
        exchanges_db = [(obj.code, obj.name or obj.code) for obj in Exchange.objects.all()]
        countries_db = [(obj.code, obj.name or obj.code) for obj in Country.objects.all()]
        industries_db = sorted(industries_db, key=lambda x: x[1].lower())
        sectors_db = sorted(sectors_db, key=lambda x: x[1].lower())
        exchanges_db = sorted(exchanges_db, key=lambda x: x[1].lower())
        countries_db = sorted(countries_db, key=lambda x: x[1].lower())
        return {
            "industry_choices": [("", "Any")] + (industries_db or list(INDUSTRY_CHOICES[1:])),
            "sector_choices": [("", "Any")] + (sectors_db or list(SECTOR_CHOICES[1:])),
            "exchange_choices": exchanges_db or list(EXCHANGE_CHOICES[1:]),
            "country_choices": [("", "Any")] + (countries_db or list(COUNTRY_CHOICES[1:])),
        }

    def _latest_updated(model) -> Any:
        return model.objects.order_by("-last_updated").values_list("last_updated", flat=True).first()

    def _is_stale(model) -> bool:
        latest = _latest_updated(model)
        if latest is None:
            return True
        return latest < (timezone.now() - timedelta(days=_LOOKUP_MAX_AGE_DAYS))

    needs_refresh = _is_stale(Industry) or _is_stale(Sector) or _is_stale(Exchange) or _is_stale(Country)

    if needs_refresh and api_key:
        try:
            client = FMPClient(api_key=api_key, timeout_s=15.0, max_retries=1)
            industries = _extract_choices(
                client.get_json("/stable/available-industries"),
                value_keys=("industry", "name", "value"),
                label_keys=("industry", "name", "value"),
            )
            sectors = _extract_choices(
                client.get_json("/stable/available-sectors"),
                value_keys=("sector", "name", "value"),
                label_keys=("sector", "name", "value"),
            )
            exchanges = _extract_choices(
                client.get_json("/stable/available-exchanges"),
                value_keys=("exchangeShortName", "exchange", "symbol", "name", "value"),
                label_keys=("exchangeShortName", "name", "exchange", "symbol", "value"),
            )
            countries = _extract_choices(
                client.get_json("/stable/available-countries"),
                value_keys=("countryCode", "code", "country", "name", "value"),
                label_keys=("country", "name", "countryCode", "code", "value"),
            )

            for value, _label in industries:
                Industry.objects.update_or_create(name=value, defaults={})
            for value, _label in sectors:
                Sector.objects.update_or_create(name=value, defaults={})
            for code, label in exchanges:
                Exchange.objects.update_or_create(code=code, defaults={"name": label or code})
            for code, label in countries:
                Country.objects.update_or_create(code=code, defaults={"name": label or code})
        except Exception:
            pass

    try:
        data = _db_choices()
    except Exception:
        data = fallback

    _FILTER_CHOICES_CACHE["data"] = data
    _FILTER_CHOICES_CACHE["ts"] = now
    return data


def _parse_iso_date(value: Any):
    if not value:
        return None
    try:
        return timezone.datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _candidate_supports_date_window(candidates: list[tuple[str, dict]]) -> bool:
    return candidates_support_date_window(candidates)


def _clone_candidates_with_date_window(
    candidates: list[tuple[str, dict]],
    *,
    from_date,
    to_date,
) -> list[tuple[str, dict]]:
    return with_date_window(candidates, from_date=from_date, to_date=to_date)


def _run_with_retries(fetch_fn, *, max_attempts: int, base_delay_s: float) -> tuple[Any, int]:
    return run_with_retries(fetch_fn, max_attempts=max_attempts, base_delay_s=base_delay_s)


def _section_coverage_ratio(symbol_obj: Symbol, section_key: str, target_start, today) -> float:
    ranges = dict(symbol_obj.historical_date_ranges or {})
    section_range = ranges.get(section_key) if isinstance(ranges, dict) else None
    if not isinstance(section_range, dict):
        return 0.0
    min_date = _parse_iso_date(section_range.get("min_date"))
    max_date = _parse_iso_date(section_range.get("max_date"))
    count = int(section_range.get("count") or 0)
    if min_date is None or max_date is None or count <= 0:
        return 0.0
    window_days = max(1, (today - target_start).days + 1)
    covered_start = max(min_date, target_start)
    covered_end = min(max_date, today)
    if covered_end < covered_start:
        return 0.0
    covered_days = (covered_end - covered_start).days + 1
    ratio = float(covered_days) / float(window_days)
    return max(0.0, min(1.0, ratio))


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
    if fallback_dates:
        return max(fallback_dates)
    return None


def _historical_section_fetched_recently(
    symbol_obj: Symbol,
    section_key: str,
    *,
    target_end,
    threshold_days: int,
    state: SymbolSectionState | None = None,
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
        # Backfill only the missing head window when partial history exists.
        fetch_mode = "head"
        fetch_ranges = [(target_start, min_date - timedelta(days=1))]

    return fetch_mode, fetch_ranges


# historical_symbol_refresh_needed and _refresh_all_symbol_sections (and the
# supporting _historical_section_* decision logic) have been moved to
# fmp/refresh.py as the canonical owner of refresh planning/orchestration.
# Views that need them for UI can import the public versions from .refresh if required.


def _serialize_universe_download_job(job: UniverseDownloadJob) -> dict[str, Any]:
    total = max(1, int(job.total or 0))
    progress_pct = int((int(job.completed or 0) / total) * 100) if total else 0
    progress_pct = max(0, min(100, progress_pct))
    return {
        "job_id": str(job.pk),
        "status": job.status,
        "total": int(job.total or 0),
        "completed": int(job.completed or 0),
        "success_count": int(job.success_count or 0),
        "failed_count": int(job.failed_count or 0),
        "progress_pct": progress_pct,
        "current_symbol": str(job.current_symbol or ""),
        "errors": list(job.errors or []),
        "metrics": dict(job.metrics or {}),
        "celery_task_id": str(job.celery_task_id or ""),
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


@require_GET
def universe_screener_download_status(request: HttpRequest, job_id: str) -> JsonResponse:
    job = UniverseDownloadJob.objects.filter(pk=str(job_id).strip()).first()
    if job is None:
        raise Http404("Download job not found.")
    return JsonResponse(_serialize_universe_download_job(job))


@require_POST
def universe_screener_download_start(request: HttpRequest) -> JsonResponse:
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        return JsonResponse({"error": "Missing FMP_API_KEY in environment/.env."}, status=400)

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}
    symbols_raw = payload.get("symbols") if isinstance(payload, dict) else []
    if not isinstance(symbols_raw, list):
        symbols_raw = []
    symbols: list[str] = []
    seen: set[str] = set()
    for item in symbols_raw:
        code = str(item).strip().upper()
        if not code or code in seen:
            continue
        seen.add(code)
        symbols.append(code)

    if not symbols:
        return JsonResponse({"error": "No symbols were provided."}, status=400)

    if len(symbols) > 5000:
        return JsonResponse({"error": "Too many symbols requested (max 5000)."}, status=400)

    job = UniverseDownloadJob.objects.create(
        status=UniverseDownloadJob.STATUS_PENDING,
        symbols=symbols,
        total=len(symbols),
        completed=0,
        success_count=0,
        failed_count=0,
        current_symbol="",
        errors=[],
        metrics={},
    )

    try:
        from .tasks import run_universe_download_job_task
    except Exception as exc:
        job.status = UniverseDownloadJob.STATUS_FAILED
        job.errors = [f"Celery task module unavailable: {exc}"]
        job.finished_at = timezone.now()
        job.save(update_fields=["status", "errors", "finished_at", "updated_at"])
        return JsonResponse(
            {"error": "Celery is not available. Install/start Celery worker and broker."},
            status=503,
        )

    try:
        result = run_universe_download_job_task.delay(str(job.pk), api_key)
        job.celery_task_id = str(getattr(result, "id", "") or "")
        job.save(update_fields=["celery_task_id", "updated_at"])
    except Exception as exc:
        job.status = UniverseDownloadJob.STATUS_FAILED
        job.errors = [f"Failed to enqueue Celery task: {exc}"]
        job.finished_at = timezone.now()
        job.save(update_fields=["status", "errors", "finished_at", "updated_at"])
        return JsonResponse({"error": f"Could not enqueue job: {exc}"}, status=503)

    return JsonResponse(
        {
            "job_id": str(job.pk),
            "status": job.status,
            "total": len(symbols),
            "started_at": job.created_at.isoformat() if job.created_at else None,
        }
    )


@require_GET
def universe_screener(request: HttpRequest) -> JsonResponse:
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        return JsonResponse(
            {
                "error": "Missing FMP_API_KEY in environment/.env.",
            },
            status=400,
        )

    try:
        symbols, records = screen_companies_fmp(
            api_key=api_key,
            limit=_parse_int(request.GET.get("limit"), "limit", 10000),
            marketCapMoreThan=_market_cap_millions_to_dollars(
                _parse_float(request.GET.get("marketCapMoreThan"), "marketCapMoreThan")
            ),
            marketCapLowerThan=_market_cap_millions_to_dollars(
                _parse_float(request.GET.get("marketCapLowerThan"), "marketCapLowerThan")
            ),
            sector=request.GET.get("sector") or None,
            industry=request.GET.get("industry") or None,
            betaMoreThan=_parse_float(request.GET.get("betaMoreThan"), "betaMoreThan"),
            betaLowerThan=_parse_float(request.GET.get("betaLowerThan"), "betaLowerThan"),
            priceMoreThan=_parse_float(request.GET.get("priceMoreThan"), "priceMoreThan"),
            priceLowerThan=_parse_float(request.GET.get("priceLowerThan"), "priceLowerThan"),
            dividendMoreThan=_parse_float(request.GET.get("dividendMoreThan"), "dividendMoreThan"),
            dividendLowerThan=_parse_float(request.GET.get("dividendLowerThan"), "dividendLowerThan"),
            volumeMoreThan=_volume_millions_to_units(
                _parse_float(request.GET.get("volumeMoreThan"), "volumeMoreThan")
            ),
            volumeLowerThan=_volume_millions_to_units(
                _parse_float(request.GET.get("volumeLowerThan"), "volumeLowerThan")
            ),
            exchange=request.GET.get("exchange") or None,
            country=request.GET.get("country") or None,
            isEtf=_parse_bool(request.GET.get("isEtf")),
            isFund=_parse_bool(request.GET.get("isFund")),
            isActivelyTrading=_parse_bool(request.GET.get("isActivelyTrading")),
            includeAllShareClasses=_parse_bool(request.GET.get("includeAllShareClasses")),
        )
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=502)

    saved_count = _save_symbols(records)

    return JsonResponse(
        {
            "count": len(symbols),
            "saved_count": saved_count,
            "symbols": list(symbols),
            "records": records,
        }
    )


@xframe_options_exempt
def universe_screener_form(request: HttpRequest):
    api_key = os.getenv("FMP_API_KEY")
    dynamic_choices = _fetch_filter_choices(api_key)
    records: list[dict] = []
    columns: list[str] = []
    header_labels: list[str] = []
    symbol_col_index = -1
    display_rows: list[list] = []
    error: str | None = None
    data_source_label = "FMP API"

    if request.method == "POST":
        form = UniverseScreenerForm(request.POST, **dynamic_choices)
        if form.is_valid():
            data = form.cleaned_data
            selected_exchanges = data.get("exchange") or []
            data_source = str(data.get("data_source") or "db").strip().lower()
            data_source_label = "Django DB" if data_source == "db" else "FMP API"
            exchange_param = ",".join(selected_exchanges) if selected_exchanges else None
            try:
                if data_source == "db":
                    _, records = _screen_companies_db(
                        limit=data.get("limit") or 10000,
                        marketCapMoreThan=_market_cap_millions_to_dollars(data.get("marketCapMoreThan")),
                        marketCapLowerThan=_market_cap_millions_to_dollars(data.get("marketCapLowerThan")),
                        sector=data.get("sector") or None,
                        industry=data.get("industry") or None,
                        betaMoreThan=data.get("betaMoreThan"),
                        betaLowerThan=data.get("betaLowerThan"),
                        priceMoreThan=data.get("priceMoreThan"),
                        priceLowerThan=data.get("priceLowerThan"),
                        dividendMoreThan=data.get("dividendMoreThan"),
                        dividendLowerThan=data.get("dividendLowerThan"),
                        volumeMoreThan=_volume_millions_to_units(data.get("volumeMoreThan")),
                        volumeLowerThan=_volume_millions_to_units(data.get("volumeLowerThan")),
                        exchange_values=list(selected_exchanges),
                        country=data.get("country") or None,
                        isEtf=_parse_bool(data.get("isEtf")),
                        isFund=_parse_bool(data.get("isFund")),
                        isActivelyTrading=_parse_bool(data.get("isActivelyTrading")),
                    )
                else:
                    if not api_key:
                        raise ValueError("Missing FMP_API_KEY in environment/.env.")
                    _, records = screen_companies_fmp(
                        api_key=api_key,
                        limit=data.get("limit") or 10000,
                        marketCapMoreThan=_market_cap_millions_to_dollars(data.get("marketCapMoreThan")),
                        marketCapLowerThan=_market_cap_millions_to_dollars(data.get("marketCapLowerThan")),
                        sector=data.get("sector") or None,
                        industry=data.get("industry") or None,
                        betaMoreThan=data.get("betaMoreThan"),
                        betaLowerThan=data.get("betaLowerThan"),
                        priceMoreThan=data.get("priceMoreThan"),
                        priceLowerThan=data.get("priceLowerThan"),
                        dividendMoreThan=data.get("dividendMoreThan"),
                        dividendLowerThan=data.get("dividendLowerThan"),
                        volumeMoreThan=_volume_millions_to_units(data.get("volumeMoreThan")),
                        volumeLowerThan=_volume_millions_to_units(data.get("volumeLowerThan")),
                        exchange=exchange_param,
                        country=data.get("country") or None,
                        isEtf=_parse_bool(data.get("isEtf")),
                        isFund=_parse_bool(data.get("isFund")),
                        isActivelyTrading=_parse_bool(data.get("isActivelyTrading")),
                        includeAllShareClasses=_parse_bool(data.get("includeAllShareClasses")),
                    )
                _save_symbols(records)
                selected_symbols: list[str] = []
                seen_symbols: set[str] = set()
                for row in records:
                    symbol_value = str(row.get("symbol") or "").strip().upper()
                    if not symbol_value or symbol_value in seen_symbols:
                        continue
                    seen_symbols.add(symbol_value)
                    selected_symbols.append(symbol_value)
                request.session["universe_screener_symbols"] = selected_symbols
                try:
                    workflow_filters = {
                        "data_source": str(data.get("data_source") or "api"),
                        "limit": data.get("limit") or 10000,
                        "marketCapMoreThan": data.get("marketCapMoreThan"),
                        "marketCapLowerThan": data.get("marketCapLowerThan"),
                        "sector": data.get("sector") or "",
                        "industry": data.get("industry") or "",
                        "betaMoreThan": data.get("betaMoreThan"),
                        "betaLowerThan": data.get("betaLowerThan"),
                        "priceMoreThan": data.get("priceMoreThan"),
                        "priceLowerThan": data.get("priceLowerThan"),
                        "dividendMoreThan": data.get("dividendMoreThan"),
                        "dividendLowerThan": data.get("dividendLowerThan"),
                        "volumeMoreThan": data.get("volumeMoreThan"),
                        "volumeLowerThan": data.get("volumeLowerThan"),
                        "exchange": list(selected_exchanges),
                        "country": data.get("country") or "",
                        "isEtf": data.get("isEtf") or "",
                        "isFund": data.get("isFund") or "",
                        "isActivelyTrading": data.get("isActivelyTrading") or "",
                        "includeAllShareClasses": data.get("includeAllShareClasses") or "",
                    }
                    WorkflowState.objects.update_or_create(
                        key="default",
                        defaults={
                            "universe_symbols": selected_symbols,
                            "universe_filters": workflow_filters,
                        },
                    )
                except Exception:
                    pass
                if records:
                    columns = _collect_columns(records)
                    header_labels = [_prettify_header(col) for col in columns]
                    symbol_col_index = columns.index("symbol") if "symbol" in columns else -1
                    display_rows = [[_format_cell_for_column(col, row.get(col)) for col in columns] for row in records]
            except Exception as exc:
                error = str(exc)
    else:
        form = UniverseScreenerForm(
            initial={
                "data_source": "db",
                "limit": 10000,
                "marketCapMoreThan": 5000,
                "country": _default_country_value(dynamic_choices.get("country_choices")),
                "exchange": _default_us_exchange_values(dynamic_choices.get("exchange_choices")),
                "isFund": "false",
                "includeAllShareClasses": "false",
            },
            **dynamic_choices,
        )

    return render(
        request,
        "fmp/universe_screener_form.html",
        {
            "form": form,
            "records": records,
            "columns": columns,
            "header_labels": header_labels,
            "symbol_col_index": symbol_col_index,
            "display_rows": display_rows,
            "count": len(records),
            "data_source_label": data_source_label,
            "error": error,
        },
    )


def _render_macro_endpoint_form(
    request: HttpRequest,
    *,
    form,
    page_title: str,
    submit_label: str,
    series_summaries: list[dict],
    count: int,
    points_count: int,
    data_source: str,
    error: str | None,
):
    return render(
        request,
        "fmp/macro_form.html",
        {
            "form": form,
            "page_title": page_title,
            "submit_label": submit_label,
            "series_summaries": series_summaries,
            "count": count,
            "points_count": points_count,
            "data_source": data_source,
            "error": error,
        },
    )


def economic_indicators_form(request: HttpRequest):
    series_summaries: list[dict] = []
    error: str | None = None
    data_source = ""
    points_count = 0

    if request.method == "POST":
        form = EconomicIndicatorsForm(request.POST)
        if form.is_valid():
            api_key = os.getenv("FMP_API_KEY")
            if not api_key:
                error = "Missing FMP_API_KEY in environment/.env."
            else:
                data = form.cleaned_data
                series = tuple(data.get("economic_series") or [])
                try:
                    requested_codes = list(dict.fromkeys(series))
                    df_sparse = pd.DataFrame()
                    if requested_codes:
                        df_sparse = _load_series_dataframe_from_db(
                            requested_codes,
                            data["start_date"],
                            data["end_date"],
                            series_model=EconomicIndicatorSeries,
                            observation_model=EconomicIndicatorObservation,
                            threshold_days=30,
                        )
                    if df_sparse.empty:
                        df_sparse = _fetch_economic_indicators_from_api(
                            api_key=api_key,
                            start_date=data["start_date"].isoformat(),
                            end_date=data["end_date"].isoformat(),
                            series=series,
                        )
                        _store_series_dataframe(
                            df_sparse,
                            series_model=EconomicIndicatorSeries,
                            observation_model=EconomicIndicatorObservation,
                        )
                        data_source = "FMP API"
                    else:
                        data_source = "DB cache"
                    df = df_sparse
                    if not df.empty:
                        points_count = len(df)
                        df = df.sort_index(ascending=False)
                        series_summaries = _build_series_summaries(
                            list(df.columns),
                            series_model=EconomicIndicatorSeries,
                        )
                except Exception as exc:
                    error = str(exc)
    else:
        form = EconomicIndicatorsForm()

    return _render_macro_endpoint_form(
        request,
        form=form,
        page_title="Economic Indicators",
        submit_label="Fetch Economic Indicators",
        series_summaries=series_summaries,
        count=len(series_summaries),
        points_count=points_count,
        data_source=data_source,
        error=error,
    )


def treasury_rates_form(request: HttpRequest):
    series_summaries: list[dict] = []
    error: str | None = None
    data_source = ""
    points_count = 0

    if request.method == "POST":
        form = TreasuryRatesForm(request.POST)
        if form.is_valid():
            api_key = os.getenv("FMP_API_KEY")
            if not api_key:
                error = "Missing FMP_API_KEY in environment/.env."
            else:
                data = form.cleaned_data
                requested_codes = list(TreasuryRateSeries.objects.order_by("code").values_list("code", flat=True))
                try:
                    df_sparse = pd.DataFrame()
                    if requested_codes:
                        df_sparse = _load_series_dataframe_from_db(
                            requested_codes,
                            data["start_date"],
                            data["end_date"],
                            series_model=TreasuryRateSeries,
                            observation_model=TreasuryRateObservation,
                            threshold_days=1,
                        )
                    if df_sparse.empty:
                        df_sparse = _fetch_treasury_rates_from_api(
                            api_key=api_key,
                            start_date=data["start_date"].isoformat(),
                            end_date=data["end_date"].isoformat(),
                        )
                        _store_series_dataframe(
                            df_sparse,
                            series_model=TreasuryRateSeries,
                            observation_model=TreasuryRateObservation,
                        )
                        data_source = "FMP API"
                    else:
                        data_source = "DB cache"
                    if not df_sparse.empty:
                        points_count = len(df_sparse)
                        series_summaries = _build_series_summaries(
                            list(df_sparse.columns),
                            series_model=TreasuryRateSeries,
                        )
                except Exception as exc:
                    error = str(exc)
    else:
        form = TreasuryRatesForm()

    return _render_macro_endpoint_form(
        request,
        form=form,
        page_title="Treasury Rates",
        submit_label="Fetch Treasury Rates",
        series_summaries=series_summaries,
        count=len(series_summaries),
        points_count=points_count,
        data_source=data_source,
        error=error,
    )


def macro_series_detail(request: HttpRequest, code: str):
    series_obj = EconomicIndicatorSeries.objects.filter(code=code).first()
    observation_model = EconomicIndicatorObservation
    back_href = reverse("economic-indicators-form")
    back_label = "Back to Economic Indicators"
    category_label = "economic"
    if series_obj is None:
        series_obj = TreasuryRateSeries.objects.filter(code=code).first()
        observation_model = TreasuryRateObservation
        back_href = reverse("treasury-rates-form")
        back_label = "Back to Treasury Rates"
        category_label = "treasury"
    if series_obj is None:
        raise Http404()
    obs_qs = (
        observation_model.objects.filter(series=series_obj)
        .order_by("observation_date")
        .only("observation_date", "value")
    )
    labels: list[str] = []
    values: list[float | None] = []
    display_rows: list[list] = []
    for obs in obs_qs.iterator():
        date_str = obs.observation_date.isoformat()
        labels.append(date_str)
        values.append(obs.value if obs.value is None else float(obs.value))
        display_rows.append([date_str, _format_cell(obs.value)])

    return render(
        request,
        "fmp/macro_series_detail.html",
        {
            "series_obj": series_obj,
            "category_label": category_label,
            "back_href": back_href,
            "back_label": back_label,
            "labels_json": json.dumps(labels),
            "values_json": json.dumps(_json_safe(values)),
            "display_rows": display_rows,
            "points_count": len(labels),
        },
    )


def symbol_chart(request: HttpRequest, symbol: str):
    from data.warehouse import _symbol_is_etf, load_warehouse_price_frame

    symbol_obj = get_object_or_404(Symbol, symbol__iexact=symbol)
    mode = str(request.GET.get("price_mode") or "adjusted").strip().lower()
    if mode not in {"adjusted", "unadjusted"}:
        mode = "adjusted"
    mode_label = "Adjusted" if mode == "adjusted" else "Unadjusted"

    labels: list[str] = []
    opens: list[float | None] = []
    highs: list[float | None] = []
    lows: list[float | None] = []
    closes: list[float | None] = []
    volumes: list[float | None] = []

    price_frame = load_warehouse_price_frame(symbol_obj.symbol, is_etf=_symbol_is_etf(symbol_obj))
    if price_frame is not None and not price_frame.empty:
        price_frame = price_frame.copy()
        price_frame.index = pd.to_datetime(price_frame.index, errors="coerce")
        price_frame = price_frame[~price_frame.index.isna()].sort_index()

    price_rows = price_frame.iterrows() if price_frame is not None and not price_frame.empty else []
    for idx, row in price_rows:
        date_str = pd.Timestamp(idx).strftime("%Y-%m-%d")
        if mode == "adjusted":
            open_v = _safe_float(row.get("adj_open"))
            high_v = _safe_float(row.get("adj_high"))
            low_v = _safe_float(row.get("adj_low"))
            close_v = _safe_float(row.get("adj_close"))
        else:
            open_v = _safe_float(row.get("open"))
            high_v = _safe_float(row.get("high"))
            low_v = _safe_float(row.get("low"))
            close_v = _safe_float(row.get("close"))
        volume_v = _safe_float(row.get("volume"))

        if close_v is None:
            continue

        # Cross-mode fallback if keys differ in payload.
        if open_v is None:
            open_v = _safe_float(row.get("adj_open")) or _safe_float(row.get("open"))
        if high_v is None:
            high_v = _safe_float(row.get("adj_high")) or _safe_float(row.get("high"))
        if low_v is None:
            low_v = _safe_float(row.get("adj_low")) or _safe_float(row.get("low"))
        if close_v is None:
            close_v = _safe_float(row.get("adj_close")) or _safe_float(row.get("close"))

        labels.append(date_str)
        opens.append(open_v)
        highs.append(high_v)
        lows.append(low_v)
        closes.append(close_v)
        volumes.append(volume_v)

    close_by_date = {d: c for d, c in zip(labels, closes) if c is not None}

    def _all_event_points() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        event_sections = [
            ("dividends", "Dividend"),
            ("splits", "Split"),
            ("earnings", "Earnings"),
            ("sec_filings", "SEC Filing"),
            ("ratings_historical", "Rating"),
            ("insider_trading", "Insider Trade"),
        ]
        for section, event_type in event_sections:
            qs = (
                SymbolSectionHistorical.objects.filter(symbol=symbol_obj, section_key=section)
                .order_by("record_date", "updated_at")
                .only("record_date", "payload")
            )
            for r in qs.iterator():
                payload = r.payload if isinstance(r.payload, dict) else {}
                d = _extract_record_date(payload) or r.record_date
                if not d:
                    continue
                date_str = d.isoformat()
                close_val = close_by_date.get(date_str)
                if close_val is None:
                    continue
                details: list[str] = []
                if section == "dividends":
                    amount = _safe_float(payload.get("dividend"))
                    if amount is None:
                        amount = _safe_float(payload.get("adjDividend"))
                    if amount is not None:
                        details.append(f"Amount: ${amount:,.4f}".rstrip("0").rstrip("."))
                    frequency = payload.get("frequency")
                    if frequency:
                        details.append(f"Frequency: {frequency}")
                elif section == "splits":
                    ratio = payload.get("splitRatio")
                    numerator = payload.get("numerator")
                    denominator = payload.get("denominator")
                    if ratio:
                        details.append(f"Ratio: {ratio}")
                    elif numerator and denominator:
                        details.append(f"Ratio: {numerator}:{denominator}")
                elif section == "earnings":
                    eps = _safe_float(payload.get("eps") or payload.get("epsActual"))
                    if eps is not None:
                        details.append(f"EPS: {eps:,.4f}".rstrip("0").rstrip("."))
                    revenue = _safe_float(payload.get("revenue") or payload.get("revenueActual"))
                    if revenue is not None:
                        details.append(f"Revenue: {_abbrev_number(revenue)}")
                    period = payload.get("period")
                    if period:
                        details.append(f"Period: {period}")
                elif section == "sec_filings":
                    form_type = payload.get("type") or payload.get("formType") or payload.get("form")
                    if form_type:
                        details.append(f"Form: {form_type}")
                    title = payload.get("finalLink") or payload.get("link") or payload.get("title")
                    if title:
                        details.append(f"Ref: {str(title)[:120]}")
                elif section == "ratings_historical":
                    new_rating = payload.get("newGrade") or payload.get("rating")
                    prev_rating = payload.get("previousGrade")
                    analyst = payload.get("gradingCompany") or payload.get("analystCompany")
                    if new_rating:
                        details.append(f"New: {new_rating}")
                    if prev_rating:
                        details.append(f"Prev: {prev_rating}")
                    if analyst:
                        details.append(f"Firm: {analyst}")
                elif section == "insider_trading":
                    person = payload.get("reportingName") or payload.get("name")
                    txn_type = payload.get("transactionType")
                    securities = payload.get("securitiesTransacted")
                    if person:
                        details.append(f"Person: {person}")
                    if txn_type:
                        details.append(f"Type: {txn_type}")
                    if securities is not None:
                        try:
                            details.append(f"Qty: {int(float(securities)):,}")
                        except Exception:
                            details.append(f"Qty: {securities}")
                out.append({"x": date_str, "y": close_val, "type": event_type, "details": details})
        return out

    events = _all_event_points()

    return render(
        request,
        "fmp/symbol_chart.html",
        {
            "symbol_obj": symbol_obj,
            "labels_json": json.dumps(labels),
            "opens_json": json.dumps(opens),
            "highs_json": json.dumps(highs),
            "lows_json": json.dumps(lows),
            "closes_json": json.dumps(closes),
            "volumes_json": json.dumps(volumes),
            "points_count": len(labels),
            "price_mode": mode,
            "mode_label": mode_label,
            "events_json": json.dumps(events),
            "events_count": len(events),
        },
    )


def symbol_detail(request: HttpRequest, symbol: str):
    symbol_obj = get_object_or_404(Symbol, symbol__iexact=symbol)
    force_refresh_section = ""
    force_refresh_all_historical = False
    if request.method == "POST":
        force_refresh_section = str(request.POST.get("force_refresh_section") or "").strip()
        force_refresh_all_historical = str(request.POST.get("force_refresh_all_historical") or "").strip() in {"1", "true", "True"}
    payload_pretty = json.dumps(symbol_obj.payload or {}, indent=2, sort_keys=True)
    price_header_labels: list[str] = []
    price_rows: list[list] = []
    price_error: str | None = None
    price_unadj_header_labels: list[str] = []
    price_unadj_rows: list[list] = []
    price_unadj_error: str | None = None
    metrics_header_labels: list[str] = []
    metrics_rows: list[list] = []
    ratios_header_labels: list[str] = []
    ratios_rows: list[list] = []
    extra_sections: list[dict] = []
    data_error: str | None = None
    price_source_label = "quant-warehouse adjusted price history"
    historical_ranges = dict(symbol_obj.historical_date_ranges or {})

    api_key = os.getenv("FMP_API_KEY")
    section_defs = get_symbol_endpoint_definitions(symbol_obj)

    # Trust cached coverage on normal page loads; syncing every request is expensive.
    historical_section_keys = [s.key for s in section_defs if s.kind == "historical"]
    if request.method == "POST" or not historical_ranges:
        _sync_symbol_historical_ranges_from_db(symbol_obj, historical_section_keys)
        symbol_obj.refresh_from_db(fields=["historical_date_ranges"])
        historical_ranges = dict(symbol_obj.historical_date_ranges or {})

    client = FMPClient(api_key=api_key, timeout_s=30.0, max_retries=2) if api_key else None
    if not api_key:
        data_error = "Missing FMP_API_KEY in environment/.env."

    cached_states = _load_cached_section_states(symbol_obj)
    refresh_requested = bool(force_refresh_all_historical or force_refresh_section)
    refreshed_historical = False
    section_errors: dict[str, str] = {}

    for section in section_defs:
        section_key = section.key
        title = section.title
        kind = section.kind
        threshold_days = int(section.threshold_days)
        max_rows = min(int(section.max_rows), 10) if kind == "historical" else int(section.max_rows)
        candidates = section.candidates
        filter_symbol = bool(section.filter_symbol)
        error_msg = None
        state = cached_states.get(section_key)

        if refresh_requested:
            should_refresh = bool(force_refresh_all_historical or force_refresh_section == section_key)
        elif kind == "historical":
            section_range = historical_ranges.get(section_key) if isinstance(historical_ranges, dict) else None
            should_refresh = not section_range or not section_range.get("count")
        else:
            should_refresh = not state or not state.last_fetched_at
        if should_refresh and client is not None:
            try:
                if kind == "historical":
                    records = fetch_historical_records(client, section)
                else:
                    raw = _fetch_first_success(client, candidates)
                    records = _to_records(raw)
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
                            peer_records = _to_records(raw)
                    records = peer_records

                if filter_symbol:
                    records = _filter_records_for_symbol(records, symbol_obj.symbol)

                if kind == "snapshot":
                    _save_snapshot_section(symbol_obj, section_key, raw)
                else:
                    _save_historical_section(
                        symbol_obj,
                        section_key,
                        records,
                        dedupe_by_date=bool(section.dedupe_by_date),
                    )
                    _update_symbol_historical_range(symbol_obj, section_key)
                    refreshed_historical = True
                _mark_section_fetched(symbol_obj, section_key, kind)
                cached_states[section_key] = SymbolSectionState(
                    symbol=symbol_obj,
                    section_key=section_key,
                    kind=kind,
                    last_fetched_at=timezone.now(),
                )
            except Exception as exc:
                if "HTTP 404" not in str(exc):
                    error_msg = str(exc)
                    section_errors[section_key] = error_msg

    snapshot_payloads = _load_cached_snapshots(symbol_obj)
    history_rows_by_section = _load_cached_history_rows(
        symbol_obj,
        {section.key: min(int(section.max_rows), 10) for section in section_defs if section.kind == "historical"},
    )

    for section in section_defs:
        section_key = section.key
        title = section.title
        kind = section.kind
        error_msg = section_errors.get(section_key)

        if kind == "snapshot":
            payload = snapshot_payloads.get(section_key)
            records = _to_records(payload)
            _, header_labels, rows = _records_to_table(records, max_rows=1)
            first_row = records[0] if records and isinstance(records[0], dict) else {}
            snapshot_items = [
                {
                    "label": _prettify_header(str(key)),
                    "value": _format_cell_for_column(str(key), _normalize_cell(value)),
                }
                for key, value in first_row.items()
            ]
        else:
            records = history_rows_by_section.get(section_key, [])
            _, header_labels, rows = _records_to_table(records, max_rows=min(int(section.max_rows), 10))
            snapshot_items = []

        if section_key in {"prices_div_adj", "prices_unadjusted"}:
            from data.warehouse import _symbol_is_etf, load_warehouse_price_frame

            price_frame = load_warehouse_price_frame(symbol_obj.symbol, is_etf=_symbol_is_etf(symbol_obj))
            if price_frame is not None and not price_frame.empty:
                frame = price_frame.copy()
                frame.index = pd.to_datetime(frame.index, errors="coerce")
                frame = frame[~frame.index.isna()].sort_index()
                records = []
                for idx, row in frame.iterrows():
                    date_value = pd.Timestamp(idx).strftime("%Y-%m-%d")
                    if section_key == "prices_div_adj":
                        records.append(
                            {
                                "date": date_value,
                                "adj_open": row.get("adj_open"),
                                "adj_high": row.get("adj_high"),
                                "adj_low": row.get("adj_low"),
                                "adj_close": row.get("adj_close"),
                                "volume": row.get("volume"),
                            }
                        )
                    else:
                        records.append(
                            {
                                "date": date_value,
                                "open": row.get("open"),
                                "high": row.get("high"),
                                "low": row.get("low"),
                                "close": row.get("close"),
                                "volume": row.get("volume"),
                            }
                        )
                _, header_labels, rows = _records_to_table(records, max_rows=min(int(section.max_rows), 10))
            if section_key == "prices_div_adj":
                price_header_labels, price_rows = header_labels, rows
                price_error = error_msg
            else:
                price_unadj_header_labels, price_unadj_rows = header_labels, rows
                price_unadj_error = error_msg
            continue
        if section_key == "key_metrics":
            metrics_header_labels, metrics_rows = header_labels, rows
            continue
        if section_key == "ratios":
            ratios_header_labels, ratios_rows = header_labels, rows
            continue

        extra_sections.append(
            {
                "title": title,
                "section_key": section_key,
                "kind": kind,
                "header_labels": header_labels,
                "rows": rows,
                "snapshot_items": snapshot_items,
                "error": error_msg,
                "date_range": historical_ranges.get(section_key),
            }
        )

    if refreshed_historical:
        symbol_obj.refresh_from_db(fields=["historical_date_ranges"])
        historical_ranges = dict(symbol_obj.historical_date_ranges or {})
    historical_coverage_rows = [
        {
            "section_key": k,
            "section_label": _prettify_header(k),
            "min_date": (v or {}).get("min_date"),
            "max_date": (v or {}).get("max_date"),
            "count": (v or {}).get("count") or 0,
        }
        for k, v in sorted(historical_ranges.items(), key=lambda kv: kv[0])
        if k in historical_section_keys
    ]

    return render(
        request,
        "fmp/symbol_detail.html",
        {
            "symbol_obj": symbol_obj,
            "payload_pretty": payload_pretty,
            "price_header_labels": price_header_labels,
            "price_rows": price_rows,
            "price_error": price_error,
            "price_unadj_header_labels": price_unadj_header_labels,
            "price_unadj_rows": price_unadj_rows,
            "price_unadj_error": price_unadj_error,
            "metrics_header_labels": metrics_header_labels,
            "metrics_rows": metrics_rows,
            "ratios_header_labels": ratios_header_labels,
            "ratios_rows": ratios_rows,
            "extra_sections": extra_sections,
            "data_error": data_error,
            "price_source_label": price_source_label,
            "price_date_range": historical_ranges.get("prices_div_adj"),
            "price_unadj_date_range": historical_ranges.get("prices_unadjusted"),
            "metrics_date_range": historical_ranges.get("key_metrics"),
            "ratios_date_range": historical_ranges.get("ratios"),
            "historical_coverage_rows": historical_coverage_rows,
        },
    )
