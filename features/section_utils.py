from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from django.db import connection

from fmp.models import Symbol, SymbolSectionHistorical
from data import broadcast_asof_to_target_index


@dataclass(frozen=True)
class BuiltFeatureSet:
    df: pd.DataFrame
    feature_cols: list[str]


_SECTION_RECORD_CACHE: dict[tuple[int, str], list[tuple[Any, dict[str, Any]]]] = {}
_DATE_KEY_CANDIDATES = (
    "accepteddate",
    "filingdate",
    "fillingdate",
    "transactiondate",
    "date",
    "calendar_date",
)
_EXCLUDED_PAYLOAD_FIELDS = {
    "accepteddate",
    "filingdate",
    "fillingdate",
    "transactiondate",
    "date",
    "calendar_date",
    "symbol",
    "link",
    "finallink",
    "cik",
    "reportedcurrency",
    "currency",
    "fiscalyear",
    "calendaryear",
    "calendar_year",
    "period",
}


@lru_cache(maxsize=1)
def legacy_symbol_section_table_exists() -> bool:
    return SymbolSectionHistorical._meta.db_table in connection.introspection.table_names()


def prime_section_record_cache(symbols: Sequence[Symbol], section_keys: Sequence[str]) -> None:
    from data.warehouse import read_fundamentals_from_warehouse

    if read_fundamentals_from_warehouse():
        return
    if not legacy_symbol_section_table_exists():
        return

    symbol_rows = [symbol for symbol in list(symbols or []) if getattr(symbol, "id", None)]
    normalized_section_keys = [str(section_key).strip() for section_key in list(section_keys or []) if str(section_key).strip()]
    if not symbol_rows or not normalized_section_keys:
        return

    symbol_ids = [int(symbol.id) for symbol in symbol_rows]
    wanted_keys = {
        (int(symbol.id), str(section_key))
        for symbol in symbol_rows
        for section_key in normalized_section_keys
    }
    missing = [item for item in wanted_keys if item not in _SECTION_RECORD_CACHE]
    if not missing:
        return

    grouped: dict[tuple[int, str], list[tuple[Any, dict[str, Any]]]] = {key: [] for key in wanted_keys}
    qs = (
        SymbolSectionHistorical.objects.filter(symbol_id__in=symbol_ids, section_key__in=normalized_section_keys)
        .only("symbol_id", "section_key", "record_date", "payload")
        .order_by("symbol_id", "section_key", "record_date", "updated_at")
    )
    for item in qs.iterator():
        grouped.setdefault((int(item.symbol_id), str(item.section_key)), []).append(
            (item.record_date, item.payload if isinstance(item.payload, dict) else {})
        )
    for key in wanted_keys:
        _SECTION_RECORD_CACHE[key] = list(grouped.get(key) or [])


def clear_section_record_cache() -> None:
    _SECTION_RECORD_CACHE.clear()


def load_section_payload(
    symbol_obj: Symbol,
    section_key: str,
    *,
    prefix: str,
    keep_fields: Iterable[str] | None = None,
    filing_lag_days: int = 0,
) -> pd.DataFrame:
    from data.warehouse import (
        read_fundamentals_from_warehouse,
        warehouse_section_for_django,
        warehouse_section_to_indexed_frame,
    )

    if read_fundamentals_from_warehouse() and warehouse_section_for_django(section_key) is not None:
        frame = warehouse_section_to_indexed_frame(
            symbol_obj.symbol,
            section_key,
            prefix=prefix,
            keep_fields=keep_fields,
            filing_lag_days=filing_lag_days,
        )
        if not frame.empty:
            return frame

    keep = {str(v).lower().strip() for v in keep_fields or []}
    rows: list[dict[str, Any]] = []
    cached_rows = _SECTION_RECORD_CACHE.get((int(symbol_obj.id), str(section_key)))
    if cached_rows is None:
        if not legacy_symbol_section_table_exists():
            return pd.DataFrame()
        qs = (
            SymbolSectionHistorical.objects.filter(symbol=symbol_obj, section_key=section_key)
            .order_by("record_date", "updated_at")
            .only("record_date", "payload")
        )
        iterator = ((item.record_date, item.payload if isinstance(item.payload, dict) else {}) for item in qs.iterator())
    else:
        iterator = iter(cached_rows)
    for record_date, payload in iterator:
        row = payload_to_row(
            payload=payload,
            symbol=symbol_obj.symbol,
            prefix=prefix,
            keep_fields=keep if keep else None,
            record_date=record_date,
            filing_lag_days=filing_lag_days,
        )
        if row:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col in {"date", "symbol"}:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            df[col] = converted
    df = df.sort_values(["date", "symbol"]).drop_duplicates(subset=["date", "symbol"], keep="last")
    return df.set_index(["date", "symbol"]).sort_index()


def load_combined_sparse_sections(
    symbol_obj: Symbol,
    section_to_fields: dict[str, Sequence[str]],
    filing_lag_days: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for section_key, fields in section_to_fields.items():
        frame = load_section_payload(
            symbol_obj,
            section_key,
            prefix=section_prefix(section_key),
            keep_fields=fields,
            filing_lag_days=filing_lag_days,
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.join(frame, how="outer")
    return merged


def payload_to_row(
    *,
    payload: dict[str, Any],
    symbol: str,
    prefix: str,
    keep_fields: set[str] | None,
    record_date,
    filing_lag_days: int,
) -> dict[str, Any]:
    raw = {str(k).lower().strip(): v for k, v in payload.items()}
    date_val = None
    for key in _DATE_KEY_CANDIDATES:
        if raw.get(key):
            date_val = raw.get(key)
            break
    ts = _normalize_payload_timestamp(date_val, record_date=record_date, filing_lag_days=filing_lag_days)
    if ts is None:
        return {}
    row: dict[str, Any] = {
        "date": ts,
        "symbol": str(symbol).upper(),
    }
    for key, value in raw.items():
        if key in _EXCLUDED_PAYLOAD_FIELDS:
            continue
        if keep_fields is not None and key not in keep_fields:
            continue
        row[f"{prefix}{key}"] = value
    return row


def _normalize_payload_timestamp(date_val: Any, *, record_date: Any, filing_lag_days: int) -> pd.Timestamp | None:
    candidate = date_val if date_val not in (None, "") else record_date
    if candidate in (None, ""):
        return None
    if isinstance(candidate, pd.Timestamp):
        ts = candidate
    else:
        text_value = candidate
        if not hasattr(candidate, "year"):
            text_value = str(candidate).strip()
            if not text_value:
                return None
            if len(text_value) >= 10:
                text_value = text_value[:10]
        try:
            ts = pd.Timestamp(text_value)
        except Exception:
            return None
    if pd.isna(ts):
        return None
    ts = pd.Timestamp(ts).normalize()
    if filing_lag_days:
        ts = ts + pd.Timedelta(days=filing_lag_days)
    return ts


def section_prefix(section_key: str) -> str:
    return {
        "key_metrics": "km__",
        "ratios": "rt__",
        "income_statement": "is__",
        "income_statement_growth": "isg__",
        "cash_flow": "cf__",
        "cash_flow_growth": "cfg__",
        "balance_sheet": "bs__",
        "balance_sheet_growth": "bsg__",
        "financial_growth": "fg__",
        "earnings": "earn__",
        "analyst_estimates": "ae__",
        "ratings_historical": "rating__",
        "grades_historical": "grade__",
        "insider_trading": "insider__",
        "positions_summary": "ps__",
    }.get(section_key, f"{section_key}__")


def build_passthrough_section_features(
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    *,
    section_key: str,
    prefix: str,
    filing_lag_days: int = 45,
) -> BuiltFeatureSet:
    sparse = load_section_payload(
        symbol_obj,
        section_key,
        prefix=prefix,
        keep_fields=None,
        filing_lag_days=filing_lag_days,
    )
    if sparse.empty:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    numeric_cols = [c for c in sparse.columns if c.startswith(prefix) and pd.api.types.is_numeric_dtype(sparse[c])]
    if not numeric_cols:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    daily = broadcast_sparse(sparse[numeric_cols].sort_index(), target_index)
    return BuiltFeatureSet(df=daily, feature_cols=[c for c in daily.columns if c.startswith(prefix)])


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(2, min(window, 4))).mean()
    std = series.rolling(window, min_periods=max(2, min(window, 4))).std()
    return safe_ratio(series - mean, std)


def safe_ratio(a, b):
    if a is None or b is None:
        return np.nan
    if not isinstance(a, pd.Series):
        a = pd.Series(a)
    if not isinstance(b, pd.Series):
        b = pd.Series(b, index=a.index)
    denom = pd.to_numeric(b, errors="coerce").replace(0.0, np.nan)
    numer = pd.to_numeric(a, errors="coerce")
    return numer / denom


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series | None:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return None


def daily_price_series(
    df_prices: pd.DataFrame | None,
    target_index: pd.MultiIndex,
    *,
    price_col: str = "close",
) -> pd.Series | None:
    if df_prices is None or df_prices.empty or price_col not in df_prices.columns:
        return None
    close = pd.to_numeric(df_prices[price_col], errors="coerce").sort_index()
    if isinstance(close.index, pd.MultiIndex):
        return close.reindex(target_index)
    target_date_index = target_dates(target_index)
    aligned = close.reindex(target_date_index, method="ffill")
    return pd.Series(aligned.to_numpy(), index=target_index, dtype="float64")


def add_daily_price_linked_features(
    daily: pd.DataFrame,
    target_index: pd.MultiIndex,
    *,
    df_prices: pd.DataFrame | None = None,
    market_cap: pd.Series | None = None,
    share_count_candidates: Sequence[str] = (),
    price_denominated: Sequence[tuple[Sequence[str], str]] = (),
    market_cap_denominated: Sequence[tuple[Sequence[str], str]] = (),
    negate_market_cap_sources: Sequence[str] = (),
) -> tuple[pd.DataFrame, list[str]]:
    if daily.empty:
        return daily, []
    out = daily.copy()
    close = daily_price_series(df_prices, target_index)
    if market_cap is None and close is not None and share_count_candidates:
        shares = first_existing(out, share_count_candidates)
        if shares is not None:
            market_cap = shares.reindex(out.index) * close.reindex(out.index)
    elif market_cap is not None:
        market_cap = pd.to_numeric(market_cap, errors="coerce").reindex(out.index)

    added: list[str] = []

    def _add(candidates: Sequence[str], output_col: str, denominator: pd.Series | None, *, negate: bool = False) -> None:
        if denominator is None:
            return
        source = first_existing(out, candidates)
        if source is None:
            return
        values = -source if negate else source
        linked = safe_ratio(values.reindex(out.index), denominator.reindex(out.index)).replace([np.inf, -np.inf], np.nan)
        if linked.notna().any():
            out[output_col] = linked
            added.append(output_col)

    for candidates, output_col in price_denominated:
        _add(candidates, output_col, close)

    negate_set = {str(value).strip() for value in negate_market_cap_sources}
    for candidates, output_col in market_cap_denominated:
        _add(candidates, output_col, market_cap, negate=str(output_col) in negate_set)

    return out, added


def _growth_as_percent(series: pd.Series) -> pd.Series:
    growth = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = growth.dropna().abs()
    if not valid.empty and float(valid.median()) <= 2.0:
        growth = growth * 100.0
    return growth


def add_growth_adjusted_valuation_features(
    daily: pd.DataFrame,
    *,
    valuation_frame: pd.DataFrame | None = None,
    specs: Sequence[tuple[Sequence[str], Sequence[str], str]] = (),
) -> tuple[pd.DataFrame, list[str]]:
    if daily.empty or valuation_frame is None or valuation_frame.empty:
        return daily, []
    out = daily.copy()
    valuation = valuation_frame.reindex(out.index)
    added: list[str] = []
    for growth_candidates, valuation_candidates, output_col in specs:
        growth = first_existing(out, growth_candidates)
        valuation_series = first_existing(valuation, valuation_candidates)
        if growth is None or valuation_series is None:
            continue
        growth_pct = _growth_as_percent(growth).where(lambda s: s > 0.0)
        values = safe_ratio(valuation_series.reindex(out.index), growth_pct.reindex(out.index)).replace([np.inf, -np.inf], np.nan)
        if values.notna().any():
            out[output_col] = values
            added.append(output_col)
    return out, added


def target_dates(target_index: pd.MultiIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(target_index.get_level_values("date"))).normalize()


def days_since_last_event(target_dates_index: pd.DatetimeIndex, event_dates: Sequence[Any]) -> pd.Series:
    event_index = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(event_dates)), errors="coerce").dropna()).normalize().sort_values().unique()
    if len(event_index) == 0:
        return pd.Series(np.nan, index=target_dates_index)
    last_seen = np.searchsorted(event_index, target_dates_index.values.astype("datetime64[ns]"), side="right") - 1
    out = np.full(len(target_dates_index), np.nan, dtype=float)
    valid = last_seen >= 0
    if valid.any():
        prior = event_index[last_seen[valid]]
        out[valid] = (target_dates_index[valid] - prior).days.astype(float)
    return pd.Series(out, index=target_dates_index)


def days_since_for_target(target_index: pd.MultiIndex, by_date_values: pd.Series) -> pd.Series:
    date_index = pd.DatetimeIndex(pd.to_datetime(target_index.get_level_values("date"))).normalize()
    values = by_date_values.reindex(date_index)
    return pd.Series(values.to_numpy(), index=target_index)


def broadcast_sparse(sparse_df: pd.DataFrame, target_index: pd.MultiIndex) -> pd.DataFrame:
    if sparse_df.empty:
        return pd.DataFrame(index=target_index)
    return broadcast_asof_to_target_index(sparse_df=sparse_df, target_index=target_index, on="date", by=("symbol",))


__all__ = [
    "BuiltFeatureSet",
    "broadcast_sparse",
    "build_passthrough_section_features",
    "clear_section_record_cache",
    "add_daily_price_linked_features",
    "add_growth_adjusted_valuation_features",
    "daily_price_series",
    "days_since_for_target",
    "days_since_last_event",
    "first_existing",
    "load_combined_sparse_sections",
    "load_section_payload",
    "payload_to_row",
    "prime_section_record_cache",
    "rolling_zscore",
    "safe_ratio",
    "section_prefix",
    "target_dates",
]
