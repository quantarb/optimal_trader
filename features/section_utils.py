from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from fmp.models import Symbol, SymbolSectionHistorical
from modules.data.pit import broadcast_asof_to_target_index


@dataclass(frozen=True)
class BuiltFeatureSet:
    df: pd.DataFrame
    feature_cols: list[str]


def load_section_payload(
    symbol_obj: Symbol,
    section_key: str,
    *,
    prefix: str,
    keep_fields: Iterable[str] | None = None,
    filing_lag_days: int = 0,
) -> pd.DataFrame:
    qs = (
        SymbolSectionHistorical.objects.filter(symbol=symbol_obj, section_key=section_key)
        .order_by("record_date", "updated_at")
        .only("record_date", "payload")
    )
    keep = {str(v).lower().strip() for v in keep_fields or []}
    rows: list[dict[str, Any]] = []
    for item in qs.iterator():
        payload = item.payload if isinstance(item.payload, dict) else {}
        row = payload_to_row(
            payload=payload,
            symbol=symbol_obj.symbol,
            prefix=prefix,
            keep_fields=keep if keep else None,
            record_date=item.record_date,
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
    for key in ("accepteddate", "filingdate", "fillingdate", "transactiondate", "date", "calendar_date"):
        if raw.get(key):
            date_val = raw.get(key)
            break
    if date_val is None and record_date is not None:
        date_val = record_date
    ts = pd.to_datetime(date_val, errors="coerce")
    if pd.isna(ts):
        return {}
    ts = pd.Timestamp(ts).normalize()
    if filing_lag_days:
        ts = ts + pd.Timedelta(days=filing_lag_days)
    row: dict[str, Any] = {
        "date": ts,
        "symbol": str(symbol).upper(),
    }
    exclude = {
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
    for key, value in raw.items():
        if key in exclude:
            continue
        if keep_fields is not None and key not in keep_fields:
            continue
        row[f"{prefix}{key}"] = value
    return row


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
