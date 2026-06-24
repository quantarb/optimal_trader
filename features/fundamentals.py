from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np
import pandas as pd
from quant_warehouse.feature_engineering.fundamentals import (
    broadcast_fundamentals_to_daily as _broadcast_quant_warehouse_fundamentals_to_daily,
    fetch_fundamentals_data as _fetch_quant_warehouse_fundamentals_data,
)

from fmp.models import Symbol, SymbolSectionHistorical
from data import asof_join_pit
from features.section_utils import legacy_symbol_section_table_exists


def fetch_fundamentals_data(
    symbols: Sequence[str],
    api_key: str,
    period: str = "quarter",
    limit: int = 160,
    verbose: bool = True,
    use_filing_lag: bool = True,
    filing_lag_days: int = 45,
) -> pd.DataFrame:
    """
    Compatibility entrypoint.
    Loads sparse fundamentals from quant-warehouse, falling back to Django only
    when warehouse data is absent.
    """
    warehouse_df = _fetch_quant_warehouse_fundamentals_data(
        symbols=symbols,
        api_key=api_key,
        period=period,
        limit=limit,
        verbose=False,
        use_filing_lag=use_filing_lag,
        filing_lag_days=filing_lag_days,
    )
    if not warehouse_df.empty:
        return warehouse_df

    del api_key, period, limit
    dfs_per_symbol: List[pd.DataFrame] = []
    for sym in symbols:
        symbol_obj = Symbol.objects.filter(symbol__iexact=str(sym)).first()
        if not symbol_obj:
            continue
        df_km = _load_section(symbol_obj, "key_metrics", "km__", use_filing_lag, filing_lag_days)
        df_rt = _load_section(symbol_obj, "ratios", "rt__", use_filing_lag, filing_lag_days)
        if df_km.empty and df_rt.empty:
            continue
        merge_keys = ["date", "symbol", "period"]
        for d in (df_km, df_rt):
            if d.empty:
                continue
            for key in merge_keys:
                if key not in d.columns:
                    d[key] = np.nan
            d["symbol"] = d["symbol"].astype(str)
        if df_km.empty:
            df_merged = df_rt
        elif df_rt.empty:
            df_merged = df_km
        else:
            df_merged = pd.merge(df_km, df_rt, on=merge_keys, how="outer")
        dfs_per_symbol.append(df_merged)

    if not dfs_per_symbol:
        if verbose:
            print("[fundamentals] WARN: No DB-backed fundamentals found.")
        return pd.DataFrame()
    fund_df = pd.concat(dfs_per_symbol, ignore_index=True)
    if "date" in fund_df.columns and "symbol" in fund_df.columns:
        fund_df = fund_df.sort_values(["symbol", "date"]).set_index(["date", "symbol"])
    return _enforce_numeric_features(fund_df)


def broadcast_fundamentals_to_daily(
    fund_df: pd.DataFrame,
    target_daily_index: pd.Index,
) -> pd.DataFrame:
    return _broadcast_quant_warehouse_fundamentals_to_daily(fund_df, target_daily_index)


def _load_section(
    symbol_obj: Symbol,
    section_key: str,
    prefix: str,
    use_filing_lag: bool,
    filing_lag_days: int,
) -> pd.DataFrame:
    from data.warehouse import (
        read_fundamentals_from_warehouse,
        warehouse_section_for_django,
        warehouse_section_to_payload_rows,
    )

    if read_fundamentals_from_warehouse() and warehouse_section_for_django(section_key) is not None:
        rows = warehouse_section_to_payload_rows(
            symbol_obj.symbol,
            section_key,
            prefix=prefix,
            filing_lag_days=filing_lag_days if use_filing_lag else 0,
        )
        if rows:
            return pd.DataFrame(rows)
    if not legacy_symbol_section_table_exists():
        return pd.DataFrame()

    qs = (
        SymbolSectionHistorical.objects.filter(symbol=symbol_obj, section_key=section_key)
        .order_by("record_date", "updated_at")
        .only("record_date", "payload")
    )
    rows: list[dict[str, Any]] = []
    for item in qs.iterator():
        payload = item.payload if isinstance(item.payload, dict) else {}
        row = _clean_payload_response(payload, symbol_obj.symbol, prefix)
        if not row:
            continue
        if use_filing_lag and row.get("date") is not None:
            row["date"] = row["date"] + pd.to_timedelta(filing_lag_days, unit="D")
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _clean_payload_response(payload: dict[str, Any], symbol: str, prefix: str) -> dict[str, Any]:
    out = {str(k).lower().strip(): v for k, v in payload.items()}
    out["symbol"] = str(symbol)
    date_col = None
    for candidate in ("date", "fillingdate", "accepteddate", "calendar_date"):
        if candidate in out:
            date_col = candidate
            break
    if not date_col:
        return {}
    date_value = pd.to_datetime(out.get(date_col), errors="coerce")
    if pd.isna(date_value):
        return {}
    row: dict[str, Any] = {
        "date": pd.Timestamp(date_value),
        "symbol": str(symbol),
        "period": out.get("period"),
    }
    exclude = {
        "date",
        "symbol",
        "period",
        "fiscalyear",
        "fiscalyearended",
        "calendaryear",
        "calendar_year",
        "reportedcurrency",
        "currency",
        "link",
        "finallink",
        "accepteddate",
        "fillingdate",
        "filingdate",
        "cik",
    }
    for key, value in out.items():
        if key in exclude:
            continue
        row[f"{prefix}{key}"] = value
    return row


def _enforce_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.startswith("km__") or col.startswith("rt__"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _asof_join_fundamentals(target_df: pd.DataFrame, fund_df: pd.DataFrame) -> pd.DataFrame:
    t = target_df.reset_index() if isinstance(target_df.index, pd.MultiIndex) else target_df.copy()
    f = fund_df.reset_index() if isinstance(fund_df.index, pd.MultiIndex) else fund_df.copy()
    t["date"] = pd.to_datetime(t["date"])
    f["date"] = pd.to_datetime(f["date"])
    t["symbol"] = t["symbol"].astype(str)
    f["symbol"] = f["symbol"].astype(str)
    return asof_join_pit(left=t, right=f, on="date", by=("symbol",))
