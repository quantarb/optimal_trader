from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from fmp.models import Symbol
from features.balance_sheet_features import build_balance_sheet_features
from features.balance_sheet_growth_features import build_balance_sheet_growth_features
from features.cash_flow_features import build_cash_flow_features
from features.cash_flow_growth_features import build_cash_flow_growth_features
from features.earnings_features import build_earnings_features
from features.financial_growth_features import build_financial_growth_features
from features.income_statement_features import build_income_statement_features
from features.income_statement_growth_features import build_income_statement_growth_features
from features.key_metrics_features import build_key_metrics_features
from features.ratios_features import build_ratios_features
from features.section_utils import clear_section_record_cache, prime_section_record_cache
from features.time_features import build_time_calendar_features
from django.db import close_old_connections


def _build_target_index_by_symbol(target_index: pd.MultiIndex) -> dict[str, pd.MultiIndex]:
    if target_index is None or len(target_index) == 0:
        return {}
    symbol_values = pd.Index(target_index.get_level_values("symbol")).astype(str).str.strip().str.upper().to_numpy()
    date_values = pd.DatetimeIndex(pd.to_datetime(target_index.get_level_values("date"))).normalize().to_numpy()
    order = np.argsort(symbol_values, kind="stable")
    symbol_sorted = symbol_values[order]
    date_sorted = date_values[order]
    unique_symbols, start_positions = np.unique(symbol_sorted, return_index=True)
    target_index_by_symbol: dict[str, pd.MultiIndex] = {}
    for idx, symbol in enumerate(unique_symbols):
        start = int(start_positions[idx])
        end = int(start_positions[idx + 1]) if idx + 1 < len(start_positions) else len(symbol_sorted)
        dates = pd.DatetimeIndex(date_sorted[start:end])
        target_index_by_symbol[str(symbol)] = pd.MultiIndex.from_arrays(
            [dates, np.repeat(str(symbol), len(dates))],
            names=["date", "symbol"],
        )
    return target_index_by_symbol


def _empty_symbol_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([pd.DatetimeIndex([]), []], names=["date", "symbol"])


def _price_frame_for_symbol(price_panel: pd.DataFrame, symbol: str) -> pd.DataFrame:
    code = str(symbol).strip().upper()
    if not code:
        return pd.DataFrame()
    try:
        return price_panel.xs(code, level="symbol")
    except Exception:
        return pd.DataFrame()


def _fmp_endpoint_builders(filing_lag_days: int) -> dict[str, Any]:
    return {
        "key_metrics": lambda symbol_obj, idx, px, market_cap, valuation_context: build_key_metrics_features(symbol_obj, idx, df_prices=px, filing_lag_days=filing_lag_days),
        "ratios": lambda symbol_obj, idx, px, market_cap, valuation_context: build_ratios_features(symbol_obj, idx, df_prices=px, filing_lag_days=filing_lag_days),
        "income_statement": lambda symbol_obj, idx, px, market_cap, valuation_context: build_income_statement_features(symbol_obj, idx, df_prices=px, filing_lag_days=filing_lag_days),
        "income_statement_growth": lambda symbol_obj, idx, px, market_cap, valuation_context: build_income_statement_growth_features(symbol_obj, idx, valuation_frame=valuation_context, filing_lag_days=filing_lag_days),
        "balance_sheet": lambda symbol_obj, idx, px, market_cap, valuation_context: build_balance_sheet_features(symbol_obj, idx, df_prices=px, market_cap=market_cap, filing_lag_days=filing_lag_days),
        "balance_sheet_growth": lambda symbol_obj, idx, px, market_cap, valuation_context: build_balance_sheet_growth_features(symbol_obj, idx, valuation_frame=valuation_context, filing_lag_days=filing_lag_days),
        "cash_flow": lambda symbol_obj, idx, px, market_cap, valuation_context: build_cash_flow_features(symbol_obj, idx, df_prices=px, market_cap=market_cap, filing_lag_days=filing_lag_days),
        "cash_flow_growth": lambda symbol_obj, idx, px, market_cap, valuation_context: build_cash_flow_growth_features(symbol_obj, idx, valuation_frame=valuation_context, filing_lag_days=filing_lag_days),
        "financial_growth": lambda symbol_obj, idx, px, market_cap, valuation_context: build_financial_growth_features(symbol_obj, idx, valuation_frame=valuation_context, filing_lag_days=filing_lag_days),
        "earnings": lambda symbol_obj, idx, px, market_cap, valuation_context: build_earnings_features(symbol_obj, idx),
        "time_calendar": lambda symbol_obj, idx, px, market_cap, valuation_context: build_time_calendar_features(symbol_obj, idx),
    }


def _build_fmp_endpoint_features_for_symbol(
    code: str,
    symbol_obj: Symbol,
    *,
    symbol_index_by_symbol: dict[str, pd.MultiIndex],
    price_panel: pd.DataFrame,
    endpoint_builders,
):
    close_old_connections()
    try:
        symbol_index = symbol_index_by_symbol.get(str(code).strip().upper(), _empty_symbol_index())
        if len(symbol_index) == 0:
            return code, {}, {}, "empty_index", 0.0
        symbol_prices = _price_frame_for_symbol(price_panel, code)
        symbol_market_cap = None
        valuation_context = pd.DataFrame(index=symbol_index)
        frames: dict[str, pd.DataFrame] = {}
        cols_by_endpoint: dict[str, list[str]] = {}
        start_time = time.perf_counter()
        for endpoint_name, builder in endpoint_builders.items():
            built = builder(symbol_obj, symbol_index, symbol_prices, symbol_market_cap, valuation_context)
            active_cols = [c for c in built.feature_cols if c in built.df.columns and pd.api.types.is_numeric_dtype(built.df[c])]
            if endpoint_name == "key_metrics" and "km__marketcap" in built.df.columns:
                symbol_market_cap = pd.to_numeric(built.df["km__marketcap"], errors="coerce")
            if endpoint_name in {"key_metrics", "ratios", "income_statement", "balance_sheet", "cash_flow"} and not built.df.empty:
                valuation_context = pd.concat([valuation_context, built.df], axis=1)
                if valuation_context.columns.has_duplicates:
                    valuation_context = valuation_context.loc[:, ~valuation_context.columns.duplicated(keep="last")]
            if not active_cols:
                continue
            frames[endpoint_name] = built.df[active_cols]
            cols_by_endpoint[endpoint_name] = list(active_cols)
        return code, frames, cols_by_endpoint, "ok", time.perf_counter() - start_time
    except Exception as exc:
        return code, {}, {}, f"error:{type(exc).__name__}: {exc}", 0.0
    finally:
        close_old_connections()


def build_fmp_endpoint_feature_families(
    *,
    symbols,
    target_index: pd.MultiIndex,
    price_panel: pd.DataFrame,
    filing_lag_days: int = 45,
    max_workers: int = 1,
    progress_logger=None,
):
    endpoint_builders = _fmp_endpoint_builders(int(filing_lag_days))
    normalized_symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
    symbol_objs = {
        str(obj.symbol).strip().upper(): obj
        for obj in Symbol.objects.filter(symbol__in=normalized_symbols)
    }
    symbol_index_by_symbol = _build_target_index_by_symbol(target_index)
    endpoint_frames = {name: [] for name in endpoint_builders}
    endpoint_cols = {name: [] for name in endpoint_builders}
    total = len(normalized_symbols)
    workers = max(1, int(max_workers or 1))
    errors: list[tuple[str, str]] = []

    def _record_result(result, idx):
        code, frames, cols_by_endpoint, status, elapsed = result
        if str(status).startswith("error:"):
            errors.append((code, status))
        for endpoint_name, frame in frames.items():
            endpoint_frames[endpoint_name].append(frame)
            for col in cols_by_endpoint.get(endpoint_name, []):
                if col not in endpoint_cols[endpoint_name]:
                    endpoint_cols[endpoint_name].append(col)
        if callable(progress_logger) and (idx == 1 or idx % 25 == 0 or idx == total or str(status).startswith("error:")):
            progress_logger(
                f"FMP feature build progress: {idx:,}/{total:,} symbols processed"
                f" | latest={code} | status={status} | elapsed={elapsed:.2f}s"
            )

    try:
        prime_section_record_cache(list(symbol_objs.values()), list(endpoint_builders.keys()))
        if callable(progress_logger):
            progress_logger(f"FMP feature build start | symbols={total:,} | workers={workers:,} | families={len(endpoint_builders):,}")
        if workers == 1:
            for idx, code in enumerate(normalized_symbols, start=1):
                symbol_obj = symbol_objs.get(code) or Symbol.objects.filter(symbol__iexact=code).first()
                if symbol_obj is None:
                    _record_result((code, {}, {}, "missing_symbol", 0.0), idx)
                    continue
                result = _build_fmp_endpoint_features_for_symbol(
                    code,
                    symbol_obj,
                    symbol_index_by_symbol=symbol_index_by_symbol,
                    price_panel=price_panel,
                    endpoint_builders=endpoint_builders,
                )
                _record_result(result, idx)
        else:
            futures = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for code in normalized_symbols:
                    symbol_obj = symbol_objs.get(code)
                    if symbol_obj is None:
                        continue
                    futures.append(
                        executor.submit(
                            _build_fmp_endpoint_features_for_symbol,
                            code,
                            symbol_obj,
                            symbol_index_by_symbol=symbol_index_by_symbol,
                            price_panel=price_panel,
                            endpoint_builders=endpoint_builders,
                        )
                    )
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    _record_result(future.result(), completed)
    finally:
        clear_section_record_cache()

    family_frames = {}
    family_cols = {}
    for endpoint_name, frames in endpoint_frames.items():
        if not frames:
            continue
        frame = pd.concat(frames, axis=0).sort_index()
        if frame.index.has_duplicates:
            frame = frame[~frame.index.duplicated(keep="last")]
        cols = [c for c in endpoint_cols[endpoint_name] if c in frame.columns and pd.api.types.is_numeric_dtype(frame[c])]
        cols = [c for c in cols if frame[c].notna().any()]
        if cols:
            family_frames[endpoint_name] = frame.loc[:, cols].astype(np.float32, copy=False)
            family_cols[endpoint_name] = cols
    if errors and callable(progress_logger):
        progress_logger(f"FMP feature build completed with {len(errors):,} symbol errors | first={errors[0]}")
    return family_frames, family_cols
