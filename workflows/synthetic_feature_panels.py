from __future__ import annotations

import numpy as np
import pandas as pd

from fmp.models import Symbol
from features.feature_builders import (
    build_industry_pe_features,
    build_industry_performance_features,
    build_price_technical_features,
    build_sector_pe_features,
    build_sector_performance_features,
    build_ta_classic_technical_features,
)
from features.views import _load_adjusted_prices

def build_technical_dataframe_from_django(*, symbols, start_date=None, end_date=None):
    start_ts = pd.Timestamp(start_date) if start_date is not None else None
    end_ts = pd.Timestamp(end_date) if end_date is not None else None
    frames = []
    feature_cols = []
    technical_family_frames = {}
    technical_family_cols = {}
    for sym in symbols:
        code = str(sym).strip().upper()
        if not code:
            continue

        symbol_obj = Symbol.objects.filter(symbol__iexact=code).only("id", "symbol").first()
        if symbol_obj is None:
            continue

        df_prices = _load_adjusted_prices(
            symbol_obj,
            start_ts.date() if start_ts is not None else None,
            end_ts.date() if end_ts is not None else None,
        )
        if df_prices.empty:
            continue

        built = build_price_technical_features(code, df_prices)
        if built.df.empty:
            continue

        px = df_prices[["open", "high", "low", "close", "volume"]].copy()
        px["symbol"] = code
        px = px.reset_index().set_index(["date", "symbol"]).sort_index()

        panel = px.join(built.df[built.feature_cols], how="left")
        frames.append(panel)

        for family_name, family_built in build_ta_classic_technical_features(code, df_prices).items():
            active_cols = [
                col
                for col in family_built.feature_cols
                if col in family_built.df.columns and pd.api.types.is_numeric_dtype(family_built.df[col])
            ]
            if not active_cols:
                continue
            technical_family_frames.setdefault(family_name, []).append(family_built.df.loc[:, active_cols])
            family_cols = technical_family_cols.setdefault(family_name, [])
            for col in active_cols:
                if col not in family_cols:
                    family_cols.append(col)
        for col in built.feature_cols:
            if col not in feature_cols:
                feature_cols.append(col)

    if not frames:
        empty_index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["date", "symbol"])
        return pd.DataFrame(index=empty_index), feature_cols, {}, {}

    technical_df = pd.concat(frames, axis=0).sort_index()
    if technical_df.index.has_duplicates:
        technical_df = technical_df[~technical_df.index.duplicated(keep="last")]
    split_family_frames = {}
    split_family_cols = {}
    for family_name, family_frames in technical_family_frames.items():
        family_frame = pd.concat(family_frames, axis=0).sort_index()
        if family_frame.index.has_duplicates:
            family_frame = family_frame[~family_frame.index.duplicated(keep="last")]
        cols = [c for c in technical_family_cols.get(family_name, []) if c in family_frame.columns]
        cols = [c for c in cols if pd.api.types.is_numeric_dtype(family_frame[c]) and family_frame[c].notna().any()]
        if cols:
            split_family_frames[family_name] = family_frame.loc[:, cols].astype(np.float32, copy=False)
            split_family_cols[family_name] = cols
    return technical_df, feature_cols, split_family_frames, split_family_cols

def _target_index_for_symbol(target_index, symbol):
    code = str(symbol).strip().upper()
    mask = target_index.get_level_values("symbol").astype(str).str.upper() == code
    dates = pd.DatetimeIndex(pd.to_datetime(target_index.get_level_values("date")[mask])).normalize()
    return pd.MultiIndex.from_arrays([dates, [code] * len(dates)], names=["date", "symbol"])

def _price_frame_for_symbol(price_panel, symbol):
    code = str(symbol).strip().upper()
    try:
        return price_panel.xs(code, level="symbol")
    except Exception:
        return pd.DataFrame()

def build_classification_performance_feature_families(*, symbols, target_index, price_panel, progress_logger=None):
    builders = {
        "sector_performance": lambda obj, idx, px: build_sector_performance_features(obj, idx, df_prices=px),
        "industry_performance": lambda obj, idx, px: build_industry_performance_features(obj, idx, df_prices=px),
        "sector_pe": lambda obj, idx, px: build_sector_pe_features(obj, idx),
        "industry_pe": lambda obj, idx, px: build_industry_pe_features(obj, idx),
    }
    normalized_symbols = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    symbol_objs = {
        str(obj.symbol).strip().upper(): obj
        for obj in Symbol.objects.filter(symbol__in=normalized_symbols).only(
            "id", "symbol", "exchange", "sector", "industry"
        )
    }
    family_parts = {name: [] for name in builders}
    family_cols = {name: [] for name in builders}
    total = len(normalized_symbols)
    for position, code in enumerate(normalized_symbols, start=1):
        symbol_obj = symbol_objs.get(code)
        if symbol_obj is None:
            continue
        symbol_index = _target_index_for_symbol(target_index, code)
        if len(symbol_index) == 0:
            continue
        symbol_prices = _price_frame_for_symbol(price_panel, code)
        for family_name, builder in builders.items():
            built = builder(symbol_obj, symbol_index, symbol_prices)
            active_cols = [
                col for col in built.feature_cols
                if col in built.df.columns and pd.api.types.is_numeric_dtype(built.df[col])
            ]
            if not active_cols:
                continue
            family_parts[family_name].append(built.df.loc[:, active_cols])
            for col in active_cols:
                if col not in family_cols[family_name]:
                    family_cols[family_name].append(col)
        if callable(progress_logger) and (position == 1 or position % 100 == 0 or position == total):
            progress_logger(
                f"Classification performance feature build: {position:,}/{total:,} symbols processed"
            )
    family_frames = {}
    active_family_cols = {}
    for family_name, parts in family_parts.items():
        if not parts:
            continue
        frame = pd.concat(parts, axis=0).sort_index()
        if frame.index.has_duplicates:
            frame = frame.loc[~frame.index.duplicated(keep="last")]
        cols = [col for col in family_cols[family_name] if col in frame.columns and frame[col].notna().any()]
        if not cols:
            continue
        family_frames[family_name] = frame.loc[:, cols].astype(np.float32, copy=False)
        active_family_cols[family_name] = cols
    return family_frames, active_family_cols

__all__ = ['build_classification_performance_feature_families', 'build_technical_dataframe_from_django']
