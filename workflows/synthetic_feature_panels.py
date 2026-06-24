from __future__ import annotations

import numpy as np
import pandas as pd

from data.warehouse import load_warehouse_price_frames
from quant_warehouse.feature_engineering import build_price_technical_features, build_price_ta_classic_feature_families


def _normalized_symbols(symbols):
    return [str(symbol).strip().upper() for symbol in list(symbols or []) if str(symbol).strip()]

def build_technical_dataframe_from_django(*, symbols, start_date=None, end_date=None):
    normalized_symbols = _normalized_symbols(symbols)
    price_frames = load_warehouse_price_frames(
        normalized_symbols,
        start_date=str(start_date)[:10] if start_date is not None else None,
        end_date=str(end_date)[:10] if end_date is not None else None,
    )
    frames = []
    feature_cols = []
    technical_family_frames = {}
    technical_family_cols = {}
    for code in normalized_symbols:
        df_prices = price_frames.get(code)
        if df_prices is None or df_prices.empty:
            continue

        built = build_price_technical_features(code, df_prices)
        if built.df.empty:
            continue

        px = df_prices[["open", "high", "low", "close", "volume"]].copy()
        px["symbol"] = code
        px = px.reset_index().set_index(["date", "symbol"]).sort_index()

        panel = px.join(built.df[built.feature_cols], how="left")
        frames.append(panel)

        for family_name, family_built in build_price_ta_classic_feature_families(code, df_prices).items():
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
    del symbols, target_index, price_panel, progress_logger
    return {}, {}

__all__ = ['build_classification_performance_feature_families', 'build_technical_dataframe_from_django']
