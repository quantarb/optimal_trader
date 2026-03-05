from __future__ import annotations

import re

import pandas as pd

from features.section_utils import BuiltFeatureSet
from features.technical import BASE_PRICE_COLS, compute_features_worldclass


def build_prices_div_adj_features(symbol: str, df_prices: pd.DataFrame) -> BuiltFeatureSet:
    if df_prices.empty:
        return BuiltFeatureSet(df=pd.DataFrame(), feature_cols=[])

    df_daily = compute_features_worldclass(df_prices.copy())
    feature_cols = [
        col
        for col in df_daily.columns
        if col not in BASE_PRICE_COLS and col != "symbol" and pd.api.types.is_numeric_dtype(df_daily[col])
    ]
    rename_map = {col: f"px__{_to_snake(col)}" for col in feature_cols}

    out = df_daily[feature_cols].rename(columns=rename_map).copy()
    out["symbol"] = str(symbol)
    out = out.reset_index().rename(columns={out.index.name or "index": "date"}).set_index(["date", "symbol"]).sort_index()
    renamed_feature_cols = [rename_map[col] for col in feature_cols]
    return BuiltFeatureSet(df=out, feature_cols=renamed_feature_cols)


def _to_snake(value: str) -> str:
    text = str(value).replace("%", "pct")
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
    text = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", text)
    text = re.sub(r"([A-Za-z])([0-9])", r"\1_\2", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()
