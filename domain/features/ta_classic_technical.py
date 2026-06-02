from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from domain.features.specs import BuiltFeatureSet
from domain.features.technical import BASE_PRICE_COLS, _ensure_dt_index, _to_snake
from utils.normalize import normalize_cols


TA_CLASSIC_FAMILY_PREFIXES: dict[str, str] = {
    "technical_candles": "ta_candle__",
    "technical_cycles": "ta_cycle__",
    "technical_math": "ta_math__",
    "technical_momentum": "ta_momentum__",
    "technical_overlap": "ta_overlap__",
    "technical_performance": "ta_performance__",
}


@dataclass(frozen=True)
class TaIndicatorSpec:
    name: str
    fn_name: str
    inputs: tuple[str, ...]
    kwargs: dict[str, object] | None = None
    min_rows: int = 1


def build_price_ta_classic_feature_families(
    symbol: str,
    df_prices: pd.DataFrame,
) -> dict[str, BuiltFeatureSet]:
    """Build split pandas-ta-classic technical feature families for a single symbol."""

    if df_prices.empty:
        return _empty_family_sets()
    ta = _import_pandas_ta_classic()
    prices = _prepare_price_frame(df_prices)
    if prices.empty:
        return _empty_family_sets()

    result: dict[str, BuiltFeatureSet] = {}
    for family_name, specs in _indicator_specs().items():
        frame = pd.DataFrame(index=prices.index)
        for spec in specs:
            indicator = _compute_indicator(ta, prices, spec)
            if indicator.empty:
                continue
            for column in indicator.columns:
                out_col = _feature_column_name(family_name, spec.name, column)
                frame[out_col] = pd.to_numeric(indicator[column], errors="coerce")
        feature_cols = _usable_feature_cols(frame)
        result[family_name] = _to_built_feature_set(symbol, frame, feature_cols)
    return result


def _import_pandas_ta_classic():
    try:
        import pandas_ta_classic as ta
    except ImportError as exc:
        raise ImportError(
            "pandas-ta-classic is required for split technical feature families. "
            "Install it in the optimal_trader environment with `pip install pandas-ta-classic`."
        ) from exc
    return ta


def _empty_family_sets() -> dict[str, BuiltFeatureSet]:
    return {
        family_name: BuiltFeatureSet(df=pd.DataFrame(), feature_cols=[])
        for family_name in TA_CLASSIC_FAMILY_PREFIXES
    }


def _prepare_price_frame(df_prices: pd.DataFrame) -> pd.DataFrame:
    out = normalize_cols(df_prices)
    out = _ensure_dt_index(out)
    missing = [column for column in BASE_PRICE_COLS if column not in out.columns]
    if missing:
        raise ValueError(f"df_prices missing required columns for pandas-ta-classic features: {missing}")
    out = out.loc[:, list(BASE_PRICE_COLS)].copy()
    for column in BASE_PRICE_COLS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=["open", "high", "low", "close"]).sort_index()


def _compute_indicator(ta, prices: pd.DataFrame, spec: TaIndicatorSpec) -> pd.DataFrame:
    if len(prices) < int(spec.min_rows):
        return pd.DataFrame(index=prices.index)
    fn: Callable[..., object] | None = getattr(ta, spec.fn_name, None)
    if fn is None:
        return pd.DataFrame(index=prices.index)
    kwargs = dict(spec.kwargs or {})
    call_args = {input_name: prices[input_name] for input_name in spec.inputs if input_name in prices.columns}
    if "open" in call_args:
        call_args["open_"] = call_args.pop("open")
    try:
        raw = fn(**call_args, **kwargs)
    except Exception:
        return pd.DataFrame(index=prices.index)
    if raw is None:
        return pd.DataFrame(index=prices.index)
    if isinstance(raw, pd.Series):
        name = str(raw.name or spec.name)
        return raw.rename(name).to_frame().reindex(prices.index)
    if isinstance(raw, pd.DataFrame):
        return raw.reindex(prices.index)
    return pd.DataFrame(index=prices.index)


def _feature_column_name(family_name: str, spec_name: str, raw_column: str) -> str:
    prefix = TA_CLASSIC_FAMILY_PREFIXES[family_name]
    raw = _to_snake(raw_column)
    base = _to_snake(spec_name)
    if raw.startswith(base):
        core = raw
    else:
        core = f"{base}_{raw}"
    return f"{prefix}{core}"


def _usable_feature_cols(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for column in frame.columns:
        series = frame[column].replace([np.inf, -np.inf], np.nan)
        if not pd.api.types.is_numeric_dtype(series):
            continue
        if series.notna().any():
            frame[column] = series.ffill().bfill().fillna(0.0).astype(np.float32)
            cols.append(column)
    return list(dict.fromkeys(cols))


def _to_built_feature_set(symbol: str, frame: pd.DataFrame, feature_cols: list[str]) -> BuiltFeatureSet:
    if frame.empty or not feature_cols:
        return BuiltFeatureSet(df=pd.DataFrame(), feature_cols=[])
    out = frame.loc[:, feature_cols].copy()
    out["symbol"] = str(symbol).strip().upper()
    out = out.reset_index().rename(columns={out.index.name or "index": "date"}).set_index(["date", "symbol"]).sort_index()
    return BuiltFeatureSet(df=out, feature_cols=list(feature_cols))


def _indicator_specs() -> dict[str, tuple[TaIndicatorSpec, ...]]:
    return {
        "technical_candles": (
            TaIndicatorSpec("doji", "cdl_doji", ("open", "high", "low", "close"), min_rows=10),
            TaIndicatorSpec("inside", "cdl_inside", ("open", "high", "low", "close"), {"asbool": False}),
            TaIndicatorSpec("ha", "ha", ("open", "high", "low", "close")),
            TaIndicatorSpec("candle_z", "cdl_z", ("open", "high", "low", "close"), {"length": 20}, min_rows=20),
        ),
        "technical_cycles": (
            TaIndicatorSpec("ebsw_40_10", "ebsw", ("close",), {"length": 40, "bars": 10}, min_rows=40),
            TaIndicatorSpec("ht_dcperiod", "ht_dcperiod", ("close",)),
            TaIndicatorSpec("ht_dcphase", "ht_dcphase", ("close",)),
            TaIndicatorSpec("ht_phasor", "ht_phasor", ("close",)),
            TaIndicatorSpec("ht_sine", "ht_sine", ("close",)),
            TaIndicatorSpec("ht_trendmode", "ht_trendmode", ("close",)),
        ),
        "technical_math": (
            TaIndicatorSpec("zscore_20", "zscore", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("zscore_63", "zscore", ("close",), {"length": 63}, min_rows=63),
            TaIndicatorSpec("entropy_20", "entropy", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("stdev_20", "stdev", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("variance_20", "variance", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("skew_20", "skew", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("kurtosis_20", "kurtosis", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("slope_20", "slope", ("close",), {"length": 20}, min_rows=20),
        ),
        "technical_momentum": (
            TaIndicatorSpec("rsi_14", "rsi", ("close",), {"length": 14}, min_rows=14),
            TaIndicatorSpec("macd", "macd", ("close",), min_rows=26),
            TaIndicatorSpec("stoch", "stoch", ("high", "low", "close"), min_rows=14),
            TaIndicatorSpec("cci_20", "cci", ("high", "low", "close"), {"length": 20}, min_rows=20),
            TaIndicatorSpec("roc_10", "roc", ("close",), {"length": 10}, min_rows=10),
            TaIndicatorSpec("mom_10", "mom", ("close",), {"length": 10}, min_rows=10),
            TaIndicatorSpec("willr_14", "willr", ("high", "low", "close"), {"length": 14}, min_rows=14),
            TaIndicatorSpec("ppo", "ppo", ("close",), min_rows=26),
            TaIndicatorSpec("cmo_14", "cmo", ("close",), {"length": 14}, min_rows=14),
            TaIndicatorSpec("bop", "bop", ("open", "high", "low", "close")),
            TaIndicatorSpec("ao", "ao", ("high", "low"), min_rows=34),
        ),
        "technical_overlap": (
            TaIndicatorSpec("sma_10", "sma", ("close",), {"length": 10}, min_rows=10),
            TaIndicatorSpec("sma_20", "sma", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("sma_50", "sma", ("close",), {"length": 50}, min_rows=50),
            TaIndicatorSpec("ema_12", "ema", ("close",), {"length": 12}, min_rows=12),
            TaIndicatorSpec("ema_26", "ema", ("close",), {"length": 26}, min_rows=26),
            TaIndicatorSpec("ema_50", "ema", ("close",), {"length": 50}, min_rows=50),
            TaIndicatorSpec("dema_20", "dema", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("tema_20", "tema", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("hma_20", "hma", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("wma_20", "wma", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("bbands_20", "bbands", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("hl2", "hl2", ("high", "low")),
            TaIndicatorSpec("hlc3", "hlc3", ("high", "low", "close")),
            TaIndicatorSpec("ohlc4", "ohlc4", ("open", "high", "low", "close")),
        ),
        "technical_performance": (
            TaIndicatorSpec("pct_return_1", "percent_return", ("close",), {"length": 1}, min_rows=2),
            TaIndicatorSpec("pct_return_5", "percent_return", ("close",), {"length": 5}, min_rows=5),
            TaIndicatorSpec("pct_return_20", "percent_return", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("log_return_1", "log_return", ("close",), {"length": 1}, min_rows=2),
            TaIndicatorSpec("log_return_5", "log_return", ("close",), {"length": 5}, min_rows=5),
            TaIndicatorSpec("log_return_20", "log_return", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("drawdown", "drawdown", ("close",)),
        ),
    }
