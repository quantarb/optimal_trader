"""Canonical public API.

This repo previously had multiple competing entrypoints (spec builders, runners,
notebook recipes). Those have been removed or deprecated.

Everything here is explicit:
- train/infer windows are required
- data flow is function calls with typed artifacts
"""

from __future__ import annotations

import os
import re
import math
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from modules.data.build import build_dataset, build_technical_panel
from modules.data.feature_name_map import PRETTY_NAME_MAP
from modules.data.preparation import (
    Entry2ExitTextConfig,
    MLDatasetConfig,
    prepare_entry2exit_dataset as _prepare_entry2exit_dataset_unified,
    prepare_ml_dataset as _prepare_ml_dataset_unified,
)
from modules.features.fundamentals_fmp import fetch_fundamentals_data, broadcast_fundamentals_to_daily
from modules.qcore.contracts import DatasetArtifacts, ModelArtifact, PredictionsArtifact
from modules.qcore.window import TimeWindow
from modules.utils.panel import ensure_panel_index
from modules.engine.backtest import run_backtest
from modules.features.macro_fmp import fetch_macro_series, broadcast_macro_to_daily, MacroFeatureConfig
from modules.features.time_features import build_time_features, TimeFeatureConfig
from modules.labels.events import build_label_panel, deduplicate_labels
from modules.labels.ranking import add_rank_regression_labels
from modules.labels.strategy_solver import solve_longs_by_frequency, solve_shorts_by_frequency

_ACRONYM_TOKENS: dict[str, str] = {
    "atr": "ATR",
    "bb": "BB",
    "clv": "CLV",
    "cpi": "CPI",
    "ebit": "EBIT",
    "ebitda": "EBITDA",
    "eps": "EPS",
    "ev": "EV",
    "fcf": "FCF",
    "gdp": "GDP",
    "km": "KM",
    "llm": "LLM",
    "macd": "MACD",
    "obv": "OBV",
    "ohlcv": "OHLCV",
    "pe": "PE",
    "pit": "PIT",
    "pnl": "PnL",
    "px": "Px",
    "ret": "Ret",
    "rsi": "RSI",
    "rt": "RT",
    "sma": "SMA",
    "ema": "EMA",
    "ttm": "TTM",
    "ust": "UST",
}


def _filter_df_by_date_bounds(
    df: pd.DataFrame,
    *,
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    """
    Filter by inclusive date bounds when both dates are provided.
    Supports MultiIndex with "date", DatetimeIndex, or a "date" column.
    """
    if start_date is None and end_date is None:
        return df
    if (start_date is None) != (end_date is None):
        raise ValueError("Provide both start_date and end_date, or neither.")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if start_ts > end_ts:
        raise ValueError("start_date must be <= end_date.")

    if df.empty:
        return df

    if isinstance(df.index, pd.MultiIndex) and "date" in (df.index.names or []):
        dts = pd.to_datetime(df.index.get_level_values("date"))
        mask = (dts >= start_ts) & (dts <= end_ts)
        return df.loc[mask]

    if isinstance(df.index, pd.DatetimeIndex):
        dts = pd.to_datetime(df.index)
        mask = (dts >= start_ts) & (dts <= end_ts)
        return df.loc[mask]

    if "date" in df.columns:
        dts = pd.to_datetime(df["date"], errors="coerce")
        mask = (dts >= start_ts) & (dts <= end_ts)
        return df.loc[mask]

    return df


def _extract_date_bounds(df: pd.DataFrame) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if df is None or df.empty:
        return None, None
    if isinstance(df.index, pd.MultiIndex) and "date" in (df.index.names or []):
        dts = pd.to_datetime(df.index.get_level_values("date"), errors="coerce")
    elif isinstance(df.index, pd.DatetimeIndex):
        dts = pd.to_datetime(df.index, errors="coerce")
    elif "date" in df.columns:
        dts = pd.to_datetime(df["date"], errors="coerce")
    else:
        return None, None
    dts = pd.Series(dts).dropna()
    if dts.empty:
        return None, None
    return pd.Timestamp(dts.min()), pd.Timestamp(dts.max())


def _resolve_fundamental_limit(
    *,
    start_date: Optional[str],
    end_date: Optional[str],
    period: str,
) -> int:
    """
    Ensure fundamentals fetch limit can cover requested date span.
    FMP fundamentals are period/limit-based; they are not date-ranged at source.
    """
    base = 160
    if start_date is None or end_date is None:
        return base

    s = pd.Timestamp(start_date)
    e = pd.Timestamp(end_date)
    if s > e:
        return base

    years = max(0.0, float((e - s).days) / 365.25)
    p = str(period).lower()
    if p.startswith("q"):
        needed = int(math.ceil(years * 4.0)) + 8
    elif p.startswith("a") or p.startswith("y"):
        needed = int(math.ceil(years)) + 2
    else:
        return base

    return max(base, needed)


def _strip_source_prefix(name: str) -> tuple[str, Optional[str]]:
    raw = str(name).strip()
    m = re.match(r"^(macro|km|kt|rt)(?:__|_)+(.*)$", raw, flags=re.IGNORECASE)
    if not m:
        return raw, None
    src = m.group(1).lower()
    rest = m.group(2).strip()
    src_map = {
        "macro": "Macro",
        "km": "Km",
        "kt": "Km",
        "rt": "Rt",
    }
    return (rest or raw), src_map.get(src)


def _to_pascal_case(name: str) -> str:
    """
    Convert snake/mixed delimiters to PascalCase.
    Examples:
      - ret_1d -> Ret1d
      - macro__ust_year10 -> UstYear10
      - km__enterprisevalue -> KmEnterprisevalue
    """
    raw, _ = _strip_source_prefix(name)
    lookup = re.sub(r"[^A-Za-z0-9]+", "", raw).lower()
    if lookup in PRETTY_NAME_MAP:
        return PRETTY_NAME_MAP[lookup]
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", raw) if p]
    if not parts:
        return str(name)
    out: list[str] = []
    for p in parts:
        key = p.lower()
        if key in _ACRONYM_TOKENS:
            out.append(_ACRONYM_TOKENS[key])
            continue
        # Preserve true mixed-case tokens (e.g., DistSMA5, MACDSignal).
        has_upper = any(ch.isupper() for ch in p[1:])
        has_lower = any(ch.islower() for ch in p[1:])
        if has_upper and has_lower:
            out.append(p[:1].upper() + p[1:])
        else:
            out.append(p[:1].upper() + p[1:].lower())
    return "".join(out)


def _camelize_dataframe_columns(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Camel-case dataframe columns and return remapped feature columns.
    Handles collisions deterministically by suffixing 2, 3, ...
    """
    if df.empty and not feature_cols:
        return df, []

    rename_map: dict[Any, str] = {}
    used_names: set[str] = set()
    for col in df.columns:
        base = _to_pascal_case(col)
        _, src = _strip_source_prefix(str(col))
        candidate = base

        if candidate in used_names and src:
            candidate = f"{base}{src}"

        if candidate in used_names:
            n = 2
            while f"{base}{n}" in used_names:
                n += 1
            candidate = f"{base}{n}"

        new_col = candidate
        used_names.add(new_col)
        rename_map[col] = new_col

    out = df.rename(columns=rename_map).copy()
    remapped = [rename_map.get(c, str(c)) for c in list(feature_cols)]
    return out, remapped


def _drop_source_duplicate_columns(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Drop source-suffixed duplicates when a base feature exists.
    Example: keep CurrentRatio, drop CurrentRatioRt/CurrentRatioKm.

    This is an upstream schema cleanup so all model families see the same
    deduplicated feature set.
    """
    cols = list(df.columns)
    colset = set(cols)
    drop_cols: list[str] = []
    for c in cols:
        if c.endswith("Rt") and c[:-2] in colset:
            drop_cols.append(c)
        elif c.endswith("Km") and c[:-2] in colset:
            drop_cols.append(c)

    if not drop_cols:
        return df, list(feature_cols)

    out = df.drop(columns=drop_cols, errors="ignore")
    fset = set(out.columns)
    remapped_features = [c for c in list(feature_cols) if c in fset]
    return out, remapped_features


_DAILY_PRICE_LINKED_FUNDAMENTAL_COLUMNS: tuple[str, ...] = (
    "km__marketcap",
    "km__enterprisevalue",
    "km__evtosales",
    "km__evtooperatingcashflow",
    "km__evtofreecashflow",
    "km__evtoebitda",
    "rt__enterprisevaluemultiple",
    "rt__debttomarketcap",
    "km__earningsyield",
    "km__freecashflowyield",
    "rt__dividendyield",
    "rt__dividendyieldpercentage",
    "rt__pricetoearningsratio",
    "rt__pricetoearningsgrowthratio",
    "rt__forwardpricetoearningsgrowthratio",
    "rt__pricetobookratio",
    "rt__pricetosalesratio",
    "rt__pricetofreecashflowratio",
    "rt__pricetooperatingcashflowratio",
    "rt__pricetofairvalue",
)


def _safe_divide(numer: pd.Series, denom: pd.Series) -> pd.Series:
    out = pd.to_numeric(numer, errors="coerce") / pd.to_numeric(denom, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def _add_fundamental_update_timing_features(
    dense_fund_df: pd.DataFrame,
    *,
    sparse_effective_index: pd.MultiIndex,
) -> pd.DataFrame:
    """
    Add timing features based on sparse fundamental update dates.

    Features added:
      - km__dayssincelastfundamentalupdate
      - km__isnewfundamentalupdate
    """
    if dense_fund_df.empty or not isinstance(dense_fund_df.index, pd.MultiIndex):
        return dense_fund_df

    out = dense_fund_df.copy()
    idx = out.index

    if "date" not in (idx.names or []) or "symbol" not in (idx.names or []):
        return out

    # Mark rows where a new sparse fundamentals snapshot becomes effective.
    event = pd.Series(0.0, index=idx, dtype="float64")
    if sparse_effective_index is not None and len(sparse_effective_index) > 0:
        sparse_idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp(d), str(s)) for d, s in sparse_effective_index.to_list()],
            names=["date", "symbol"],
        )
        aligned = idx.intersection(sparse_idx)
        if len(aligned) > 0:
            event.loc[aligned] = 1.0

    # Days since latest effective sparse update per symbol.
    work = pd.DataFrame(
        {
            "_date": pd.to_datetime(idx.get_level_values("date")),
            "_symbol": idx.get_level_values("symbol").astype(str),
            "_event": event.values,
        },
        index=idx,
    )
    work = work.sort_values(["_symbol", "_date"])
    work["_last_update_date"] = work["_date"].where(work["_event"] > 0.0)
    work["_last_update_date"] = work.groupby("_symbol", sort=False)["_last_update_date"].ffill()
    days = (work["_date"] - work["_last_update_date"]).dt.days.astype("float64")

    out["km__dayssincelastfundamentalupdate"] = days
    out["km__isnewfundamentalupdate"] = work["_event"].astype("float64")
    return out


def _extract_daily_close_series(
    daily_prices: Optional[Union[pd.DataFrame, pd.Series]],
    *,
    target_index: Optional[pd.Index] = None,
    preferred_col: str = "Close",
) -> Optional[pd.Series]:
    if daily_prices is None:
        return None

    if isinstance(daily_prices, pd.Series):
        out = pd.to_numeric(daily_prices, errors="coerce")
    else:
        df = daily_prices.copy()
        candidates = [
            str(preferred_col),
            str(preferred_col).lower(),
            "Close",
            "close",
            "AdjClose",
            "adjclose",
            "Adj_Close",
            "adj_close",
        ]
        col = next((c for c in candidates if c in df.columns), None)
        if col is None:
            return None
        out = pd.to_numeric(df[col], errors="coerce")

    if isinstance(out.index, pd.MultiIndex):
        names = list(out.index.names or [])
        if "date" not in names or "symbol" not in names:
            return None
    else:
        return None

    out = out.rename("close_daily")
    if target_index is not None:
        out = out.reindex(target_index)
    return out


def _recompute_price_linked_fundamentals_daily(
    fund_df: pd.DataFrame,
    close_daily: pd.Series,
) -> tuple[pd.DataFrame, list[str]]:
    if fund_df.empty or close_daily is None or close_daily.empty:
        return fund_df, []

    out = fund_df.copy()
    close = pd.to_numeric(close_daily, errors="coerce")
    close = close.reindex(out.index)
    base = out.copy()

    def has(col: str) -> bool:
        return col in out.columns

    def upd(col: str, values: pd.Series, updated_cols: list[str]) -> None:
        if not has(col):
            return
        out[col] = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if col not in updated_cols:
            updated_cols.append(col)

    updated: list[str] = []

    # Direct price-per-share recomputes.
    if has("rt__netincomepershare"):
        eps = pd.to_numeric(out["rt__netincomepershare"], errors="coerce")
        upd("rt__pricetoearningsratio", _safe_divide(close, eps), updated)
        upd("km__earningsyield", _safe_divide(eps, close), updated)

    if has("rt__bookvaluepershare"):
        bvps = pd.to_numeric(out["rt__bookvaluepershare"], errors="coerce")
        pb = _safe_divide(close, bvps)
        upd("rt__pricetobookratio", pb, updated)
        upd("rt__pricetofairvalue", pb, updated)

    if has("rt__revenuepershare"):
        rev_ps = pd.to_numeric(out["rt__revenuepershare"], errors="coerce")
        upd("rt__pricetosalesratio", _safe_divide(close, rev_ps), updated)

    if has("rt__freecashflowpershare"):
        fcf_ps = pd.to_numeric(out["rt__freecashflowpershare"], errors="coerce")
        upd("rt__pricetofreecashflowratio", _safe_divide(close, fcf_ps), updated)
        upd("km__freecashflowyield", _safe_divide(fcf_ps, close), updated)

    if has("rt__operatingcashflowpershare"):
        ocf_ps = pd.to_numeric(out["rt__operatingcashflowpershare"], errors="coerce")
        upd("rt__pricetooperatingcashflowratio", _safe_divide(close, ocf_ps), updated)

    if has("rt__dividendpershare"):
        div_ps = pd.to_numeric(out["rt__dividendpershare"], errors="coerce")
        div_yld = _safe_divide(div_ps, close)
        upd("rt__dividendyield", div_yld, updated)
        upd("rt__dividendyieldpercentage", 100.0 * div_yld, updated)

    # Infer a filing-date reference price from already-computed valuation fields.
    ref_candidates: list[pd.Series] = []
    if has("rt__pricetoearningsratio") and has("rt__netincomepershare"):
        ref_candidates.append(
            pd.to_numeric(base["rt__pricetoearningsratio"], errors="coerce")
            * pd.to_numeric(base["rt__netincomepershare"], errors="coerce")
        )
    if has("rt__pricetobookratio") and has("rt__bookvaluepershare"):
        ref_candidates.append(
            pd.to_numeric(base["rt__pricetobookratio"], errors="coerce")
            * pd.to_numeric(base["rt__bookvaluepershare"], errors="coerce")
        )
    if has("rt__pricetosalesratio") and has("rt__revenuepershare"):
        ref_candidates.append(
            pd.to_numeric(base["rt__pricetosalesratio"], errors="coerce")
            * pd.to_numeric(base["rt__revenuepershare"], errors="coerce")
        )
    if has("rt__pricetofreecashflowratio") and has("rt__freecashflowpershare"):
        ref_candidates.append(
            pd.to_numeric(base["rt__pricetofreecashflowratio"], errors="coerce")
            * pd.to_numeric(base["rt__freecashflowpershare"], errors="coerce")
        )
    if has("rt__pricetooperatingcashflowratio") and has("rt__operatingcashflowpershare"):
        ref_candidates.append(
            pd.to_numeric(base["rt__pricetooperatingcashflowratio"], errors="coerce")
            * pd.to_numeric(base["rt__operatingcashflowpershare"], errors="coerce")
        )
    if has("rt__dividendpershare") and has("rt__dividendyield"):
        ref_candidates.append(
            _safe_divide(
                pd.to_numeric(base["rt__dividendpershare"], errors="coerce"),
                pd.to_numeric(base["rt__dividendyield"], errors="coerce"),
            )
        )

    if ref_candidates:
        ref_px = pd.concat(ref_candidates, axis=1).median(axis=1, skipna=True)
    else:
        ref_px = pd.Series(np.nan, index=out.index, dtype="float64")

    # Shares outstanding implied from stale market cap + stale reference price.
    if has("km__marketcap"):
        stale_mcap = pd.to_numeric(base["km__marketcap"], errors="coerce")
        implied_shares = _safe_divide(stale_mcap, ref_px)
        daily_mcap = close * implied_shares
        upd("km__marketcap", daily_mcap, updated)
    else:
        stale_mcap = pd.Series(np.nan, index=out.index, dtype="float64")
        daily_mcap = pd.Series(np.nan, index=out.index, dtype="float64")

    # Keep balance-sheet debt/net-debt amount fixed between filings.
    stale_ev = pd.to_numeric(base["km__enterprisevalue"], errors="coerce") if has("km__enterprisevalue") else pd.Series(np.nan, index=out.index, dtype="float64")
    net_debt_amt = stale_ev - stale_mcap
    daily_ev = daily_mcap + net_debt_amt
    upd("km__enterprisevalue", daily_ev, updated)

    if has("rt__debttomarketcap"):
        debt_amt = pd.to_numeric(base["rt__debttomarketcap"], errors="coerce") * stale_mcap
        upd("rt__debttomarketcap", _safe_divide(debt_amt, daily_mcap), updated)

    # EV-based multiples: keep denominators fixed, update via EV ratio.
    ev_ratio = _safe_divide(daily_ev, stale_ev)
    for c in (
        "km__evtosales",
        "km__evtooperatingcashflow",
        "km__evtofreecashflow",
        "km__evtoebitda",
        "rt__enterprisevaluemultiple",
    ):
        if has(c):
            upd(c, pd.to_numeric(base[c], errors="coerce") * ev_ratio, updated)

    # PEG-style fields scale with price when growth inputs are fixed.
    px_ratio = _safe_divide(close, ref_px)
    for c in ("rt__pricetoearningsgrowthratio", "rt__forwardpricetoearningsgrowthratio"):
        if has(c):
            upd(c, pd.to_numeric(base[c], errors="coerce") * px_ratio, updated)

    return out, updated


def _compact_fin_feature_name(name: str) -> str:
    k = str(name)

    exact = {
        "Open": "O",
        "High": "H",
        "Low": "L",
        "Close": "C",
        "Volume": "Vol",
        "EntryPx": "EPx",
        "ExitPx": "XPx",
        "TradeDurationDays": "DurD",
        "TradeReturn": "Ret",
        "FederalFundsRate": "FFR",
        "Unemployment": "Unemp",
        "Inflation": "Infl",
        "MarketCap": "MktCap",
        "EnterpriseValue": "EV",
        "CurrentRatio": "CurrR",
        "QuickRatio": "QuickR",
        "CashRatio": "CashR",
        "DebtServiceCoverageRatio": "DSCR",
        "InterestCoverageRatio": "ICR",
        "DividendYield": "DivYld",
        "DividendYieldPercentage": "DivYldPct",
        "ReturnOnAssets": "ROA",
        "ReturnOnEquity": "ROE",
        "ReturnOnInvestedCapital": "ROIC",
        "ReturnOnCapitalEmployed": "ROCE",
        "OperatingReturnOnAssets": "OpROA",
        "EarningsYield": "EarningsYld",
        "FreeCashFlowYield": "FCFYld",
        "NetDebtToEBITDA": "NetDebt2EBITDA",
        "OperatingCashFlowRatio": "OCFR",
        "OperatingCashFlowSalesRatio": "OCF2Sales",
        "PriceToEarningsRatio": "PE",
        "PriceToBookRatio": "PB",
        "PriceToSalesRatio": "PS",
        "PriceToFreeCashFlowRatio": "PFCF",
        "PriceToOperatingCashFlowRatio": "POCF",
    }
    if k in exact:
        return exact[k]

    m = re.match(r"^Ret(\d+)d$", k)
    if m:
        return f"R{m.group(1)}D"
    m = re.match(r"^CumRet(\d+)d$", k)
    if m:
        return f"CR{m.group(1)}D"
    m = re.match(r"^DistSMA(\d+)$", k)
    if m:
        return f"SMADev{m.group(1)}"
    m = re.match(r"^SMASlope(\d+)$", k)
    if m:
        return f"SMASlp{m.group(1)}"
    m = re.match(r"^DistEMA(\d+)$", k)
    if m:
        return f"EMADev{m.group(1)}"
    m = re.match(r"^ZClose(\d+)$", k)
    if m:
        return f"ZC{m.group(1)}"
    m = re.match(r"^BBPos(\d+)$", k)
    if m:
        return f"BBP{m.group(1)}"
    m = re.match(r"^ATRPct(\d+)$", k)
    if m:
        return f"ATRP{m.group(1)}"
    m = re.match(r"^VolRegimeZ(\d+)$", k)
    if m:
        return f"VolRZ{m.group(1)}"
    m = re.match(r"^BreakoutUp(\d+)$", k)
    if m:
        return f"BrkUp{m.group(1)}"
    m = re.match(r"^BreakoutDn(\d+)$", k)
    if m:
        return f"BrkDn{m.group(1)}"
    m = re.match(r"^PosInChannel(\d+)$", k)
    if m:
        return f"ChPos{m.group(1)}"
    m = re.match(r"^DistHh(\d+)$", k)
    if m:
        return f"DHH{m.group(1)}"
    m = re.match(r"^DistLl(\d+)$", k)
    if m:
        return f"DLL{m.group(1)}"
    m = re.match(r"^VolZ(\d+)$", k)
    if m:
        return f"VZ{m.group(1)}"
    m = re.match(r"^USTMonth(\d+)$", k)
    if m:
        return f"UST{m.group(1)}M"
    m = re.match(r"^USTYear(\d+)$", k)
    if m:
        return f"UST{m.group(1)}Y"

    repl = [
        ("OperatingCashFlow", "OCF"),
        ("FreeCashFlow", "FCF"),
        ("CashFlow", "CF"),
        ("EnterpriseValue", "EV"),
        ("MarketCap", "MktCap"),
        ("ReturnOn", "RO"),
        ("PriceTo", "P2"),
        ("DebtTo", "D2"),
        ("LongTerm", "LT"),
        ("ShortTerm", "ST"),
        ("WorkingCapital", "WC"),
        ("InvestedCapital", "IC"),
        ("CapitalExpenditure", "CapEx"),
        ("Dividend", "Div"),
        ("PerShare", "PS"),
        ("Coverage", "Cov"),
        ("Turnover", "TO"),
        ("Outstanding", "Out"),
        ("Inventory", "Inv"),
        ("Receivables", "AR"),
        ("Payables", "AP"),
        ("Revenue", "Rev"),
        ("Income", "Inc"),
        ("Assets", "Ast"),
        ("Equity", "Eq"),
        ("Yield", "Yld"),
        ("Margin", "Mgn"),
        ("EffectiveTaxRate", "ETR"),
    ]
    out = k
    for a, b in repl:
        out = out.replace(a, b)
    return out


def _compact_dataframe_columns(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[pd.DataFrame, list[str]]:
    if df.empty and not feature_cols:
        return df, []

    rename_map: dict[Any, str] = {}
    used: set[str] = set()
    for c in df.columns:
        base = _compact_fin_feature_name(str(c))
        new_c = base
        if new_c in used and str(c) != new_c:
            n = 2
            while f"{base}{n}" in used:
                n += 1
            new_c = f"{base}{n}"
        used.add(new_c)
        rename_map[c] = new_c

    out = df.rename(columns=rename_map).copy()
    remapped = [rename_map.get(c, str(c)) for c in list(feature_cols)]
    return out, remapped



def _summarize_dataset_for_llm(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """
    Prints a statistical summary formatted for LLM reasoning.
    Highlights: Missingness, Infinite values, Constants, and Date ranges.
    """
    print("\n" + "="*60)
    print("  LLM DATA DEBUG REPORT  ")
    print("="*60)

    # 1. Scope
    n_rows, n_cols = df.shape
    n_symbols = df.index.get_level_values("symbol").nunique() if "symbol" in df.index.names else 0

    dates = df.index.get_level_values("date") if "date" in df.index.names else pd.Series([])
    d_start = dates.min().date() if not dates.empty else "N/A"
    d_end = dates.max().date() if not dates.empty else "N/A"

    print(f"SHAPE:       {n_rows:,} rows x {n_cols} cols")
    print(f"ENTITIES:    {n_symbols} symbols")
    print(f"TIME RANGE:  {d_start} to {d_end}")
    src_start = df.attrs.get("source_start_date")
    src_end = df.attrs.get("source_end_date")
    if src_start is not None and src_end is not None:
        try:
            ss = pd.Timestamp(src_start).date()
            se = pd.Timestamp(src_end).date()
            print(f"SOURCE RANGE (PRE-BROADCAST): {ss} to {se}")
        except Exception:
            pass

    # 2. Feature Health (NaNs)
    # Only check numeric columns to avoid object-dtype errors
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        print("WARNING: No numeric columns found.")
        return

    nan_counts = num_df.isna().sum()
    nan_pcts = (nan_counts / n_rows) * 100
    bad_cols = nan_pcts[nan_pcts > 0].sort_values(ascending=False)

    # Earliest date with at least one numeric value present (not symbol-level).
    try:
        row_has_numeric = num_df.notna().any(axis=1).to_numpy()
        if row_has_numeric.any():
            if isinstance(df.index, pd.MultiIndex) and "date" in (df.index.names or []):
                dts = pd.to_datetime(df.index.get_level_values("date"), errors="coerce")
            elif isinstance(df.index, pd.DatetimeIndex):
                dts = pd.to_datetime(df.index, errors="coerce")
            elif "date" in df.columns:
                dts = pd.to_datetime(df["date"], errors="coerce")
            else:
                dts = pd.Series([], dtype="datetime64[ns]")
            dts = pd.Series(dts)
            dts = dts[row_has_numeric]
            dts = dts.dropna()
            if not dts.empty:
                print(f"EARLIEST USABLE NUMERIC DATE: {pd.Timestamp(dts.min()).date()}")
    except Exception:
        pass

    print(f"\n[MISSING DATA] {len(bad_cols)} cols have NaNs.")
    if not bad_cols.empty:
        print("Top 5 worst offenders:")
        for col, pct in bad_cols.head(5).items():
            print(f"  - {col:<30} : {pct:.1f}% missing")

    # 3. Infinite Values
    inf_counts = np.isinf(num_df).sum()
    inf_cols = inf_counts[inf_counts > 0].sort_values(ascending=False)

    if not inf_cols.empty:
        print(f"\n[INFINITE VALUES] {len(inf_cols)} cols have +/- inf.")
        print("Top 5 worst offenders:")
        for col, count in inf_cols.head(5).items():
            print(f"  - {col:<30} : {count} rows")
    else:
        print("\n[INFINITE VALUES] None detected (Good).")

    # 4. Constant Columns (Zero Variance)
    # We use nunique() <= 1 as a proxy for constant, which is faster than std() on large data
    const_cols = [c for c in num_df.columns if num_df[c].nunique() <= 1]
    if const_cols:
        print(f"\n[CONSTANT COLS] {len(const_cols)} cols have zero variance (useless features).")
        print(f"  - Examples: {const_cols[:5]}")
    else:
        print("\n[CONSTANT COLS] None detected (Good).")

    print("="*60 + "\n")


def summarize_technical_features(
    technical_df: pd.DataFrame,
    technical_cols: Optional[Sequence[str]] = None,
) -> None:
    """
    Focused diagnostics for technical features only, using the same
    LLM debug report format as fundamentals/macro.
    """
    if technical_df is None or technical_df.empty:
        print("[technical-report] Empty dataframe.")
        return

    if technical_cols is None:
        base_exclude = {"symbol", "Open", "High", "Low", "Close", "Volume"}
        technical_cols = [
            c for c in technical_df.columns
            if c not in base_exclude and pd.api.types.is_numeric_dtype(technical_df[c])
        ]
    cols = [c for c in technical_cols if c in technical_df.columns]
    if not cols:
        print("[technical-report] No technical feature columns detected.")
        return

    df = technical_df[cols].copy()
    names = ", ".join(sorted({str(c) for c in cols}))
    print(f"[technical] Active numeric technical features ({len(cols)}): {names}")
    _summarize_dataset_for_llm(df, cols)


def build_dataset_artifacts(
    *,
    ctx,
    symbols: Sequence[str],
    train_window: TimeWindow,
    infer_window: TimeWindow,
    k_params: Dict[str, int],
    execution_params: Dict[str, Any],
    weighting: Dict[str, Any],
    add_rank_labels: bool = True,
    add_rank_tasks_to_mtl: bool = True,
    debug_data_quality: bool = False,
    data_quality_overrides: Optional[Dict[str, Any]] = None,
    skip_on_error: bool = True,
    verbose_data: bool = True,
) -> DatasetArtifacts:
    out = build_dataset(
        ctx=ctx,
        symbols=list(symbols),
        train_start=str(train_window.start.date()),
        train_end=str(train_window.end.date()),
        infer_start=str(infer_window.start.date()),
        infer_end=str(infer_window.end.date()),
        k_params=dict(k_params),
        execution_params=dict(execution_params),
        weighting=dict(weighting),
        add_rank_labels=bool(add_rank_labels),
        add_rank_tasks_to_mtl=bool(add_rank_tasks_to_mtl),
        debug_data_quality=bool(debug_data_quality),
        data_quality_overrides=dict(data_quality_overrides) if data_quality_overrides else None,
        skip_on_error=bool(skip_on_error),
        verbose_data=bool(verbose_data),
    )
    return DatasetArtifacts(
        daily_by_symbol=out["daily_by_symbol"],
        training_df=out["training_df"],
        inference_panel=ensure_panel_index(out["inference_panel"]),
        feature_cols=out["feature_cols"],
        meta=out.get("meta", {}) or {},
    )


def build_technical_dataframe(
    *,
    ctx,
    symbols: Sequence[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    execution_params: Optional[Dict[str, Any]] = None,
    debug_data_quality: bool = False,
    data_quality_overrides: Optional[Dict[str, Any]] = None,
    skip_on_error: bool = True,
    verbose_data: bool = True,
    compact_feature_names: bool = False,
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Builds a clean dataframe of Prices + Technicals (MultiIndex: date, symbol).
    """
    data_dir = os.path.dirname(ctx.store.db_path) or "."

    panel, feature_cols, skipped = build_technical_panel(
        universe=list(symbols),
        api_key=ctx.api_key,
        data_dir=data_dir,
        execution_params=dict(execution_params) if execution_params else None,
        db_name=os.path.basename(ctx.store.db_path),
        sleep_s=ctx.sleep_s,
        skip_on_error=skip_on_error,
        verbose_data=verbose_data,
        debug_data_quality=debug_data_quality,
        data_quality_overrides=data_quality_overrides,
    )

    if skipped:
        print(f"[WARNING] Skipped {len(skipped)} symbols during build.")

    panel = _filter_df_by_date_bounds(
        panel,
        start_date=start_date,
        end_date=end_date,
    )
    panel, feature_cols = _camelize_dataframe_columns(panel, feature_cols)
    if compact_feature_names:
        panel, feature_cols = _compact_dataframe_columns(panel, feature_cols)

    if verbose_data and not panel.empty:
        _summarize_dataset_for_llm(panel, feature_cols)

    return panel, feature_cols


def build_fundamental_dataframe(
    *,
    ctx,
    symbols: Sequence[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "quarter",
    verbose: bool = True,
    use_filing_lag: bool = True,
    filing_lag_days: int = 45,
    compact_feature_names: bool = False,
    target_index: Optional[pd.Index] = None,
    daily_prices: Optional[Union[pd.DataFrame, pd.Series]] = None,
    daily_price_col: str = "Close",
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Builds a clean dataframe of Fundamentals.

    If 'target_index' is provided (e.g. technical_df.index), the function
    returns a DENSE daily dataframe matching that index.
    Otherwise, it returns the SPARSE quarterly dataframe.

    If `daily_prices` is provided as a (date, symbol) panel with a close column,
    price-linked valuation features (P/E, P/B, MarketCap, EV-based multiples, etc.)
    are recomputed daily while statement-linked inputs remain as-of filled.
    """
    # 1. Fetch Sparse Data
    effective_limit = _resolve_fundamental_limit(
        start_date=start_date,
        end_date=end_date,
        period=period,
    )
    if verbose:
        print(f"[fundamentals] Using computed limit={effective_limit} for date range.")

    df = fetch_fundamentals_data(
        symbols=list(symbols),
        api_key=ctx.api_key,
        period=period,
        limit=effective_limit,
        verbose=verbose,
        use_filing_lag=use_filing_lag,
        filing_lag_days=filing_lag_days,
    )
    src_start, src_end = _extract_date_bounds(df)

    df = _filter_df_by_date_bounds(
        df,
        start_date=start_date,
        end_date=end_date,
    )

    # 2. Identify Features
    feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # 3. Broadcast to Daily (if requested)
    if target_index is not None and not df.empty:
        sparse_effective_index = df.index if isinstance(df.index, pd.MultiIndex) else None
        if verbose: print("[fundamentals] Broadcasting to daily frequency (smearing forward)...")
        df = broadcast_fundamentals_to_daily(df, target_index)
        df = _filter_df_by_date_bounds(
            df,
            start_date=start_date,
            end_date=end_date,
        )
        if sparse_effective_index is not None:
            df = _add_fundamental_update_timing_features(
                df,
                sparse_effective_index=sparse_effective_index,
            )

        # Re-verify features (ensure types didn't break during merge)
        feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        close_daily = _extract_daily_close_series(
            daily_prices,
            target_index=target_index,
            preferred_col=daily_price_col,
        )
        if close_daily is not None:
            df, updated = _recompute_price_linked_fundamentals_daily(df, close_daily)
            if verbose and updated:
                names = ", ".join(sorted({_to_pascal_case(c) for c in updated}))
                print(f"[fundamentals] Recomputed daily price-linked features ({len(updated)}): {names}")
            feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        elif verbose and daily_prices is not None:
            print(
                f"[fundamentals] Skipped daily valuation recompute: could not find "
                f"'{daily_price_col}'/close column on a MultiIndex (date, symbol)."
            )

    df, feature_cols = _camelize_dataframe_columns(df, feature_cols)
    df, feature_cols = _drop_source_duplicate_columns(df, feature_cols)
    if compact_feature_names:
        df, feature_cols = _compact_dataframe_columns(df, feature_cols)
    if src_start is not None and src_end is not None:
        df.attrs["source_start_date"] = src_start
        df.attrs["source_end_date"] = src_end

    if verbose and not df.empty:
        _summarize_dataset_for_llm(df, feature_cols)

    return df, feature_cols


def train_model(
    *,
    trainer,
    dataset: DatasetArtifacts,
    feature_cols: Sequence[str],
    target_col: str,
    weight_col: Optional[str] = None,
) -> ModelArtifact:
    return trainer.fit(
        train_df=dataset.training_df,
        feature_cols=list(feature_cols),
        target_col=str(target_col),
        weight_col=str(weight_col) if weight_col else None,
    )


def predict_panel(
    *,
    predictor,
    model_artifact: ModelArtifact,
    panel: pd.DataFrame,
) -> PredictionsArtifact:
    panel = ensure_panel_index(panel)
    return predictor.predict(model_artifact=model_artifact, panel=panel)


def backtest(
    *,
    panel: pd.DataFrame,
    strategy,
    title: Optional[str] = None,
    **engine_kwargs: Any,
):
    panel = ensure_panel_index(panel)
    return run_backtest(panel=panel, strategy=strategy, title=title, **engine_kwargs)
def build_macro_dataframe(
        *,
        ctx,
        start_date: str,
        end_date: str,
        config: Optional[MacroFeatureConfig] = None,
        target_index: Optional[pd.Index] = None,
        verbose: bool = True,
        compact_feature_names: bool = False,
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Builds a clean dataframe of Macroeconomic features.

    If 'target_index' is provided, it broadcasts the macro data to match
    the shape (Date, Symbol) of the target.
    """
    df = fetch_macro_series(
        api_key=ctx.api_key,
        start_date=start_date,
        end_date=end_date,
        config=config,
        verbose=verbose
    )
    src_start, src_end = _extract_date_bounds(df)

    # Broadcast if requested
    if target_index is not None and not df.empty:
        if verbose: print("[macro] Broadcasting to daily frequency...")
        df = broadcast_macro_to_daily(df, target_index)

    feature_cols = list(df.columns)
    df, feature_cols = _camelize_dataframe_columns(df, feature_cols)
    if compact_feature_names:
        df, feature_cols = _compact_dataframe_columns(df, feature_cols)
    if src_start is not None and src_end is not None:
        df.attrs["source_start_date"] = src_start
        df.attrs["source_end_date"] = src_end

    if verbose and not df.empty:
        _summarize_dataset_for_llm(df, feature_cols)

    return df, feature_cols


def build_time_dataframe(
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        target_index: Optional[pd.Index] = None,
        config: Optional[TimeFeatureConfig] = None,
        verbose: bool = True,
        compact_feature_names: bool = False,
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Builds a clean dataframe of numeric calendar features.

    If 'target_index' is provided, the output index matches it
    (supports DatetimeIndex or MultiIndex with a 'date' level).
    """
    df = build_time_features(
        start_date=start_date,
        end_date=end_date,
        target_index=target_index,
        config=config,
    )

    feature_cols = list(df.columns)
    df, feature_cols = _camelize_dataframe_columns(df, feature_cols)
    if compact_feature_names:
        df, feature_cols = _compact_dataframe_columns(df, feature_cols)

    if verbose and not df.empty:
        _summarize_dataset_for_llm(df, feature_cols)

    return df, feature_cols

def _summarize_labels_for_llm(df: pd.DataFrame, dedup_stats: Optional[Dict] = None) -> None:
    """Prints a structured table for Oracle performance and deduplication stats."""
    print("\n" + "=" * 80)
    print("  ORACLE LABEL PERFORMANCE & DEDUPLICATION SUMMARY")
    print("=" * 80)

    if dedup_stats:
        print(f"DEDUPLICATION METRICS:")
        print(f"  - Raw Signal Count:    {dedup_stats['raw_count']:,}")
        print(f"  - Unique Signal Count: {dedup_stats['unique_count']:,}")
        print(f"  - Redundancy Removed:  {dedup_stats['pct_removed']:.1f}%")
        print("-" * 80)

    if "trade_return" not in df.columns or "side" not in df.columns:
        print("Missing required columns for performance statistics.")
        return

    report_rows = []
    horizons = sorted(df['horizon'].unique())

    for horizon in horizons:
        h_df = df[df['horizon'] == horizon]
        for side in ["long", "short"]:
            s_df = h_df[h_df['side'] == side]
            if s_df.empty: continue

            n_trades = len(s_df) // 2
            avg_ret = s_df['trade_return'].mean()
            win_rate = (s_df['trade_return'] > 0).mean()

            avg_dur = np.nan
            if 'trade_duration_days' in s_df.columns:
                avg_dur = s_df['trade_duration_days'].mean()

            report_rows.append({
                "Horizon": horizon,
                "Side": "BUY" if side == "long" else "SHORT",
                "Trades": n_trades,
                "Mean Return %": round(avg_ret * 100, 2),
                "Win Rate %": round(win_rate * 100, 1),
                "Avg Duration": round(avg_dur, 1)
            })

    summary_table = pd.DataFrame(report_rows)
    if not summary_table.empty:
        # Numeric sort logic
        def _extract_k(h):
            import re
            match = re.search(r'k(\d+)', str(h))
            return int(match.group(1)) if match else 0

        summary_table["_k_val"] = summary_table["Horizon"].apply(_extract_k)
        summary_table = summary_table.sort_values(["_k_val", "Side"], ascending=[True, False]).drop(columns=["_k_val"])
        print(summary_table.to_string(index=False))

    print("=" * 80 + "\n")


def build_label_dataframe(
        *,
        daily_by_symbol: Dict[str, pd.DataFrame],
        k_params: Dict[str, Union[int, List[int]]],
        execution_params: Dict[str, Any],
        weighting: Dict[str, Any],
        add_rank_labels: bool = True,
        deduplicate: bool = True,
        verbose: bool = True,
) -> pd.DataFrame:
    """Standard API entry point with deduplication tracking."""

    # To track stats, we run the logic once
    df_raw = build_label_panel(
        daily_by_symbol=daily_by_symbol,
        solve_longs_by_frequency_fn=solve_longs_by_frequency,
        solve_shorts_by_frequency_fn=solve_shorts_by_frequency,
        k_params=k_params,
        execution_params=execution_params,
        weighting=weighting,
        add_rank_labels=False,  # Rank happens after dedup
        deduplicate=False  # We want raw count first
    )

    raw_count = len(df_raw)

    # 2. Apply Deduplication and Ranking
    if deduplicate:
        df_final = deduplicate_labels(df_raw)
    else:
        df_final = df_raw

    if add_rank_labels:
        df_final = add_rank_regression_labels(df_final)

    unique_count = len(df_final)
    pct_removed = ((raw_count - unique_count) / raw_count * 100) if raw_count > 0 else 0

    stats = {
        "raw_count": raw_count,
        "unique_count": unique_count,
        "pct_removed": pct_removed
    }

    if verbose and not df_final.empty:
        _summarize_labels_for_llm(df_final, dedup_stats=stats if deduplicate else None)

    return df_final

def prepare_ml_dataset(
        *,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        target_cols: Union[str, List[str]] = "target",
        weight_col: Optional[str] = "sample_weight",
        drop_nan_features: bool = True,
        verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    return _prepare_ml_dataset_unified(
        features_df=features_df,
        labels_df=labels_df,
        target_cols=target_cols,
        weight_col=weight_col,
        config=MLDatasetConfig(drop_nan_features=bool(drop_nan_features)),
        verbose=verbose,
    )


def prepare_entry2exit_dataset(
    *,
    features_df: pd.DataFrame,
    labels_df: Optional[pd.DataFrame] = None,
    trades_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[Sequence[str]] = None,
    numeric_precision: int = 2,
    scientific_for_large_numbers: bool = True,
    scientific_threshold: float = 1_000_000.0,
    dedupe_source_duplicate_features: bool = True,
    compact_feature_names: bool = False,
    drop_missing_entry_rows: bool = True,
) -> pd.DataFrame:
    """
    Build canonical Entry->Exit text dataset from features + optimal trades.

    Provide exactly one of:
      - labels_df (will be converted to trades via labels_panel_to_trades_df), or
      - trades_df (already prepared trade pairs).
    """
    return _prepare_entry2exit_dataset_unified(
        features_df=features_df,
        labels_df=labels_df,
        trades_df=trades_df,
        feature_cols=feature_cols,
        config=Entry2ExitTextConfig(
            numeric_precision=int(numeric_precision),
            scientific_for_large_numbers=bool(scientific_for_large_numbers),
            scientific_threshold=float(scientific_threshold),
            dedupe_source_duplicate_features=bool(dedupe_source_duplicate_features),
            compact_feature_names=bool(compact_feature_names),
            drop_missing_entry_rows=bool(drop_missing_entry_rows),
        ),
    )
