from __future__ import annotations

import re
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data import PRETTY_NAME_MAP

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
    print("\n" + "=" * 60)
    print("  LLM DATA DEBUG REPORT  ")
    print("=" * 60)

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

    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        print("WARNING: No numeric columns found.")
        return

    nan_counts = num_df.isna().sum()
    nan_pcts = (nan_counts / n_rows) * 100
    bad_cols = nan_pcts[nan_pcts > 0].sort_values(ascending=False)

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

    inf_counts = np.isinf(num_df).sum()
    inf_cols = inf_counts[inf_counts > 0].sort_values(ascending=False)

    if not inf_cols.empty:
        print(f"\n[INFINITE VALUES] {len(inf_cols)} cols have +/- inf.")
        print("Top 5 worst offenders:")
        for col, count in inf_cols.head(5).items():
            print(f"  - {col:<30} : {count} rows")
    else:
        print("\n[INFINITE VALUES] None detected (Good).")

    const_cols = [c for c in num_df.columns if num_df[c].nunique() <= 1]
    if const_cols:
        print(f"\n[CONSTANT COLS] {len(const_cols)} cols have zero variance (useless features).")
        print(f"  - Examples: {const_cols[:5]}")
    else:
        print("\n[CONSTANT COLS] None detected (Good).")

    print("=" * 60 + "\n")


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


__all__ = [
    "_camelize_dataframe_columns",
    "_compact_dataframe_columns",
    "_drop_source_duplicate_columns",
    "_extract_date_bounds",
    "_filter_df_by_date_bounds",
    "_summarize_dataset_for_llm",
    "summarize_technical_features",
]
