from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from modules.data.fmp_client import FMPClient


def _to_df(raw: Any) -> pd.DataFrame:
    if raw is None:
        return pd.DataFrame()
    if isinstance(raw, pd.DataFrame):
        return raw
    if isinstance(raw, list):
        return pd.DataFrame(raw)
    if isinstance(raw, dict):
        return pd.DataFrame([raw])
    return pd.DataFrame()


def build_large_liquid_universe_single_call(
    *,
    api_key: str,
    marketCapMoreThan: Optional[float] = None,
    marketCapLowerThan: Optional[float] = None,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
    betaMoreThan: Optional[float] = None,
    betaLowerThan: Optional[float] = None,
    priceMoreThan: Optional[float] = None,
    priceLowerThan: Optional[float] = None,
    dividendMoreThan: Optional[float] = None,
    dividendLowerThan: Optional[float] = None,
    volumeMoreThan: Optional[float] = None,
    volumeLowerThan: Optional[float] = None,
    exchange: Optional[str] = None,
    country: Optional[str] = None,
    isEtf: Optional[bool] = None,
    isFund: Optional[bool] = None,
    isActivelyTrading: Optional[bool] = None,
    limit: int = 10_000,
    includeAllShareClasses: Optional[bool] = None,
) -> tuple[str, ...]:
    """
    Build a liquid equity universe from one FMP screener call using only
    official endpoint parameters.

    Returns:
      Tuple of symbols sorted alphabetically.
    """
    client = FMPClient(api_key=api_key)

    params: dict[str, Any] = {"limit": int(limit)}
    optional_params: dict[str, Any] = {
        "marketCapMoreThan": marketCapMoreThan,
        "marketCapLowerThan": marketCapLowerThan,
        "sector": sector,
        "industry": industry,
        "betaMoreThan": betaMoreThan,
        "betaLowerThan": betaLowerThan,
        "priceMoreThan": priceMoreThan,
        "priceLowerThan": priceLowerThan,
        "dividendMoreThan": dividendMoreThan,
        "dividendLowerThan": dividendLowerThan,
        "volumeMoreThan": volumeMoreThan,
        "volumeLowerThan": volumeLowerThan,
        "exchange": exchange,
        "country": country,
        "isEtf": isEtf,
        "isFund": isFund,
        "isActivelyTrading": isActivelyTrading,
        "includeAllShareClasses": includeAllShareClasses,
    }
    for k, v in optional_params.items():
        if v is not None:
            params[k] = v

    payload = client.get_json(
        "/stable/company-screener",
        params=params,
    )
    df = _to_df(payload)
    if df.empty:
        return tuple()

    symbol_col = "symbol" if "symbol" in df.columns else None
    if symbol_col is None:
        return tuple()

    syms = (
        df[symbol_col]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    return tuple(sorted(syms))
