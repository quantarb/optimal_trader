from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional
import time

import pandas as pd
import requests


@dataclass(frozen=True)
class FMPClientConfig:
    api_key: str
    sleep_s: float = 0.0
    timeout_s: float = 30.0
    max_retries: int = 3


class FMPInvalidNameError(RuntimeError):
    """Raised when FMP returns a 200 response with body 'Invalid name'."""


class FMPClient:
    """Thin FinancialModelingPrep client used by research workflows."""

    def __init__(
        self,
        api_key: str,
        *,
        sleep_s: float = 0.0,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = str(api_key)
        self.sleep_s = float(sleep_s)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self._session = session or requests.Session()
        self._base_api = "https://financialmodelingprep.com/api"
        self._base_root = "https://financialmodelingprep.com"

    def _make_url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        if path.startswith("/stable/"):
            return self._base_root + path
        return self._base_api + path

    def get_json(self, path: str, *, params: Optional[Mapping[str, Any]] = None) -> Any:
        url = self._make_url(path)
        payload: Dict[str, Any] = dict(params or {})
        payload["apikey"] = self.api_key
        last_err: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            if self.sleep_s > 0:
                time.sleep(self.sleep_s)
            response = self._session.get(url, params=payload, timeout=self.timeout_s)
            if response.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {response.status_code} (attempt {attempt}): {response.text[:200]}"
                time.sleep(0.5 * attempt)
                continue
            if response.status_code != 200:
                raise RuntimeError(
                    f"FMP request failed: {path} HTTP {response.status_code}\n"
                    f"URL: {response.url}\n"
                    f"Content-Type: {response.headers.get('Content-Type')}\n"
                    f"Body (first 400 chars):\n{response.text[:400]}"
                )
            body = (response.text or "").strip()
            if body.lower() == "invalid name":
                raise FMPInvalidNameError(f"FMP returned 'Invalid name' for {path}\nURL: {response.url}")
            try:
                return response.json()
            except Exception as exc:
                raise RuntimeError(
                    f"FMP returned non-JSON for {path} (HTTP 200)\n"
                    f"URL: {response.url}\n"
                    f"Content-Type: {response.headers.get('Content-Type')}\n"
                    f"Body (first 400 chars):\n{response.text[:400]}\n"
                    f"Parse error: {repr(exc)}"
                ) from exc
        raise RuntimeError(f"FMP request failed after retries: {path}\nLast error: {last_err}")

    def get_df(self, path: str, *, params: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
        data = self.get_json(path, params=params)
        if data is None:
            return pd.DataFrame()
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            if "Error Message" in data:
                return pd.DataFrame([data])
            return pd.DataFrame([data])
        return pd.DataFrame()

    def _stable_df(
        self,
        path: str,
        *,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        extra_params: Optional[Mapping[str, Any]] = None,
    ) -> pd.DataFrame:
        params: Dict[str, Any] = dict(extra_params or {})
        if from_date:
            params["from"] = str(from_date)
        if to_date:
            params["to"] = str(to_date)
        return self.get_df(path, params=params)

    def _stable_symbol_period_limit(
        self,
        path: str,
        *,
        symbol: str,
        period: str = "quarter",
        limit: int = 400,
    ) -> pd.DataFrame:
        params: Dict[str, Any] = {"symbol": str(symbol)}
        if period:
            params["period"] = str(period)
        if limit:
            params["limit"] = int(limit)
        return self.get_df(path, params=params)

    def economic_indicators(
        self,
        name: str,
        *,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        return self._stable_df(
            "/stable/economic-indicators",
            from_date=from_date,
            to_date=to_date,
            extra_params={"name": str(name)},
        )

    def treasury_rates(
        self,
        *,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        return self._stable_df("/stable/treasury-rates", from_date=from_date, to_date=to_date)

    def key_metrics(self, symbol: str, *, period: str = "quarter", limit: int = 400) -> pd.DataFrame:
        return self._stable_symbol_period_limit("/stable/key-metrics", symbol=symbol, period=period, limit=limit)

    def ratios(self, symbol: str, *, period: str = "quarter", limit: int = 400) -> pd.DataFrame:
        return self._stable_symbol_period_limit("/stable/ratios", symbol=symbol, period=period, limit=limit)


def fundamentals_to_daily_panel(
    fundamentals_df: pd.DataFrame,
    *,
    symbols: Optional[Iterable[str]] = None,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Convert sparse fundamentals into a daily as-of filled panel."""

    if fundamentals_df is None or fundamentals_df.empty:
        return pd.DataFrame()
    df = fundamentals_df.copy()
    if "date" not in df.columns:
        raise ValueError("fundamentals_df must contain a 'date' column")
    if "symbol" not in df.columns:
        raise ValueError("fundamentals_df must contain a 'symbol' column")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["symbol"] = df["symbol"].astype(str)
    df = df.sort_values(["symbol", "date"])
    if symbols is not None:
        symbol_set = {str(symbol) for symbol in symbols}
        df = df[df["symbol"].isin(symbol_set)]

    if start_date is None:
        start_date = pd.Timestamp(df["date"].min())
    if end_date is None:
        end_date = pd.Timestamp(df["date"].max())
    date_index = pd.date_range(start=start_date, end=end_date, freq="D")

    frames: list[pd.DataFrame] = []
    for symbol, group in df.groupby("symbol", sort=True):
        group = group.set_index("date").sort_index()
        daily = group.reindex(date_index).ffill()
        daily["symbol"] = symbol
        daily.index.name = "date"
        frames.append(daily.reset_index())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).set_index(["date", "symbol"]).sort_index()
