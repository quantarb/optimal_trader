from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional
import time

import fmpsdk
import pandas as pd


@dataclass(frozen=True)
class FMPClientConfig:
    api_key: str
    sleep_s: float = 0.0
    timeout_s: float = 30.0
    max_retries: int = 3


class FMPInvalidNameError(RuntimeError):
    """Raised when FMP returns a 200 response with body 'Invalid name'."""


class FMPClient:
    """Thin fmpsdk-backed client used by research workflows."""

    def __init__(
        self,
        api_key: str,
        *,
        sleep_s: float = 0.0,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        session: Optional[Any] = None,
    ) -> None:
        self.api_key = str(api_key)
        self.sleep_s = float(sleep_s)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self._session = session

    def _sdk_endpoint(self, path: str) -> str:
        endpoint = str(path or "").strip()
        if endpoint.startswith("https://financialmodelingprep.com"):
            return endpoint
        if endpoint.startswith("stable/"):
            return endpoint[len("stable/") :]
        if endpoint.startswith("/stable/"):
            return endpoint[len("/stable/") :]
        return endpoint.lstrip("/")

    def get_json(self, path: str, *, params: Optional[Mapping[str, Any]] = None) -> Any:
        endpoint = self._sdk_endpoint(path)
        payload = self._sdk_params(params)
        last_err: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            if self.sleep_s > 0:
                time.sleep(self.sleep_s)
            try:
                data = fmpsdk.request(self.api_key, endpoint, **payload)
            except Exception as exc:
                last_err = repr(exc)
            else:
                if isinstance(data, str) and data.strip().lower() == "invalid name":
                    raise FMPInvalidNameError(f"FMP returned 'Invalid name' for {path}")
                if data is not None:
                    return data
                last_err = f"fmpsdk returned None (attempt {attempt})"
            if attempt < self.max_retries:
                time.sleep(0.5 * attempt)
        raise RuntimeError(f"FMP request failed after retries via fmpsdk: {path}\nLast error: {last_err}")

    def get_df(self, path: str, *, params: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
        return self._data_to_df(self.get_json(path, params=params))

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
        return self._named_sdk_df(
            fmpsdk.economic_indicator,
            name=str(name),
            from_date=from_date,
            to_date=to_date,
        )

    def treasury_rates(
        self,
        *,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        if not from_date or not to_date:
            return self._stable_df("/stable/treasury-rates", from_date=from_date, to_date=to_date)
        return self._named_sdk_df(fmpsdk.treasury_rates, from_date=from_date, to_date=to_date)

    def historical_sector_performance(
        self,
        sector: str,
        exchange: str,
        *,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        return self._stable_df(
            "/stable/historical-sector-performance",
            from_date=from_date,
            to_date=to_date,
            extra_params={"sector": str(sector), "exchange": str(exchange)},
        )

    def historical_industry_performance(
        self,
        industry: str,
        exchange: str,
        *,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        return self._stable_df(
            "/stable/historical-industry-performance",
            from_date=from_date,
            to_date=to_date,
            extra_params={"industry": str(industry), "exchange": str(exchange)},
        )

    def historical_sector_pe(
        self,
        sector: str,
        exchange: str,
        *,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        return self._stable_df(
            "/stable/historical-sector-pe",
            from_date=from_date,
            to_date=to_date,
            extra_params={"sector": str(sector), "exchange": str(exchange)},
        )

    def historical_industry_pe(
        self,
        industry: str,
        exchange: str,
        *,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        return self._stable_df(
            "/stable/historical-industry-pe",
            from_date=from_date,
            to_date=to_date,
            extra_params={"industry": str(industry), "exchange": str(exchange)},
        )

    def key_metrics(self, symbol: str, *, period: str = "quarter", limit: int = 400) -> pd.DataFrame:
        return self._named_sdk_df(fmpsdk.key_metrics, symbol=str(symbol), period=period, limit=limit)

    def ratios(self, symbol: str, *, period: str = "quarter", limit: int = 400) -> pd.DataFrame:
        return self._named_sdk_df(fmpsdk.financial_ratios, symbol=str(symbol), period=period, limit=limit)

    @staticmethod
    def _sdk_params(params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for key, value in dict(params or {}).items():
            if str(key).startswith("__") or value is None:
                continue
            payload[str(key)] = value
        return payload

    def _named_sdk_df(self, fn: Any, **kwargs: Any) -> pd.DataFrame:
        payload = self._sdk_params(kwargs)
        last_err: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            if self.sleep_s > 0:
                time.sleep(self.sleep_s)
            try:
                data = fn(self.api_key, **payload)
            except Exception as exc:
                last_err = repr(exc)
            else:
                if data is not None:
                    return self._data_to_df(data)
                last_err = f"fmpsdk returned None (attempt {attempt})"
            if attempt < self.max_retries:
                time.sleep(0.5 * attempt)
        raise RuntimeError(f"FMP request failed after retries via fmpsdk: {getattr(fn, '__name__', fn)}\nLast error: {last_err}")

    @staticmethod
    def _data_to_df(data: Any) -> pd.DataFrame:
        if data is None:
            return pd.DataFrame()
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            if "Error Message" in data:
                return pd.DataFrame([data])
            return pd.DataFrame([data])
        return pd.DataFrame()


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
