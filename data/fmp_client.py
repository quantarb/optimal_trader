# ============================================================
# modules/data/fmp_client.py
#
# FMP client with:
#   - correct routing: /stable/* is root; everything else under /api
#   - robust JSON parsing (handles "Invalid name" 200 responses)
#   - stable fundamentals: /stable/key-metrics and /stable/ratios
#   - macro: /stable/economic-indicators and /stable/treasury-rates
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional
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
    """Raised when FMP returns a 200 response with body 'Invalid name' for economic indicators."""
    pass


class FMPClient:
    """
    Minimal-but-solid FMP client.

    Routing rules:
      - /stable/* endpoints are *NOT* under /api
      - everything else is under /api
    """

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

    # -----------------------------
    # Core HTTP helpers
    # -----------------------------
    def _make_url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        if path.startswith("/stable/"):
            return self._base_root + path
        return self._base_api + path

    def get_json(self, path: str, *, params: Optional[Mapping[str, Any]] = None) -> Any:
        """
        Fetch JSON with retries and helpful errors.

        IMPORTANT:
          FMP sometimes returns HTTP 200 but a plain-text body like "Invalid name"
          (not JSON). We detect that and raise FMPInvalidNameError so callers can
          try alternate names.
        """
        url = self._make_url(path)
        p: Dict[str, Any] = dict(params or {})
        p["apikey"] = self.api_key

        last_err: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            if self.sleep_s > 0:
                time.sleep(self.sleep_s)

            r = self._session.get(url, params=p, timeout=self.timeout_s)

            # Retry on throttling/transient server errors
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code} (attempt {attempt}): {r.text[:200]}"
                time.sleep(0.5 * attempt)
                continue

            if r.status_code != 200:
                raise RuntimeError(
                    f"FMP request failed: {path} HTTP {r.status_code}\n"
                    f"URL: {r.url}\n"
                    f"Content-Type: {r.headers.get('Content-Type')}\n"
                    f"Body (first 400 chars):\n{r.text[:400]}"
                )

            # Handle "Invalid name" which FMP returns as plain text (still HTTP 200)
            body = (r.text or "").strip()
            if body.lower() == "invalid name":
                raise FMPInvalidNameError(
                    f"FMP returned 'Invalid name' for {path}\nURL: {r.url}"
                )

            # Normal JSON parse
            try:
                return r.json()
            except Exception as e:
                # Still not JSON: raise with context
                raise RuntimeError(
                    f"FMP returned non-JSON for {path} (HTTP 200)\n"
                    f"URL: {r.url}\n"
                    f"Content-Type: {r.headers.get('Content-Type')}\n"
                    f"Body (first 400 chars):\n{r.text[:400]}\n"
                    f"Parse error: {repr(e)}"
                )

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

    # -----------------------------
    # Macro (stable) endpoints
    # -----------------------------
    def economic_indicators(
        self,
        name: str,
        *,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        https://financialmodelingprep.com/stable/economic-indicators?name=GDP
        """
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
        """
        https://financialmodelingprep.com/stable/treasury-rates
        """
        return self._stable_df("/stable/treasury-rates", from_date=from_date, to_date=to_date)

    # -----------------------------
    # Fundamentals (stable, non-legacy)
    # -----------------------------
    def key_metrics(self, symbol: str, *, period: str = "quarter", limit: int = 400) -> pd.DataFrame:
        return self._stable_symbol_period_limit(
            "/stable/key-metrics",
            symbol=symbol,
            period=period,
            limit=limit,
        )

    def ratios(self, symbol: str, *, period: str = "quarter", limit: int = 400) -> pd.DataFrame:
        return self._stable_symbol_period_limit(
            "/stable/ratios",
            symbol=symbol,
            period=period,
            limit=limit,
        )


def fundamentals_to_daily_panel(
    fundamentals_df: pd.DataFrame,
    *,
    symbols: Optional[Iterable[str]] = None,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Convert sparse fundamentals into a daily as-of-filled panel.

    Expected fundamentals_df columns:
      - symbol
      - date (already normalized upstream)
      - numeric fundamental feature columns

    Returns:
      DataFrame indexed by ['date', 'symbol'] with fundamentals forward-filled.
      Never back-fills (no look-ahead).
    """
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
        sym_set = {str(s) for s in symbols}
        df = df[df["symbol"].isin(sym_set)]

    if df.empty:
        return pd.DataFrame()

    min_date = pd.to_datetime(start_date) if start_date is not None else df["date"].min()
    max_date = pd.to_datetime(end_date) if end_date is not None else df["date"].max()

    all_days = pd.date_range(min_date, max_date, freq="D")

    panels: List[pd.DataFrame] = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.drop(columns=["symbol"], errors="ignore").set_index("date").sort_index()
        g_daily = g.reindex(all_days).ffill()
        g_daily.index.name = "date"
        g_daily = g_daily.reset_index()
        g_daily["symbol"] = sym
        panels.append(g_daily)

    out = pd.concat(panels, axis=0, ignore_index=True)
    out = out.set_index(["date", "symbol"]).sort_index()
    return out
