from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


PAPER_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"


def build_equal_weight_order_plan(
    target_symbols: Sequence[str],
    latest_prices: Mapping[str, float],
    current_positions: Mapping[str, float] | None = None,
    *,
    portfolio_value: float,
    gross_exposure: float = 0.95,
    liquidate_unselected: bool = True,
) -> list[dict[str, Any]]:
    """Build integer-share market orders for an equal-weight long-only portfolio."""
    symbols = list(dict.fromkeys(str(symbol).strip().upper() for symbol in target_symbols if str(symbol).strip()))
    positions = {
        str(symbol).strip().upper(): float(qty)
        for symbol, qty in dict(current_positions or {}).items()
        if str(symbol).strip()
    }
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive")
    if not 0 < gross_exposure <= 1:
        raise ValueError("gross_exposure must be in (0, 1]")
    if not symbols:
        raise ValueError("target_symbols must contain at least one symbol")

    dollars_per_symbol = float(portfolio_value) * float(gross_exposure) / len(symbols)
    target_qty: dict[str, int] = {}
    for symbol in symbols:
        price = float(latest_prices.get(symbol, 0.0) or 0.0)
        if price <= 0:
            raise ValueError(f"Missing positive latest price for {symbol}")
        target_qty[symbol] = int(dollars_per_symbol // price)

    orders: list[dict[str, Any]] = []
    all_symbols = set(target_qty)
    if liquidate_unselected:
        all_symbols.update(symbol for symbol, qty in positions.items() if qty > 0)

    for symbol in sorted(all_symbols):
        current = int(positions.get(symbol, 0.0))
        target = int(target_qty.get(symbol, 0))
        delta = target - current
        if delta == 0:
            continue
        orders.append(
            {
                "symbol": symbol,
                "side": "buy" if delta > 0 else "sell",
                "qty": abs(delta),
                "current_qty": current,
                "target_qty": target,
                "order_type": "market",
                "time_in_force": "day",
            }
        )
    return orders


@dataclass(frozen=True)
class AlpacaPaperClient:
    api_key: str
    api_secret: str
    base_url: str = PAPER_BASE_URL
    data_base_url: str = DATA_BASE_URL
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "AlpacaPaperClient":
        api_key = str(os.getenv("ALPACA_PAPER_API_KEY") or "").strip()
        api_secret = str(os.getenv("ALPACA_PAPER_API_SECRET") or "").strip()
        if not api_key or not api_secret:
            raise RuntimeError(
                "Set ALPACA_PAPER_API_KEY and ALPACA_PAPER_API_SECRET before using Alpaca paper trading."
            )
        return cls(api_key=api_key, api_secret=api_secret)

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        data_api: bool = False,
    ) -> Any:
        base_url = self.base_url.rstrip("/")
        if base_url != PAPER_BASE_URL:
            raise RuntimeError(f"Refusing non-paper Alpaca URL: {base_url}")
        request_base_url = self.data_base_url.rstrip("/") if data_api else base_url
        body = None if payload is None else json.dumps(dict(payload)).encode("utf-8")
        request = urllib.request.Request(
            f"{request_base_url}{path}",
            data=body,
            method=method.upper(),
            headers={
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Alpaca paper API {exc.code}: {detail}") from exc
        return json.loads(raw) if raw else None

    def get_account(self) -> dict[str, Any]:
        return dict(self._request("GET", "/v2/account"))

    def get_positions(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/v2/positions"))

    def get_open_orders(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/v2/orders?status=open&direction=asc"))

    def get_option_contracts(
        self,
        underlying_symbol: str,
        *,
        option_type: str = "call",
        expiration_date_gte: str | None = None,
        expiration_date_lte: str | None = None,
    ) -> list[dict[str, Any]]:
        params = {
            "underlying_symbols": str(underlying_symbol).strip().upper(),
            "status": "active",
            "type": str(option_type).strip().lower(),
            "limit": 10000,
        }
        if expiration_date_gte:
            params["expiration_date_gte"] = str(expiration_date_gte)
        if expiration_date_lte:
            params["expiration_date_lte"] = str(expiration_date_lte)
        path = "/v2/options/contracts?" + urllib.parse.urlencode(params)
        response = dict(self._request("GET", path) or {})
        return list(response.get("option_contracts") or response.get("contracts") or [])

    def get_option_contract(self, symbol_or_id: str) -> dict[str, Any]:
        encoded = urllib.parse.quote(str(symbol_or_id).strip(), safe="")
        return dict(self._request("GET", f"/v2/options/contracts/{encoded}") or {})

    def get_option_snapshots(self, symbols: Sequence[str]) -> dict[str, Any]:
        normalized = list(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()))
        if not normalized:
            return {}
        snapshots: dict[str, Any] = {}
        for start in range(0, len(normalized), 100):
            params = urllib.parse.urlencode({"symbols": ",".join(normalized[start : start + 100])})
            response = dict(
                self._request(
                    "GET",
                    f"/v1beta1/options/snapshots?{params}",
                    data_api=True,
                )
                or {}
            )
            snapshots.update(response.get("snapshots") or {})
        return snapshots

    def cancel_order(self, order_id: str) -> None:
        encoded = urllib.parse.quote(str(order_id).strip(), safe="")
        self._request("DELETE", f"/v2/orders/{encoded}")

    def submit_order(self, order: Mapping[str, Any]) -> dict[str, Any]:
        payload = {
            "symbol": str(order["symbol"]),
            "side": str(order["side"]),
            "qty": str(int(order["qty"])),
            "type": str(order.get("order_type") or "market"),
            "time_in_force": str(order.get("time_in_force") or "day"),
        }
        if order.get("limit_price") is not None:
            payload["limit_price"] = str(round(float(order["limit_price"]), 2))
        return dict(self._request("POST", "/v2/orders", payload))

    def submit_orders(self, orders: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for order in orders:
            if str(order.get("action") or "").startswith("cancel_"):
                self.cancel_order(str(order.get("order_id") or ""))
                results.append({"id": order.get("order_id"), "status": "canceled", **dict(order)})
            else:
                results.append(self.submit_order(order))
        return results
