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


def build_directional_option_order_plan(
    ranked_directions: Sequence[Mapping[str, Any]],
    selected_contracts: Sequence[Mapping[str, Any]],
    current_positions: Sequence[Mapping[str, Any]] | None = None,
    *,
    max_underlyings: int = 20,
) -> list[dict[str, Any]]:
    """Reconcile ranker-selected calls/puts against meta_stack directions."""
    if not 1 <= int(max_underlyings) <= 20:
        raise ValueError("max_underlyings must be between 1 and 20")

    directions: dict[str, str] = {}
    for row in ranked_directions:
        underlying = str(row.get("symbol") or row.get("underlying_symbol") or "").strip().upper()
        direction = str(row.get("direction") or row.get("meta_stack_direction") or "").strip().lower()
        if not underlying or underlying in directions:
            continue
        if direction not in {"long", "short", "hold"}:
            raise ValueError(f"Invalid meta_stack direction for {underlying}: {direction!r}")
        directions[underlying] = direction

    selections: dict[tuple[str, str], dict[str, Any]] = {}
    selected_underlyings: set[str] = set()
    for raw in selected_contracts:
        row = dict(raw)
        underlying = str(row.get("underlying_symbol") or row.get("symbol") or "").strip().upper()
        contract_symbol = str(row.get("contract_symbol") or row.get("option_symbol") or "").strip().upper()
        option_type = str(row.get("option_type") or row.get("type") or "").strip().lower()
        if not underlying or not contract_symbol or option_type not in {"call", "put"}:
            raise ValueError("Each selected contract requires an underlying, contract symbol, and call/put type.")
        key = (underlying, option_type)
        if key in selections:
            raise ValueError(f"Multiple selected {option_type} contracts for {underlying}")
        selections[key] = {**row, "underlying_symbol": underlying, "contract_symbol": contract_symbol, "option_type": option_type}
        selected_underlyings.add(underlying)
    if len(selected_underlyings) > int(max_underlyings):
        raise ValueError(f"Option selections contain {len(selected_underlyings)} unique underlyings; limit is {max_underlyings}.")
    unknown = sorted(selected_underlyings.difference(directions))
    if unknown:
        raise ValueError(f"Missing meta_stack directions for selected contracts: {unknown}")

    positions: list[dict[str, Any]] = []
    held_underlyings: set[str] = set()
    for raw in current_positions or []:
        row = dict(raw)
        underlying = str(row.get("underlying_symbol") or "").strip().upper()
        contract_symbol = str(row.get("contract_symbol") or row.get("option_symbol") or row.get("symbol") or "").strip().upper()
        option_type = str(row.get("option_type") or row.get("type") or "").strip().lower()
        quantity = float(row.get("qty") or row.get("quantity") or 0)
        if not underlying or not contract_symbol or option_type not in {"call", "put"} or quantity == 0:
            raise ValueError("Each current option position requires an underlying, contract symbol, call/put type, and quantity.")
        positions.append({**row, "underlying_symbol": underlying, "contract_symbol": contract_symbol, "option_type": option_type, "qty": quantity})
        held_underlyings.add(underlying)
    if len(held_underlyings) > int(max_underlyings):
        raise ValueError(f"Current option account has {len(held_underlyings)} unique underlyings; limit is {max_underlyings}.")
    missing = sorted(held_underlyings.difference(directions))
    if missing:
        raise ValueError(f"Missing meta_stack directions for current option positions: {missing}")

    retained_contracts: set[str] = set()
    orders: list[dict[str, Any]] = []
    for position in positions:
        underlying = position["underlying_symbol"]
        option_type = position["option_type"]
        direction = directions[underlying]
        desired_type = "call" if direction == "long" else "put" if direction == "short" else None
        selected = selections.get((underlying, desired_type)) if desired_type else None
        retain = direction == "hold" or (
            option_type == desired_type and selected is not None and position["contract_symbol"] == selected["contract_symbol"]
        )
        if retain:
            retained_contracts.add(position["contract_symbol"])
            continue
        orders.append({
            "symbol": position["contract_symbol"],
            "underlying_symbol": underlying,
            "option_type": option_type,
            "action": f"sell_to_close_{option_type}",
            "side": "sell",
            "qty": abs(position["qty"]),
        })

    for underlying, direction in directions.items():
        if direction == "hold":
            continue
        option_type = "call" if direction == "long" else "put"
        selected = selections.get((underlying, option_type))
        if selected is None:
            raise ValueError(f"Missing selected {option_type} contract for {underlying} {direction} signal")
        if selected["contract_symbol"] in retained_contracts:
            continue
        orders.append({
            "symbol": selected["contract_symbol"],
            "underlying_symbol": underlying,
            "option_type": option_type,
            "action": f"buy_to_open_{option_type}",
            "side": "buy",
            "qty": int(selected.get("qty") or 1),
        })
    return orders


def build_directional_equity_order_plan(
    ranked_directions: Sequence[Mapping[str, Any]],
    latest_prices: Mapping[str, float],
    current_positions: Mapping[str, float] | None = None,
    *,
    portfolio_value: float,
    max_positions: int = 20,
    gross_exposure: float = 0.95,
) -> list[dict[str, Any]]:
    """Reconcile ranked long/short/hold signals without exceeding account capacity."""
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive")
    if not 0 < gross_exposure <= 1:
        raise ValueError("gross_exposure must be in (0, 1]")
    if not 1 <= int(max_positions) <= 20:
        raise ValueError("max_positions must be between 1 and 20")

    signals: list[tuple[str, str]] = []
    seen: set[str] = set()
    for row in ranked_directions:
        symbol = str(row.get("symbol") or "").strip().upper()
        direction = str(row.get("direction") or row.get("meta_stack_direction") or "").strip().lower()
        if not symbol or symbol in seen:
            continue
        if direction not in {"long", "short", "hold"}:
            raise ValueError(f"Invalid meta_stack direction for {symbol}: {direction!r}")
        signals.append((symbol, direction))
        seen.add(symbol)

    positions = {
        str(symbol).strip().upper(): float(quantity)
        for symbol, quantity in dict(current_positions or {}).items()
        if str(symbol).strip() and float(quantity) != 0
    }
    if len(positions) > int(max_positions):
        raise ValueError(f"Current account has {len(positions)} unique positions; limit is {max_positions}.")
    missing = sorted(set(positions).difference(seen))
    if missing:
        raise ValueError(f"Missing meta_stack directions for current positions: {missing}")

    direction_by_symbol = dict(signals)
    retained = {
        symbol
        for symbol, quantity in positions.items()
        if direction_by_symbol[symbol] == "hold"
        or (direction_by_symbol[symbol] == "long" and quantity > 0)
        or (direction_by_symbol[symbol] == "short" and quantity < 0)
    }
    candidates = [(symbol, direction) for symbol, direction in signals if direction != "hold" and symbol not in retained]
    selected = candidates[: max(0, int(max_positions) - len(retained))]

    orders: list[dict[str, Any]] = []
    selected_symbols = {symbol for symbol, _ in selected}
    for symbol, quantity in positions.items():
        if symbol in retained:
            continue
        orders.append(
            {
                "symbol": symbol,
                "action": "close_opposite_signal" if symbol in selected_symbols else "close_for_capacity",
                "side": "sell" if quantity > 0 else "buy",
                "qty": int(abs(quantity)),
                "order_type": "market",
                "time_in_force": "day",
            }
        )

    target_count = len(retained) + len(selected)
    dollars_per_position = float(portfolio_value) * float(gross_exposure) / target_count if target_count else 0.0
    for symbol, direction in selected:
        price = float(latest_prices.get(symbol, 0.0) or 0.0)
        if price <= 0:
            raise ValueError(f"Missing positive latest price for {symbol}")
        quantity = int(dollars_per_position // price)
        if quantity <= 0:
            continue
        orders.append(
            {
                "symbol": symbol,
                "action": f"open_{direction}",
                "side": "buy" if direction == "long" else "sell",
                "qty": quantity,
                "order_type": "market",
                "time_in_force": "day",
            }
        )
    return orders


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
