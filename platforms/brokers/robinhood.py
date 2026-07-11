from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from platforms.brokers.option_pricing import (
    normalize_option_limit_price,
    option_limit_tick_size,
)

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

_REPO_DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"
_BUY_OPTION_BID_MULTIPLIER = 1.00
_OPTION_CONTRACT_MULTIPLIER = 100.0
_ROBINHOOD_OPTION_API_ATTEMPTS = 3
_OPTION_MARKET_DATA_CANDIDATE_LIMIT = 16
ROBINHOOD_OPTION_PLAN_CODE_VERSION = "option_lookup_v2"


def _load_repo_env() -> None:
    if load_dotenv is not None:
        try:
            load_dotenv(dotenv_path=_REPO_DOTENV_PATH, override=True)
            return
        except Exception:
            pass
    if not _REPO_DOTENV_PATH.exists():
        return
    try:
        for raw_line in _REPO_DOTENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = str(key).strip()
            if not key:
                continue
            cleaned = str(value).strip().strip('"').strip("'")
            os.environ[key] = cleaned
    except Exception:
        return


_load_repo_env()


def _require_robin_stocks():
    try:
        import robin_stocks.robinhood as rh  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "The `robin_stocks` package is required for Robinhood automation. "
            "Install it with `pip install robin-stocks`."
        ) from exc
    return rh


def _coerce_robinhood_market_row(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, Mapping):
        return dict(payload)
    if isinstance(payload, pd.DataFrame):
        if payload.empty:
            return {}
        return _coerce_robinhood_market_row(payload.iloc[0].to_dict())
    if isinstance(payload, pd.Series):
        return payload.to_dict()
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return {}
        try:
            decoded = json.loads(text)
        except Exception:
            return {}
        return _coerce_robinhood_market_row(decoded)
    if isinstance(payload, (list, tuple)):
        if not payload:
            return {}
        first = payload[0]
        if isinstance(first, Mapping):
            return dict(first)
        if isinstance(first, (list, tuple)) and len(first) == 2:
            try:
                return dict(payload)
            except Exception:
                return _coerce_robinhood_market_row(first)
        return _coerce_robinhood_market_row(first)
    return {}


def _is_broken_pipe_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, BrokenPipeError):
            return True
        if isinstance(current, OSError) and getattr(current, "errno", None) == 32:
            return True
        if "broken pipe" in str(current).lower():
            return True
        current = current.__cause__ or current.__context__
    return False


def _call_robinhood_option_api(label: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, _ROBINHOOD_OPTION_API_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not _is_broken_pipe_error(exc) or attempt >= _ROBINHOOD_OPTION_API_ATTEMPTS:
                break
            time.sleep(0.25 * attempt)
    if last_exc is not None and _is_broken_pipe_error(last_exc):
        raise RuntimeError(
            f"Robinhood option API call failed during {label} after "
            f"{_ROBINHOOD_OPTION_API_ATTEMPTS} attempt(s): {last_exc}"
        ) from last_exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Robinhood option API call failed during {label}.")


def _is_robinhood_option_api_error(exc: BaseException) -> bool:
    return _is_broken_pipe_error(exc) or "robinhood option api call failed" in str(exc).lower()


def _positive_float(value: Any) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(number) and float(number) > 0.0:
        return float(number)
    return None


def _first_positive_float_from_keys(*payloads: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[float | None, str]:
    for payload in payloads:
        data = dict(payload or {})
        for key in keys:
            value = _positive_float(data.get(key))
            if value is not None:
                return value, key
    return None, ""


def _first_value_from_keys(*payloads: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[Any, str]:
    for payload in payloads:
        data = dict(payload or {})
        for key in keys:
            value = data.get(key)
            if value not in (None, ""):
                return value, key
    return None, ""


def _normalize_option_contract_quantity(quantity: Any) -> tuple[float | None, str]:
    qty = pd.to_numeric(pd.Series([quantity]), errors="coerce").iloc[0]
    if pd.isna(qty) or float(qty) <= 0.0:
        return None, "missing"
    qty_float = float(qty)
    share_equivalent_contracts = qty_float / float(_OPTION_CONTRACT_MULTIPLIER)
    if qty_float >= _OPTION_CONTRACT_MULTIPLIER and abs(share_equivalent_contracts - round(share_equivalent_contracts)) < 1e-9:
        return float(share_equivalent_contracts), "share_equivalent_divided_by_100"
    return qty_float, "contracts"


def _infer_robinhood_option_type(*values: Any) -> str:
    text = " ".join(str(value or "").strip().lower() for value in values)
    if "put" in text:
        return "put"
    if "call" in text:
        return "call"
    return ""


def _infer_robinhood_option_action(
    *,
    side: str,
    effect: str,
    option_type: str,
    opening_strategy: str,
    closing_strategy: str,
    direction: str,
) -> str:
    opt = str(option_type or "").strip().lower()
    if opt not in {"call", "put"}:
        return ""
    side_text = str(side or "").strip().lower()
    effect_text = str(effect or "").strip().lower()
    opening_text = str(opening_strategy or "").strip().lower()
    closing_text = str(closing_strategy or "").strip().lower()
    direction_text = str(direction or "").strip().lower()
    if side_text == "buy" and (effect_text == "open" or opening_text):
        return f"buy_to_open_{opt}"
    if side_text == "sell" and (effect_text == "close" or closing_text):
        return f"sell_to_close_{opt}"
    if opening_text and direction_text in {"debit", "buy"}:
        return f"buy_to_open_{opt}"
    if closing_text and direction_text in {"credit", "sell"}:
        return f"sell_to_close_{opt}"
    if opening_text and not closing_text:
        return f"buy_to_open_{opt}"
    if closing_text and not opening_text:
        return f"sell_to_close_{opt}"
    return ""


def _option_previous_close_price(payload: Mapping[str, Any] | pd.Series | dict[str, Any]) -> float | None:
    data = payload.to_dict() if isinstance(payload, pd.Series) else dict(payload or {})
    for key in ("previous_close_price", "previous_close", "prev_close", "previous_close_mark_price"):
        value = _positive_float(data.get(key))
        if value is not None:
            return value
    return None


def _multiplier_label(value: float) -> str:
    from decimal import Decimal

    decimal_value = Decimal(str(float(value))).normalize()
    text = format(decimal_value, "f").rstrip("0").rstrip(".")
    return text.replace("-", "neg_").replace(".", "_")


def _buy_option_bid_limit_source() -> str:
    return f"bid_price_x_{_multiplier_label(_BUY_OPTION_BID_MULTIPLIER)}"


def _buy_option_bid_limit_column() -> str:
    return _buy_option_bid_limit_source()


def _option_limit_tick_size(price: float | None) -> float | None:
    return option_limit_tick_size(price)


def _round_option_limit_price(price: float | None) -> float | None:
    return normalize_option_limit_price(price, side="nearest")


def _floor_option_limit_price(price: float | None) -> float | None:
    return normalize_option_limit_price(price, side="buy")


def _robinhood_strike_candidates(strike: Any) -> list[str]:
    number = pd.to_numeric(pd.Series([strike]), errors="coerce").iloc[0]
    if pd.isna(number):
        return []
    value = float(number)
    candidates = [str(value), f"{value:.2f}", f"{value:.4f}"]
    if value.is_integer():
        candidates.insert(0, str(int(value)))
    return list(dict.fromkeys(candidates))


def _lookup_robinhood_option_market_row(
    rh: Any,
    symbol: str,
    expiry: str,
    strike: Any,
    option_type: str,
) -> dict[str, Any]:
    clean_symbol = str(symbol or "").strip().upper()
    clean_expiry = str(expiry or "").strip()
    clean_type = str(option_type or "").strip().lower()
    if not clean_symbol or not clean_expiry or clean_type not in {"call", "put"}:
        return {}
    for strike_text in _robinhood_strike_candidates(strike):
        try:
            market_data = rh.get_option_market_data(clean_symbol, clean_expiry, strike_text, clean_type) or []
        except Exception:
            market_data = []
        market_row = _coerce_robinhood_market_row(market_data)
        if market_row:
            return market_row
    try:
        options = rh.find_options_by_expiration(
            clean_symbol,
            expirationDate=clean_expiry,
            optionType=clean_type,
        ) or []
    except Exception:
        return {}
    wanted_strike = pd.to_numeric(pd.Series([strike]), errors="coerce").iloc[0]
    if pd.isna(wanted_strike):
        return {}
    for item in options:
        option_row = _coerce_robinhood_market_row(item)
        option_strike = pd.to_numeric(pd.Series([option_row.get("strike_price")]), errors="coerce").iloc[0]
        if pd.notna(option_strike) and abs(float(option_strike) - float(wanted_strike)) < 1e-6:
            return option_row
    return {}


def _option_market_price_fields(market_row: Mapping[str, Any] | dict[str, Any]) -> dict[str, Any]:
    row = dict(market_row or {})
    bid_price = pd.to_numeric(pd.Series([row.get("bid_price")]), errors="coerce").iloc[0]
    ask_price = pd.to_numeric(pd.Series([row.get("ask_price")]), errors="coerce").iloc[0]
    mark_price = pd.to_numeric(pd.Series([row.get("adjusted_mark_price", row.get("mark_price"))]), errors="coerce").iloc[0]
    return {
        "bid_price": None if pd.isna(bid_price) else float(bid_price),
        "ask_price": None if pd.isna(ask_price) else float(ask_price),
        "mark_price": None if pd.isna(mark_price) else float(mark_price),
        "previous_close_price": _option_previous_close_price(row),
    }


def _sell_option_limit_price(row: Mapping[str, Any] | pd.Series | dict[str, Any]) -> tuple[float | None, str]:
    data = row.to_dict() if isinstance(row, pd.Series) else dict(row or {})
    # For sell orders (credit received), target the ask to maximize premium.
    # Fall back to mark / existing price / bid.
    for key in ("ask_price", "mark_price", "limit_order_price", "price", "average_price", "bid_price"):
        value = _positive_float(data.get(key))
        if value is not None:
            return normalize_option_limit_price(float(value), side="sell"), key
    return None, ""


def _buy_option_limit_price(row: Mapping[str, Any] | pd.Series | dict[str, Any]) -> tuple[float | None, str]:
    data = row.to_dict() if isinstance(row, pd.Series) else dict(row or {})
    bid_price = _positive_float(data.get("bid_price"))
    if bid_price is not None:
        tick_price = _floor_option_limit_price(_BUY_OPTION_BID_MULTIPLIER * bid_price)
        return tick_price, _buy_option_bid_limit_source() if tick_price is not None else ""
    mark_price = _positive_float(data.get("mark_price"))
    if mark_price is not None:
        tick_price = _floor_option_limit_price(float(mark_price))
        return tick_price, "mark_price" if tick_price is not None else ""
    existing_price = _positive_float(data.get("price"))
    if existing_price is not None:
        tick_price = _floor_option_limit_price(float(existing_price))
        return tick_price, "existing_price" if tick_price is not None else ""
    return None, ""


def resolve_short_score_col(score_col: str) -> str:
    mapping = {
        "prob_buy": "prob_short",
        "buy_score_mean_raw3": "short_score_mean_raw3",
        "buy_score_mean_raw_pct6": "short_score_mean_raw_pct6",
        "buy_score_pct_mean": "short_score_pct_mean",
        "buy_score_pct_product": "short_score_pct_product",
        "buy_score_raw": "short_score_raw",
        "buy_score": "short_score",
    }
    key = str(score_col)
    if key in mapping:
        return str(mapping[key])
    if key.startswith("buy_"):
        return "short_" + key[len("buy_") :]
    raise KeyError(f"No short-score mapping configured for: {score_col}")


def resolve_component_cols(score_col: str) -> list[str]:
    mapping = {
        "prob_buy": ["prob_buy"],
        "buy_score_mean_raw3": ["prob_buy", "pred_rf_reg", "ae_familiarity"],
        "buy_score_mean_raw_pct6": [
            "prob_buy",
            "pred_rf_reg",
            "ae_familiarity",
            "prob_buy_pct",
            "pred_rf_reg_pct",
            "ae_familiarity_pct",
        ],
        "buy_score_pct_mean": ["prob_buy_pct", "pred_rf_reg_pct", "ae_familiarity_pct"],
        "buy_score_pct_product": ["prob_buy_pct", "pred_rf_reg_pct", "ae_familiarity_pct"],
        "buy_score_raw": ["prob_buy", "pred_rf_reg", "ae_familiarity"],
        "buy_score": ["prob_buy", "pred_rf_reg", "ae_familiarity"],
        "short_score_mean_raw3": ["prob_short", "pred_rf_reg", "ae_familiarity"],
        "short_score_mean_raw_pct6": [
            "prob_short",
            "pred_rf_reg",
            "ae_familiarity",
            "prob_short_pct",
            "pred_rf_reg_pct",
            "ae_familiarity_pct",
        ],
        "short_score_pct_mean": ["prob_short_pct", "pred_rf_reg_pct", "ae_familiarity_pct"],
        "short_score_pct_product": ["prob_short_pct", "pred_rf_reg_pct", "ae_familiarity_pct"],
        "short_score_raw": ["prob_short", "pred_rf_reg", "ae_familiarity"],
        "short_score": ["prob_short", "pred_rf_reg", "ae_familiarity"],
        "prob_short": ["prob_short"],
    }
    if str(score_col) not in mapping:
        raise KeyError(f"No component mapping configured for: {score_col}")
    return list(mapping[str(score_col)])


def robinhood_login(
    *,
    username: str | None = None,
    password: str | None = None,
    mfa_code: str | None = None,
    store_session: bool = True,
) -> dict[str, Any]:
    rh = _require_robin_stocks()
    _load_repo_env()
    user = str(username or os.getenv("ROBINHOOD_USERNAME") or "").strip()
    pwd = str(password or os.getenv("ROBINHOOD_PASSWORD") or "").strip()
    mfa = str(mfa_code or os.getenv("ROBINHOOD_MFA_CODE") or "").strip()
    if not user or not pwd:
        raise RuntimeError(
            "Robinhood credentials are required. Set `ROBINHOOD_USERNAME` and "
            "`ROBINHOOD_PASSWORD` in your `.env` file or environment."
        )
    login_kwargs: dict[str, Any] = {
        "username": user,
        "password": pwd,
        "store_session": bool(store_session),
    }
    if mfa:
        login_kwargs["mfa_code"] = mfa
    result = rh.login(**login_kwargs)
    if not result:
        raise RuntimeError("Robinhood login failed.")
    return _coerce_robinhood_market_row(result)


def load_robinhood_account_snapshot(*, account_number: str | None = None) -> dict[str, Any]:
    rh = _require_robin_stocks()
    portfolio = rh.load_portfolio_profile(account_number=account_number) or {}
    account = rh.load_account_profile(account_number=account_number) or {}
    phoenix = {}
    try:
        phoenix = rh.load_phoenix_account() or {}
    except Exception:
        phoenix = {}
    return {
        "portfolio": _coerce_robinhood_market_row(portfolio),
        "account": _coerce_robinhood_market_row(account),
        "phoenix": _coerce_robinhood_market_row(phoenix),
    }


def load_robinhood_stock_positions(*, account_number: str | None = None) -> pd.DataFrame:
    rh = _require_robin_stocks()
    positions = rh.get_open_stock_positions(account_number=account_number) or []
    rows: list[dict[str, Any]] = []
    for item in positions:
        if not item:
            continue
        instrument_url = str(item.get("instrument") or "")
        try:
            instrument = rh.get_instrument_by_url(instrument_url) if instrument_url else {}
        except Exception:
            instrument = {}
        symbol = str((instrument or {}).get("symbol") or item.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        quantity = pd.to_numeric(pd.Series([item.get("quantity")]), errors="coerce").iloc[0]
        if pd.isna(quantity) or float(quantity) == 0.0:
            continue
        rows.append(
            {
                "symbol": symbol,
                "quantity": float(quantity),
                "average_buy_price": pd.to_numeric(pd.Series([item.get("average_buy_price")]), errors="coerce").iloc[0],
                "instrument": instrument_url,
                "raw": item,
            }
        )
    return pd.DataFrame(rows)


def _extract_url_id(url: str) -> str:
    parts = [part for part in str(url or "").strip().split("/") if part]
    return str(parts[-1]) if parts else ""


def load_robinhood_option_positions(*, account_number: str | None = None) -> pd.DataFrame:
    rh = _require_robin_stocks()
    positions = rh.get_open_option_positions(account_number=account_number) or []
    rows: list[dict[str, Any]] = []
    for item in positions:
        if not item:
            continue
        quantity = pd.to_numeric(pd.Series([item.get("quantity")]), errors="coerce").iloc[0]
        if pd.isna(quantity) or float(quantity) == 0.0:
            continue
        option_url = str(item.get("option") or item.get("option_instrument") or item.get("instrument") or "").strip()
        option_id = _extract_url_id(option_url)
        instrument = {}
        if option_id:
            try:
                instrument = rh.get_option_instrument_data_by_id(option_id) or {}
            except Exception:
                instrument = {}
        symbol = str((instrument or {}).get("chain_symbol") or item.get("chain_symbol") or "").strip().upper()
        option_type = str((instrument or {}).get("type") or item.get("type") or "").strip().lower()
        expiry = str((instrument or {}).get("expiration_date") or item.get("expiration_date") or "").strip()
        strike = pd.to_numeric(pd.Series([(instrument or {}).get("strike_price", item.get("strike_price"))]), errors="coerce").iloc[0]
        if not symbol or option_type not in {"call", "put"} or not expiry or pd.isna(strike):
            continue
        market_row = _lookup_robinhood_option_market_row(rh, symbol, expiry, strike, option_type)
        price_fields = _option_market_price_fields(market_row)
        rows.append(
            {
                "symbol": symbol,
                "option_type": option_type,
                "expiry_date": expiry,
                "strike_price": float(strike),
                "quantity": float(quantity),
                "option_id": option_id,
                "option_url": option_url,
                "average_price": pd.to_numeric(pd.Series([item.get("average_price")]), errors="coerce").iloc[0],
                **price_fields,
                "raw": item,
            }
        )
    return pd.DataFrame(rows)


def _is_open_robinhood_order_state(value: Any) -> bool:
    state = str(value or "").strip().lower()
    if not state:
        return True
    closed_states = {"cancelled", "canceled", "filled", "rejected", "failed", "expired"}
    return state not in closed_states


def load_robinhood_open_option_orders(*, account_number: str | None = None) -> pd.DataFrame:
    rh = _require_robin_stocks()
    if hasattr(rh, "get_open_option_orders"):
        try:
            orders = rh.get_open_option_orders(account_number=account_number) or []
        except TypeError:
            orders = rh.get_open_option_orders() or []
    elif hasattr(rh, "get_all_open_option_orders"):
        try:
            orders = rh.get_all_open_option_orders(account_number=account_number) or []
        except TypeError:
            orders = rh.get_all_open_option_orders() or []
    elif hasattr(rh, "get_all_option_orders"):
        try:
            raw_orders = rh.get_all_option_orders(account_number=account_number) or []
        except TypeError:
            raw_orders = rh.get_all_option_orders() or []
        orders = [
            item
            for item in raw_orders
            if _is_open_robinhood_order_state(_coerce_robinhood_market_row(item).get("state"))
        ]
    else:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for item in orders:
        order = _coerce_robinhood_market_row(item)
        if not order or not _is_open_robinhood_order_state(order.get("state")):
            continue
        legs = order.get("legs") or order.get("option_legs") or []
        if not isinstance(legs, list) or not legs:
            legs = [order]
        for raw_leg in legs:
            leg = _coerce_robinhood_market_row(raw_leg)
            option_url = str(
                leg.get("option")
                or leg.get("option_instrument")
                or leg.get("instrument")
                or order.get("option")
                or order.get("option_instrument")
                or order.get("instrument")
                or ""
            ).strip()
            option_id = _extract_url_id(option_url)
            instrument = {}
            if option_id:
                try:
                    instrument = rh.get_option_instrument_data_by_id(option_id) or {}
                except Exception:
                    instrument = {}
            symbol = str(
                (instrument or {}).get("chain_symbol")
                or leg.get("chain_symbol")
                or order.get("chain_symbol")
                or order.get("chain_symbol_id")
                or ""
            ).strip().upper()
            option_type = str(
                (instrument or {}).get("type")
                or leg.get("option_type")
                or leg.get("type")
                or order.get("option_type")
                or order.get("type")
                or ""
            ).strip().lower()
            opening_strategy = str(order.get("opening_strategy") or leg.get("opening_strategy") or "").strip().lower()
            closing_strategy = str(order.get("closing_strategy") or leg.get("closing_strategy") or "").strip().lower()
            if option_type not in {"call", "put"}:
                option_type = _infer_robinhood_option_type(opening_strategy, closing_strategy, order.get("strategy"), leg.get("strategy"))
            expiry = str(
                (instrument or {}).get("expiration_date")
                or leg.get("expiration_date")
                or order.get("expiration_date")
                or ""
            ).strip()
            strike = pd.to_numeric(
                pd.Series([(instrument or {}).get("strike_price", leg.get("strike_price", order.get("strike_price")))]),
                errors="coerce",
            ).iloc[0]
            side = str(leg.get("side") or order.get("side") or "").strip().lower()
            effect = str(leg.get("position_effect") or order.get("position_effect") or "").strip().lower()
            direction = str(order.get("direction") or leg.get("direction") or "").strip().lower()
            raw_quantity, raw_quantity_source = _first_value_from_keys(
                leg,
                order,
                keys=(
                    "quantity",
                    "pending_quantity",
                    "processed_quantity",
                    "cumulative_quantity",
                    "filled_quantity",
                    "remaining_quantity",
                ),
            )
            quantity = pd.to_numeric(pd.Series([raw_quantity]), errors="coerce").iloc[0]
            contract_quantity, quantity_source = _normalize_option_contract_quantity(raw_quantity)
            if raw_quantity_source:
                quantity_source = f"{quantity_source}:{raw_quantity_source}"
            if not symbol or option_type not in {"call", "put"} or not expiry or pd.isna(strike):
                continue
            action = _infer_robinhood_option_action(
                side=side,
                effect=effect,
                option_type=option_type,
                opening_strategy=opening_strategy,
                closing_strategy=closing_strategy,
                direction=direction,
            )
            order_price, order_price_source = _first_positive_float_from_keys(
                leg,
                order,
                keys=(
                    "price",
                    "limit_price",
                    "premium",
                    "processed_premium",
                    "pending_premium",
                    "opening_price",
                    "closing_price",
                    "average_price",
                ),
            )
            rows.append(
                {
                    "symbol": symbol,
                    "action": action,
                    "side": side,
                    "position_effect": effect,
                    "direction": direction,
                    "opening_strategy": opening_strategy,
                    "closing_strategy": closing_strategy,
                    "quantity": None if pd.isna(quantity) else float(quantity),
                    "contract_quantity": contract_quantity,
                    "quantity_source": quantity_source,
                    "contract_multiplier": float(_OPTION_CONTRACT_MULTIPLIER),
                    "expiry_date": expiry,
                    "strike_price": float(strike),
                    "option_type": option_type,
                    "order_type": str(order.get("type") or order.get("order_type") or "").strip().lower(),
                    "state": str(order.get("state") or "").strip().lower(),
                    "price": order_price,
                    "limit_price": order_price,
                    "limit_price_source": order_price_source,
                    "order_id": str(order.get("id") or ""),
                    "cancel_url": str(order.get("cancel_url") or ""),
                    "option_id": option_id,
                    "option_url": option_url,
                    "raw": order,
                }
            )
    return pd.DataFrame(rows)


def _resolve_nearest_option_expiry(as_of_date: object, expiration_dates: list[str], target_hold_days: int) -> str | None:
    base_date = pd.Timestamp(as_of_date).normalize()
    candidates = []
    for value in expiration_dates:
        expiry_ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(expiry_ts):
            continue
        expiry_ts = pd.Timestamp(expiry_ts).normalize()
        days = int((expiry_ts - base_date).days)
        if days <= 0:
            continue
        candidates.append((abs(days - int(target_hold_days)), days, expiry_ts.strftime("%Y-%m-%d")))
    if not candidates:
        return None
    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    return str(candidates[0][2])


def _breakeven_move_pct(
    *,
    spot_price: float,
    strike_price: float,
    premium: float,
    option_type: str,
) -> float:
    spot = float(spot_price)
    strike = float(strike_price)
    cost = float(premium)
    if str(option_type or "").strip().lower() == "call":
        return ((strike + cost) - spot) / spot
    return (spot - (strike - cost)) / spot


def _rank_option_candidates(
    options: list[dict[str, Any]],
    *,
    target_strike: float,
    option_type: str,
    spot_price: float | None = None,
    min_breakeven_move_pct: float | None = None,
) -> list[dict[str, Any]]:
    right = str(option_type or "").strip().lower()
    valid: list[dict[str, Any]] = []
    for item in options:
        if not item:
            continue
        strike = pd.to_numeric(pd.Series([item.get("strike_price")]), errors="coerce").iloc[0]
        bid_price = pd.to_numeric(pd.Series([item.get("bid_price")]), errors="coerce").iloc[0]
        ask_price = pd.to_numeric(pd.Series([item.get("ask_price")]), errors="coerce").iloc[0]
        mark_price = pd.to_numeric(pd.Series([item.get("adjusted_mark_price", item.get("mark_price"))]), errors="coerce").iloc[0]
        if pd.isna(strike):
            continue
        enriched = _coerce_robinhood_market_row(item)
        enriched["strike_price"] = float(strike)
        enriched["bid_price"] = None if pd.isna(bid_price) else float(bid_price)
        enriched["ask_price"] = None if pd.isna(ask_price) else float(ask_price)
        enriched["mark_price"] = None if pd.isna(mark_price) else float(mark_price)
        premium = _positive_float(bid_price) or _positive_float(mark_price) or _positive_float(ask_price)
        if premium is not None and spot_price is not None and float(spot_price) > 0.0:
            enriched["breakeven_move_pct"] = _breakeven_move_pct(
                spot_price=float(spot_price),
                strike_price=float(strike),
                premium=float(premium),
                option_type=right,
            )
            enriched["breakeven_price"] = (
                float(strike) + float(premium)
                if right == "call"
                else float(strike) - float(premium)
            )
            enriched["breakeven_premium"] = float(premium)
            enriched["breakeven_premium_source"] = (
                "bid_price"
                if _positive_float(bid_price) is not None
                else "mark_price"
                if _positive_float(mark_price) is not None
                else "ask_price"
            )
        valid.append(enriched)
    if spot_price is not None and min_breakeven_move_pct is not None:
        threshold = float(min_breakeven_move_pct)
        qualifying = [
            item
            for item in valid
            if _positive_float(item.get("breakeven_move_pct")) is not None
            and float(item["breakeven_move_pct"]) >= threshold
        ]
        if qualifying:
            if right == "call":
                return sorted(qualifying, key=lambda item: (float(item["strike_price"]), float(item["breakeven_move_pct"])))
            return sorted(qualifying, key=lambda item: (-float(item["strike_price"]), float(item["breakeven_move_pct"])))
    if right == "call":
        return sorted(
            valid,
            key=lambda item: (
                0 if float(item["strike_price"]) >= float(target_strike) else 1,
                abs(float(item["strike_price"]) - float(target_strike)),
                float(item["strike_price"]),
            ),
        )
    return sorted(
        valid,
        key=lambda item: (
            0 if float(item["strike_price"]) <= float(target_strike) else 1,
            abs(float(item["strike_price"]) - float(target_strike)),
            -float(item["strike_price"]),
        ),
    )


def _pre_rank_option_instruments(
    options: list[dict[str, Any]],
    *,
    target_strike: float,
    option_type: str,
) -> list[dict[str, Any]]:
    return _rank_option_candidates(
        options,
        target_strike=float(target_strike),
        option_type=str(option_type),
        spot_price=None,
        min_breakeven_move_pct=None,
    )


def _merge_option_market_data(
    rh: Any,
    *,
    symbol: str,
    option: Mapping[str, Any],
) -> dict[str, Any]:
    enriched = dict(option or {})
    option_id = str(enriched.get("id") or "").strip()
    if not option_id or not hasattr(rh, "get_option_market_data_by_id"):
        return enriched
    market_payload = _call_robinhood_option_api(
        f"get_option_market_data_by_id({symbol}, {option_id})",
        rh.get_option_market_data_by_id,
        option_id,
    )
    market_row = _coerce_robinhood_market_row(market_payload)
    if market_row:
        enriched.update(market_row)
    return enriched


def _find_robinhood_option_instruments(
    rh: Any,
    *,
    symbol: str,
    expiry_date: str,
    option_type: str,
) -> list[dict[str, Any]]:
    if hasattr(rh, "find_tradable_options"):
        options = _call_robinhood_option_api(
            f"find_tradable_options({symbol}, {expiry_date}, {option_type})",
            rh.find_tradable_options,
            symbol,
            expirationDate=expiry_date,
            optionType=option_type,
        ) or []
    else:
        options = _call_robinhood_option_api(
            f"find_options_by_expiration({symbol}, {expiry_date}, {option_type})",
            rh.find_options_by_expiration,
            symbol,
            expirationDate=expiry_date,
            optionType=option_type,
        ) or []
    return [_coerce_robinhood_market_row(item) for item in options if item]


def select_robinhood_long_option_contract(
    *,
    symbol: str,
    spot_price: float,
    option_type: str,
    as_of_date: object,
    target_hold_days: int,
    strike_multiplier: float,
    min_breakeven_move_pct: float = 0.05,
) -> dict[str, Any]:
    rh = _require_robin_stocks()
    clean_symbol = str(symbol).strip().upper()
    clean_option_type = str(option_type).strip().lower()
    chain = _coerce_robinhood_market_row(
        _call_robinhood_option_api(
            f"get_chains({clean_symbol})",
            rh.get_chains,
            clean_symbol,
        )
        or {}
    )
    expirations = list(chain.get("expiration_dates") or [])
    expiry_date = _resolve_nearest_option_expiry(as_of_date, expirations, int(target_hold_days))
    if not expiry_date:
        raise RuntimeError(f"No future Robinhood option expiry found for {symbol}.")
    options = _find_robinhood_option_instruments(
        rh,
        symbol=clean_symbol,
        expiry_date=expiry_date,
        option_type=clean_option_type,
    )
    if not options:
        raise RuntimeError(f"No Robinhood {option_type} contracts found for {symbol} at {expiry_date}.")
    target_strike = float(spot_price) * float(strike_multiplier)
    pre_ranked = _pre_rank_option_instruments(
        options,
        target_strike=target_strike,
        option_type=option_type,
    )
    options_to_enrich = pre_ranked[:_OPTION_MARKET_DATA_CANDIDATE_LIMIT]
    enriched_options = [
        _merge_option_market_data(rh, symbol=clean_symbol, option=item)
        for item in options_to_enrich
    ]
    ranked = _rank_option_candidates(
        enriched_options,
        target_strike=target_strike,
        option_type=option_type,
        spot_price=float(spot_price),
        min_breakeven_move_pct=float(min_breakeven_move_pct),
    )
    if not ranked:
        raise RuntimeError(f"No eligible Robinhood {option_type} contracts found for {symbol} at {expiry_date}.")
    selected = ranked[0]
    bid_price = pd.to_numeric(pd.Series([selected.get("bid_price")]), errors="coerce").iloc[0]
    ask_price = pd.to_numeric(pd.Series([selected.get("ask_price")]), errors="coerce").iloc[0]
    mark_price = pd.to_numeric(pd.Series([selected.get("adjusted_mark_price", selected.get("mark_price"))]), errors="coerce").iloc[0]
    if pd.isna(ask_price):
        ask_price = mark_price
    if pd.isna(bid_price):
        bid_price = mark_price
    return {
        "symbol": clean_symbol,
        "option_type": clean_option_type,
        "expiry_date": str(selected.get("expiration_date") or expiry_date),
        "strike_price": float(pd.to_numeric(pd.Series([selected.get("strike_price")]), errors="coerce").iloc[0]),
        "ask_price": None if pd.isna(ask_price) else float(ask_price),
        "bid_price": None if pd.isna(bid_price) else float(bid_price),
        "mark_price": None if pd.isna(mark_price) else float(mark_price),
        "previous_close_price": _option_previous_close_price(selected),
        "breakeven_price": selected.get("breakeven_price"),
        "breakeven_move_pct": selected.get("breakeven_move_pct"),
        "breakeven_move_threshold": float(min_breakeven_move_pct),
        "breakeven_premium": selected.get("breakeven_premium"),
        "breakeven_premium_source": selected.get("breakeven_premium_source"),
        "id": str(selected.get("id") or ""),
        "raw": selected,
    }


def build_robinhood_option_trade_plan(
    *,
    latest_scored_df: pd.DataFrame,
    current_option_positions: pd.DataFrame | None,
    pending_option_orders: pd.DataFrame | None = None,
    missing_symbol_scorer: Callable[..., pd.DataFrame | None] | None = None,
    top_k: int,
    score_col: str,
    component_threshold: float,
    account_equity: float,
    strategy_allocation: float | None = None,
    as_of_date: object,
    option_bucket: str = "otm_option",
    tenor_days: int = 90,
    max_contracts_per_position: int | None = None,
) -> dict[str, Any]:
    plan_log_lines: list[str] = []
    plan_log_lines.append(f"Robinhood option plan code version: {ROBINHOOD_OPTION_PLAN_CODE_VERSION}.")
    current_df = current_option_positions.copy() if current_option_positions is not None else pd.DataFrame()
    pending_df = pending_option_orders.copy() if pending_option_orders is not None else pd.DataFrame()

    # Track contracts (symbol, expiry, strike, type) that already have a pending sell_to_close.
    # This lets us skip generating duplicate sell orders for the same contract.
    pending_sell_contracts: set[tuple[str, str, float, str]] = set()
    if not pending_df.empty:
        pcopy = pending_df.copy()
        if "symbol" in pcopy.columns:
            pcopy["symbol"] = pcopy["symbol"].astype(str).str.strip().str.upper()
        if "action" in pcopy.columns:
            sell_mask = pcopy["action"].astype(str).str.startswith("sell_to_close")
            for _, prow in pcopy.loc[sell_mask].iterrows():
                psym = str(prow.get("symbol") or "").strip().upper()
                pexp = str(prow.get("expiry_date") or "").strip()
                pstrike = pd.to_numeric(pd.Series([prow.get("strike_price")]), errors="coerce").iloc[0]
                ptype = str(prow.get("option_type") or "").strip().lower()
                if psym and pexp and pd.notna(pstrike) and ptype:
                    pending_sell_contracts.add((psym, pexp, float(pstrike), ptype))
    work = latest_scored_df.copy()
    work.index = pd.Index([str(idx).strip().upper() for idx in work.index], name="symbol")
    held_symbols = {
        str(value).strip().upper()
        for value in current_df.get("symbol", pd.Series(dtype=str))
        if str(value).strip()
    }
    missing_held_symbols = sorted(symbol for symbol in held_symbols if symbol not in work.index)
    if missing_held_symbols and missing_symbol_scorer is not None:
        plan_log_lines.append(
            f"Step 1a: scoring {int(len(missing_held_symbols))} held symbol(s) missing from the scored frame: {', '.join(missing_held_symbols)}."
        )
        scored_missing = missing_symbol_scorer(
            symbols=missing_held_symbols,
            as_of_date=as_of_date,
            latest_scored_df=work.copy(),
        )
        if isinstance(scored_missing, pd.DataFrame) and not scored_missing.empty:
            scored_rows = scored_missing.copy()
            if "symbol" in scored_rows.columns:
                scored_rows.index = pd.Index(scored_rows["symbol"].astype(str).str.strip().str.upper(), name="symbol")
            scored_rows.index = pd.Index([str(idx).strip().upper() for idx in scored_rows.index], name="symbol")
            work = pd.concat([work.loc[~work.index.isin(scored_rows.index)], scored_rows], axis=0)
            plan_log_lines.append(f"Step 1a: appended {int(len(scored_rows))} on-the-fly scored row(s).")
    missing_held_symbols_after_score = sorted(symbol for symbol in held_symbols if symbol not in work.index)
    work["prob_buy"] = pd.to_numeric(work.get("prob_buy", work.get("clf__prob_1")), errors="coerce")
    work["prob_short"] = pd.to_numeric(work.get("prob_short"), errors="coerce")
    missing_short = work["prob_short"].isna()
    work.loc[missing_short, "prob_short"] = 1.0 - work.loc[missing_short, "prob_buy"].fillna(0.0)
    work["close"] = pd.to_numeric(work.get("close"), errors="coerce")

    long_score_col = str(score_col)
    short_score_col = resolve_short_score_col(long_score_col)
    long_component_cols = resolve_component_cols(long_score_col)
    short_component_cols = resolve_component_cols(short_score_col)
    required_cols = list(
        dict.fromkeys(
            ["close", "prob_buy", "prob_short", long_score_col, short_score_col, *long_component_cols, *short_component_cols]
        )
    )
    missing_cols = [col for col in required_cols if col not in work.columns]
    if missing_cols:
        raise KeyError(f"Missing required latest-score columns for Robinhood option plan: {missing_cols}")
    work.loc[:, required_cols] = work[required_cols].apply(pd.to_numeric, errors="coerce")
    work["long_entry_ok"] = _build_entry_ok_by_side(
        work,
        side_score_col=long_score_col,
        component_cols=long_component_cols,
        price_col="close",
        component_threshold=component_threshold,
    )
    work["short_entry_ok"] = _build_entry_ok_by_side(
        work,
        side_score_col=short_score_col,
        component_cols=short_component_cols,
        price_col="close",
        component_threshold=component_threshold,
    )

    plan_log_lines.append(f"Step 1: scanned {int(len(current_df))} current Robinhood option position row(s).")
    grouped_current: dict[str, dict[str, Any]] = {}
    exit_contract_rows: list[dict[str, Any]] = []
    unscored_retained_symbols: set[str] = set(missing_held_symbols_after_score)
    if not current_df.empty:
        for symbol, group in current_df.groupby(current_df["symbol"].astype(str).str.upper(), dropna=False):
            group = group.copy()
            option_types = sorted({str(value).strip().lower() for value in group["option_type"] if str(value).strip()})
            plan_log_lines.append(
                f"Step 1 option position: {str(symbol).upper()} | rows={int(len(group))} | types={', '.join(option_types) if option_types else 'unknown'}."
            )
            if option_types == ["call"]:
                grouped_current[str(symbol).upper()] = {"side": 1, "rows": group.to_dict(orient="records")}
            elif option_types == ["put"]:
                grouped_current[str(symbol).upper()] = {"side": -1, "rows": group.to_dict(orient="records")}
            else:
                for _, row in group.iterrows():
                    exit_contract_rows.append(
                        {
                            "symbol": str(symbol).upper(),
                            "action": "sell_to_close_call" if str(row.get("option_type")).strip().lower() == "call" else "sell_to_close_put",
                            "reason": "mixed_existing_option_positions",
                            "quantity": int(round(float(row.get("quantity") or 0.0))),
                            "expiry_date": str(row.get("expiry_date") or ""),
                            "strike_price": float(row.get("strike_price") or 0.0),
                            "option_type": str(row.get("option_type") or "").strip().lower(),
                            "order_type": "limit",
                            "price": _sell_option_limit_price(row)[0] if _sell_option_limit_price(row)[0] is not None else np.nan,
                            "limit_order_price": _sell_option_limit_price(row)[0] if _sell_option_limit_price(row)[0] is not None else np.nan,
                            "limit_price_source": _sell_option_limit_price(row)[1],
                            "bid_price": row.get("bid_price"),
                            "mark_price": row.get("mark_price"),
                            "average_price": row.get("average_price"),
                        }
                    )

    retained_side_by_symbol: dict[str, int] = {}
    for symbol, current in grouped_current.items():
        if symbol not in work.index:
            retained_side_by_symbol[symbol] = int(current["side"])
            plan_log_lines.append(
                f"Step 2: kept {symbol} because no scored row was available after on-the-fly scoring attempt."
            )
            continue
        row = work.loc[symbol]
        price_ok = pd.notna(row["close"]) and float(row["close"]) > 0.0
        probs_ok = pd.notna(row["prob_buy"]) and pd.notna(row["prob_short"])
        if (not price_ok) or (not probs_ok):
            for pos_row in current["rows"]:
                exit_contract_rows.append(
                    {
                        "symbol": symbol,
                        "action": "sell_to_close_call" if current["side"] > 0 else "sell_to_close_put",
                        "reason": "invalid_live_inputs",
                        "quantity": int(round(float(pos_row.get("quantity") or 0.0))),
                        "expiry_date": str(pos_row.get("expiry_date") or ""),
                        "strike_price": float(pos_row.get("strike_price") or 0.0),
                        "option_type": str(pos_row.get("option_type") or "").strip().lower(),
                        "order_type": "limit",
                        "price": _sell_option_limit_price(pos_row)[0] if _sell_option_limit_price(pos_row)[0] is not None else np.nan,
                        "limit_order_price": _sell_option_limit_price(pos_row)[0] if _sell_option_limit_price(pos_row)[0] is not None else np.nan,
                        "limit_price_source": _sell_option_limit_price(pos_row)[1],
                        "bid_price": pos_row.get("bid_price"),
                        "mark_price": pos_row.get("mark_price"),
                        "average_price": pos_row.get("average_price"),
                    }
                )
            continue
        if current["side"] > 0:
            if bool(float(row["prob_short"]) > float(row["prob_buy"])):
                exit_reason = "classifier_flipped_short"
                plan_log_lines.append(f"Step 2 option exit: {symbol} | side=call | reason={exit_reason}.")
                for pos_row in current["rows"]:
                    exit_contract_rows.append(
                        {
                            "symbol": symbol,
                            "action": "sell_to_close_call",
                            "reason": exit_reason,
                            "quantity": int(round(float(pos_row.get("quantity") or 0.0))),
                            "expiry_date": str(pos_row.get("expiry_date") or ""),
                            "strike_price": float(pos_row.get("strike_price") or 0.0),
                            "option_type": "call",
                            "order_type": "limit",
                            "price": _sell_option_limit_price(pos_row)[0] if _sell_option_limit_price(pos_row)[0] is not None else np.nan,
                            "limit_order_price": _sell_option_limit_price(pos_row)[0] if _sell_option_limit_price(pos_row)[0] is not None else np.nan,
                            "limit_price_source": _sell_option_limit_price(pos_row)[1],
                            "bid_price": pos_row.get("bid_price"),
                            "mark_price": pos_row.get("mark_price"),
                            "average_price": pos_row.get("average_price"),
                        }
                    )
                continue
            retained_side_by_symbol[symbol] = 1
        else:
            if bool(float(row["prob_buy"]) >= float(row["prob_short"])):
                exit_reason = "classifier_flipped_long"
                plan_log_lines.append(f"Step 2 option exit: {symbol} | side=put | reason={exit_reason}.")
                for pos_row in current["rows"]:
                    exit_contract_rows.append(
                        {
                            "symbol": symbol,
                            "action": "sell_to_close_put",
                            "reason": exit_reason,
                            "quantity": int(round(float(pos_row.get("quantity") or 0.0))),
                            "expiry_date": str(pos_row.get("expiry_date") or ""),
                            "strike_price": float(pos_row.get("strike_price") or 0.0),
                            "option_type": "put",
                            "order_type": "limit",
                            "price": _sell_option_limit_price(pos_row)[0] if _sell_option_limit_price(pos_row)[0] is not None else np.nan,
                            "limit_order_price": _sell_option_limit_price(pos_row)[0] if _sell_option_limit_price(pos_row)[0] is not None else np.nan,
                            "limit_price_source": _sell_option_limit_price(pos_row)[1],
                            "bid_price": pos_row.get("bid_price"),
                            "mark_price": pos_row.get("mark_price"),
                            "average_price": pos_row.get("average_price"),
                        }
                    )
                continue
            retained_side_by_symbol[symbol] = -1

    # Remove any exit/sell orders for contracts that already have a pending sell order.
    # This prevents submitting duplicate sell orders for the same option contract.
    if pending_sell_contracts and exit_contract_rows:
        kept_exits: list[dict[str, Any]] = []
        for erow in exit_contract_rows:
            ekey = (
                str(erow.get("symbol") or "").strip().upper(),
                str(erow.get("expiry_date") or "").strip(),
                float(pd.to_numeric(pd.Series([erow.get("strike_price")]), errors="coerce").iloc[0] or 0.0),
                str(erow.get("option_type") or "").strip().lower(),
            )
            if ekey in pending_sell_contracts:
                plan_log_lines.append(
                    f"Step 2 option skip duplicate sell: {ekey[0]} | {ekey[3]} {ekey[1]} strike={ekey[2]} | reason=pending_sell_order_already_exists."
                )
                continue
            kept_exits.append(erow)
        exit_contract_rows = kept_exits

    capacity = max(0, int(top_k))
    exiting_symbols = {str(row.get("symbol") or "").strip().upper() for row in exit_contract_rows if str(row.get("symbol") or "").strip()}
    pending_entry_symbols: set[str] = set()
    pending_cancel_rows: list[dict[str, Any]] = []
    if not pending_df.empty:
        pending_df = pending_df.copy()
        if "symbol" in pending_df.columns:
            pending_df["symbol"] = pending_df["symbol"].astype(str).str.strip().str.upper()
        if "action" in pending_df.columns:
            pending_symbols = (
                pending_df["symbol"].astype(str).str.upper()
                if "symbol" in pending_df.columns
                else pd.Series("", index=pending_df.index, dtype=str)
            )
            pending_entries = pending_df[
                pending_df["action"].astype(str).str.startswith("buy_to_open")
                & ~pending_symbols.isin(exiting_symbols)
            ].copy()
        else:
            pending_entries = pd.DataFrame()
        if not pending_entries.empty:
            for _, pending_row in pending_entries.iterrows():
                pending_symbol = str(pending_row.get("symbol") or "").strip().upper()
                pending_action = str(pending_row.get("action") or "").strip().lower()
                if not pending_symbol or pending_symbol not in work.index:
                    continue
                score_row = work.loc[pending_symbol]
                if pd.isna(score_row.get("prob_buy")) or pd.isna(score_row.get("prob_short")):
                    continue
                cancel_reason = ""
                if pending_action == "buy_to_open_call" and float(score_row["prob_short"]) > float(score_row["prob_buy"]):
                    cancel_reason = "classifier_flipped_short"
                elif pending_action == "buy_to_open_put" and float(score_row["prob_buy"]) > float(score_row["prob_short"]):
                    cancel_reason = "classifier_flipped_long"
                if cancel_reason:
                    pending_cancel_rows.append(
                        {
                            "symbol": pending_symbol,
                            "action": "cancel_buy_to_open_call" if pending_action == "buy_to_open_call" else "cancel_buy_to_open_put",
                            "reason": cancel_reason,
                            "quantity": pending_row.get("quantity"),
                            "expiry_date": str(pending_row.get("expiry_date") or ""),
                            "strike_price": pending_row.get("strike_price"),
                            "option_type": str(pending_row.get("option_type") or ("call" if pending_action == "buy_to_open_call" else "put")).strip().lower(),
                            "order_type": str(pending_row.get("order_type") or ""),
                            "order_id": str(pending_row.get("order_id") or ""),
                            "cancel_url": str(pending_row.get("cancel_url") or ""),
                            "price": pending_row.get("price", np.nan),
                        }
                    )
                    plan_log_lines.append(
                        f"Step 3 pending option cancel: {pending_symbol} | action={pending_action} | reason={cancel_reason}."
                    )
            canceled_pending_order_ids = {
                str(row.get("order_id") or "").strip()
                for row in pending_cancel_rows
                if str(row.get("order_id") or "").strip()
            }
            canceled_pending_symbols = {
                str(row.get("symbol") or "").strip().upper()
                for row in pending_cancel_rows
                if str(row.get("symbol") or "").strip()
            }
            if canceled_pending_order_ids and "order_id" in pending_entries.columns:
                pending_entries = pending_entries.loc[
                    ~pending_entries["order_id"].astype(str).str.strip().isin(canceled_pending_order_ids)
                ].copy()
            elif canceled_pending_symbols:
                pending_entries = pending_entries.loc[
                    ~pending_entries["symbol"].astype(str).str.strip().str.upper().isin(canceled_pending_symbols)
                ].copy()
        pending_entry_symbols = {
            str(value).strip().upper()
            for value in pending_entries.get("symbol", pd.Series(dtype=str))
            if str(value).strip()
        }
        for pending_symbol in sorted(pending_entry_symbols):
            plan_log_lines.append(f"Step 3 pending option order: {pending_symbol} | counts against top_k capacity.")
    if pending_sell_contracts:
        plan_log_lines.append(
            f"Step 3 pending option sell orders: {len(pending_sell_contracts)} sell_to_close contract(s) already pending (duplicates skipped to avoid resubmitting sells)."
        )
    occupied_symbols = set(retained_side_by_symbol) | pending_entry_symbols
    slots_left = max(0, capacity - len(occupied_symbols))
    plan_log_lines.append(
        f"Step 2: classifier scan generated {int(len(exit_contract_rows))} limit exit order(s) and kept {int(len(retained_side_by_symbol))} existing underlying(s)."
    )
    if unscored_retained_symbols:
        plan_log_lines.append(
            f"Step 2: {int(len(unscored_retained_symbols))} held underlying(s) were kept without exit because scores were unavailable: {', '.join(sorted(unscored_retained_symbols))}."
        )
    plan_log_lines.append(
        f"Step 3: scanned {int(len(pending_df))} outstanding option order row(s), including {int(len(pending_entry_symbols))} pending buy-to-open underlying(s) after {int(len(pending_cancel_rows))} cancel request(s)."
    )
    plan_log_lines.append(
        f"Step 4: top_k={int(capacity)}, occupied slots={int(len(occupied_symbols))}, remaining buy slots={int(slots_left)}."
    )
    candidate_rows: list[tuple[float, str, int]] = []
    for symbol, row in work.iterrows():
        if symbol in occupied_symbols or symbol in exiting_symbols:
            continue
        if not (pd.notna(row["close"]) and float(row["close"]) > 0.0):
            continue
        long_ok = bool(row["long_entry_ok"]) and pd.notna(row[long_score_col])
        short_ok = bool(row["short_entry_ok"]) and pd.notna(row[short_score_col])
        if not long_ok and not short_ok:
            continue
        if long_ok and short_ok:
            long_value = float(row[long_score_col])
            short_value = float(row[short_score_col])
            side = 1 if long_value >= short_value else -1
            score_value = max(long_value, short_value)
        elif long_ok:
            side = 1
            score_value = float(row[long_score_col])
        else:
            side = -1
            score_value = float(row[short_score_col])
        candidate_rows.append((score_value, symbol, side))

    option_buckets = {
        "atm_option": {"call": 1.00, "put": 1.00},
        "otm_option": {"call": 1.05, "put": 0.95},
        "ditm_option": {"call": 0.90, "put": 1.10},
    }
    bucket_cfg = dict(option_buckets.get(str(option_bucket), option_buckets["otm_option"]))
    desired_contract_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    target_weight = (1.0 / float(capacity)) if capacity > 0 else 0.0
    capital_base = float(max(strategy_allocation if strategy_allocation is not None else account_equity, 0.0))
    target_dollars = capital_base * target_weight
    effective_max_contracts = None if max_contracts_per_position is None else max(1, int(max_contracts_per_position))
    chosen_entries: list[tuple[str, int, float]] = []
    for candidate_rank, (score_value, symbol, side) in enumerate(
        sorted(candidate_rows, key=lambda row: (row[0], row[1]), reverse=True),
        start=1,
    ):
        if len(chosen_entries) >= slots_left:
            break
        option_type = "call" if side > 0 else "put"
        strike_multiplier = float(bucket_cfg["call"] if option_type == "call" else bucket_cfg["put"])
        row = work.loc[symbol]
        plan_log_lines.append(
            f"Step 5 option candidate: {symbol} | type={option_type} | score={float(score_value):.6f} | slot_budget=${float(target_dollars):,.2f}."
        )
        try:
            contract = select_robinhood_long_option_contract(
                symbol=symbol,
                spot_price=float(row["close"]),
                option_type=option_type,
                as_of_date=as_of_date,
                target_hold_days=int(tenor_days),
                strike_multiplier=strike_multiplier,
            )
        except Exception as exc:
            skip_reason = "option_lookup_failed" if _is_robinhood_option_api_error(exc) else "no_eligible_option_contract"
            plan_log_lines.append(f"Step 5 option skip: {symbol} | reason={skip_reason} | error={str(exc)}.")
            skipped_rows.append(
                {
                    "symbol": symbol,
                    "direction": "Long" if side > 0 else "Short",
                    "candidate_rank": int(candidate_rank),
                    "reason": skip_reason,
                    "error": str(exc),
                }
            )
            if skip_reason == "option_lookup_failed":
                plan_log_lines.append(
                    "Step 5: stopping new option lookups because Robinhood option data is unavailable."
                )
                break
            continue
        quote_contract_price = (
            _positive_float(contract.get("bid_price"))
            or _positive_float(contract.get("mark_price"))
            or _positive_float(contract.get("ask_price"))
        )
        quote_contract_price_source = (
            "bid_price"
            if _positive_float(contract.get("bid_price")) is not None
            else "mark_price"
            if _positive_float(contract.get("mark_price")) is not None
            else "ask_price"
        )
        quote_contract_value = float(quote_contract_price) * 100.0 if quote_contract_price is not None else float("nan")
        limit_price, limit_price_source = _buy_option_limit_price(contract)
        contract_value = float(limit_price) * 100.0 if limit_price is not None else float("nan")
        if (not np.isfinite(contract_value)) or contract_value <= 0.0:
            plan_log_lines.append(f"Step 5 option skip: {symbol} | reason=invalid_contract_value.")
            skipped_rows.append(
                {
                    "symbol": symbol,
                    "direction": "Long" if side > 0 else "Short",
                    "candidate_rank": int(candidate_rank),
                    "reason": "invalid_contract_value",
                    "contract_value": contract_value,
                    "contract_price_source": limit_price_source,
                    "quote_contract_value": quote_contract_value,
                    "quote_contract_price_source": quote_contract_price_source,
                    "target_dollars": target_dollars,
                }
            )
            continue
        if target_dollars <= 0.0 or contract_value > target_dollars:
            plan_log_lines.append(
                f"Step 5 option skip: {symbol} | reason=contract_value_exceeds_slot_budget | contract_value=${float(contract_value):,.2f} | slot_budget=${float(target_dollars):,.2f}."
            )
            skipped_rows.append(
                {
                    "symbol": symbol,
                    "direction": "Long" if side > 0 else "Short",
                    "candidate_rank": int(candidate_rank),
                    "reason": "contract_value_exceeds_slot_budget",
                    "contract_value": float(contract_value),
                    "contract_price_source": limit_price_source,
                    "quote_contract_value": quote_contract_value,
                    "quote_contract_price_source": quote_contract_price_source,
                    "target_dollars": float(target_dollars),
                }
            )
            continue
        target_contracts = int(np.floor(target_dollars / contract_value))
        if effective_max_contracts is not None:
            target_contracts = min(int(target_contracts), int(effective_max_contracts))
        if target_contracts <= 0:
            plan_log_lines.append(
                f"Step 5 option skip: {symbol} | reason=non_positive_target_contracts | contract_value=${float(contract_value):,.2f} | slot_budget=${float(target_dollars):,.2f}."
            )
            skipped_rows.append(
                {
                    "symbol": symbol,
                    "direction": "Long" if side > 0 else "Short",
                    "candidate_rank": int(candidate_rank),
                    "reason": "non_positive_target_contracts",
                    "contract_value": float(contract_value),
                    "contract_price_source": limit_price_source,
                    "quote_contract_value": quote_contract_value,
                    "quote_contract_price_source": quote_contract_price_source,
                    "target_dollars": float(target_dollars),
                }
            )
            continue
        plan_log_lines.append(
            f"Step 5 option buy: {symbol} | type={option_type} | contracts={int(target_contracts)} | limit_contract_value=${float(contract_value):,.2f} | quote_contract_value=${float(quote_contract_value):,.2f}."
        )
        desired_contract_rows.append(
            {
                "symbol": symbol,
                "direction": "Long" if side > 0 else "Short",
                "combined_score": float(score_value),
                "target_weight": target_weight,
                "target_dollars": target_dollars,
                "contract_value": float(contract_value),
                "contract_price_source": limit_price_source,
                "limit_order_price": float(limit_price),
                "limit_price_source": limit_price_source,
                "quote_contract_value": float(quote_contract_value) if np.isfinite(quote_contract_value) else np.nan,
                "quote_contract_price_source": quote_contract_price_source,
                "target_contracts": int(target_contracts),
                **contract,
            }
        )
        chosen_entries.append((symbol, side, float(score_value)))
    filled_entry_slots = int(len(desired_contract_rows))
    unfilled_entry_slots = max(0, int(slots_left) - filled_entry_slots)
    if skipped_rows:
        replacement_status = "replaced" if unfilled_entry_slots == 0 else "partially_replaced"
        for skipped_row in skipped_rows:
            skipped_row["replacement_status"] = replacement_status
    plan_log_lines.append(
        f"Step 5: generated {filled_entry_slots} new buy limit order(s) for {int(slots_left)} available slot(s) after {int(len(skipped_rows))} skipped candidate(s)."
    )
    if unfilled_entry_slots:
        plan_log_lines.append(f"Step 5: {int(unfilled_entry_slots)} entry slot(s) remained unfilled because no eligible replacement candidate was available.")

    desired_df = pd.DataFrame(desired_contract_rows)
    actions_rows: list[dict[str, Any]] = list(pending_cancel_rows) + list(exit_contract_rows)
    for _, row in desired_df.iterrows():
        limit_price = _positive_float(row.get("limit_order_price"))
        limit_price_source = str(row.get("limit_price_source") or "").strip() or _buy_option_limit_price(row)[1]
        bid_limit = _BUY_OPTION_BID_MULTIPLIER * float(row.get("bid_price")) if _positive_float(row.get("bid_price")) is not None else np.nan
        bid_limit_column = _buy_option_bid_limit_column()
        actions_rows.append(
            {
                "symbol": str(row["symbol"]).upper(),
                "action": "buy_to_open_call" if str(row["option_type"]).lower() == "call" else "buy_to_open_put",
                "reason": "leaderboard_top_k_entry",
                "quantity": int(row["target_contracts"]),
                "expiry_date": str(row["expiry_date"]),
                "strike_price": float(row["strike_price"]),
                "option_type": str(row["option_type"]).lower(),
                "order_type": "limit",
                "price": float(limit_price) if limit_price is not None else np.nan,
                "limit_order_price": float(limit_price) if limit_price is not None else np.nan,
                "limit_price_source": limit_price_source,
                "bid_price": row.get("bid_price"),
                bid_limit_column: bid_limit,
                "previous_close_price": row.get("previous_close_price"),
                "mark_price": row.get("mark_price"),
                "breakeven_price": row.get("breakeven_price"),
                "breakeven_move_pct": row.get("breakeven_move_pct"),
                "breakeven_move_threshold": row.get("breakeven_move_threshold"),
                "contract_value": row.get("contract_value"),
                "quote_contract_value": row.get("quote_contract_value"),
                "combined_score": float(row["combined_score"]),
                "direction": str(row["direction"]),
                "target_weight": float(row["target_weight"]),
                "target_dollars": float(row["target_dollars"]),
            }
        )

    retained_rows: list[dict[str, Any]] = []
    for symbol, side in retained_side_by_symbol.items():
        retained_group = grouped_current.get(symbol, {}).get("rows", [])
        for pos_row in retained_group:
            has_score = symbol in work.index
            retained_rows.append(
                {
                    "symbol": symbol,
                    "action": "hold_call" if side > 0 else "hold_put",
                    "reason": "signal_still_valid" if has_score else "score_unavailable_hold",
                    "quantity": int(round(float(pos_row.get("quantity") or 0.0))),
                    "expiry_date": str(pos_row.get("expiry_date") or ""),
                    "strike_price": float(pos_row.get("strike_price") or 0.0),
                    "option_type": "call" if side > 0 else "put",
                    "price": np.nan,
                    "combined_score": float(work.loc[symbol, long_score_col] if side > 0 else work.loc[symbol, short_score_col]) if has_score else np.nan,
                    "direction": "Long" if side > 0 else "Short",
                    "target_weight": target_weight,
                    "target_dollars": target_dollars,
                }
            )
    actions_rows.extend(retained_rows)

    actions = pd.DataFrame(actions_rows)
    if not actions.empty:
        if "combined_score" not in actions.columns:
            actions["combined_score"] = np.nan
        priority = {
            "cancel_buy_to_open_call": 0,
            "cancel_buy_to_open_put": 1,
            "sell_to_close_call": 2,
            "sell_to_close_put": 3,
            "buy_to_open_call": 4,
            "buy_to_open_put": 5,
            "hold_call": 6,
            "hold_put": 7,
        }
        actions["_priority"] = actions["action"].map(priority).fillna(99)
        actions = actions.sort_values(["_priority", "combined_score", "symbol"], ascending=[True, False, True], kind="stable").drop(columns=["_priority"])

    actionable_orders = actions.loc[
        actions["action"].isin(["cancel_buy_to_open_call", "cancel_buy_to_open_put", "sell_to_close_call", "sell_to_close_put", "buy_to_open_call", "buy_to_open_put"])
    ].copy() if not actions.empty else pd.DataFrame()
    if not actionable_orders.empty:
        cancel_mask = actionable_orders["action"].astype(str).str.startswith("cancel_")
        actionable_orders["skip_submit"] = (
            pd.to_numeric(actionable_orders.get("quantity"), errors="coerce").fillna(0.0).le(0.0)
            & ~cancel_mask
        )
    else:
        actionable_orders["skip_submit"] = pd.Series(dtype=bool)

    target_portfolio_rows: list[dict[str, Any]] = []
    for symbol, side in retained_side_by_symbol.items():
        if symbol not in work.index:
            target_portfolio_rows.append(
                {
                    "symbol": symbol,
                    "direction": "Long" if side > 0 else "Short",
                    "status": "hold_unscored",
                    "combined_score": np.nan,
                }
            )
            continue
        target_portfolio_rows.append(
            {
                "symbol": symbol,
                "direction": "Long" if side > 0 else "Short",
                "status": "hold",
                "combined_score": float(work.loc[symbol, long_score_col] if side > 0 else work.loc[symbol, short_score_col]),
            }
        )
    for _, row in desired_df.iterrows():
        target_portfolio_rows.append(
            {
                "symbol": str(row["symbol"]).upper(),
                "direction": str(row["direction"]),
                "status": "buy",
                "combined_score": float(row["combined_score"]),
                "expiry_date": str(row["expiry_date"]),
                "strike_price": float(row["strike_price"]),
                "option_type": str(row["option_type"]).lower(),
                "target_contracts": int(row["target_contracts"]),
            }
        )
    target_portfolio = pd.DataFrame(target_portfolio_rows)
    if not target_portfolio.empty:
        target_portfolio = target_portfolio.sort_values(["status", "combined_score", "symbol"], ascending=[True, False, True], kind="stable").reset_index(drop=True)

    summary = pd.DataFrame(
        [
            {
                "top_k": int(capacity),
                "current_underlyings": int(len(grouped_current)),
                "positions_kept": int(len(retained_side_by_symbol)),
                "positions_exited": int(len(exit_contract_rows)),
                "new_entries": int(len(desired_df)),
                "target_positions": int(len(retained_side_by_symbol) + len(desired_df)),
                "calls_to_open": int((actions["action"] == "buy_to_open_call").sum()) if not actions.empty else 0,
                "puts_to_open": int((actions["action"] == "buy_to_open_put").sum()) if not actions.empty else 0,
                "contracts_to_close": int((actions["action"].astype(str).str.startswith("sell_to_close")).sum()) if not actions.empty else 0,
                "orders_to_cancel": int((actions["action"].astype(str).str.startswith("cancel_")).sum()) if not actions.empty else 0,
                "target_weight_per_position": float(target_weight),
                "account_equity": float(max(account_equity, 0.0)),
                "strategy_allocation": float(capital_base),
                "option_bucket": str(option_bucket),
                "tenor_days": int(tenor_days),
                "pending_buy_underlyings": int(len(pending_entry_symbols)),
                "occupied_slots": int(len(occupied_symbols)),
                "remaining_buy_slots": int(slots_left),
                "filled_entry_slots": int(filled_entry_slots),
                "unfilled_entry_slots": int(unfilled_entry_slots),
                "skipped_entry_candidates": int(len(skipped_rows)),
                "held_symbols_missing_scores": int(len(unscored_retained_symbols)),
            }
        ]
    )
    return {
        "summary": summary,
        "target_portfolio": target_portfolio,
        "desired_contracts": desired_df.reset_index(drop=True),
        "actions": actions.reset_index(drop=True),
        "actionable_orders": actionable_orders.reset_index(drop=True),
        "skipped_symbols": pd.DataFrame(skipped_rows).reset_index(drop=True),
        "pending_option_orders": pending_df.reset_index(drop=True),
        "plan_log_lines": plan_log_lines,
        "code_version": ROBINHOOD_OPTION_PLAN_CODE_VERSION,
    }


def enrich_robinhood_option_prices(orders_df: pd.DataFrame) -> pd.DataFrame:
    rh = _require_robin_stocks()
    if orders_df is None or orders_df.empty:
        return pd.DataFrame() if orders_df is None else orders_df.copy()
    enriched = orders_df.copy()
    for idx, row in enriched.iterrows():
        symbol = str(row.get("symbol") or "").strip().upper()
        option_type = str(row.get("option_type") or "").strip().lower()
        expiry = str(row.get("expiry_date") or "").strip()
        strike = pd.to_numeric(pd.Series([row.get("strike_price")]), errors="coerce").iloc[0]
        if str(row.get("order_type") or "").strip().lower() == "market":
            continue
        if not symbol or option_type not in {"call", "put"} or not expiry or pd.isna(strike):
            continue
        market_row = _lookup_robinhood_option_market_row(rh, symbol, expiry, strike, option_type)
        price_fields = _option_market_price_fields(market_row)
        bid_price = _positive_float(price_fields.get("bid_price")) or _positive_float(row.get("bid_price"))
        ask_price = _positive_float(price_fields.get("ask_price")) or _positive_float(row.get("ask_price"))
        mark_price = _positive_float(price_fields.get("mark_price")) or _positive_float(row.get("mark_price"))
        previous_close_price = price_fields.get("previous_close_price") or _option_previous_close_price(row.to_dict())
        bid_limit_column = _buy_option_bid_limit_column()
        if bid_price is not None:
            enriched.at[idx, "bid_price"] = float(bid_price)
            enriched.at[idx, bid_limit_column] = _BUY_OPTION_BID_MULTIPLIER * float(bid_price)
        if ask_price is not None:
            enriched.at[idx, "ask_price"] = float(ask_price)
        if mark_price is not None:
            enriched.at[idx, "mark_price"] = float(mark_price)
        if previous_close_price is not None:
            enriched.at[idx, "previous_close_price"] = float(previous_close_price)
        if str(row.get("action") or "").startswith("sell_to_close"):
            pricing_row = row.to_dict()
            if ask_price is not None:
                pricing_row["ask_price"] = float(ask_price)
            if bid_price is not None:
                pricing_row["bid_price"] = float(bid_price)
            if mark_price is not None:
                pricing_row["mark_price"] = float(mark_price)
            limit_price, limit_price_source = _sell_option_limit_price(pricing_row)
            enriched.at[idx, "price"] = limit_price if limit_price is not None else row.get("price")
            enriched.at[idx, "limit_order_price"] = limit_price if limit_price is not None else row.get("price")
            enriched.at[idx, "limit_price_source"] = limit_price_source or "existing_price"
        elif str(row.get("action") or "").startswith("buy_to_open"):
            pricing_row = row.to_dict()
            if pd.notna(bid_price):
                pricing_row["bid_price"] = float(bid_price)
            if pd.notna(mark_price):
                pricing_row["mark_price"] = float(mark_price)
            if previous_close_price is not None:
                pricing_row["previous_close_price"] = float(previous_close_price)
            limit_price, limit_price_source = _buy_option_limit_price(pricing_row)
            enriched.at[idx, "price"] = limit_price if limit_price is not None else row.get("price")
            enriched.at[idx, "limit_order_price"] = limit_price if limit_price is not None else row.get("price")
            enriched.at[idx, "limit_price_source"] = limit_price_source or "existing_price"
    return enriched


def annotate_robinhood_option_limit_savings(orders_df: pd.DataFrame) -> pd.DataFrame:
    """Add pending-order savings/missed columns without overwriting the order's submitted limit."""
    rh = _require_robin_stocks()
    if orders_df is None or orders_df.empty:
        return pd.DataFrame() if orders_df is None else orders_df.copy()
    annotated = orders_df.copy()
    for idx, row in annotated.iterrows():
        action = str(row.get("action") or "").strip().lower()
        if not action.startswith("buy_to_open"):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        option_type = str(row.get("option_type") or "").strip().lower()
        expiry = str(row.get("expiry_date") or "").strip()
        strike = pd.to_numeric(pd.Series([row.get("strike_price")]), errors="coerce").iloc[0]
        existing_contract_quantity = _positive_float(row.get("contract_quantity"))
        if existing_contract_quantity is not None:
            contract_quantity, quantity_source = existing_contract_quantity, str(row.get("quantity_source") or "contract_quantity")
        else:
            contract_quantity, quantity_source = _normalize_option_contract_quantity(row.get("quantity"))
        submitted_limit, submitted_limit_source = _first_positive_float_from_keys(
            row.to_dict() if isinstance(row, pd.Series) else dict(row or {}),
            keys=(
                "limit_price",
                "price",
                "limit_order_price",
                "premium",
                "processed_premium",
                "pending_premium",
                "opening_price",
                "closing_price",
                "average_price",
            ),
        )
        if not symbol or option_type not in {"call", "put"} or not expiry or pd.isna(strike) or contract_quantity is None or submitted_limit is None:
            continue

        market_row = _lookup_robinhood_option_market_row(rh, symbol, expiry, strike, option_type)
        price_fields = _option_market_price_fields(market_row)
        bid_price = _positive_float(price_fields.get("bid_price")) or _positive_float(row.get("bid_price"))
        mark_price = _positive_float(price_fields.get("mark_price")) or _positive_float(row.get("mark_price"))
        ask_price = _positive_float(price_fields.get("ask_price")) or _positive_float(row.get("ask_price"))
        current_reference_price = bid_price or mark_price or ask_price
        discount_rate = float(_BUY_OPTION_BID_MULTIPLIER)
        inferred_original_reference_price = float(submitted_limit) / discount_rate
        bid_limit_column = _buy_option_bid_limit_column()
        target_bid_limit_column = f"target_{_buy_option_bid_limit_source()}_limit_price"
        if bid_price is not None:
            annotated.at[idx, "bid_price"] = float(bid_price)
            annotated.at[idx, bid_limit_column] = _BUY_OPTION_BID_MULTIPLIER * float(bid_price)
            annotated.at[idx, target_bid_limit_column] = _floor_option_limit_price(_BUY_OPTION_BID_MULTIPLIER * float(bid_price))
        if mark_price is not None:
            annotated.at[idx, "mark_price"] = float(mark_price)
        if ask_price is not None:
            annotated.at[idx, "ask_price"] = float(ask_price)
        qty = float(contract_quantity)
        original_strategy_qty = qty * discount_rate
        original_strategy_qty_floor = float(np.floor(original_strategy_qty))
        discount_saved_per_share = inferred_original_reference_price - float(submitted_limit)
        discount_saved_per_contract = discount_saved_per_share * float(_OPTION_CONTRACT_MULTIPLIER)
        discount_saved_total = discount_saved_per_contract * original_strategy_qty
        submitted_limit_notional = float(submitted_limit) * float(_OPTION_CONTRACT_MULTIPLIER) * qty
        original_strategy_notional = inferred_original_reference_price * float(_OPTION_CONTRACT_MULTIPLIER) * original_strategy_qty
        annotated.at[idx, "contract_quantity"] = float(qty)
        annotated.at[idx, "inferred_original_strategy_contract_quantity"] = float(original_strategy_qty)
        annotated.at[idx, "inferred_original_strategy_contract_quantity_floor"] = float(original_strategy_qty_floor)
        annotated.at[idx, "quantity_source"] = quantity_source
        annotated.at[idx, "contract_multiplier"] = float(_OPTION_CONTRACT_MULTIPLIER)
        annotated.at[idx, "submitted_limit_notional"] = float(submitted_limit_notional)
        annotated.at[idx, "inferred_original_strategy_notional"] = float(original_strategy_notional)
        annotated.at[idx, "submitted_limit_price"] = float(submitted_limit)
        annotated.at[idx, "discount_rate"] = float(discount_rate)
        annotated.at[idx, "limit_price"] = float(submitted_limit)
        annotated.at[idx, "limit_price_source"] = submitted_limit_source
        annotated.at[idx, "current_qty"] = float(qty)
        annotated.at[idx, "original_strategy_price"] = float(inferred_original_reference_price)
        annotated.at[idx, "original_strategy_qty"] = float(original_strategy_qty)
        annotated.at[idx, "inferred_original_reference_price"] = float(inferred_original_reference_price)
        annotated.at[idx, "discount_saved_per_share"] = float(discount_saved_per_share)
        annotated.at[idx, "discount_saved_per_contract"] = float(discount_saved_per_contract)
        annotated.at[idx, "discount_saved_total"] = float(discount_saved_total)
        if current_reference_price is None:
            continue
        missed_per_share = float(current_reference_price) - inferred_original_reference_price
        missed_per_contract = missed_per_share * float(_OPTION_CONTRACT_MULTIPLIER)
        missed_total = missed_per_contract * original_strategy_qty
        annotated.at[idx, "limit_savings_reference_price"] = float(current_reference_price)
        annotated.at[idx, "limit_savings_reference_source"] = "bid_price" if bid_price is not None else "mark_price" if mark_price is not None else "ask_price"
        annotated.at[idx, "missed_move_per_share"] = float(missed_per_share)
        annotated.at[idx, "missed_move_per_contract"] = float(missed_per_contract)
        annotated.at[idx, "missed_move_total"] = float(missed_total)
        annotated.at[idx, "missed_move_label"] = "missed_upside" if missed_total > 0.0 else "avoided_loss" if missed_total < 0.0 else "flat"
    return annotated


def submit_robinhood_option_orders(
    *,
    orders_df: pd.DataFrame,
    account_number: str | None = None,
    time_in_force: str = "gtc",
) -> pd.DataFrame:
    rh = _require_robin_stocks()
    if orders_df is None or orders_df.empty:
        return pd.DataFrame(columns=["symbol", "action", "submitted", "posted_to_robinhood", "robinhood_state", "order_id", "response"])

    def _result_row(symbol: str, action: str, response: Any, *, skipped: str | None = None) -> dict[str, Any]:
        payload = _coerce_robinhood_market_row(response)
        state = str(payload.get("state") or payload.get("derived_state") or "").strip().lower()
        order_id = str(payload.get("id") or "").strip()
        posted = bool(order_id)
        failed_states = {"unconfirmed", "rejected", "failed", "cancelled", "canceled", "expired"}
        submitted = bool(payload) if not state else bool(posted and state not in failed_states)
        if str(action).startswith("cancel_") and payload:
            submitted = True
        if skipped:
            submitted = False
        return {
            "symbol": symbol,
            "action": action,
            "submitted": bool(submitted),
            "posted_to_robinhood": bool(posted),
            "robinhood_state": state or ("skipped" if skipped else ""),
            "order_id": order_id,
            "price": _positive_float(payload.get("price")),
            "quantity": payload.get("quantity"),
            "pending_quantity": payload.get("pending_quantity"),
            "processed_quantity": payload.get("processed_quantity"),
            "estimated_total_net_amount": payload.get("estimated_total_net_amount"),
            "cancel_url": str(payload.get("cancel_url") or ""),
            "message": skipped or str(payload.get("detail") or payload.get("message") or payload.get("error") or ""),
            "response": response,
        }

    responses: list[dict[str, Any]] = []
    seen_sell_contracts: set[tuple[str, str, float, str]] = set()
    for _, row in orders_df.iterrows():
        symbol = str(row.get("symbol") or "").strip().upper()
        action = str(row.get("action") or "").strip().lower()
        option_type = str(row.get("option_type") or "").strip().lower()
        order_type = str(row.get("order_type") or "limit").strip().lower()
        expiry_date = str(row.get("expiry_date") or "").strip()
        strike_price = pd.to_numeric(pd.Series([row.get("strike_price")]), errors="coerce").iloc[0]
        quantity = pd.to_numeric(pd.Series([row.get("quantity")]), errors="coerce").iloc[0]
        if action in ("sell_to_close_call", "sell_to_close_put"):
            skey = (symbol, expiry_date, float(strike_price) if pd.notna(strike_price) else 0.0, option_type)
            if skey in seen_sell_contracts:
                responses.append(_result_row(symbol, action, {"skipped": "duplicate_sell_order"}, skipped="duplicate_sell_order"))
                continue
            seen_sell_contracts.add(skey)
        if action == "cancel_buy_to_open_call" or action == "cancel_buy_to_open_put":
            order_id = str(row.get("order_id") or "").strip()
            if not order_id:
                responses.append(_result_row(symbol, action, {"skipped": "missing_order_id"}, skipped="missing_order_id"))
                continue
            response = rh.cancel_option_order(order_id)
            responses.append(_result_row(symbol, action, response if response else {"cancelled_order_id": order_id, "state": "cancelled"}))
            continue
        if order_type == "market":
            limit_price = None
        elif action == "buy_to_open_call" or action == "buy_to_open_put":
            limit_price = (
                _positive_float(row.get("limit_order_price"))
                or _positive_float(row.get("limit_price"))
            )
            if limit_price is None:
                limit_price, _limit_price_source = _buy_option_limit_price(row)
        elif action == "sell_to_close_call" or action == "sell_to_close_put":
            # Use ask for sell orders to target a higher credit (better for the seller).
            limit_price = (
                _positive_float(row.get("ask_price"))
                or _positive_float(row.get("limit_order_price"))
                or _positive_float(row.get("price"))
                or _positive_float(row.get("bid_price"))
            )
        else:
            limit_price = _positive_float(row.get("price"))
        if order_type != "market":
            if action == "buy_to_open_call" or action == "buy_to_open_put":
                limit_price = normalize_option_limit_price(limit_price, side="buy")
            elif action == "sell_to_close_call" or action == "sell_to_close_put":
                limit_price = normalize_option_limit_price(limit_price, side="sell")
            else:
                limit_price = _round_option_limit_price(limit_price)
        if not symbol or option_type not in {"call", "put"} or not expiry_date or pd.isna(strike_price) or pd.isna(quantity) or int(quantity) <= 0:
            responses.append(_result_row(symbol, action, {"skipped": "invalid_order_payload"}, skipped="invalid_order_payload"))
            continue
        if order_type != "market" and limit_price is None:
            responses.append(_result_row(symbol, action, {"skipped": "invalid_limit_price"}, skipped="invalid_limit_price"))
            continue
        order_time_in_force = "gfd" if order_type == "market" else str(time_in_force or "gtc").strip().lower()
        if action == "buy_to_open_call" or action == "buy_to_open_put":
            response = rh.order_buy_option_limit(
                "open",
                "debit",
                float(limit_price),
                symbol,
                int(quantity),
                expiry_date,
                float(strike_price),
                optionType=option_type,
                account_number=account_number,
                timeInForce=order_time_in_force,
                jsonify=True,
            )
        elif (action == "sell_to_close_call" or action == "sell_to_close_put") and order_type == "market":
            try:
                response = rh.order_sell_option_market(
                    "close",
                    "credit",
                    symbol,
                    int(quantity),
                    expiry_date,
                    float(strike_price),
                    optionType=option_type,
                    account_number=account_number,
                    timeInForce=order_time_in_force,
                    jsonify=True,
                )
            except TypeError:
                response = rh.order_sell_option_market(
                    "close",
                    "credit",
                    symbol,
                    int(quantity),
                    expiry_date,
                    float(strike_price),
                    optionType=option_type,
                    jsonify=True,
                )
        elif action == "sell_to_close_call" or action == "sell_to_close_put":
            response = rh.order_sell_option_limit(
                "close",
                "credit",
                float(limit_price),
                symbol,
                int(quantity),
                expiry_date,
                float(strike_price),
                optionType=option_type,
                account_number=account_number,
                timeInForce=order_time_in_force,
                jsonify=True,
            )
        else:
            responses.append(_result_row(symbol, action, {"skipped": "unsupported_action"}, skipped="unsupported_action"))
            continue
        responses.append(_result_row(symbol, action, response))
    return pd.DataFrame(responses)


def _build_entry_ok_by_side(
    work: pd.DataFrame,
    *,
    side_score_col: str,
    component_cols: list[str],
    price_col: str,
    component_threshold: float,
) -> pd.Series:
    entry_ok = pd.to_numeric(work.get(side_score_col), errors="coerce").notna()
    entry_ok &= pd.to_numeric(work.get(price_col), errors="coerce").gt(0.0).fillna(False)
    for col in component_cols:
        values = pd.to_numeric(work.get(col), errors="coerce")
        entry_ok &= values.gt(float(component_threshold)).fillna(False)
    return pd.Series(entry_ok, index=work.index, dtype=bool)


def build_robinhood_live_trade_plan(
    *,
    latest_scored_df: pd.DataFrame,
    current_positions: pd.DataFrame | None,
    top_k: int,
    score_col: str,
    component_threshold: float,
    account_equity: float,
    price_col: str = "close",
) -> dict[str, Any]:
    work = latest_scored_df.copy()
    if work.empty:
        return {
            "summary": pd.DataFrame([{"top_k": int(top_k), "target_positions": 0, "buy_orders": 0, "sell_orders": 0, "short_orders": 0, "cover_orders": 0}]),
            "target_portfolio": pd.DataFrame(),
            "actions": pd.DataFrame(),
            "watchlist": pd.DataFrame(),
        }

    work.index = pd.Index([str(idx).strip().upper() for idx in work.index], name="symbol")
    work["prob_buy"] = pd.to_numeric(work.get("prob_buy", work.get("clf__prob_1")), errors="coerce")
    work["prob_short"] = pd.to_numeric(work.get("prob_short"), errors="coerce")
    missing_short = work["prob_short"].isna()
    work.loc[missing_short, "prob_short"] = 1.0 - work.loc[missing_short, "prob_buy"].fillna(0.0)
    work[price_col] = pd.to_numeric(work.get(price_col), errors="coerce")

    long_score_col = str(score_col)
    short_score_col = resolve_short_score_col(long_score_col)
    long_component_cols = resolve_component_cols(long_score_col)
    short_component_cols = resolve_component_cols(short_score_col)

    required_cols = list(
        dict.fromkeys(
            [price_col, "prob_buy", "prob_short", long_score_col, short_score_col, *long_component_cols, *short_component_cols]
        )
    )
    missing_cols = [col for col in required_cols if col not in work.columns]
    if missing_cols:
        raise KeyError(f"Missing required latest-score columns for Robinhood plan: {missing_cols}")

    work.loc[:, required_cols] = work[required_cols].apply(pd.to_numeric, errors="coerce")
    work["long_entry_ok"] = _build_entry_ok_by_side(
        work,
        side_score_col=long_score_col,
        component_cols=long_component_cols,
        price_col=price_col,
        component_threshold=component_threshold,
    )
    work["short_entry_ok"] = _build_entry_ok_by_side(
        work,
        side_score_col=short_score_col,
        component_cols=short_component_cols,
        price_col=price_col,
        component_threshold=component_threshold,
    )

    current_df = current_positions.copy() if current_positions is not None else pd.DataFrame(columns=["symbol", "quantity"])
    current_map: dict[str, dict[str, Any]] = {}
    for _, row in current_df.iterrows():
        symbol = str(row.get("symbol") or "").strip().upper()
        quantity = pd.to_numeric(pd.Series([row.get("quantity")]), errors="coerce").iloc[0]
        if not symbol or pd.isna(quantity) or float(quantity) == 0.0:
            continue
        current_map[symbol] = {
            "quantity": float(quantity),
            "side": 1 if float(quantity) > 0.0 else -1,
        }

    retained_side_by_symbol: dict[str, int] = {}
    unscored_retained_symbols: set[str] = set()
    exits: list[dict[str, Any]] = []
    for symbol, current in current_map.items():
        if symbol not in work.index:
            retained_side_by_symbol[symbol] = int(current["side"])
            unscored_retained_symbols.add(symbol)
            continue
        row = work.loc[symbol]
        price_ok = pd.notna(row[price_col]) and float(row[price_col]) > 0.0
        probs_ok = pd.notna(row["prob_buy"]) and pd.notna(row["prob_short"])
        if (not price_ok) or (not probs_ok):
            exits.append(
                {
                    "symbol": symbol,
                    "action": "sell" if current["side"] > 0 else "cover",
                    "reason": "invalid_live_inputs",
                    "current_quantity": abs(float(current["quantity"])),
                }
            )
            continue
        if current["side"] > 0:
            if bool(float(row["prob_short"]) > float(row["prob_buy"])) or (not bool(row["long_entry_ok"])):
                exits.append(
                    {
                        "symbol": symbol,
                        "action": "sell",
                        "reason": "classifier_flipped_short" if bool(float(row["prob_short"]) > float(row["prob_buy"])) else "long_gate_failed",
                        "current_quantity": abs(float(current["quantity"])),
                    }
                )
                continue
            retained_side_by_symbol[symbol] = 1
        else:
            if bool(float(row["prob_buy"]) >= float(row["prob_short"])) or (not bool(row["short_entry_ok"])):
                exits.append(
                    {
                        "symbol": symbol,
                        "action": "cover",
                        "reason": "classifier_flipped_long" if bool(float(row["prob_buy"]) >= float(row["prob_short"])) else "short_gate_failed",
                        "current_quantity": abs(float(current["quantity"])),
                    }
                )
                continue
            retained_side_by_symbol[symbol] = -1

    capacity = max(0, int(top_k))
    slots_left = max(0, capacity - len(retained_side_by_symbol))
    candidate_rows: list[tuple[float, str, int]] = []
    for symbol, row in work.iterrows():
        if symbol in retained_side_by_symbol:
            continue
        if not (pd.notna(row[price_col]) and float(row[price_col]) > 0.0):
            continue
        long_ok = bool(row["long_entry_ok"]) and pd.notna(row[long_score_col])
        short_ok = bool(row["short_entry_ok"]) and pd.notna(row[short_score_col])
        if not long_ok and not short_ok:
            continue
        if long_ok and short_ok:
            long_value = float(row[long_score_col])
            short_value = float(row[short_score_col])
            best_side = 1 if long_value >= short_value else -1
            best_score = max(long_value, short_value)
        elif long_ok:
            best_side = 1
            best_score = float(row[long_score_col])
        else:
            best_side = -1
            best_score = float(row[short_score_col])
        candidate_rows.append((best_score, symbol, best_side))

    chosen_entries: list[tuple[str, int, float]] = []
    for score_value, symbol, side in sorted(candidate_rows, key=lambda row: (row[0], row[1]), reverse=True):
        if len(chosen_entries) >= slots_left:
            break
        chosen_entries.append((symbol, int(side), float(score_value)))

    target_side_by_symbol = dict(retained_side_by_symbol)
    for symbol, side, _score in chosen_entries:
        target_side_by_symbol[symbol] = int(side)

    target_weight = (1.0 / float(capacity)) if capacity > 0 and target_side_by_symbol else 0.0
    safe_equity = float(max(account_equity, 0.0))
    target_dollars = safe_equity * target_weight

    action_rows: list[dict[str, Any]] = []
    for exit_row in exits:
        symbol = str(exit_row["symbol"])
        row = work.loc[symbol] if symbol in work.index else pd.Series(dtype=object)
        action_rows.append(
            {
                "symbol": symbol,
                "action": str(exit_row["action"]),
                "reason": str(exit_row["reason"]),
                "order_quantity": int(round(float(exit_row.get("current_quantity") or 0.0))),
                "target_weight": 0.0,
                "target_dollars": 0.0,
                "price": pd.to_numeric(pd.Series([row.get(price_col)]), errors="coerce").iloc[0],
                "prob_buy": pd.to_numeric(pd.Series([row.get("prob_buy")]), errors="coerce").iloc[0],
                "prob_short": pd.to_numeric(pd.Series([row.get("prob_short")]), errors="coerce").iloc[0],
                "combined_score": np.nan,
                "direction": "Long" if exit_row["action"] == "sell" else "Short",
            }
        )

    for symbol, side in retained_side_by_symbol.items():
        has_score = symbol in work.index
        row = work.loc[symbol] if has_score else pd.Series(dtype=object)
        score_value = float(row[long_score_col]) if has_score and side > 0 else float(row[short_score_col]) if has_score else np.nan
        action_rows.append(
            {
                "symbol": symbol,
                "action": "hold",
                "reason": "signal_still_valid" if has_score else "score_unavailable_hold",
                "order_quantity": 0,
                "target_weight": target_weight,
                "target_dollars": target_dollars,
                "price": float(row[price_col]) if has_score else np.nan,
                "prob_buy": float(row["prob_buy"]) if has_score else np.nan,
                "prob_short": float(row["prob_short"]) if has_score else np.nan,
                "combined_score": score_value,
                "direction": "Long" if side > 0 else "Short",
            }
        )

    for symbol, side, score_value in chosen_entries:
        row = work.loc[symbol]
        planned_shares = 0
        if pd.notna(row[price_col]) and float(row[price_col]) > 0.0 and target_dollars > 0.0:
            planned_shares = int(np.floor(target_dollars / float(row[price_col])))
        action_rows.append(
            {
                "symbol": symbol,
                "action": "buy" if side > 0 else "short",
                "reason": "leaderboard_top_k_entry",
                "order_quantity": max(int(planned_shares), 0),
                "target_weight": target_weight,
                "target_dollars": target_dollars,
                "price": float(row[price_col]),
                "prob_buy": float(row["prob_buy"]),
                "prob_short": float(row["prob_short"]),
                "combined_score": score_value,
                "direction": "Long" if side > 0 else "Short",
            }
        )

    actions = pd.DataFrame(action_rows)
    if not actions.empty:
        action_priority = {"sell": 0, "cover": 1, "buy": 2, "short": 3, "hold": 4}
        actions["_priority"] = actions["action"].map(action_priority).fillna(99)
        actions = actions.sort_values(["_priority", "combined_score", "symbol"], ascending=[True, False, True], kind="stable").drop(columns=["_priority"])

    actionable = actions.loc[actions["action"].isin(["buy", "sell", "short", "cover"])].copy() if not actions.empty else pd.DataFrame()
    actionable["robinhood_order_action"] = actionable["action"].map(
        {"buy": "BUY", "cover": "BUY", "sell": "SELL", "short": "SELL"}
    )
    actionable["position_effect"] = actionable["action"].map(
        {"buy": "open_long", "cover": "close_short", "sell": "close_long", "short": "open_short"}
    )
    actionable["skip_submit"] = actionable["order_quantity"].fillna(0).astype(float).le(0.0)

    target_portfolio_rows: list[dict[str, Any]] = []
    for symbol, side in sorted(target_side_by_symbol.items()):
        if symbol not in work.index:
            target_portfolio_rows.append(
                {
                    "symbol": symbol,
                    "direction": "Long" if side > 0 else "Short",
                    "target_weight": target_weight,
                    "target_dollars": target_dollars,
                    "price": np.nan,
                    "combined_score": np.nan,
                    "prob_buy": np.nan,
                    "prob_short": np.nan,
                    "status": "hold_unscored",
                }
            )
            continue
        row = work.loc[symbol]
        score_value = float(row[long_score_col]) if side > 0 else float(row[short_score_col])
        target_portfolio_rows.append(
            {
                "symbol": symbol,
                "direction": "Long" if side > 0 else "Short",
                "target_weight": target_weight,
                "target_dollars": target_dollars,
                "price": float(row[price_col]),
                "combined_score": score_value,
                "prob_buy": float(row["prob_buy"]),
                "prob_short": float(row["prob_short"]),
            }
        )
    target_portfolio = pd.DataFrame(target_portfolio_rows).sort_values(["combined_score", "symbol"], ascending=[False, True], kind="stable") if target_portfolio_rows else pd.DataFrame()

    watchlist_rows: list[dict[str, Any]] = []
    for score_value, symbol, side in sorted(candidate_rows, key=lambda row: (row[0], row[1]), reverse=True)[: max(20, capacity * 3 if capacity > 0 else 20)]:
        row = work.loc[symbol]
        watchlist_rows.append(
            {
                "symbol": symbol,
                "direction": "Long" if side > 0 else "Short",
                "combined_score": float(score_value),
                "price": float(row[price_col]),
                "prob_buy": float(row["prob_buy"]),
                "prob_short": float(row["prob_short"]),
            }
        )
    watchlist = pd.DataFrame(watchlist_rows)

    summary = pd.DataFrame(
        [
            {
                "top_k": int(capacity),
                "current_positions": int(len(current_map)),
                "positions_kept": int(len(retained_side_by_symbol)),
                "positions_exited": int(len(exits)),
                "new_entries": int(len(chosen_entries)),
                "target_positions": int(len(target_side_by_symbol)),
                "buy_orders": int((actions["action"] == "buy").sum()) if not actions.empty else 0,
                "sell_orders": int((actions["action"] == "sell").sum()) if not actions.empty else 0,
                "short_orders": int((actions["action"] == "short").sum()) if not actions.empty else 0,
                "cover_orders": int((actions["action"] == "cover").sum()) if not actions.empty else 0,
                "target_weight_per_position": target_weight,
                "held_symbols_missing_scores": int(len(unscored_retained_symbols)),
                "account_equity": safe_equity,
            }
        ]
    )
    return {
        "summary": summary,
        "target_portfolio": target_portfolio.reset_index(drop=True),
        "actions": actions.reset_index(drop=True),
        "actionable_orders": actionable.reset_index(drop=True),
        "watchlist": watchlist.reset_index(drop=True),
    }


def submit_robinhood_equity_orders(
    *,
    orders_df: pd.DataFrame,
    extended_hours: bool = False,
    account_number: str | None = None,
) -> pd.DataFrame:
    rh = _require_robin_stocks()
    if orders_df is None or orders_df.empty:
        return pd.DataFrame(columns=["symbol", "action", "submitted", "response"])

    responses: list[dict[str, Any]] = []
    for _, row in orders_df.iterrows():
        symbol = str(row.get("symbol") or "").strip().upper()
        action = str(row.get("action") or "").strip().lower()
        qty = pd.to_numeric(pd.Series([row.get("order_quantity")]), errors="coerce").iloc[0]
        if not symbol or pd.isna(qty) or int(qty) <= 0:
            responses.append({"symbol": symbol, "action": action, "submitted": False, "response": {"skipped": "non_positive_quantity"}})
            continue
        quantity = int(qty)
        if action in {"buy", "cover"}:
            response = rh.order_buy_market(
                symbol,
                quantity,
                account_number=account_number,
                timeInForce="gfd",
                extendedHours=bool(extended_hours),
                jsonify=True,
            )
        elif action in {"sell", "short"}:
            response = rh.order_sell_market(
                symbol,
                quantity,
                account_number=account_number,
                timeInForce="gfd",
                extendedHours=bool(extended_hours),
                jsonify=True,
            )
        else:
            responses.append({"symbol": symbol, "action": action, "submitted": False, "response": {"skipped": "unsupported_action"}})
            continue
        responses.append({"symbol": symbol, "action": action, "submitted": True, "response": response})
    return pd.DataFrame(responses)
