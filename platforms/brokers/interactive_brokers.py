from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pandas as pd

from platforms.brokers.option_pricing import normalize_option_limit_price


def load_ib_components() -> dict[str, Any]:
    try:
        from ib_insync import IB, Stock, Option, MarketOrder, LimitOrder, Order, TagValue, util
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ib_insync is required for IBKR trading. Install it in the notebook environment before enabling broker mode."
        ) from exc
    return {
        "IB": IB,
        "Stock": Stock,
        "Option": Option,
        "MarketOrder": MarketOrder,
        "LimitOrder": LimitOrder,
        "Order": Order,
        "TagValue": TagValue,
        "util": util,
    }


def cancel_all_ib_open_orders(ib, wait_seconds: float = 1.0) -> pd.DataFrame:
    _prime_ib_open_orders(ib)
    open_trades = list(ib.openTrades())
    open_rows: list[dict[str, Any]] = []
    for trade in open_trades:
        contract = getattr(trade, "contract", None)
        order = getattr(trade, "order", None)
        order_status = getattr(trade, "orderStatus", None)
        open_rows.append(
            {
                "symbol": str(getattr(contract, "symbol", "")).strip().upper(),
                "secType": str(getattr(contract, "secType", "")),
                "order_id": float(getattr(order, "orderId", np.nan)),
                "action": str(getattr(order, "action", "")),
                "order_type": str(getattr(order, "orderType", "")),
                "status": str(getattr(order_status, "status", "")),
            }
        )
    ib.reqGlobalCancel()
    if wait_seconds and float(wait_seconds) > 0:
        ib.sleep(float(wait_seconds))
    return pd.DataFrame(open_rows)


def _safe_ib_request(ib, label: str, awaitable, timeout: float | None = None, default=None):
    try:
        if timeout is None:
            return ib._run(awaitable)
        timeout_value = float(timeout)
        if timeout_value > 0:
            return ib._run(asyncio.wait_for(awaitable, timeout_value))
        return ib._run(awaitable)
    except Exception:
        return default


def _prime_ib_open_orders(ib, timeout: float = 5.0):
    if list(ib.openTrades()):
        return
    _safe_ib_request(ib, "open orders", ib.reqOpenOrdersAsync(), timeout=timeout, default=[])


def _prime_ib_positions(ib, timeout: float = 5.0):
    if getattr(getattr(ib, "wrapper", None), "positions", None):
        try:
            if any(ib.wrapper.positions.values()):
                return
        except Exception:
            pass
    _safe_ib_request(ib, "positions", ib.reqPositionsAsync(), timeout=timeout, default=[])


def _resolve_ib_account(ib, broker_cfg: dict[str, Any] | None = None) -> str:
    account = str((broker_cfg or {}).get("account", "")).strip()
    if account:
        return account
    try:
        accounts = list(getattr(ib.client, "getAccounts", lambda: [])() or [])
    except Exception:
        accounts = []
    return str(accounts[0]).strip() if accounts else ""


def _prime_ib_account_updates(ib, broker_cfg: dict[str, Any] | None = None, timeout: float = 5.0):
    account = _resolve_ib_account(ib, broker_cfg)
    if not account:
        return
    if getattr(getattr(ib, "wrapper", None), "accountValues", None):
        try:
            if any(v.account == account for v in ib.wrapper.accountValues.values()):
                return
        except Exception:
            pass
    _safe_ib_request(ib, "account updates", ib.reqAccountUpdatesAsync(account), timeout=timeout, default=None)


def _prime_ib_account_summary(ib, timeout: float = 5.0):
    acct_summary = getattr(getattr(ib, "wrapper", None), "acctSummary", None)
    if acct_summary:
        return
    _safe_ib_request(ib, "account summary", ib.reqAccountSummaryAsync(), timeout=timeout, default=None)


def connect_ibkr(broker_cfg: dict[str, Any]) -> tuple[Any, dict[str, Any], pd.DataFrame]:
    mods = load_ib_components()
    try:
        mods["util"].startLoop()
    except Exception:
        pass
    ib = mods["IB"]()
    host = str(broker_cfg.get("host", "127.0.0.1"))
    port = int(broker_cfg.get("port", 7497))
    client_id = int(broker_cfg.get("client_id", 17))
    connect_timeout = float(broker_cfg.get("connect_timeout", 8.0))
    sync_timeout = float(broker_cfg.get("sync_timeout", 5.0))
    try:
        ib.RequestTimeout = sync_timeout
    except Exception:
        pass
    try:
        ib.client.connect(
            host=host,
            port=port,
            clientId=client_id,
            timeout=connect_timeout,
        )
        if client_id == 0:
            try:
                ib.reqAutoOpenOrders(True)
            except Exception:
                pass
    except Exception as exc:
        raise RuntimeError(
            f"Failed to connect to IBKR at {host}:{port} with clientId={client_id}. "
            "The module now uses a lightweight socket connect instead of IB.connect() full synchronization. "
            "If this still fails, make sure TWS or IB Gateway is running, API access is enabled, and the configured port matches your IBKR settings."
        ) from exc
    cancelled_orders = pd.DataFrame()
    if bool(broker_cfg.get("cancel_open_orders_on_connect", True)):
        cancelled_orders = cancel_all_ib_open_orders(
            ib,
            wait_seconds=float(broker_cfg.get("cancel_wait_seconds", 1.0)),
        )
    return ib, mods, cancelled_orders


def ensure_ib_connection(
    *,
    broker_cfg: dict[str, Any],
    ib=None,
    mods=None,
) -> tuple[Any, dict[str, Any], bool, pd.DataFrame]:
    ib_instance = ib
    mods_instance = mods
    owns_connection = False
    cancelled_orders = pd.DataFrame()
    is_connected = bool(getattr(ib_instance, "isConnected", lambda: False)()) if ib_instance is not None else False
    if not is_connected:
        ib_instance, mods_instance, cancelled_orders = connect_ibkr(broker_cfg)
        owns_connection = True
    elif mods_instance is None:
        mods_instance = load_ib_components()
    return ib_instance, mods_instance, owns_connection, cancelled_orders


def fetch_ib_option_positions(ib, account: str | None = None, right: str = "C") -> pd.DataFrame:
    _prime_ib_positions(ib)
    rows: list[dict[str, Any]] = []
    positions = ib.positions(account) if account else ib.positions()
    for pos in positions:
        contract = pos.contract
        if getattr(contract, "secType", "") != "OPT":
            continue
        if right and str(getattr(contract, "right", "")).upper() != str(right).upper():
            continue
        rows.append(
            {
                "symbol": str(getattr(contract, "symbol", "")).strip().upper(),
                "localSymbol": getattr(contract, "localSymbol", ""),
                "conId": int(getattr(contract, "conId", 0) or 0),
                "expiry": str(getattr(contract, "lastTradeDateOrContractMonth", "")),
                "strike": float(getattr(contract, "strike", np.nan)),
                "right": str(getattr(contract, "right", "")),
                "exchange": str(getattr(contract, "exchange", "")),
                "tradingClass": str(getattr(contract, "tradingClass", "")),
                "currency": str(getattr(contract, "currency", "")),
                "position": int(round(float(getattr(pos, "position", 0) or 0))),
                "avgCost": float(getattr(pos, "avgCost", np.nan)),
                "contract": contract,
            }
        )
    return pd.DataFrame(rows)


def fetch_ib_all_positions(ib, account: str | None = None) -> pd.DataFrame:
    _prime_ib_positions(ib)
    rows: list[dict[str, Any]] = []
    positions = ib.positions(account) if account else ib.positions()
    for pos in positions:
        contract = pos.contract
        rows.append(
            {
                "symbol": str(getattr(contract, "symbol", "")).strip().upper(),
                "localSymbol": getattr(contract, "localSymbol", ""),
                "secType": str(getattr(contract, "secType", "")),
                "conId": int(getattr(contract, "conId", 0) or 0),
                "expiry": str(getattr(contract, "lastTradeDateOrContractMonth", "")),
                "strike": float(getattr(contract, "strike", np.nan)),
                "right": str(getattr(contract, "right", "")),
                "exchange": str(getattr(contract, "exchange", "")),
                "tradingClass": str(getattr(contract, "tradingClass", "")),
                "currency": str(getattr(contract, "currency", "")),
                "position": int(round(float(getattr(pos, "position", 0) or 0))),
                "avgCost": float(getattr(pos, "avgCost", np.nan)),
                "contract": contract,
            }
        )
    return pd.DataFrame(rows)


def fetch_ib_portfolio_snapshot(ib, account: str | None = None) -> pd.DataFrame:
    _prime_ib_account_updates(ib, {"account": account} if account else None)
    rows: list[dict[str, Any]] = []
    portfolio_items = ib.portfolio(account) if account else ib.portfolio()
    for item in portfolio_items:
        contract = getattr(item, "contract", None)
        rows.append(
            {
                "symbol": str(getattr(contract, "symbol", "")).strip().upper(),
                "localSymbol": getattr(contract, "localSymbol", ""),
                "secType": str(getattr(contract, "secType", "")),
                "conId": int(getattr(contract, "conId", 0) or 0),
                "expiry": str(getattr(contract, "lastTradeDateOrContractMonth", "")),
                "strike": float(getattr(contract, "strike", np.nan)),
                "right": str(getattr(contract, "right", "")),
                "exchange": str(getattr(contract, "exchange", "")),
                "currency": str(getattr(contract, "currency", "")),
                "position": float(getattr(item, "position", np.nan)),
                "marketPrice": float(getattr(item, "marketPrice", np.nan)),
                "marketValue": float(getattr(item, "marketValue", np.nan)),
                "averageCost": float(getattr(item, "averageCost", np.nan)),
                "unrealizedPNL": float(getattr(item, "unrealizedPNL", np.nan)),
                "realizedPNL": float(getattr(item, "realizedPNL", np.nan)),
                "account": str(getattr(item, "account", "")),
                "contract": contract,
            }
        )
    portfolio_df = pd.DataFrame(rows)
    if portfolio_df.empty:
        return portfolio_df
    return portfolio_df.sort_values(
        ["secType", "symbol", "expiry", "strike"],
        ascending=[True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _get_account_net_liquidation(ib, broker_cfg: dict[str, Any]) -> float:
    account_value = np.nan
    _prime_ib_account_summary(ib)
    if broker_cfg.get("account"):
        try:
            summary_rows = ib.accountSummary(account=str(broker_cfg.get("account")))
        except TypeError:
            summary_rows = ib.accountSummary()
    else:
        summary_rows = ib.accountSummary()
    try:
        for item in summary_rows:
            if getattr(item, "tag", "") == "NetLiquidation":
                account_value = float(getattr(item, "value", np.nan))
                break
    except Exception:
        pass
    return account_value


def _estimate_option_price(selected: dict[str, Any]) -> float:
    ask = float(selected.get("ask", np.nan))
    bid = float(selected.get("bid", np.nan))
    market_price = float(selected.get("market_price", np.nan))
    last = float(selected.get("last", np.nan))
    close = float(selected.get("close", np.nan))
    model_price = float(selected.get("model_price", np.nan))
    if np.isfinite(ask) and ask > 0:
        return ask
    if np.isfinite(market_price) and market_price > 0:
        return market_price
    if np.isfinite(model_price) and model_price > 0:
        return model_price
    if np.isfinite(last) and last > 0:
        return last
    if np.isfinite(close) and close > 0:
        return close
    if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if np.isfinite(bid) and bid > 0:
        return bid
    return np.nan


def _extract_ticker_float(ticker, attr: str) -> float:
    try:
        return float(getattr(ticker, attr, np.nan))
    except Exception:
        return np.nan


def _extract_greek_price(ticker) -> float:
    for attr in ("modelGreeks", "askGreeks", "bidGreeks", "lastGreeks"):
        try:
            greeks = getattr(ticker, attr, None)
            if greeks is None:
                continue
            price = float(getattr(greeks, "optPrice", np.nan))
            if np.isfinite(price) and price > 0:
                return price
        except Exception:
            continue
    return np.nan


def _fetch_option_quote(ib, option_contract, option_cfg: dict[str, Any]) -> dict[str, float]:
    if not bool(option_cfg.get("include_market_data", True)):
        return {
            "bid": np.nan,
            "ask": np.nan,
            "last": np.nan,
            "close": np.nan,
            "market_price": np.nan,
            "model_price": np.nan,
        }

    try:
        ib.reqMarketDataType(int(option_cfg.get("market_data_type", 3)))
    except Exception:
        pass

    quote = {
        "bid": np.nan,
        "ask": np.nan,
        "last": np.nan,
        "close": np.nan,
        "market_price": np.nan,
        "model_price": np.nan,
    }

    try:
        tickers = ib.reqTickers(option_contract)
        if tickers:
            ticker = tickers[0]
            quote["bid"] = _extract_ticker_float(ticker, "bid")
            quote["ask"] = _extract_ticker_float(ticker, "ask")
            quote["last"] = _extract_ticker_float(ticker, "last")
            quote["close"] = _extract_ticker_float(ticker, "close")
            try:
                quote["market_price"] = float(ticker.marketPrice()) if hasattr(ticker, "marketPrice") else np.nan
            except Exception:
                quote["market_price"] = np.nan
            quote["model_price"] = _extract_greek_price(ticker)
    except Exception:
        pass

    if any(np.isfinite(v) and v > 0 for v in quote.values()):
        return quote

    live_ticker = None
    try:
        live_ticker = ib.reqMktData(option_contract, "", False, False)
        ib.sleep(float(option_cfg.get("market_data_wait_seconds", 1.5)))
        if live_ticker is not None:
            quote["bid"] = _extract_ticker_float(live_ticker, "bid")
            quote["ask"] = _extract_ticker_float(live_ticker, "ask")
            quote["last"] = _extract_ticker_float(live_ticker, "last")
            quote["close"] = _extract_ticker_float(live_ticker, "close")
            try:
                quote["market_price"] = float(live_ticker.marketPrice()) if hasattr(live_ticker, "marketPrice") else np.nan
            except Exception:
                quote["market_price"] = np.nan
            quote["model_price"] = _extract_greek_price(live_ticker)
    except Exception:
        pass
    finally:
        if live_ticker is not None:
            try:
                ib.cancelMktData(option_contract)
            except Exception:
                pass
    return quote


def fetch_ib_open_orders(ib, account: str | None = None) -> pd.DataFrame:
    _prime_ib_open_orders(ib)
    rows: list[dict[str, Any]] = []
    for trade in list(ib.openTrades()):
        contract = getattr(trade, "contract", None)
        order = getattr(trade, "order", None)
        order_status = getattr(trade, "orderStatus", None)
        account_name = str(getattr(order, "account", "") or getattr(order_status, "account", "") or "")
        if account and account_name and account_name != str(account):
            continue
        rows.append(
            {
                "symbol": str(getattr(contract, "symbol", "")).strip().upper(),
                "localSymbol": str(getattr(contract, "localSymbol", "")),
                "secType": str(getattr(contract, "secType", "")),
                "conId": int(getattr(contract, "conId", 0) or 0),
                "expiry": str(getattr(contract, "lastTradeDateOrContractMonth", "")),
                "strike": float(getattr(contract, "strike", np.nan)),
                "right": str(getattr(contract, "right", "")),
                "exchange": str(getattr(contract, "exchange", "")),
                "currency": str(getattr(contract, "currency", "")),
                "orderId": int(getattr(order, "orderId", 0) or 0),
                "permId": int(getattr(order, "permId", 0) or 0),
                "action": str(getattr(order, "action", "")),
                "orderType": str(getattr(order, "orderType", "")),
                "totalQuantity": float(getattr(order, "totalQuantity", np.nan)),
                "lmtPrice": float(getattr(order, "lmtPrice", np.nan)),
                "auxPrice": float(getattr(order, "auxPrice", np.nan)),
                "tif": str(getattr(order, "tif", "")),
                "status": str(getattr(order_status, "status", "")),
                "filled": float(getattr(order_status, "filled", np.nan)),
                "remaining": float(getattr(order_status, "remaining", np.nan)),
                "avgFillPrice": float(getattr(order_status, "avgFillPrice", np.nan)),
                "account": account_name,
                "contract": contract,
                "order": order,
                "trade": trade,
            }
        )
    orders_df = pd.DataFrame(rows)
    if orders_df.empty:
        return orders_df
    return orders_df.sort_values(
        ["status", "secType", "symbol", "expiry", "strike", "orderId"],
        ascending=[True, True, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)


def choose_ib_expiry(
    expirations,
    min_days_to_expiry: int,
    max_days_to_expiry: int | None = None,
    as_of_date=None,
) -> str:
    if as_of_date is None:
        as_of_date = pd.Timestamp.today().normalize()
    expiry_info: list[tuple[str, pd.Timestamp, int]] = []
    for raw in sorted(set(expirations or [])):
        try:
            expiry_ts = pd.Timestamp(str(raw))
        except Exception:
            continue
        dte = int((expiry_ts.normalize() - as_of_date.normalize()).days)
        if dte < int(min_days_to_expiry):
            continue
        if max_days_to_expiry is not None and dte > int(max_days_to_expiry):
            continue
        expiry_info.append((str(raw), expiry_ts, dte))
    if expiry_info:
        expiry_info.sort(key=lambda item: (item[2], item[1]))
        return expiry_info[0][0]
    fallback: list[tuple[str, pd.Timestamp, int]] = []
    for raw in sorted(set(expirations or [])):
        try:
            expiry_ts = pd.Timestamp(str(raw))
        except Exception:
            continue
        dte = int((expiry_ts.normalize() - as_of_date.normalize()).days)
        if dte >= 0:
            fallback.append((str(raw), expiry_ts, dte))
    if not fallback:
        raise RuntimeError("No valid option expirations returned by IBKR for the requested underlying.")
    fallback.sort(key=lambda item: (abs(item[2] - int(min_days_to_expiry)), item[1]))
    return fallback[0][0]


def _rank_ib_expiries(
    expirations,
    *,
    min_days_to_expiry: int,
    max_days_to_expiry: int | None = None,
    as_of_date=None,
) -> list[str]:
    if as_of_date is None:
        as_of_date = pd.Timestamp.today().normalize()
    preferred: list[tuple[str, pd.Timestamp, int]] = []
    fallback: list[tuple[str, pd.Timestamp, int]] = []
    target_dte = int(min_days_to_expiry)
    for raw in sorted(set(expirations or [])):
        try:
            expiry_ts = pd.Timestamp(str(raw))
        except Exception:
            continue
        dte = int((expiry_ts.normalize() - as_of_date.normalize()).days)
        if dte < 0:
            continue
        item = (str(raw), expiry_ts, dte)
        if dte >= target_dte and (max_days_to_expiry is None or dte <= int(max_days_to_expiry)):
            preferred.append(item)
        else:
            fallback.append(item)
    preferred.sort(key=lambda item: (abs(item[2] - target_dte), item[1]))
    fallback.sort(key=lambda item: (abs(item[2] - target_dte), item[1]))
    return [item[0] for item in preferred + fallback]


def _rank_ib_strikes(
    strikes,
    *,
    target_strike: float,
    right: str,
) -> list[float]:
    valid = sorted(float(x) for x in set(strikes or []) if np.isfinite(x) and float(x) > 0)
    if not valid:
        return []
    right_value = str(right or "C").upper()
    if right_value == "C":
        return sorted(
            valid,
            key=lambda strike: (
                0 if strike >= float(target_strike) else 1,
                abs(float(strike) - float(target_strike)),
                float(strike),
            ),
        )
    return sorted(
        valid,
        key=lambda strike: (
            0 if strike <= float(target_strike) else 1,
            abs(float(strike) - float(target_strike)),
            -float(strike),
        ),
    )


def select_otm_call_contract(ib, mods, symbol: str, spot_price: float, option_cfg: dict[str, Any], as_of_date=None) -> dict[str, Any]:
    stock = mods["Stock"](
        str(symbol).upper(),
        str(option_cfg.get("exchange", "SMART")),
        str(option_cfg.get("currency", "USD")),
    )
    qualified_stock = ib.qualifyContracts(stock)
    if not qualified_stock:
        raise RuntimeError(f"Unable to qualify IBKR stock contract for {symbol}.")
    stock = qualified_stock[0]

    chains = ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)
    if not chains:
        raise RuntimeError(f"No option chain metadata returned by IBKR for {symbol}.")

    preferred_exchange = str(option_cfg.get("exchange", "SMART")).upper()
    filtered_chains = [chain for chain in chains if str(getattr(chain, "exchange", "")).upper() == preferred_exchange]
    candidate_chains = filtered_chains or list(chains)
    candidate_chains = sorted(
        candidate_chains,
        key=lambda chain: (
            0 if str(getattr(chain, "tradingClass", "")).upper() == str(symbol).upper() else 1,
            str(getattr(chain, "tradingClass", "")),
        ),
    )

    min_dte = int(option_cfg.get("min_days_to_expiry", option_cfg.get("tenor_days", 60)))
    max_dte = option_cfg.get("max_days_to_expiry")
    right = str(option_cfg.get("right", "C")).upper()
    target_strike = float(spot_price) * float(option_cfg.get("strike_multiplier", 1.05))
    option_contract = None
    attempted_contracts: list[str] = []
    for chain in candidate_chains:
        expiry_candidates = _rank_ib_expiries(
            chain.expirations,
            min_days_to_expiry=min_dte,
            max_days_to_expiry=max_dte,
            as_of_date=as_of_date,
        )
        strike_candidates = _rank_ib_strikes(
            chain.strikes,
            target_strike=target_strike,
            right=right,
        )
        if not expiry_candidates or not strike_candidates:
            continue
        for expiry in expiry_candidates[: max(1, int(option_cfg.get("max_expiry_candidates", 8)))]:
            for strike in strike_candidates[: max(1, int(option_cfg.get("max_strike_candidates", 12)))]:
                candidate_contract = mods["Option"](
                    str(symbol).upper(),
                    str(expiry),
                    float(strike),
                    right,
                    str(option_cfg.get("exchange", "SMART")),
                    tradingClass=str(getattr(chain, "tradingClass", "") or str(symbol).upper()),
                    currency=str(option_cfg.get("currency", "USD")),
                )
                qualified_option = ib.qualifyContracts(candidate_contract)
                if qualified_option:
                    option_contract = qualified_option[0]
                    break
                attempted_contracts.append(
                    f"{symbol} {expiry} {float(strike):g} {right} {str(getattr(chain, 'tradingClass', '')).upper()}"
                )
            if option_contract is not None:
                break
        if option_contract is not None:
            break
    if option_contract is None:
        attempted_preview = ", ".join(attempted_contracts[:5])
        if len(attempted_contracts) > 5:
            attempted_preview += ", ..."
        raise RuntimeError(
            f"Unable to qualify IBKR option contract for {symbol}. "
            f"Tried nearest expiry/strike candidates around target strike {target_strike:.2f}"
            + (f": {attempted_preview}" if attempted_preview else ".")
        )

    quote = _fetch_option_quote(ib, option_contract, option_cfg)

    return {
        "contract": option_contract,
        "symbol": str(symbol).upper(),
        "expiry": str(option_contract.lastTradeDateOrContractMonth),
        "strike": float(option_contract.strike),
        "right": str(option_contract.right),
        "exchange": str(option_contract.exchange),
        "tradingClass": str(option_contract.tradingClass),
        "currency": str(option_contract.currency),
        "conId": int(option_contract.conId),
        "multiplier": float(getattr(option_contract, "multiplier", 100) or 100),
        "spot_price": float(spot_price),
        "bid": quote["bid"],
        "ask": quote["ask"],
        "last": quote["last"],
        "close": quote["close"],
        "market_price": quote["market_price"],
        "model_price": quote["model_price"],
    }


def build_ib_otm_option_trade_plan(
    ib,
    mods,
    live_plan: dict[str, Any],
    current_option_positions: pd.DataFrame | None,
    option_cfg: dict[str, Any],
    broker_cfg: dict[str, Any],
    *,
    as_of_date=None,
    instrument: str = "otm_call",
) -> dict[str, pd.DataFrame]:
    target_portfolio = live_plan["target_portfolio"].copy()
    desired_contract_rows: list[dict[str, Any]] = []
    desired_contract_map: dict[tuple[str, str, float, str], dict[str, Any]] = {}
    skipped_rows: list[dict[str, Any]] = []

    account_value = _get_account_net_liquidation(ib, broker_cfg)
    contracts_per_position = max(0, int(option_cfg.get("contracts_per_position", 1)))
    size_by_target_weight = bool(option_cfg.get("size_positions_by_target_weight", True))
    max_contracts_per_position = option_cfg.get("max_contracts_per_position")
    for symbol, row in target_portfolio.iterrows():
        try:
            selected = select_otm_call_contract(
                ib,
                mods,
                symbol=symbol,
                spot_price=float(row[option_cfg.get("underlying_price_col", "close")]),
                option_cfg=option_cfg,
                as_of_date=as_of_date,
            )
        except Exception as exc:
            skipped_rows.append(
                {
                    "symbol": str(symbol).upper(),
                    "spot_price": float(row.get(option_cfg.get("underlying_price_col", "close"), np.nan)),
                    "target_weight": float(row.get("target_weight", 0.0)),
                    "status": str(row.get("status", "")),
                    "reason": "no_eligible_option_contract",
                    "error": str(exc),
                }
            )
            continue
        selected["target_weight"] = float(row.get("target_weight", 0.0))
        selected["status"] = str(row.get("status", "buy"))
        quote_price = _estimate_option_price(selected)
        multiplier = float(selected.get("multiplier", 100.0) or 100.0)
        estimated_contract_value = quote_price * multiplier if np.isfinite(quote_price) and quote_price > 0 and multiplier > 0 else np.nan
        target_contracts = contracts_per_position
        sizing_mode = "fixed_contracts_fallback"
        if (
            size_by_target_weight
            and np.isfinite(account_value)
            and account_value > 0
            and np.isfinite(estimated_contract_value)
            and estimated_contract_value > 0
        ):
            target_dollar = float(selected["target_weight"]) * float(account_value)
            target_contracts = max(1, int(np.floor(target_dollar / estimated_contract_value)))
            sizing_mode = "target_weight"
        if max_contracts_per_position is not None:
            target_contracts = min(int(target_contracts), max(0, int(max_contracts_per_position)))
        selected["quote_price"] = quote_price
        selected["estimated_contract_value"] = estimated_contract_value
        selected["target_contracts"] = int(target_contracts)
        selected["sizing_mode"] = sizing_mode
        desired_contract_rows.append(selected)
        desired_contract_map[(selected["symbol"], selected["expiry"], selected["strike"], selected["right"])] = selected

    desired_df = pd.DataFrame(desired_contract_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    current_positions = current_option_positions.copy() if current_option_positions is not None else pd.DataFrame()
    if current_positions.empty:
        current_positions = pd.DataFrame(columns=["symbol", "expiry", "strike", "right", "position", "conId", "contract"])

    order_rows: list[dict[str, Any]] = []
    existing_map: dict[tuple[str, str, float, str], pd.Series] = {}
    for _, row in current_positions.iterrows():
        key = (str(row["symbol"]).upper(), str(row["expiry"]), float(row["strike"]), str(row["right"]).upper())
        existing_map[key] = row

    all_keys = sorted(set(existing_map) | set(desired_contract_map))
    for key in all_keys:
        desired = desired_contract_map.get(key)
        existing = existing_map.get(key)
        current_qty = int(existing["position"]) if existing is not None else 0
        target_qty = int(desired["target_contracts"]) if desired is not None else 0
        delta = int(target_qty - current_qty)
        action = "hold"
        order_qty = 0
        reason = "already_at_target"
        contract = desired["contract"] if desired is not None else existing["contract"]
        bid = desired.get("bid", np.nan) if desired is not None else np.nan
        ask = desired.get("ask", np.nan) if desired is not None else np.nan
        if delta > 0:
            action = "BUY"
            order_qty = delta
            reason = "open_new_otm_call" if current_qty == 0 else "increase_to_target"
        elif delta < 0:
            action = "SELL"
            order_qty = abs(delta)
            reason = "close_obsolete_contract" if target_qty == 0 else "reduce_to_target"
        order_rows.append(
            {
                "symbol": key[0],
                "expiry": key[1],
                "strike": key[2],
                "right": key[3],
                "secType": "OPT",
                "conId": int(getattr(contract, "conId", 0) or 0),
                "current_contracts": current_qty,
                "target_contracts": target_qty,
                "order_action": action,
                "order_contracts": order_qty,
                "reason": reason,
                "quote_price": desired.get("quote_price", np.nan) if desired is not None else np.nan,
                "estimated_contract_value": desired.get("estimated_contract_value", np.nan) if desired is not None else np.nan,
                "sizing_mode": desired.get("sizing_mode", "") if desired is not None else "",
                "bid": bid,
                "ask": ask,
                "contract": contract,
            }
        )

    orders_df = pd.DataFrame(order_rows)
    if not orders_df.empty:
        action_order = {"SELL": 0, "BUY": 1, "hold": 2}
        orders_df["_order"] = orders_df["order_action"].map(action_order).fillna(99)
        orders_df = orders_df.sort_values(
            ["_order", "symbol", "expiry", "strike"],
            ascending=[True, True, True, True],
            kind="stable",
        ).drop(columns=["_order"])

    summary = pd.DataFrame(
        [
            {
                "instrument": str(instrument),
                "target_underlyings": int(len(target_portfolio)),
                "eligible_underlyings": int(len(desired_df)),
                "skipped_underlyings": int(len(skipped_df)),
                "current_option_positions": int(len(current_positions)),
                "buy_orders": int((orders_df["order_action"] == "BUY").sum()) if not orders_df.empty else 0,
                "sell_orders": int((orders_df["order_action"] == "SELL").sum()) if not orders_df.empty else 0,
                "account_net_liquidation": account_value,
                "contracts_per_position": contracts_per_position,
                "size_positions_by_target_weight": size_by_target_weight,
                "min_days_to_expiry": int(option_cfg.get("min_days_to_expiry", option_cfg.get("tenor_days", 60))),
                "strike_multiplier": float(option_cfg.get("strike_multiplier", 1.05)),
                "submit_orders": bool(broker_cfg.get("submit_orders", False)),
            }
        ]
    )

    return {
        "summary": summary,
        "desired_contracts": desired_df,
        "skipped_symbols": skipped_df,
        "current_option_positions": current_positions,
        "orders": orders_df,
    }


def build_ib_full_liquidation_plan(current_positions: pd.DataFrame | None, broker_cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    positions = current_positions.copy() if current_positions is not None else pd.DataFrame()
    if positions.empty:
        summary = pd.DataFrame(
            [
                {
                    "positions_to_liquidate": 0,
                    "buy_to_cover_orders": 0,
                    "sell_orders": 0,
                    "submit_orders": bool(broker_cfg.get("submit_orders", False)),
                }
            ]
        )
        return {"summary": summary, "positions": positions, "orders": pd.DataFrame()}

    order_rows: list[dict[str, Any]] = []
    for _, row in positions.iterrows():
        qty = int(row.get("position", 0) or 0)
        if qty == 0:
            continue
        action = "SELL" if qty > 0 else "BUY"
        order_rows.append(
            {
                "symbol": row.get("symbol", ""),
                "localSymbol": row.get("localSymbol", ""),
                "secType": row.get("secType", ""),
                "expiry": row.get("expiry", ""),
                "strike": row.get("strike", np.nan),
                "right": row.get("right", ""),
                "conId": int(row.get("conId", 0) or 0),
                "current_contracts": qty,
                "target_contracts": 0,
                "order_action": action,
                "order_contracts": abs(qty),
                "reason": "liquidate_all_positions",
                "contract": row.get("contract"),
                "bid": np.nan,
                "ask": np.nan,
            }
        )

    orders_df = pd.DataFrame(order_rows)
    if not orders_df.empty:
        action_order = {"SELL": 0, "BUY": 1, "hold": 2}
        orders_df["_order"] = orders_df["order_action"].map(action_order).fillna(99)
        orders_df = orders_df.sort_values(
            ["_order", "secType", "symbol", "expiry", "strike"],
            ascending=[True, True, True, True, True],
            kind="stable",
        ).drop(columns=["_order"])

    summary = pd.DataFrame(
        [
            {
                "positions_to_liquidate": int(len(orders_df)),
                "buy_to_cover_orders": int((orders_df["order_action"] == "BUY").sum()) if not orders_df.empty else 0,
                "sell_orders": int((orders_df["order_action"] == "SELL").sum()) if not orders_df.empty else 0,
                "submit_orders": bool(broker_cfg.get("submit_orders", False)),
            }
        ]
    )
    return {"summary": summary, "positions": positions, "orders": orders_df}


def build_ib_order(mods, row: pd.Series, broker_cfg: dict[str, Any], option_cfg: dict[str, Any]):
    action = str(row["order_action"]).upper()
    quantity = int(row["order_contracts"])
    if action not in {"BUY", "SELL"} or quantity <= 0:
        return None
    order_type = str(option_cfg.get("order_type", "MKT")).upper()
    sec_type = str(row.get("secType", "OPT")).upper()
    if order_type == "CLOSEPX":
        if sec_type != "OPT":
            raise RuntimeError(f"ClosePx is only configured here for options, but got secType={sec_type} for {row['symbol']}.")
        order = mods["Order"]()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = str(option_cfg.get("close_px_base_order_type", "MKT")).upper()
        if order.orderType == "LMT":
            if action == "BUY":
                ref_price = row.get("bid", np.nan)
                if not np.isfinite(ref_price) or float(ref_price) <= 0.0:
                    raise RuntimeError(f"Cannot build ClosePx limit buy order for {row['symbol']} without valid bid data.")
                order.lmtPrice = normalize_option_limit_price(float(ref_price), side="buy")
            else:
                ref_price = row.get("ask", np.nan)
                if not np.isfinite(ref_price) or float(ref_price) <= 0.0:
                    raise RuntimeError(f"Cannot build ClosePx limit sell order for {row['symbol']} without valid ask data.")
                order.lmtPrice = normalize_option_limit_price(float(ref_price), side="sell")
            if order.lmtPrice is None:
                raise RuntimeError(f"Cannot build ClosePx limit {action.lower()} order for {row['symbol']} with invalid option price.")
        order.algoStrategy = "ClosePx"
        order.algoParams = [
            mods["TagValue"]("maxPctVol", str(option_cfg.get("close_px_max_pct_vol", 0.1))),
            mods["TagValue"]("riskAversion", str(option_cfg.get("close_px_risk_aversion", "Neutral"))),
            mods["TagValue"]("startTime", str(option_cfg.get("close_px_start_time", "12:00:00 US/Eastern"))),
            mods["TagValue"]("forceCompletion", "1" if bool(option_cfg.get("close_px_force_completion", True)) else "0"),
        ]
    elif order_type == "LMT":
        if action == "BUY":
            ref_price = row.get("bid", np.nan)
            if not np.isfinite(ref_price) or float(ref_price) <= 0.0:
                raise RuntimeError(f"Cannot build limit buy order for {row['symbol']} without valid bid data.")
            limit_price = normalize_option_limit_price(float(ref_price), side="buy")
        else:
            ref_price = row.get("ask", np.nan)
            if not np.isfinite(ref_price) or float(ref_price) <= 0.0:
                raise RuntimeError(f"Cannot build limit sell order for {row['symbol']} without valid ask data.")
            limit_price = normalize_option_limit_price(float(ref_price), side="sell")
        if limit_price is None:
            raise RuntimeError(f"Cannot build limit {action.lower()} order for {row['symbol']} with invalid option price.")
        order = mods["LimitOrder"](action, quantity, limit_price)
    else:
        order = mods["MarketOrder"](action, quantity)
    account = str(broker_cfg.get("account", "")).strip()
    if account:
        order.account = account
    return order


def submit_ib_orders(ib, mods, orders_df: pd.DataFrame, broker_cfg: dict[str, Any], option_cfg: dict[str, Any]) -> pd.DataFrame:
    placement_rows: list[dict[str, Any]] = []
    submit_orders = bool(broker_cfg.get("submit_orders", False))
    for _, row in orders_df.iterrows():
        if str(row["order_action"]).upper() not in {"BUY", "SELL"} or int(row["order_contracts"]) <= 0:
            continue
        order = build_ib_order(mods, row, broker_cfg, option_cfg)
        status = "preview_only"
        order_id = np.nan
        if submit_orders:
            trade = ib.placeOrder(row["contract"], order)
            status = str(getattr(getattr(trade, "orderStatus", None), "status", "submitted"))
            order_id = float(getattr(getattr(trade, "order", None), "orderId", np.nan))
        placement_rows.append(
            {
                "symbol": row["symbol"],
                "expiry": row["expiry"],
                "strike": row["strike"],
                "right": row["right"],
                "secType": row.get("secType", "OPT"),
                "order_action": row["order_action"],
                "order_contracts": int(row["order_contracts"]),
                "status": status,
                "order_id": order_id,
            }
        )
    return pd.DataFrame(placement_rows)


def liquidate_all_positions(
    *,
    broker_cfg: dict[str, Any],
    option_cfg: dict[str, Any],
    ib=None,
    mods=None,
    disconnect_when_done: bool | None = None,
) -> dict[str, pd.DataFrame]:
    ib_instance, mods_instance, owns_connection, cancelled_orders = ensure_ib_connection(
        broker_cfg=broker_cfg,
        ib=ib,
        mods=mods,
    )
    if disconnect_when_done is None:
        disconnect_when_done = owns_connection
    try:
        current_positions = fetch_ib_all_positions(
            ib_instance,
            account=str(broker_cfg.get("account", "")).strip() or None,
        )
        liquidation_plan = build_ib_full_liquidation_plan(current_positions, broker_cfg)
        submission_results = submit_ib_orders(
            ib_instance,
            mods_instance,
            liquidation_plan["orders"],
            broker_cfg,
            option_cfg,
        )
        return {
            "cancelled_orders": cancelled_orders,
            "summary": liquidation_plan["summary"],
            "positions": liquidation_plan["positions"],
            "orders": liquidation_plan["orders"],
            "submission_results": submission_results,
        }
    finally:
        if disconnect_when_done:
            try:
                ib_instance.disconnect()
            except Exception:
                pass


__all__ = [
    "build_ib_full_liquidation_plan",
    "build_ib_order",
    "build_ib_otm_option_trade_plan",
    "cancel_all_ib_open_orders",
    "choose_ib_expiry",
    "connect_ibkr",
    "ensure_ib_connection",
    "fetch_ib_all_positions",
    "fetch_ib_option_positions",
    "liquidate_all_positions",
    "load_ib_components",
    "select_otm_call_contract",
    "submit_ib_orders",
]
