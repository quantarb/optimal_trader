from __future__ import annotations

import pandas as pd

from app import trading_app_v2_runtime as runtime


def test_apply_option_limit_policy_uses_bid_ask_ticks_and_preserves_cancels():
    orders = pd.DataFrame(
        [
            {"symbol": "AAPL", "action": "buy_to_open_call", "bid_price": 70.17, "ask_price": 70.25},
            {"symbol": "MSFT", "action": "sell_to_close_call", "bid_price": 70.10, "ask_price": 70.17},
            {"symbol": "TSLA", "action": "cancel_buy_to_open_call", "order_id": "order-1"},
        ]
    )

    priced = runtime.apply_option_limit_policy(orders, time_in_force="gtc")

    assert float(priced.loc[0, "limit_order_price"]) == 70.15
    assert priced.loc[0, "limit_price_source"] == "bid_price"
    assert float(priced.loc[1, "limit_order_price"]) == 70.20
    assert priced.loc[1, "limit_price_source"] == "ask_price"
    assert bool(priced.loc[2, "skip_submit"]) is False
    assert pd.isna(priced.loc[2].get("limit_order_price"))


def test_robinhood_option_orders_reconcile_before_new_entries():
    target_contracts = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "option_type": "call",
                "expiry_date": "2026-10-16",
                "strike_price": 200.0,
                "quantity": 1,
                "bid_price": 70.17,
                "ask_price": 70.25,
                "combined_score": 0.9,
            }
        ]
    )
    current_positions = pd.DataFrame(
        [
            {
                "symbol": "MSFT",
                "option_type": "call",
                "expiry_date": "2026-10-16",
                "strike_price": 400.0,
                "quantity": 1,
                "bid_price": 2.10,
                "ask_price": 2.33,
            }
        ]
    )
    pending_orders = pd.DataFrame(
        [
            {
                "symbol": "TSLA",
                "option_type": "call",
                "expiry_date": "2026-10-16",
                "strike_price": 300.0,
                "quantity": 1,
                "action": "buy_to_open_call",
                "order_id": "stale-order",
                "cancel_url": "https://example.invalid/cancel",
            }
        ]
    )

    plan = runtime.build_robinhood_option_orders(
        target_contracts=target_contracts,
        current_option_positions=current_positions,
        pending_option_orders=pending_orders,
        gate_discount_pct=90.0,
    )
    actions = plan["actions"]

    assert list(actions["action"]) == ["cancel_buy_to_open_call", "sell_to_close_call", "buy_to_open_call"]
    assert float(actions.loc[1, "limit_order_price"]) == 2.33
    assert actions.loc[1, "limit_price_source"] == "ask_price"
    assert float(actions.loc[2, "limit_order_price"]) == 70.15
    assert actions.loc[2, "limit_price_source"] == "bid_price"

