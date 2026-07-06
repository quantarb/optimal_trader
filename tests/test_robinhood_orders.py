from __future__ import annotations

import pandas as pd

from platforms.brokers import robinhood


def test_submit_robinhood_option_orders_uses_explicit_buy_limit(monkeypatch):
    calls: list[dict[str, object]] = []

    class FakeRobinhood:
        def order_buy_option_limit(self, *args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return {"id": "order-1", "state": "queued", "price": args[2]}

    monkeypatch.setattr(robinhood, "_require_robin_stocks", lambda: FakeRobinhood())

    orders = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "action": "buy_to_open_call",
                "option_type": "call",
                "expiry_date": "2026-10-16",
                "strike_price": 200.0,
                "quantity": 1,
                "order_type": "limit",
                "limit_order_price": 0.20,
                "bid_price": 2.00,
            }
        ]
    )

    result = robinhood.submit_robinhood_option_orders(orders_df=orders)

    assert bool(result.loc[0, "submitted"]) is True
    assert calls[0]["args"][2] == 0.20
    assert calls[0]["kwargs"]["timeInForce"] == "gtc"
