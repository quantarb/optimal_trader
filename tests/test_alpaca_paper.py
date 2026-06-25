from unittest import TestCase

from trading.alpaca_paper import build_equal_weight_order_plan


class AlpacaPaperOrderPlanTests(TestCase):
    def test_builds_rebalance_and_liquidation_orders(self):
        orders = build_equal_weight_order_plan(
            ["AAPL", "MSFT"],
            {"AAPL": 100.0, "MSFT": 200.0},
            {"AAPL": 2, "TSLA": 3},
            portfolio_value=1_000.0,
            gross_exposure=1.0,
        )

        self.assertEqual(
            orders,
            [
                {
                    "symbol": "AAPL",
                    "side": "buy",
                    "qty": 3,
                    "current_qty": 2,
                    "target_qty": 5,
                    "order_type": "market",
                    "time_in_force": "day",
                },
                {
                    "symbol": "MSFT",
                    "side": "buy",
                    "qty": 2,
                    "current_qty": 0,
                    "target_qty": 2,
                    "order_type": "market",
                    "time_in_force": "day",
                },
                {
                    "symbol": "TSLA",
                    "side": "sell",
                    "qty": 3,
                    "current_qty": 3,
                    "target_qty": 0,
                    "order_type": "market",
                    "time_in_force": "day",
                },
            ],
        )

    def test_rejects_missing_price(self):
        with self.assertRaisesRegex(ValueError, "Missing positive latest price for MSFT"):
            build_equal_weight_order_plan(
                ["AAPL", "MSFT"],
                {"AAPL": 100.0},
                portfolio_value=1_000.0,
            )
