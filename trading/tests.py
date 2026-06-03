from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from trading import robinhood


class RobinhoodOptionOrderPricingTests(TestCase):
    def test_expiry_selection_uses_closest_dte_even_when_shorter_than_target(self) -> None:
        expiry = robinhood._resolve_nearest_option_expiry(
            "2026-04-27",
            ["2026-06-23", "2026-07-03"],
            60,
        )

        self.assertEqual(expiry, "2026-06-23")

    def test_call_strike_selection_uses_first_five_percent_breakeven(self) -> None:
        ranked = robinhood._rank_option_candidates(
            [
                {"strike_price": "103", "bid_price": "1.00"},
                {"strike_price": "104", "bid_price": "1.00"},
                {"strike_price": "105", "bid_price": "0.50"},
            ],
            target_strike=105.0,
            option_type="call",
            spot_price=100.0,
            min_breakeven_move_pct=0.05,
        )

        self.assertEqual(float(ranked[0]["strike_price"]), 104.0)
        self.assertAlmostEqual(float(ranked[0]["breakeven_move_pct"]), 0.05)

    def test_put_strike_selection_uses_first_five_percent_breakeven(self) -> None:
        ranked = robinhood._rank_option_candidates(
            [
                {"strike_price": "97", "bid_price": "1.00"},
                {"strike_price": "96", "bid_price": "1.00"},
                {"strike_price": "95", "bid_price": "0.50"},
            ],
            target_strike=95.0,
            option_type="put",
            spot_price=100.0,
            min_breakeven_move_pct=0.05,
        )

        self.assertEqual(float(ranked[0]["strike_price"]), 96.0)
        self.assertAlmostEqual(float(ranked[0]["breakeven_move_pct"]), 0.05)

    def test_contract_selection_retries_broken_pipe_option_api_call(self) -> None:
        class FakeRobinhood:
            chain_calls = 0

            @classmethod
            def get_chains(cls, symbol):
                cls.chain_calls += 1
                if cls.chain_calls < 3:
                    raise BrokenPipeError(32, "Broken pipe")
                return {"expiration_dates": ["2026-06-19"]}

            @staticmethod
            def find_tradable_options(symbol, expirationDate=None, optionType=None):
                return [
                    {
                        "expiration_date": expirationDate,
                        "strike_price": "105",
                        "id": "contract-id",
                    }
                ]

            @staticmethod
            def get_option_market_data_by_id(option_id):
                return [{"bid_price": "1.00", "ask_price": "1.10", "mark_price": "1.05"}]

        with patch.object(robinhood, "_require_robin_stocks", return_value=FakeRobinhood):
            contract = robinhood.select_robinhood_long_option_contract(
                symbol="AAPL",
                spot_price=100.0,
                option_type="call",
                as_of_date="2026-04-27",
                target_hold_days=60,
                strike_multiplier=1.05,
            )

        self.assertEqual(str(contract["symbol"]), "AAPL")
        self.assertEqual(FakeRobinhood.chain_calls, 3)

    def test_contract_selection_enriches_only_nearby_option_candidates(self) -> None:
        class FakeRobinhood:
            market_data_calls: list[str] = []

            @staticmethod
            def get_chains(symbol):
                return {"expiration_dates": ["2026-06-19"]}

            @staticmethod
            def find_tradable_options(symbol, expirationDate=None, optionType=None):
                rows = []
                for strike in range(80, 131):
                    rows.append(
                        {
                            "expiration_date": expirationDate,
                            "strike_price": str(float(strike)),
                            "type": optionType,
                            "id": f"contract-{strike}",
                        }
                    )
                return rows

            @classmethod
            def get_option_market_data_by_id(cls, option_id):
                cls.market_data_calls.append(str(option_id))
                strike = float(str(option_id).split("-")[-1])
                return [{"bid_price": "1.00", "ask_price": "1.10", "mark_price": "1.05", "strike_price": str(strike)}]

        with patch.object(robinhood, "_require_robin_stocks", return_value=FakeRobinhood):
            contract = robinhood.select_robinhood_long_option_contract(
                symbol="AAPL",
                spot_price=100.0,
                option_type="call",
                as_of_date="2026-04-27",
                target_hold_days=60,
                strike_multiplier=1.05,
            )

        self.assertLessEqual(len(FakeRobinhood.market_data_calls), robinhood._OPTION_MARKET_DATA_CANDIDATE_LIMIT)
        self.assertIn("contract-105", FakeRobinhood.market_data_calls)
        self.assertEqual(str(contract["symbol"]), "AAPL")

    def test_plan_marks_broken_pipe_as_option_lookup_failure_not_ineligible_contract(self) -> None:
        latest = pd.DataFrame(
            [
                {"symbol": "A", "close": 100.0, "prob_buy": 0.95, "prob_short": 0.05, "buy_score": 0.95, "short_score": 0.05, "pred_rf_reg": 0.80, "ae_familiarity": 0.80},
                {"symbol": "B", "close": 100.0, "prob_buy": 0.90, "prob_short": 0.10, "buy_score": 0.90, "short_score": 0.10, "pred_rf_reg": 0.80, "ae_familiarity": 0.80},
            ]
        ).set_index("symbol")

        with patch.object(robinhood, "select_robinhood_long_option_contract", side_effect=BrokenPipeError(32, "Broken pipe")):
            plan = robinhood.build_robinhood_option_trade_plan(
                latest_scored_df=latest,
                current_option_positions=pd.DataFrame(),
                pending_option_orders=pd.DataFrame(),
                top_k=1,
                score_col="buy_score",
                component_threshold=0.50,
                account_equity=10_000.0,
                strategy_allocation=10_000.0,
                as_of_date="2026-04-27",
            )

        skipped = plan["skipped_symbols"]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(str(skipped.loc[0, "reason"]), "option_lookup_failed")
        self.assertTrue(any("stopping new option lookups" in line for line in plan["plan_log_lines"]))

    def test_enrich_buy_to_open_uses_bid_multiplier_as_limit(self) -> None:
        class FakeRobinhood:
            @staticmethod
            def get_option_market_data(symbol, expiry, strike, option_type):
                return [
                    {
                        "bid_price": "1.34",
                        "ask_price": "1.35",
                        "mark_price": "1.27",
                        "previous_close_price": "1.10",
                    }
                ]

        orders = pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "action": "buy_to_open_call",
                    "quantity": 1,
                    "option_type": "call",
                    "expiry_date": "2026-06-19",
                    "strike_price": 200.0,
                    "price": 1.35,
                }
            ]
        )

        with patch.object(robinhood, "_require_robin_stocks", return_value=FakeRobinhood):
            enriched = robinhood.enrich_robinhood_option_prices(orders)

        bid_limit_column = robinhood._buy_option_bid_limit_column()
        self.assertEqual(float(enriched.loc[0, "price"]), 0.06)
        self.assertEqual(float(enriched.loc[0, "limit_order_price"]), 0.06)
        self.assertAlmostEqual(float(enriched.loc[0, bid_limit_column]), 0.067)
        self.assertEqual(enriched.loc[0, "limit_price_source"], robinhood._buy_option_bid_limit_source())

    def test_enrich_buy_to_open_uses_bid_multiplier_when_below_previous_close(self) -> None:
        class FakeRobinhood:
            @staticmethod
            def get_option_market_data(symbol, expiry, strike, option_type):
                return [
                    {
                        "bid_price": "1.20",
                        "ask_price": "1.35",
                        "mark_price": "1.27",
                        "previous_close_price": "1.40",
                    }
                ]

        orders = pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "action": "buy_to_open_call",
                    "quantity": 1,
                    "option_type": "call",
                    "expiry_date": "2026-06-19",
                    "strike_price": 200.0,
                    "price": 1.35,
                }
            ]
        )

        with patch.object(robinhood, "_require_robin_stocks", return_value=FakeRobinhood):
            enriched = robinhood.enrich_robinhood_option_prices(orders)

        bid_limit_column = robinhood._buy_option_bid_limit_column()
        self.assertEqual(float(enriched.loc[0, "price"]), 0.06)
        self.assertEqual(float(enriched.loc[0, "limit_order_price"]), 0.06)
        self.assertEqual(float(enriched.loc[0, bid_limit_column]), 0.06)
        self.assertEqual(enriched.loc[0, "limit_price_source"], robinhood._buy_option_bid_limit_source())

    def test_enrich_sell_to_close_sets_limit_order_price_from_bid(self) -> None:
        class FakeRobinhood:
            @staticmethod
            def get_option_market_data(symbol, expiry, strike, option_type):
                return [
                    {
                        "bid_price": "1.20",
                        "ask_price": "1.35",
                        "mark_price": "1.27",
                    }
                ]

        orders = pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "action": "sell_to_close_call",
                    "quantity": 1,
                    "option_type": "call",
                    "expiry_date": "2026-06-19",
                    "strike_price": 200.0,
                    "order_type": "limit",
                    "price": np.nan,
                }
            ]
        )

        with patch.object(robinhood, "_require_robin_stocks", return_value=FakeRobinhood):
            enriched = robinhood.enrich_robinhood_option_prices(orders)

        self.assertEqual(float(enriched.loc[0, "price"]), 1.20)
        self.assertEqual(float(enriched.loc[0, "limit_order_price"]), 1.20)
        self.assertEqual(enriched.loc[0, "limit_price_source"], "bid_price")

    def test_enrich_sell_to_close_falls_back_to_existing_bid_when_quote_missing(self) -> None:
        class FakeRobinhood:
            @staticmethod
            def get_option_market_data(symbol, expiry, strike, option_type):
                return []

            @staticmethod
            def find_options_by_expiration(symbol, expirationDate, optionType):
                return []

        orders = pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "action": "sell_to_close_call",
                    "quantity": 1,
                    "option_type": "call",
                    "expiry_date": "2026-06-19",
                    "strike_price": 200.0,
                    "order_type": "limit",
                    "price": np.nan,
                    "bid_price": 1.15,
                }
            ]
        )

        with patch.object(robinhood, "_require_robin_stocks", return_value=FakeRobinhood):
            enriched = robinhood.enrich_robinhood_option_prices(orders)

        self.assertEqual(float(enriched.loc[0, "price"]), 1.15)
        self.assertEqual(float(enriched.loc[0, "limit_order_price"]), 1.15)
        self.assertEqual(enriched.loc[0, "limit_price_source"], "bid_price")

    def test_annotate_pending_buy_limit_savings_infers_original_reference(self) -> None:
        class FakeRobinhood:
            @staticmethod
            def get_option_market_data(symbol, expiry, strike, option_type):
                return [
                    {
                        "bid_price": "1.20",
                        "ask_price": "1.35",
                        "mark_price": "1.27",
                    }
                ]

        orders = pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "action": "buy_to_open_call",
                    "quantity": 2,
                    "option_type": "call",
                    "expiry_date": "2026-06-19",
                    "strike_price": 200.0,
                    "order_type": "limit",
                    "price": 0.10,
                }
            ]
        )

        with patch.object(robinhood, "_require_robin_stocks", return_value=FakeRobinhood):
            annotated = robinhood.annotate_robinhood_option_limit_savings(orders)

        self.assertEqual(float(annotated.loc[0, "submitted_limit_price"]), 0.10)
        self.assertEqual(float(annotated.loc[0, "discount_rate"]), 0.05)
        self.assertEqual(float(annotated.loc[0, "limit_price"]), 0.10)
        self.assertEqual(float(annotated.loc[0, "current_qty"]), 2.0)
        self.assertEqual(float(annotated.loc[0, "original_strategy_price"]), 2.00)
        self.assertAlmostEqual(float(annotated.loc[0, "original_strategy_qty"]), 0.1)
        self.assertEqual(float(annotated.loc[0, "inferred_original_reference_price"]), 2.00)
        self.assertEqual(float(annotated.loc[0, "contract_quantity"]), 2.0)
        self.assertAlmostEqual(float(annotated.loc[0, "inferred_original_strategy_contract_quantity"]), 0.1)
        self.assertAlmostEqual(float(annotated.loc[0, "inferred_original_strategy_contract_quantity_floor"]), 0.0)
        self.assertEqual(float(annotated.loc[0, "contract_multiplier"]), 100.0)
        self.assertAlmostEqual(float(annotated.loc[0, "submitted_limit_notional"]), 20.0)
        self.assertAlmostEqual(float(annotated.loc[0, "inferred_original_strategy_notional"]), 20.0)
        self.assertAlmostEqual(float(annotated.loc[0, "discount_saved_per_share"]), 1.90)
        self.assertAlmostEqual(float(annotated.loc[0, "discount_saved_per_contract"]), 190.0)
        self.assertAlmostEqual(float(annotated.loc[0, "discount_saved_total"]), 19.0)
        self.assertEqual(float(annotated.loc[0, "limit_savings_reference_price"]), 1.20)
        self.assertEqual(annotated.loc[0, "limit_savings_reference_source"], "bid_price")
        self.assertAlmostEqual(float(annotated.loc[0, "missed_move_per_share"]), -0.80)
        self.assertAlmostEqual(float(annotated.loc[0, "missed_move_per_contract"]), -80.0)
        self.assertAlmostEqual(float(annotated.loc[0, "missed_move_total"]), -8.0)
        self.assertEqual(annotated.loc[0, "missed_move_label"], "avoided_loss")
        target_bid_limit_column = f"target_{robinhood._buy_option_bid_limit_source()}_limit_price"
        self.assertEqual(float(annotated.loc[0, target_bid_limit_column]), 0.06)

    def test_annotate_pending_buy_limit_savings_normalizes_share_equivalent_quantity(self) -> None:
        class FakeRobinhood:
            @staticmethod
            def get_option_market_data(symbol, expiry, strike, option_type):
                return [{"bid_price": "1.20", "ask_price": "1.35", "mark_price": "1.27"}]

        orders = pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "action": "buy_to_open_call",
                    "quantity": 200,
                    "option_type": "call",
                    "expiry_date": "2026-06-19",
                    "strike_price": 200.0,
                    "order_type": "limit",
                    "price": 0.10,
                }
            ]
        )

        with patch.object(robinhood, "_require_robin_stocks", return_value=FakeRobinhood):
            annotated = robinhood.annotate_robinhood_option_limit_savings(orders)

        self.assertEqual(float(annotated.loc[0, "contract_quantity"]), 2.0)
        self.assertAlmostEqual(float(annotated.loc[0, "inferred_original_strategy_contract_quantity"]), 0.1)
        self.assertEqual(str(annotated.loc[0, "quantity_source"]), "share_equivalent_divided_by_100")
        self.assertAlmostEqual(float(annotated.loc[0, "discount_saved_total"]), 19.0)
        self.assertAlmostEqual(float(annotated.loc[0, "missed_move_total"]), -8.0)

    def test_load_option_positions_populates_bid_price(self) -> None:
        class FakeRobinhood:
            @staticmethod
            def get_open_option_positions(account_number=None):
                return [
                    {
                        "quantity": "1",
                        "option": "https://api.robinhood.com/options/instruments/option-id/",
                        "average_price": "0.85",
                    }
                ]

            @staticmethod
            def get_option_instrument_data_by_id(option_id):
                return {
                    "chain_symbol": "AAPL",
                    "type": "call",
                    "expiration_date": "2026-06-19",
                    "strike_price": "200.0000",
                }

            @staticmethod
            def get_option_market_data(symbol, expiry, strike, option_type):
                return [{"bid_price": "1.25", "mark_price": "1.30"}] if strike == "200" else []

        with patch.object(robinhood, "_require_robin_stocks", return_value=FakeRobinhood):
            positions = robinhood.load_robinhood_option_positions()

        self.assertEqual(float(positions.loc[0, "bid_price"]), 1.25)
        self.assertEqual(float(positions.loc[0, "mark_price"]), 1.30)

    def test_submit_buy_to_open_uses_bid_multiplier(self) -> None:
        orders = pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "action": "buy_to_open_call",
                    "quantity": 1,
                    "option_type": "call",
                    "expiry_date": "2026-06-19",
                    "strike_price": 200.0,
                    "price": 1.35,
                    "bid_price": 1.34,
                    "previous_close_price": 1.10,
                }
            ]
        )

        calls = []

        def order_buy_option_limit(*args, **kwargs):
            calls.append((args, kwargs))
            return {"ok": True, "price": args[2]}

        fake_rh = SimpleNamespace(order_buy_option_limit=order_buy_option_limit)
        with patch.object(robinhood, "_require_robin_stocks", return_value=fake_rh):
            results = robinhood.submit_robinhood_option_orders(orders_df=orders)

        response = results.loc[0, "response"]
        self.assertTrue(bool(results.loc[0, "submitted"]))
        self.assertEqual(float(response["price"]), 0.06)
        self.assertEqual(float(calls[0][0][2]), 0.06)
        self.assertEqual(calls[0][1]["timeInForce"], "gtc")

    def test_buy_limit_price_uses_decimal_tick_floor_without_float_artifacts(self) -> None:
        limit_price, source = robinhood._buy_option_limit_price({"bid_price": 1.999999999})

        self.assertEqual(limit_price, 0.09)
        self.assertEqual(source, robinhood._buy_option_bid_limit_source())

    def test_buy_limit_price_uses_nickel_tick_at_three_dollars_and_above(self) -> None:
        limit_price, source = robinhood._buy_option_limit_price({"bid_price": 70.17})

        self.assertEqual(limit_price, 3.50)
        self.assertEqual(source, robinhood._buy_option_bid_limit_source())

    def test_plan_exits_flipped_positions_and_counts_pending_entries_against_capacity(self) -> None:
        latest = pd.DataFrame(
            [
                {"symbol": "A", "close": 100.0, "prob_buy": 0.10, "prob_short": 0.90, "buy_score": 0.10, "short_score": 0.90, "pred_rf_reg": 0.80, "ae_familiarity": 0.80},
                {"symbol": "B", "close": 100.0, "prob_buy": 0.95, "prob_short": 0.05, "buy_score": 0.95, "short_score": 0.05, "pred_rf_reg": 0.80, "ae_familiarity": 0.80},
                {"symbol": "C", "close": 100.0, "prob_buy": 0.90, "prob_short": 0.10, "buy_score": 0.90, "short_score": 0.10, "pred_rf_reg": 0.80, "ae_familiarity": 0.80},
                {"symbol": "D", "close": 100.0, "prob_buy": 0.85, "prob_short": 0.15, "buy_score": 0.85, "short_score": 0.15, "pred_rf_reg": 0.80, "ae_familiarity": 0.80},
            ]
        ).set_index("symbol")
        current = pd.DataFrame(
            [
                {
                    "symbol": "A",
                    "option_type": "call",
                    "quantity": 1,
                    "expiry_date": "2026-06-19",
                    "strike_price": 100.0,
                }
            ]
        )
        pending = pd.DataFrame([{"symbol": "B", "action": "buy_to_open_call", "quantity": 1}])

        def fake_contract(**kwargs):
            return {
                "symbol": kwargs["symbol"],
                "option_type": kwargs["option_type"],
                "expiry_date": "2026-06-19",
                "strike_price": 105.0,
                "ask_price": 2.0,
                "bid_price": 1.5,
                "mark_price": 1.75,
                "id": "contract-id",
            }

        with patch.object(robinhood, "select_robinhood_long_option_contract", side_effect=fake_contract):
            plan = robinhood.build_robinhood_option_trade_plan(
                latest_scored_df=latest,
                current_option_positions=current,
                pending_option_orders=pending,
                top_k=2,
                score_col="buy_score",
                component_threshold=0.50,
                account_equity=10_000.0,
                strategy_allocation=10_000.0,
                as_of_date="2026-04-27",
            )

        actions = plan["actions"]
        self.assertIn("limit", set(actions.loc[actions["symbol"] == "A", "order_type"]))
        self.assertEqual(int((actions["action"].astype(str).str.startswith("buy_to_open")).sum()), 1)
        self.assertEqual(str(actions.loc[actions["action"].astype(str).str.startswith("buy_to_open"), "symbol"].iloc[0]), "C")
        summary = plan["summary"].iloc[0]
        self.assertEqual(int(summary["pending_buy_underlyings"]), 1)
        self.assertEqual(int(summary["remaining_buy_slots"]), 1)
        self.assertTrue(any("Step 1 option position: A" in line for line in plan["plan_log_lines"]))
        self.assertTrue(any("Step 3 pending option order: B" in line for line in plan["plan_log_lines"]))
        self.assertTrue(any("Step 5 option buy: C" in line for line in plan["plan_log_lines"]))

    def test_plan_cancels_pending_call_when_classifier_is_short(self) -> None:
        latest = pd.DataFrame(
            [
                {"symbol": "A", "close": 100.0, "prob_buy": 0.10, "prob_short": 0.90, "buy_score": 0.10, "short_score": 0.90, "pred_rf_reg": 0.80, "ae_familiarity": 0.80},
                {"symbol": "B", "close": 100.0, "prob_buy": 0.95, "prob_short": 0.05, "buy_score": 0.95, "short_score": 0.05, "pred_rf_reg": 0.80, "ae_familiarity": 0.80},
            ]
        ).set_index("symbol")
        pending = pd.DataFrame(
            [
                {
                    "symbol": "A",
                    "action": "buy_to_open_call",
                    "quantity": 1,
                    "order_id": "order-a",
                    "option_type": "call",
                }
            ]
        )

        def fake_contract(**kwargs):
            return {
                "symbol": kwargs["symbol"],
                "option_type": kwargs["option_type"],
                "expiry_date": "2026-06-19",
                "strike_price": 105.0,
                "ask_price": 1.0,
                "bid_price": 1.0,
                "mark_price": 1.0,
                "id": "contract-id",
            }

        with patch.object(robinhood, "select_robinhood_long_option_contract", side_effect=fake_contract):
            plan = robinhood.build_robinhood_option_trade_plan(
                latest_scored_df=latest,
                current_option_positions=pd.DataFrame(),
                pending_option_orders=pending,
                top_k=1,
                score_col="buy_score",
                component_threshold=0.50,
                account_equity=10_000.0,
                strategy_allocation=10_000.0,
                as_of_date="2026-04-27",
            )

        actions = plan["actions"]
        cancel_orders = actions.loc[actions["action"] == "cancel_buy_to_open_call"]
        self.assertEqual(len(cancel_orders), 1)
        self.assertEqual(str(cancel_orders.iloc[0]["order_id"]), "order-a")
        self.assertEqual(int(plan["summary"].iloc[0]["orders_to_cancel"]), 1)
        self.assertEqual(int(plan["summary"].iloc[0]["pending_buy_underlyings"]), 0)

    def test_plan_cancels_pending_put_when_classifier_is_long(self) -> None:
        latest = pd.DataFrame(
            [
                {"symbol": "A", "close": 100.0, "prob_buy": 0.90, "prob_short": 0.10, "buy_score": 0.90, "short_score": 0.10, "pred_rf_reg": 0.80, "ae_familiarity": 0.80},
            ]
        ).set_index("symbol")
        pending = pd.DataFrame(
            [
                {
                    "symbol": "A",
                    "action": "buy_to_open_put",
                    "quantity": 1,
                    "order_id": "order-put-a",
                    "option_type": "put",
                }
            ]
        )

        plan = robinhood.build_robinhood_option_trade_plan(
            latest_scored_df=latest,
            current_option_positions=pd.DataFrame(),
            pending_option_orders=pending,
            top_k=1,
            score_col="buy_score",
            component_threshold=0.50,
            account_equity=10_000.0,
            strategy_allocation=10_000.0,
            as_of_date="2026-04-27",
        )

        actions = plan["actions"]
        cancel_orders = actions.loc[actions["action"] == "cancel_buy_to_open_put"]
        self.assertEqual(len(cancel_orders), 1)
        self.assertEqual(str(cancel_orders.iloc[0]["order_id"]), "order-put-a")

    def test_submit_cancel_option_order_uses_robinhood_cancel_endpoint(self) -> None:
        orders = pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "action": "cancel_buy_to_open_call",
                    "order_id": "cancel-me",
                }
            ]
        )
        calls = []

        def cancel_option_order(order_id):
            calls.append(order_id)
            return {"id": order_id, "state": "cancelled"}

        fake_rh = SimpleNamespace(cancel_option_order=cancel_option_order)
        with patch.object(robinhood, "_require_robin_stocks", return_value=fake_rh):
            results = robinhood.submit_robinhood_option_orders(orders_df=orders)

        self.assertEqual(calls, ["cancel-me"])
        self.assertTrue(bool(results.loc[0, "submitted"]))

    def test_plan_ignores_pending_sell_orders_for_capacity(self) -> None:
        latest = pd.DataFrame(
            [
                {"symbol": "B", "close": 100.0, "prob_buy": 0.95, "prob_short": 0.05, "buy_score": 0.95, "short_score": 0.05, "pred_rf_reg": 0.80, "ae_familiarity": 0.80},
            ]
        ).set_index("symbol")
        pending = pd.DataFrame([{"symbol": "A", "action": "sell_to_close_call", "quantity": 1}])

        def fake_contract(**kwargs):
            return {
                "symbol": kwargs["symbol"],
                "option_type": kwargs["option_type"],
                "expiry_date": "2026-06-19",
                "strike_price": 105.0,
                "ask_price": 1.0,
                "bid_price": 1.0,
                "mark_price": 1.0,
                "id": "contract-id",
            }

        with patch.object(robinhood, "select_robinhood_long_option_contract", side_effect=fake_contract):
            plan = robinhood.build_robinhood_option_trade_plan(
                latest_scored_df=latest,
                current_option_positions=pd.DataFrame(),
                pending_option_orders=pending,
                top_k=1,
                score_col="buy_score",
                component_threshold=0.50,
                account_equity=10_000.0,
                strategy_allocation=10_000.0,
                as_of_date="2026-04-27",
            )

        summary = plan["summary"].iloc[0]
        self.assertEqual(int(summary["pending_buy_underlyings"]), 0)
        self.assertEqual(int(summary["remaining_buy_slots"]), 1)
        self.assertEqual(int((plan["actions"]["action"].astype(str).str.startswith("buy_to_open")).sum()), 1)

    def test_plan_skips_symbol_when_one_contract_exceeds_slot_budget(self) -> None:
        latest = pd.DataFrame(
            [
                {"symbol": "C", "close": 100.0, "prob_buy": 0.95, "prob_short": 0.05, "buy_score": 0.95, "short_score": 0.05, "pred_rf_reg": 0.80, "ae_familiarity": 0.80},
                {"symbol": "D", "close": 100.0, "prob_buy": 0.90, "prob_short": 0.10, "buy_score": 0.90, "short_score": 0.10, "pred_rf_reg": 0.80, "ae_familiarity": 0.80},
            ]
        ).set_index("symbol")

        def fake_contract(**kwargs):
            bid_price = 30.00 if kwargs["symbol"] == "C" else 0.50
            return {
                "symbol": kwargs["symbol"],
                "option_type": kwargs["option_type"],
                "expiry_date": "2026-06-19",
                "strike_price": 105.0,
                "ask_price": bid_price,
                "bid_price": bid_price,
                "mark_price": bid_price,
                "id": "contract-id",
            }

        with patch.object(robinhood, "select_robinhood_long_option_contract", side_effect=fake_contract):
            plan = robinhood.build_robinhood_option_trade_plan(
                latest_scored_df=latest,
                current_option_positions=pd.DataFrame(),
                pending_option_orders=pd.DataFrame(),
                top_k=1,
                score_col="buy_score",
                component_threshold=0.50,
                account_equity=100.0,
                strategy_allocation=100.0,
                as_of_date="2026-04-27",
            )

        actions = plan["actions"]
        self.assertEqual(str(actions.loc[actions["action"].astype(str).str.startswith("buy_to_open"), "symbol"].iloc[0]), "D")
        skipped = plan["skipped_symbols"]
        self.assertEqual(str(skipped.loc[0, "symbol"]), "C")
        self.assertEqual(str(skipped.loc[0, "reason"]), "contract_value_exceeds_slot_budget")
        self.assertEqual(str(skipped.loc[0, "replacement_status"]), "replaced")
        summary = plan["summary"].iloc[0]
        self.assertEqual(int(summary["filled_entry_slots"]), 1)
        self.assertEqual(int(summary["unfilled_entry_slots"]), 0)
        self.assertEqual(int(summary["skipped_entry_candidates"]), 1)
        self.assertTrue(any("Step 5 option skip: C" in line for line in plan["plan_log_lines"]))
        self.assertTrue(any("Step 5 option buy: D" in line for line in plan["plan_log_lines"]))

    def test_plan_reports_unfilled_slot_when_skipped_symbol_has_no_replacement(self) -> None:
        latest = pd.DataFrame(
            [
                {"symbol": "C", "close": 100.0, "prob_buy": 0.95, "prob_short": 0.05, "buy_score": 0.95, "short_score": 0.05, "pred_rf_reg": 0.80, "ae_familiarity": 0.80},
            ]
        ).set_index("symbol")

        def fake_contract(**kwargs):
            return {
                "symbol": kwargs["symbol"],
                "option_type": kwargs["option_type"],
                "expiry_date": "2026-06-19",
                "strike_price": 105.0,
                "ask_price": 30.0,
                "bid_price": 30.0,
                "mark_price": 30.0,
                "id": "contract-id",
            }

        with patch.object(robinhood, "select_robinhood_long_option_contract", side_effect=fake_contract):
            plan = robinhood.build_robinhood_option_trade_plan(
                latest_scored_df=latest,
                current_option_positions=pd.DataFrame(),
                pending_option_orders=pd.DataFrame(),
                top_k=1,
                score_col="buy_score",
                component_threshold=0.50,
                account_equity=100.0,
                strategy_allocation=100.0,
                as_of_date="2026-04-27",
            )

        self.assertTrue(plan["actions"].empty)
        skipped = plan["skipped_symbols"]
        self.assertEqual(str(skipped.loc[0, "replacement_status"]), "partially_replaced")
        summary = plan["summary"].iloc[0]
        self.assertEqual(int(summary["filled_entry_slots"]), 0)
        self.assertEqual(int(summary["unfilled_entry_slots"]), 1)

    def test_plan_respects_explicit_max_contracts_per_position(self) -> None:
        latest = pd.DataFrame(
            [
                {"symbol": "FTV", "close": 50.0, "prob_buy": 0.05, "prob_short": 0.95, "buy_score": 0.05, "short_score": 0.95, "pred_rf_reg": 0.80, "ae_familiarity": 0.80},
            ]
        ).set_index("symbol")

        def fake_contract(**kwargs):
            return {
                "symbol": kwargs["symbol"],
                "option_type": kwargs["option_type"],
                "expiry_date": "2026-06-19",
                "strike_price": 55.0,
                "ask_price": 0.50,
                "bid_price": 0.50,
                "mark_price": 0.50,
                "id": "contract-id",
            }

        with patch.object(robinhood, "select_robinhood_long_option_contract", side_effect=fake_contract):
            plan = robinhood.build_robinhood_option_trade_plan(
                latest_scored_df=latest,
                current_option_positions=pd.DataFrame(),
                pending_option_orders=pd.DataFrame(),
                top_k=1,
                score_col="buy_score",
                component_threshold=0.50,
                account_equity=100_000.0,
                strategy_allocation=100_000.0,
                as_of_date="2026-04-27",
                max_contracts_per_position=3,
            )

        buy_orders = plan["actions"].loc[plan["actions"]["action"].astype(str).str.startswith("buy_to_open")]
        self.assertEqual(int(buy_orders.iloc[0]["quantity"]), 3)

    def test_plan_sizes_contracts_from_submitted_limit_price(self) -> None:
        latest = pd.DataFrame(
            [
                {"symbol": "A", "close": 100.0, "prob_buy": 0.95, "prob_short": 0.05, "buy_score": 0.95, "short_score": 0.05, "pred_rf_reg": 0.80, "ae_familiarity": 0.80},
            ]
        ).set_index("symbol")

        def fake_contract(**kwargs):
            return {
                "symbol": kwargs["symbol"],
                "option_type": kwargs["option_type"],
                "expiry_date": "2026-06-19",
                "strike_price": 105.0,
                "ask_price": 26.0,
                "bid_price": 25.0,
                "mark_price": 25.5,
                "id": "contract-id",
            }

        with patch.object(robinhood, "select_robinhood_long_option_contract", side_effect=fake_contract):
            plan = robinhood.build_robinhood_option_trade_plan(
                latest_scored_df=latest,
                current_option_positions=pd.DataFrame(),
                pending_option_orders=pd.DataFrame(),
                top_k=1,
                score_col="buy_score",
                component_threshold=0.50,
                account_equity=5_000.0,
                strategy_allocation=5_000.0,
                as_of_date="2026-04-27",
            )

        buy_orders = plan["actions"].loc[plan["actions"]["action"].astype(str).str.startswith("buy_to_open")]
        self.assertEqual(int(buy_orders.iloc[0]["quantity"]), 40)
        self.assertEqual(float(buy_orders.iloc[0]["limit_order_price"]), 1.25)
        self.assertEqual(float(buy_orders.iloc[0]["contract_value"]), 125.0)

    def test_submit_limit_exit_uses_limit_price(self) -> None:
        orders = pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "action": "sell_to_close_call",
                    "quantity": 1,
                    "option_type": "call",
                    "expiry_date": "2026-06-19",
                    "strike_price": 200.0,
                    "order_type": "limit",
                    "price": 1.25,
                }
            ]
        )

        calls = []

        def order_sell_option_limit(*args, **kwargs):
            calls.append((args, kwargs))
            return {"ok": True, "order_type": "limit", "price": args[2]}

        fake_rh = SimpleNamespace(order_sell_option_limit=order_sell_option_limit)
        with patch.object(robinhood, "_require_robin_stocks", return_value=fake_rh):
            results = robinhood.submit_robinhood_option_orders(orders_df=orders)

        response = results.loc[0, "response"]
        self.assertTrue(bool(results.loc[0, "submitted"]))
        self.assertEqual(response["order_type"], "limit")
        self.assertEqual(float(response["price"]), 1.25)
        self.assertEqual(calls[0][0][0], "close")
        self.assertEqual(float(calls[0][0][2]), 1.25)

    def test_submit_limit_exit_uses_bid_price_when_price_missing(self) -> None:
        orders = pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "action": "sell_to_close_call",
                    "quantity": 1,
                    "option_type": "call",
                    "expiry_date": "2026-06-19",
                    "strike_price": 200.0,
                    "order_type": "limit",
                    "bid_price": 1.15,
                }
            ]
        )

        calls = []

        def order_sell_option_limit(*args, **kwargs):
            calls.append((args, kwargs))
            return {"ok": True, "order_type": "limit", "price": args[2]}

        fake_rh = SimpleNamespace(order_sell_option_limit=order_sell_option_limit)
        with patch.object(robinhood, "_require_robin_stocks", return_value=fake_rh):
            results = robinhood.submit_robinhood_option_orders(orders_df=orders)

        response = results.loc[0, "response"]
        self.assertTrue(bool(results.loc[0, "submitted"]))
        self.assertEqual(float(response["price"]), 1.15)
        self.assertEqual(float(calls[0][0][2]), 1.15)

    def test_submit_limit_exit_rounds_price_to_cents(self) -> None:
        orders = pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "action": "sell_to_close_call",
                    "quantity": 1,
                    "option_type": "call",
                    "expiry_date": "2026-06-19",
                    "strike_price": 200.0,
                    "order_type": "limit",
                    "bid_price": 1.1500000000000001,
                }
            ]
        )

        calls = []

        def order_sell_option_limit(*args, **kwargs):
            calls.append((args, kwargs))
            return {"ok": True, "order_type": "limit", "price": args[2]}

        fake_rh = SimpleNamespace(order_sell_option_limit=order_sell_option_limit)
        with patch.object(robinhood, "_require_robin_stocks", return_value=fake_rh):
            results = robinhood.submit_robinhood_option_orders(orders_df=orders)

        self.assertTrue(bool(results.loc[0, "submitted"]))
        self.assertEqual(calls[0][0][2], 1.15)

    def test_plan_keeps_held_option_when_score_is_missing(self) -> None:
        latest = pd.DataFrame(
            [
                {"symbol": "B", "close": 100.0, "prob_buy": 0.95, "prob_short": 0.05, "buy_score": 0.95, "short_score": 0.05, "pred_rf_reg": 0.80, "ae_familiarity": 0.80},
            ]
        ).set_index("symbol")
        current = pd.DataFrame(
            [
                {
                    "symbol": "A",
                    "option_type": "call",
                    "quantity": 1,
                    "expiry_date": "2026-06-19",
                    "strike_price": 100.0,
                }
            ]
        )

        plan = robinhood.build_robinhood_option_trade_plan(
            latest_scored_df=latest,
            current_option_positions=current,
            pending_option_orders=pd.DataFrame(),
            top_k=1,
            score_col="buy_score",
            component_threshold=0.50,
            account_equity=10_000.0,
            strategy_allocation=10_000.0,
            as_of_date="2026-04-27",
        )

        actions = plan["actions"]
        self.assertEqual(set(actions["action"]), {"hold_call"})
        self.assertEqual(str(actions.loc[0, "reason"]), "score_unavailable_hold")
        self.assertEqual(int(plan["summary"].iloc[0]["held_symbols_missing_scores"]), 1)

    def test_plan_scores_missing_held_option_before_exit_check(self) -> None:
        latest = pd.DataFrame(
            [
                {"symbol": "B", "close": 100.0, "prob_buy": 0.95, "prob_short": 0.05, "buy_score": 0.95, "short_score": 0.05, "pred_rf_reg": 0.80, "ae_familiarity": 0.80},
            ]
        ).set_index("symbol")
        current = pd.DataFrame(
            [
                {
                    "symbol": "A",
                    "option_type": "call",
                    "quantity": 1,
                    "expiry_date": "2026-06-19",
                    "strike_price": 100.0,
                }
            ]
        )

        def scorer(**kwargs):
            self.assertEqual(kwargs["symbols"], ["A"])
            return pd.DataFrame(
                [
                    {"symbol": "A", "close": 100.0, "prob_buy": 0.10, "prob_short": 0.90, "buy_score": 0.10, "short_score": 0.90, "pred_rf_reg": 0.80, "ae_familiarity": 0.80},
                ]
            )

        plan = robinhood.build_robinhood_option_trade_plan(
            latest_scored_df=latest,
            current_option_positions=current,
            pending_option_orders=pd.DataFrame(),
            missing_symbol_scorer=scorer,
            top_k=1,
            score_col="buy_score",
            component_threshold=0.50,
            account_equity=10_000.0,
            strategy_allocation=10_000.0,
            as_of_date="2026-04-27",
        )

        actions = plan["actions"]
        self.assertEqual(str(actions.loc[0, "action"]), "sell_to_close_call")
        self.assertEqual(str(actions.loc[0, "order_type"]), "limit")
        self.assertEqual(str(actions.loc[0, "reason"]), "classifier_flipped_short")

    def test_option_exit_ignores_gate_failure_without_classifier_flip(self) -> None:
        latest = pd.DataFrame(
            [
                {"symbol": "CALL", "close": 100.0, "prob_buy": 0.60, "prob_short": 0.40, "buy_score": 0.10, "short_score": 0.40, "pred_rf_reg": 0.20, "ae_familiarity": 0.20},
                {"symbol": "PUT", "close": 100.0, "prob_buy": 0.40, "prob_short": 0.60, "buy_score": 0.40, "short_score": 0.10, "pred_rf_reg": 0.20, "ae_familiarity": 0.20},
            ]
        ).set_index("symbol")
        current = pd.DataFrame(
            [
                {"symbol": "CALL", "option_type": "call", "quantity": 1, "expiry_date": "2026-06-19", "strike_price": 100.0},
                {"symbol": "PUT", "option_type": "put", "quantity": 1, "expiry_date": "2026-06-19", "strike_price": 100.0},
            ]
        )

        plan = robinhood.build_robinhood_option_trade_plan(
            latest_scored_df=latest,
            current_option_positions=current,
            pending_option_orders=pd.DataFrame(),
            top_k=2,
            score_col="buy_score",
            component_threshold=0.50,
            account_equity=10_000.0,
            strategy_allocation=10_000.0,
            as_of_date="2026-04-27",
        )

        actions = plan["actions"].sort_values("symbol").reset_index(drop=True)
        self.assertEqual(set(actions["action"]), {"hold_call", "hold_put"})
        self.assertEqual(set(actions["reason"]), {"signal_still_valid"})

    def test_put_exits_only_when_classifier_flips_long(self) -> None:
        latest = pd.DataFrame(
            [
                {"symbol": "A", "close": 100.0, "prob_buy": 0.51, "prob_short": 0.49, "buy_score": 0.51, "short_score": 0.49, "pred_rf_reg": 0.80, "ae_familiarity": 0.80},
            ]
        ).set_index("symbol")
        current = pd.DataFrame(
            [
                {"symbol": "A", "option_type": "put", "quantity": 1, "expiry_date": "2026-06-19", "strike_price": 100.0},
            ]
        )

        plan = robinhood.build_robinhood_option_trade_plan(
            latest_scored_df=latest,
            current_option_positions=current,
            pending_option_orders=pd.DataFrame(),
            top_k=1,
            score_col="buy_score",
            component_threshold=0.50,
            account_equity=10_000.0,
            strategy_allocation=10_000.0,
            as_of_date="2026-04-27",
        )

        actions = plan["actions"]
        self.assertEqual(str(actions.loc[0, "action"]), "sell_to_close_put")
        self.assertEqual(str(actions.loc[0, "reason"]), "classifier_flipped_long")

    def test_stock_plan_keeps_position_when_score_is_missing(self) -> None:
        latest = pd.DataFrame(
            [
                {"symbol": "B", "close": 100.0, "prob_buy": 0.95, "prob_short": 0.05, "buy_score": 0.95, "short_score": 0.05, "pred_rf_reg": 0.80, "ae_familiarity": 0.80},
            ]
        ).set_index("symbol")
        current = pd.DataFrame([{"symbol": "A", "quantity": 10}])

        plan = robinhood.build_robinhood_live_trade_plan(
            latest_scored_df=latest,
            current_positions=current,
            top_k=1,
            score_col="buy_score",
            component_threshold=0.50,
            account_equity=10_000.0,
        )

        actions = plan["actions"]
        self.assertEqual(set(actions["action"]), {"hold"})
        self.assertEqual(str(actions.loc[0, "reason"]), "score_unavailable_hold")
        self.assertTrue(plan["actionable_orders"].empty)
        self.assertEqual(int(plan["summary"].iloc[0]["held_symbols_missing_scores"]), 1)
