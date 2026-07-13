from __future__ import annotations

import json
import py_compile
from types import SimpleNamespace

import pandas as pd
import pytest

from app import trading_app_v2_runtime as runtime
from platforms.brokers.alpaca import (
    build_directional_equity_order_plan,
    build_directional_option_order_plan,
    build_llm_option_order_plan,
)


def test_directional_equity_plan_closes_opposites_retains_hold_and_fills_ranked_capacity():
    plan = build_directional_equity_order_plan(
        [
            {"symbol": "AAPL", "meta_stack_direction": "short"},
            {"symbol": "MSFT", "meta_stack_direction": "long"},
            {"symbol": "NVDA", "meta_stack_direction": "hold"},
            {"symbol": "TSLA", "meta_stack_direction": "long"},
            {"symbol": "AMZN", "meta_stack_direction": "short"},
        ],
        {"AAPL": 100, "MSFT": 200, "TSLA": 250, "AMZN": 200},
        {"AAPL": 3, "MSFT": -2, "NVDA": 4},
        portfolio_value=4_000,
        gross_exposure=1.0,
        max_positions=4,
    )

    assert [(row["symbol"], row["action"], row["side"]) for row in plan] == [
        ("AAPL", "close_opposite_signal", "sell"),
        ("MSFT", "close_opposite_signal", "buy"),
        ("AAPL", "open_short", "sell"),
        ("MSFT", "open_long", "buy"),
        ("TSLA", "open_long", "buy"),
    ]
    assert "NVDA" not in {row["symbol"] for row in plan}
    assert "AMZN" not in {row["symbol"] for row in plan}


def test_directional_equity_plan_fails_closed_without_signal_for_a_held_symbol():
    with pytest.raises(ValueError, match="Missing meta_stack directions"):
        build_directional_equity_order_plan(
            [{"symbol": "AAPL", "direction": "long"}],
            {"AAPL": 100},
            {"AAPL": 1, "LEGACY": 1},
            portfolio_value=1_000,
        )


def test_directional_option_plan_reverses_calls_and_puts_and_retains_hold():
    plan = build_directional_option_order_plan(
        [
            {"symbol": "AAPL", "direction": "short"},
            {"symbol": "MSFT", "direction": "long"},
            {"symbol": "NVDA", "direction": "hold"},
        ],
        [
            {"underlying_symbol": "AAPL", "contract_symbol": "AAPL_P_NEW", "option_type": "put"},
            {"underlying_symbol": "MSFT", "contract_symbol": "MSFT_C_NEW", "option_type": "call"},
        ],
        [
            {"underlying_symbol": "AAPL", "symbol": "AAPL_C_OLD", "option_type": "call", "qty": 2},
            {"underlying_symbol": "MSFT", "symbol": "MSFT_P_OLD", "option_type": "put", "qty": 1},
            {"underlying_symbol": "NVDA", "symbol": "NVDA_C_OLD", "option_type": "call", "qty": 1},
        ],
    )

    assert [(row["symbol"], row["action"]) for row in plan] == [
        ("AAPL_C_OLD", "sell_to_close_call"),
        ("MSFT_P_OLD", "sell_to_close_put"),
        ("AAPL_P_NEW", "buy_to_open_put"),
        ("MSFT_C_NEW", "buy_to_open_call"),
    ]
    assert "NVDA_C_OLD" not in {row["symbol"] for row in plan}


def test_directional_option_plan_keeps_exact_ranker_selection_and_caps_candidates():
    assert build_directional_option_order_plan(
        [{"symbol": "AAPL", "direction": "long"}],
        [{"underlying_symbol": "AAPL", "contract_symbol": "AAPL_C", "option_type": "call"}],
        [{"underlying_symbol": "AAPL", "symbol": "AAPL_C", "option_type": "call", "qty": 1}],
    ) == []

    selections = [
        {"underlying_symbol": f"S{i}", "contract_symbol": f"S{i}_C", "option_type": "call"}
        for i in range(21)
    ]
    directions = [{"symbol": f"S{i}", "direction": "long"} for i in range(21)]
    with pytest.raises(ValueError, match="limit is 20"):
        build_directional_option_order_plan(directions, selections)


@pytest.mark.parametrize(
    ("price", "expected"),
    [(1.25, 40), (2.50, 20), (60.0, 0), (None, 0)],
)
def test_option_contract_quantity_targets_one_twentieth_of_account(price, expected):
    assert runtime._option_contract_quantity(
        account_value=100_000.0,
        option_price=price,
        max_underlyings=20,
    ) == expected


def test_discounted_robinhood_bid_sizing_targets_five_thousand_dollars():
    # $5 bid at a 90% gate is a $0.50 limit; 100 shares/contract means 100
    # contracts consume the intended $5,000 sleeve.
    assert runtime.option_contract_quantity(
        account_value=100_000.0,
        option_price=5.0 * 0.10,
        max_underlyings=20,
    ) == 100


def test_llm_option_plan_reverses_opposites_and_retains_same_type_or_hold():
    plan = build_llm_option_order_plan(
        [
            {"symbol": "AAPL", "decision": "buy"},
            {"symbol": "MSFT", "decision": "sell"},
            {"symbol": "NVDA", "decision": "hold"},
        ],
        [
            {"underlying_symbol": "AAPL", "contract_symbol": "AAPL_C_NEW", "option_type": "call"},
            {"underlying_symbol": "MSFT", "contract_symbol": "MSFT_P_NEW", "option_type": "put"},
        ],
        [
            {"underlying_symbol": "AAPL", "symbol": "AAPL_P_OLD", "option_type": "put", "qty": 2},
            {"underlying_symbol": "MSFT", "symbol": "MSFT_P_OLD", "option_type": "put", "qty": 1},
            {"underlying_symbol": "NVDA", "symbol": "NVDA_C_OLD", "option_type": "call", "qty": 1},
        ],
    )

    assert [(row["symbol"], row["action"]) for row in plan] == [
        ("AAPL_P_OLD", "sell_to_close_put"),
        ("AAPL_C_NEW", "buy_to_open_call"),
    ]


def test_llm_option_plan_hold_opens_nothing_and_caps_decision_candidates():
    assert build_llm_option_order_plan(
        [{"symbol": "AAPL", "decision": "hold"}],
        [],
    ) == []

    decisions = [{"symbol": f"S{i}", "decision": "long"} for i in range(21)]
    with pytest.raises(ValueError, match="limit is 20"):
        build_llm_option_order_plan(decisions, [])


def test_llm_ranked_orders_map_relative_ratings_to_direction(monkeypatch):
    reviewed = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "NVDA"],
            "llm_rating": ["Overweight", "Underweight", "Hold"],
        }
    )
    captured = {}
    monkeypatch.setattr(
        "platforms.agents.trading_agents.review_trade_candidates",
        lambda *args, **kwargs: reviewed.copy(),
    )

    def fake_build(**kwargs):
        captured["decisions"] = kwargs["decisions"].copy()
        return pd.DataFrame()

    monkeypatch.setattr(runtime, "build_ranked_alpaca_option_orders", fake_build)
    runtime.build_llm_ranked_option_orders(
        leaderboard=reviewed,
        option_rankings=pd.DataFrame({"symbol": ["AAPL"]}),
        account_prefix="LLM",
    )

    assert captured["decisions"].set_index("symbol")["decision"].to_dict() == {
        "AAPL": "buy",
        "MSFT": "sell",
        "NVDA": "hold",
    }


def test_read_csv_if_exists_treats_empty_csv_as_empty_frame(tmp_path):
    path = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(path, index=False)

    frame = runtime._read_csv_if_exists(path)

    assert frame.empty


def test_alpaca_client_from_env_rejects_generic_credential_fallback(monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "generic-key")
    monkeypatch.setenv("ALPACA_PAPER_API_SECRET", "generic-secret")
    for name in (
        "EQUITY_ALPACA_PAPER_API_KEY",
        "EQUITY_ALPACA_PAPER_API_SECRET",
        "ALPACA_EQUITY_PAPER_API_KEY",
        "ALPACA_EQUITY_PAPER_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="dedicated Alpaca paper credentials"):
        runtime.alpaca_client_from_env("EQUITY")


def test_load_distinct_alpaca_paper_accounts_rejects_duplicate_account_ids(monkeypatch):
    account_ids = {"equity-key": "account-1", "option-key": "account-2", "llm-key": "account-2"}
    for prefix in ("EQUITY", "OPTION", "LLM"):
        monkeypatch.setenv(f"{prefix}_ALPACA_PAPER_API_KEY", f"{prefix.lower()}-key")
        monkeypatch.setenv(f"{prefix}_ALPACA_PAPER_API_SECRET", f"{prefix.lower()}-secret")

    class FakeClient:
        def __init__(self, api_key, api_secret):
            self.api_key = api_key
            self.api_secret = api_secret

        def get_account(self):
            return {"id": account_ids[self.api_key]}

    monkeypatch.setattr("platforms.brokers.alpaca.AlpacaPaperClient", FakeClient)

    with pytest.raises(RuntimeError, match="distinct Alpaca account IDs"):
        runtime.load_distinct_alpaca_paper_accounts()


def test_load_distinct_alpaca_paper_accounts_returns_three_isolated_clients(monkeypatch):
    for prefix in ("EQUITY", "OPTION", "LLM"):
        monkeypatch.setenv(f"ALPACA_{prefix}_PAPER_API_KEY", f"{prefix.lower()}-key")
        monkeypatch.setenv(f"ALPACA_{prefix}_PAPER_API_SECRET", f"{prefix.lower()}-secret")

    class FakeClient:
        def __init__(self, api_key, api_secret):
            self.api_key = api_key
            self.api_secret = api_secret

        def get_account(self):
            return {"id": f"account-for-{self.api_key}"}

    monkeypatch.setattr("platforms.brokers.alpaca.AlpacaPaperClient", FakeClient)

    clients = runtime.load_distinct_alpaca_paper_accounts()

    assert set(clients) == {"EQUITY", "OPTION", "LLM"}
    assert len({client.api_key for client in clients.values()}) == 3


def test_streamlit_app_submits_each_account_separately(tmp_path):
    app_path = runtime.write_streamlit_leaderboard_app(live_dir=tmp_path)
    script = app_path.read_text(encoding="utf-8")

    assert "Submit Orders By Account" in script
    assert "Submit {account_label} Orders" in script
    assert 'key=f"submit_{name}"' in script
    assert 'for name, orders in order_frames.items()' not in script
    assert '"alpaca_equity_paper": "equity"' in script
    assert '"alpaca_option_paper": "option"' in script
    assert '"alpaca_llm_paper": "option"' in script
    assert "asset_type=alpaca_asset_types[name]" in script


def test_streamlit_app_displays_feature_family_scores_for_all_symbols(tmp_path):
    app_path = runtime.write_streamlit_leaderboard_app(live_dir=tmp_path)
    script = app_path.read_text(encoding="utf-8")

    assert 'symbol_scores_path = LIVE_DIR / "symbol_scores.csv"' in script
    assert "Scores By Symbol" in script
    assert 'option_rankings_path = LIVE_DIR / "option_ml_rankings.csv"' in script
    assert "Option ML Rankings" in script
    assert "strategy_scores.csv" not in script
    assert "combined_feature_family_scores" not in script
    assert "Family Long Scores" not in script
    assert "Family Short Scores" not in script
    assert "Raw Family Scores" not in script


def test_streamlit_app_can_embed_in_memory_tables(tmp_path):
    app_path = runtime.write_streamlit_leaderboard_app(
        live_dir=tmp_path,
        leaderboard=pd.DataFrame(
            [{"symbol": "AAPL", "rank": 1, "selected": True, "eligible": True, "score_date": "2026-01-02"}]
        ),
        symbol_scores=pd.DataFrame([{"symbol": "AAPL", "rank": 1, "ensemble_long_score": 0.7}]),
        option_ml_rankings=pd.DataFrame([{"symbol": "AAPL", "option_ensemble_mean_score": 0.5}]),
        orders={"alpaca_equity_paper": pd.DataFrame([{"symbol": "AAPL", "qty": 1}])},
    )
    script = app_path.read_text(encoding="utf-8")

    py_compile.compile(str(app_path), doraise=True)
    assert "EMBEDDED_DATA" in script
    assert "read_embedded_frame" in script
    assert "leaderboard_latest.csv" not in script
    assert "symbol_scores.csv" not in script
    assert "option_ml_rankings.csv" not in script


def test_thetadata_oracle_backfill_accepts_in_memory_trade_frame(monkeypatch):
    calls = []

    def _fake_backfill(trades, **kwargs):
        calls.append((trades.copy(), kwargs))
        return {"status": "ok", "rows": len(trades)}

    monkeypatch.setattr(
        "quant_warehouse.migrate.backfill_thetadata_options.backfill_thetadata_options_for_oracle_trades",
        _fake_backfill,
    )
    trades = pd.DataFrame(
        [
            {"trade_id": "t1", "symbol": "AAPL", "entry_date": "2026-01-02", "exit_date": "2026-01-05"},
        ]
    )

    result = runtime.backfill_thetadata_for_oracle_trade_windows(
        trades,
        symbols=["AAPL"],
        max_trades=1,
        request_sleep=0.0,
    )

    assert result == {"status": "ok", "rows": 1}
    assert len(calls) == 1
    assert calls[0][0].equals(trades)
    assert calls[0][1]["symbols"] == ["AAPL"]
    assert calls[0][1]["max_trades"] == 1


def test_build_symbol_score_table_contains_all_symbols_and_family_scores():
    strategy_scores = pd.DataFrame(
        [
            {"strategy_source": "ensemble_mean", "symbol": "AAPL", "date": "2026-01-02", "long_score": 0.7, "short_score": 0.3, "model_count": 2},
            {"strategy_source": "ensemble_mean", "symbol": "MSFT", "date": "2026-01-02", "long_score": 0.4, "short_score": 0.6, "model_count": 2},
            {"strategy_source": "fmp.alpha", "symbol": "AAPL", "date": "2026-01-02", "long_score": 0.8, "short_score": 0.2},
            {"strategy_source": "fmp.alpha", "symbol": "MSFT", "date": "2026-01-02", "long_score": 0.1, "short_score": 0.9},
            {"strategy_source": "fmp.beta", "symbol": "TSLA", "date": "2026-01-02", "long_score": 0.6, "short_score": 0.4},
        ]
    )
    leaderboard = pd.DataFrame(
        [
            {"symbol": "AAPL", "rank": 1, "prob_buy": 0.7, "prob_short": 0.3, "selected": True, "eligible": True},
            {"symbol": "MSFT", "rank": 2, "prob_buy": 0.4, "prob_short": 0.6, "selected": False, "eligible": False},
        ]
    )

    table = runtime.build_symbol_score_table(strategy_scores, leaderboard)

    assert list(table["symbol"]) == ["AAPL", "MSFT", "TSLA"]
    assert int(table.set_index("symbol").loc["AAPL", "rank"]) == 1
    assert int(table.set_index("symbol").loc["MSFT", "rank"]) == 2
    assert int(table.set_index("symbol").loc["TSLA", "rank"]) == 3
    assert table.set_index("symbol").loc["AAPL", "ensemble_long_score"] == 0.7
    assert table.set_index("symbol").loc["AAPL", "long__fmp.alpha"] == 0.8
    assert table.set_index("symbol").loc["MSFT", "short__fmp.alpha"] == 0.9
    assert table.set_index("symbol").loc["TSLA", "long__fmp.beta"] == 0.6


def test_build_option_ml_ranking_table_uses_ensemble_mean_to_select_option(tmp_path):
    root = tmp_path / "option_family_ranker"
    family_a = root / "fmp.alpha"
    family_b = root / "fmp.beta"
    family_a.mkdir(parents=True)
    family_b.mkdir(parents=True)
    base = pd.DataFrame(
        [
            {
                "trade_id": "t1",
                "symbol": "AAPL",
                "entry_date": "2026-01-02",
                "contract_symbol": "AAPL_C_100",
                "option_type": "call",
                "option_action": "buy_call",
                "expiration": "2026-02-20",
                "strike": 100.0,
                "option_return": 0.1,
            },
            {
                "trade_id": "t1",
                "symbol": "AAPL",
                "entry_date": "2026-01-02",
                "contract_symbol": "AAPL_C_105",
                "option_type": "call",
                "option_action": "buy_call",
                "expiration": "2026-02-20",
                "strike": 105.0,
                "option_return": 0.2,
            },
            {
                "trade_id": "t2",
                "symbol": "MSFT",
                "entry_date": "2026-01-02",
                "contract_symbol": "MSFT_C_100",
                "option_type": "call",
                "option_action": "buy_call",
                "expiration": "2026-02-20",
                "strike": 100.0,
                "option_return": 0.3,
            },
        ]
    )
    frame_a = base.assign(**{"pred_fmp.alpha_rank": [0.9, 0.2, 0.7]})
    frame_b = base.assign(**{"pred_fmp.beta_rank": [0.1, 0.8, 0.7]})
    frame_a.to_parquet(family_a / "eval_scored.parquet", index=False)
    frame_b.to_parquet(family_b / "eval_scored.parquet", index=False)

    table = runtime.build_option_ml_ranking_table(
        root,
        symbols=["AAPL"],
        selected_only=False,
        one_per_symbol=False,
        tradable_as_of="2026-01-01",
    )

    by_contract = table.set_index("contract_symbol")
    assert set(table["symbol"]) == {"AAPL"}
    assert "MSFT_C_100" not in by_contract.index
    assert by_contract.loc["AAPL_C_105", "option_ensemble_mean_score"] == 0.5
    assert by_contract.loc["AAPL_C_100", "option_ensemble_mean_score"] == 0.5
    assert bool(by_contract.loc["AAPL_C_105", "selected_by_option_ensemble"]) is True
    assert int(by_contract.loc["AAPL_C_105", "option_ensemble_rank"]) == 1
    assert by_contract.loc["AAPL_C_105", "family_score__fmp.alpha"] == 0.2
    assert by_contract.loc["AAPL_C_105", "family_score__fmp.beta"] == 0.8


def test_build_option_ml_ranking_table_live_view_returns_one_selected_option_per_symbol(tmp_path):
    root = tmp_path / "option_family_ranker"
    family = root / "fmp.alpha"
    family.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "trade_id": "aapl_old",
                "symbol": "AAPL",
                "entry_date": "2026-01-02",
                "contract_symbol": "AAPL_C_100",
                "option_type": "call",
                "option_action": "buy_call",
                "expiration": "2026-02-20",
                "strike": 100.0,
                "pred_fmp.alpha_rank": 0.95,
            },
            {
                "trade_id": "aapl_new",
                "symbol": "AAPL",
                "entry_date": "2026-01-03",
                "contract_symbol": "AAPL_C_105",
                "option_type": "call",
                "option_action": "buy_call",
                "expiration": "2026-02-20",
                "strike": 105.0,
                "pred_fmp.alpha_rank": 0.80,
            },
            {
                "trade_id": "aapl_new",
                "symbol": "AAPL",
                "entry_date": "2026-01-03",
                "contract_symbol": "AAPL_C_110",
                "option_type": "call",
                "option_action": "buy_call",
                "expiration": "2026-02-20",
                "strike": 110.0,
                "pred_fmp.alpha_rank": 0.70,
            },
            {
                "trade_id": "msft_new",
                "symbol": "MSFT",
                "entry_date": "2026-01-03",
                "contract_symbol": "MSFT_C_100",
                "option_type": "call",
                "option_action": "buy_call",
                "expiration": "2026-02-20",
                "strike": 100.0,
                "pred_fmp.alpha_rank": 0.90,
            },
        ]
    ).to_parquet(family / "eval_scored.parquet", index=False)

    table = runtime.build_option_ml_ranking_table(root, symbols=["AAPL", "MSFT"], tradable_as_of="2026-01-01")

    assert list(table["symbol"]) == ["AAPL", "MSFT"]
    assert list(table["contract_symbol"]) == ["AAPL_C_105", "MSFT_C_100"]
    assert table["selected_by_option_ensemble"].astype(bool).all()
    assert table["option_ensemble_rank"].eq(1).all()


def test_build_option_ml_ranking_table_excludes_expired_options(tmp_path):
    root = tmp_path / "option_family_ranker"
    family = root / "fmp.alpha"
    family.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "trade_id": "aapl_old",
                "symbol": "AAPL",
                "entry_date": "2026-01-02",
                "contract_symbol": "AAPL_C_100",
                "option_type": "call",
                "option_action": "buy_call",
                "expiration": "2026-01-10",
                "strike": 100.0,
                "pred_fmp.alpha_rank": 0.99,
            },
            {
                "trade_id": "aapl_live",
                "symbol": "AAPL",
                "entry_date": "2026-01-03",
                "contract_symbol": "AAPL_C_105",
                "option_type": "call",
                "option_action": "buy_call",
                "expiration": "2026-08-21",
                "strike": 105.0,
                "pred_fmp.alpha_rank": 0.50,
            },
        ]
    ).to_parquet(family / "eval_scored.parquet", index=False)

    table = runtime.build_option_ml_ranking_table(root, symbols=["AAPL"], tradable_as_of="2026-07-06")

    assert list(table["contract_symbol"]) == ["AAPL_C_105"]
    assert pd.to_datetime(table["expiration"]).ge(pd.Timestamp("2026-07-06")).all()


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


def test_option_limit_policy_defaults_to_gtc():
    orders = pd.DataFrame([{"symbol": "AAPL_C", "action": "buy_to_open_call", "side": "buy", "bid_price": 1.0, "ask_price": 1.1}])
    priced = runtime.apply_option_limit_policy(orders)
    assert priced.loc[0, "time_in_force"] == "gtc"


def test_live_option_pricing_does_not_change_prior_day_contract_selection():
    intents = pd.DataFrame(
        [
            {"symbol": "AAPL_CALL", "underlying_symbol": "AAPL", "action": "buy_to_open_call", "side": "buy"},
            {"symbol": "MSFT_PUT", "underlying_symbol": "MSFT", "action": "sell_to_close_put", "side": "sell"},
        ]
    )
    quotes = pd.DataFrame(
        [
            {"symbol": "AAPL_CALL", "bid_price": 2.17, "ask_price": 2.25},
            {"symbol": "MSFT_PUT", "bid_price": 3.10, "ask_price": 3.17},
        ]
    )

    priced = runtime.generate_live_option_limit_prices(
        intents,
        quotes,
        priced_at="2026-07-11T16:00:00Z",
    )

    assert list(priced["symbol"]) == ["AAPL_CALL", "MSFT_PUT"]
    assert list(priced["underlying_symbol"]) == ["AAPL", "MSFT"]
    assert list(priced["limit_order_price"]) == [2.17, 3.2]
    assert priced["live_quote_priced_at"].nunique() == 1


def test_score_date_option_ranking_rejects_more_than_twenty_underlyings(tmp_path):
    with pytest.raises(ValueError, match="limit is 20"):
        runtime.build_score_date_option_ml_ranking_table(
            tmp_path,
            leaderboard=pd.DataFrame(),
            symbols=[f"S{i:02d}" for i in range(21)],
        )


def test_score_date_option_candidates_exclude_score_date_expirations(monkeypatch):
    chain = pd.DataFrame([{"contract_symbol": "placeholder"}])
    featured = pd.DataFrame(
        [
            {
                "contract_symbol": "AAPL_EXPIRES_TODAY",
                "snapshot_date": "2026-07-10",
                "expiration": "2026-07-10",
                "option_type": "call",
                "mid": 1.0,
            },
            {
                "contract_symbol": "AAPL_NEXT_SESSION",
                "snapshot_date": "2026-07-10",
                "expiration": "2026-07-13",
                "option_type": "call",
                "mid": 2.0,
            },
        ]
    )
    monkeypatch.setattr(
        "quant_warehouse.platforms.data_providers.thetadata.options.read_thetadata_eod_option_chain",
        lambda *args, **kwargs: chain,
    )
    monkeypatch.setattr(
        "quant_warehouse.platforms.data_providers.thetadata.feature_engineering.option_features.build_option_contract_features",
        lambda *args, **kwargs: SimpleNamespace(df=featured),
    )

    result = runtime.build_score_date_option_candidate_panel(
        leaderboard=pd.DataFrame(
            [{"symbol": "AAPL", "score_date": "2026-07-10", "selected": True, "close": 200.0}]
        ),
        score_date="2026-07-10",
    )

    assert result["contract_symbol"].tolist() == ["AAPL_NEXT_SESSION"]


def test_score_date_option_ranking_prefers_single_meta_stack(monkeypatch, tmp_path):
    model_path = tmp_path / "option_meta_stack" / "meta_stack_ranker.pkl"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    candidates = pd.DataFrame(
        [{"trade_id": "t1", "symbol": "AAPL", "entry_date": "2026-07-10", "contract_symbol": "AAPL_CALL"}]
    )
    expected = candidates.assign(pred_meta_stack_rank=0.9)
    monkeypatch.setattr(runtime, "build_score_date_option_candidate_panel", lambda **kwargs: candidates)
    monkeypatch.setattr(
        "quant_orchestrator.research_tools.score_option_meta_ranker",
        lambda path, option_candidates, equity_scores: expected,
    )

    result = runtime.build_score_date_option_ml_ranking_table(
        tmp_path,
        leaderboard=pd.DataFrame(),
        symbols=["AAPL"],
        equity_family_scores=pd.DataFrame([{"symbol": "AAPL"}]),
    )

    assert result.equals(expected)


def test_optionable_leaderboard_backfills_capacity_from_lower_equity_ranks(monkeypatch):
    leaderboard = pd.DataFrame(
        [
            {"symbol": "BAD", "rank": 1},
            {"symbol": "AAPL", "rank": 2},
            {"symbol": "MSFT", "rank": 3},
        ]
    )
    monkeypatch.setattr(
        "quant_warehouse.platforms.data_providers.thetadata.options.read_thetadata_eod_option_chain",
        lambda symbol, **kwargs: pd.DataFrame() if symbol == "BAD" else pd.DataFrame([{"symbol": symbol}]),
    )

    selected = runtime.select_optionable_leaderboard(
        leaderboard,
        score_date="2026-07-10",
        top_k=2,
    )

    assert selected["symbol"].tolist() == ["AAPL", "MSFT"]
    assert selected["option_rank"].tolist() == [1, 2]


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
    assert float(actions.loc[2, "limit_order_price"]) == 7.0
    assert actions.loc[2, "limit_price_source"] == "bid_price"
    assert float(actions.loc[2, "buy_discount_pct"]) == 90.0


def test_robinhood_100_gate_blocks_only_robinhood_copy():
    paper_orders = pd.DataFrame(
        [{"symbol": "AAPL", "action": "buy_to_open_call", "side": "buy", "qty": 1}]
    )

    robinhood_orders = runtime.apply_robinhood_submission_gate(
        paper_orders,
        gate_discount_pct=100.0,
    )

    assert "skip_submit" not in paper_orders.columns
    assert bool(robinhood_orders.loc[0, "skip_submit"]) is True
    assert robinhood_orders.loc[0, "skip_reason"] == "robinhood_gate_100_blocks_orders"


def test_prior_day_selection_rejects_live_day_inputs():
    prior = pd.DataFrame([{"symbol": "AAPL", "selection_as_of": "2026-07-10"}])
    same_day = pd.DataFrame([{"symbol": "AAPL", "selection_as_of": "2026-07-11"}])

    accepted = runtime.validate_prior_day_selection(
        prior,
        selection_date_col="selection_as_of",
        live_date="2026-07-11T16:00:00Z",
    )
    assert accepted.equals(prior)
    with pytest.raises(ValueError, match="prior-day"):
        runtime.validate_prior_day_selection(
            same_day,
            selection_date_col="selection_as_of",
            live_date="2026-07-11T16:00:00Z",
        )


def test_submission_safety_rejects_unstamped_and_stale_plans():
    order = pd.DataFrame([{"symbol": "AAPL", "side": "buy", "qty": 1}])

    with pytest.raises(ValueError, match="missing plan_created_at"):
        runtime.validate_order_plan_for_submission(order, asset_type="equity", now="2026-07-11T12:00:00Z")

    order["plan_created_at"] = "2026-07-09T12:00:00Z"
    with pytest.raises(ValueError, match="stale order plan"):
        runtime.validate_order_plan_for_submission(order, asset_type="equity", now="2026-07-11T12:00:00Z")


def test_submission_safety_rejects_duplicate_and_oversized_option_orders():
    created = "2026-07-11T11:00:00Z"
    duplicate = pd.DataFrame(
        [
            {"symbol": "AAPL260117C00200000", "side": "buy", "qty": 1, "plan_created_at": created},
            {"symbol": "AAPL260117C00200000", "side": "buy", "qty": 1, "plan_created_at": created},
        ]
    )
    with pytest.raises(ValueError, match="duplicate order rows"):
        runtime.validate_order_plan_for_submission(duplicate, asset_type="option", now="2026-07-11T12:00:00Z")

    oversized = duplicate.iloc[:1].copy()
    oversized["qty"] = 101
    with pytest.raises(ValueError, match="100 per-order limit"):
        runtime.validate_order_plan_for_submission(oversized, asset_type="option", now="2026-07-11T12:00:00Z")


def test_submission_safety_allows_stamped_cancellations_and_valid_orders():
    plan = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "action": "cancel_open_order",
                "side": "cancel",
                "qty": 0,
                "order_id": "open-1",
                "plan_created_at": "2026-07-11T11:00:00Z",
            },
            {
                "symbol": "MSFT",
                "action": "rebalance",
                "side": "buy",
                "qty": 5,
                "order_type": "market",
                "order_id": "",
                "plan_created_at": "2026-07-11T11:00:00Z",
            },
        ]
    )

    validated = runtime.validate_order_plan_for_submission(plan, asset_type="equity", now="2026-07-11T12:00:00Z")

    assert list(validated["symbol"]) == ["AAPL", "MSFT"]


def test_submit_alpaca_option_orders_uses_option_quantity_limit_without_side_effects():
    class FakeClient:
        def __init__(self):
            self.submitted = []

        def submit_orders(self, orders):
            self.submitted.extend(orders)
            return orders

    client = FakeClient()
    plan = pd.DataFrame(
        [
            {
                "symbol": "AAPL260117C00200000",
                "side": "buy",
                "qty": 101,
                "plan_created_at": pd.Timestamp.now(tz="UTC").isoformat(),
            }
        ]
    )

    with pytest.raises(ValueError, match="100 per-order limit"):
        runtime.submit_alpaca_orders(client, plan, asset_type="option")

    assert client.submitted == []
def test_resolve_option_training_panel_selects_exact_unified_contract(tmp_path):
    compatible = tmp_path / "verified"
    compatible.mkdir()
    panel = compatible / "option_candidate_panel_unified.parquet"
    pd.DataFrame(
        {"rank_y": [1.0], "label_basis": ["realized_exit_return"]}
    ).to_parquet(panel, index=False)
    (compatible / "run_summary.json").write_text(
        json.dumps({"status": "ok", "min_market_cap": 10_000_000_000}),
        encoding="utf-8",
    )

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    pd.DataFrame({"rank_y": [1.0]}).to_parquet(
        legacy / "option_candidate_panel_unified.parquet", index=False
    )
    (legacy / "run_summary.json").write_text(
        json.dumps({"status": "ok", "min_market_cap": 10_000_000_000}),
        encoding="utf-8",
    )

    assert runtime.resolve_option_training_panel(
        tmp_path, min_market_cap=10_000_000_000
    ) == panel
