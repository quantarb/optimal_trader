from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from platforms.agents.trading_agents import TradingAgentsReviewConfig, approved_symbols, review_trade_candidates


def test_review_trade_candidates_uses_isolated_worker(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        payload = json.loads(kwargs["input"])
        calls.append((command, payload, kwargs["env"]))
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(
                {
                    "request_id": payload["request_id"],
                    "decisions": [
                        {"symbol": "AAPL", "decision": "BUY", "rating": "Buy"},
                        {"symbol": "MSFT", "decision": "HOLD", "rating": "Hold"},
                    ],
                }
            ),
        )

    monkeypatch.setenv("ML_ALPACA_SECRET_KEY", "must-not-leak")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "worker-needs-this")
    monkeypatch.setattr("platforms.agents.trading_agents.subprocess.run", fake_run)

    reviewed = review_trade_candidates(
        pd.DataFrame(
            [
                {"symbol": "aapl", "score_date": "2026-07-05"},
                {"symbol": "msft", "score_date": "2026-07-05"},
            ]
        ),
        config=TradingAgentsReviewConfig(
            fast_symbol_date_only=False,
            max_workers=1,
            worker_command=("worker-python",),
        ),
    )

    assert calls[0][0] == ["worker-python"]
    assert [row["symbol"] for row in calls[0][1]["candidates"]] == ["AAPL", "MSFT"]
    assert all("evidence" not in row for row in calls[0][1]["candidates"])
    assert "ML_ALPACA_SECRET_KEY" not in calls[0][2]
    assert "DEEPSEEK_API_KEY" not in calls[0][2]
    assert calls[0][2]["TRADINGAGENTS_ENV_FILE"].endswith("TradingAgents/.env")
    assert reviewed.loc[reviewed["symbol"] == "AAPL", "llm_decision"].iloc[0] == "approved"
    assert reviewed.loc[reviewed["symbol"] == "MSFT", "llm_decision"].iloc[0] == "rejected"
    assert approved_symbols(reviewed) == {"AAPL"}


def test_review_trade_candidates_worker_failure_is_hold(monkeypatch):
    def fail(*args, **kwargs):
        raise TimeoutError("worker timed out")

    monkeypatch.setattr("platforms.agents.trading_agents.subprocess.run", fail)
    reviewed = review_trade_candidates(
        pd.DataFrame([{"symbol": "AAPL"}]),
        config=TradingAgentsReviewConfig(fast_symbol_date_only=False),
    )

    assert reviewed.loc[0, "llm_decision"] == "rejected"
    assert reviewed.loc[0, "llm_rating"] == "Hold"
    assert "tradingagents unavailable" in reviewed.loc[0, "llm_reason"]


def test_fast_review_sends_only_symbol_and_date_contract(monkeypatch):
    calls = []

    def fake_decision(symbol, trade_date, *, config):
        calls.append((symbol, trade_date))
        return ("sell", "bearish")

    monkeypatch.setattr("platforms.agents.trading_agents._deepseek_symbol_date_decision", fake_decision)
    reviewed = review_trade_candidates(
        pd.DataFrame([{"symbol": "AAPL", "score_date": "2026-07-10", "prob_buy": 0.9}]),
        config=TradingAgentsReviewConfig(max_workers=1),
    )

    assert calls == [("AAPL", "2026-07-10")]
    assert reviewed.loc[0, "llm_rating"] == "Sell"
