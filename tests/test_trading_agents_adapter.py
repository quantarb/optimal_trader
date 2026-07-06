from __future__ import annotations

import sys
from types import ModuleType

import pandas as pd

from platforms.agents.trading_agents import approved_symbols, review_trade_candidates


def test_review_trade_candidates_uses_tradingagents_graph(monkeypatch):
    calls: list[tuple[str, str]] = []

    class FakeTradingAgentsGraph:
        def __init__(self, selected_analysts, debug, config):
            self.selected_analysts = selected_analysts
            self.debug = debug
            self.config = config

        def propagate(self, symbol, trade_date):
            calls.append((symbol, trade_date))
            rating = "Buy" if symbol == "AAPL" else "Hold"
            return {"final_trade_decision": f"**Rating**: {rating}"}, rating

    root = ModuleType("tradingagents")
    graph_pkg = ModuleType("tradingagents.graph")
    graph_mod = ModuleType("tradingagents.graph.trading_graph")
    graph_mod.TradingAgentsGraph = FakeTradingAgentsGraph
    config_mod = ModuleType("tradingagents.default_config")
    config_mod.DEFAULT_CONFIG = {"llm_provider": "openai"}

    monkeypatch.setitem(sys.modules, "tradingagents", root)
    monkeypatch.setitem(sys.modules, "tradingagents.graph", graph_pkg)
    monkeypatch.setitem(sys.modules, "tradingagents.graph.trading_graph", graph_mod)
    monkeypatch.setitem(sys.modules, "tradingagents.default_config", config_mod)

    reviewed = review_trade_candidates(
        pd.DataFrame(
            [
                {"symbol": "aapl", "score_date": "2026-07-05"},
                {"symbol": "msft", "score_date": "2026-07-05"},
            ]
        )
    )

    assert calls == [("AAPL", "2026-07-05"), ("MSFT", "2026-07-05")]
    assert reviewed.loc[reviewed["symbol"] == "AAPL", "llm_decision"].iloc[0] == "approved"
    assert reviewed.loc[reviewed["symbol"] == "MSFT", "llm_decision"].iloc[0] == "rejected"
    assert approved_symbols(reviewed) == {"AAPL"}


def test_review_trade_candidates_marks_unavailable(monkeypatch):
    for name in list(sys.modules):
        if name == "tradingagents" or name.startswith("tradingagents."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    monkeypatch.setattr("platforms.agents.trading_agents.default_trading_agents_repo", lambda: __import__("pathlib").Path("/missing"))
    reviewed = review_trade_candidates(pd.DataFrame([{"symbol": "AAPL"}]))

    assert reviewed.loc[0, "llm_decision"] == "unavailable"
    assert "tradingagents unavailable" in reviewed.loc[0, "llm_reason"]
