from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from app.trading_notebook import (
    find_repo_root,
    live_trade_notebook_summary,
    make_live_trade_notebook_config,
    make_similarity_query,
)
from app.notebook_runtime import NotebookProgress


_SYNTHETIC_BACKTEST_PATH = Path(__file__).resolve().parents[1] / "workflows" / "synthetic_backtest.py"
_SPEC = importlib.util.spec_from_file_location("synthetic_backtest_under_test", _SYNTHETIC_BACKTEST_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_SYNTHETIC_BACKTEST = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SYNTHETIC_BACKTEST)
run_top_k_long_only_score_rule = _SYNTHETIC_BACKTEST.run_top_k_long_only_score_rule
prepare_capacity_rule_inputs = _SYNTHETIC_BACKTEST.prepare_capacity_rule_inputs
summarize_curve = _SYNTHETIC_BACKTEST.summarize_curve


def test_summarize_curve_returns_overall_and_yearly_metrics():
    returns = pd.Series(
        [0.10, -0.05, 0.02],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2025-01-02"]),
    )

    summary = summarize_curve(returns, [2024, 2025], "test")

    assert summary["equity_curve"].index.equals(returns.index)
    assert list(summary["yearly_df"]["test_year"]) == [2024, 2025]
    assert set(summary["yearly_df"]["mode"]) == {"test"}


def test_long_only_score_rule_respects_capacity_and_lagged_signals():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=3), ["AAPL", "MSFT"]],
        names=["date", "symbol"],
    )
    panel = pd.DataFrame(
        {
            "score": [0.9, 0.8, 0.7, 0.95, 0.6, 0.5],
            "prob_buy": [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            "prob_short": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            "close": [100.0, 100.0, 101.0, 101.0, 102.0, 102.0],
        },
        index=index,
    )

    result = run_top_k_long_only_score_rule(
        panel=panel,
        score_col="score",
        component_cols=["prob_buy"],
        component_threshold=0.5,
        price_col="close",
        top_k=1,
    )

    positions = result["positions"]
    assert int(positions.iloc[0].sum()) == 0
    assert positions.loc[pd.Timestamp("2024-01-02"), "AAPL"] == 1
    assert int(positions.iloc[1].sum()) == 1


def test_prepare_capacity_rule_inputs_is_public_and_lags_scores():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=2), ["AAPL", "MSFT"]],
        names=["date", "symbol"],
    )
    panel = pd.DataFrame(
        {
            "prob_buy": [0.8, 0.6, 0.7, 0.5],
            "prob_short": [0.2, 0.4, 0.3, 0.5],
            "close": [100.0, 200.0, 101.0, 201.0],
        },
        index=index,
    )

    inputs = prepare_capacity_rule_inputs(panel, "prob_buy", ["prob_buy"], "close")

    assert inputs["symbols"] == ["AAPL", "MSFT"]
    assert inputs["score"].iloc[0].isna().all()
    assert inputs["score"].iloc[1].tolist() == [0.8, 0.6]


def test_trading_notebook_helpers_build_config_summary_and_query(tmp_path: Path):
    repo_root = tmp_path / "repo"
    (repo_root / "app").mkdir(parents=True)
    (repo_root / "notebooks").mkdir()

    assert find_repo_root(repo_root / "notebooks") == repo_root
    cfg = make_live_trade_notebook_config(
        repo_root,
        data_start="2000-01-01",
        min_market_cap=5_000_000_000,
        refresh_fmp_data=False,
        skip_cached_inactive_symbols=True,
        refresh_macro_data=False,
        leaderboard_top_k=12,
    )
    summary = live_trade_notebook_summary(cfg, query_symbol="aapl", similar_trades_top_k=7)
    query = make_similarity_query(
        SimpleNamespace(latest_date=pd.Timestamp("2026-06-09"), artifact_dir=cfg["runtime"]["artifact_dir"]),
        cfg,
        query_symbol="aapl",
        top_k=7,
    )

    assert summary["query_symbol"] == "AAPL"
    assert summary["leaderboard_top_k"] == 12
    assert query.symbol == "AAPL"
    assert query.reference_end_date == "2026-06-08"
    assert query.top_k == 7


def test_notebook_progress_runs_callable_without_heartbeat():
    progress = NotebookProgress()

    assert progress.run("add", lambda left, right: left + right, 2, 3, heartbeat_seconds=0) == 5
