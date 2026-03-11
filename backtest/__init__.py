"""Canonical backtest namespace."""

from ml import backtest_strategy_per_stock_discrete
from backtest.backtest import (
    BacktestResult,
    ExecutionConfig,
    Strategy,
    backtest_panel,
    build_panel_from_daily_by_symbol,
    run_backtest,
)
from backtest.latest import (
    make_autoencoder_familiarity_predictor,
    run_latest_prediction_and_llm_prompt,
    run_panel_prediction_custom,
)

__all__ = [
    "BacktestResult",
    "ExecutionConfig",
    "Strategy",
    "backtest_panel",
    "backtest_strategy_per_stock_discrete",
    "build_panel_from_daily_by_symbol",
    "make_autoencoder_familiarity_predictor",
    "run_backtest",
    "run_latest_prediction_and_llm_prompt",
    "run_panel_prediction_custom",
]
