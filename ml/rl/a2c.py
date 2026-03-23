from __future__ import annotations

from .common import (
    RLConfig,
    apply_buy_cap,
    backtest_buy_and_hold_equal_weight,
    backtest_strategy_per_stock_discrete,
    make_rebalance_mask,
    run_sb3_per_symbol_workflow,
    run_sb3_workflow,
    summarize_returns,
    trade_cost_from_bps,
)


def run_a2c_workflow(*, bt_panel, cfg: RLConfig, train_split_date, years):
    return run_sb3_workflow(
        bt_panel=bt_panel,
        cfg=cfg,
        train_split_date=train_split_date,
        years=years,
        algorithm="a2c",
    )


def run_a2c_per_symbol_workflow(*, bt_panel, cfg: RLConfig, train_split_date, years):
    return run_sb3_per_symbol_workflow(
        bt_panel=bt_panel,
        cfg=cfg,
        train_split_date=train_split_date,
        years=years,
        algorithm="a2c",
    )


__all__ = [
    "RLConfig",
    "run_a2c_workflow",
    "run_a2c_per_symbol_workflow",
    "backtest_buy_and_hold_equal_weight",
    "backtest_strategy_per_stock_discrete",
    "make_rebalance_mask",
    "apply_buy_cap",
    "trade_cost_from_bps",
    "summarize_returns",
]
