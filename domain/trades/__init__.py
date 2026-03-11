"""Trade-generation domain logic."""

from domain.trades.operations import (
    apply_trade_deduplication,
    build_label_rows_from_completed_trades,
    build_label_statistics,
    trade_return_pct,
)
from domain.trades.optimal import (
    Trade,
    solve_joint_trades_by_frequency,
    solve_longs_by_frequency,
    solve_optimal_joint_trades_generic,
    solve_optimal_trades_generic,
    solve_shorts_by_frequency,
    solve_trades_by_frequency,
)
from domain.trades.panel import labels_panel_to_trades_df
from domain.trades.specs import TradeGenerationResult

__all__ = [
    "Trade",
    "TradeGenerationResult",
    "apply_trade_deduplication",
    "build_label_rows_from_completed_trades",
    "build_label_statistics",
    "labels_panel_to_trades_df",
    "solve_joint_trades_by_frequency",
    "solve_longs_by_frequency",
    "solve_optimal_joint_trades_generic",
    "solve_optimal_trades_generic",
    "solve_shorts_by_frequency",
    "solve_trades_by_frequency",
    "trade_return_pct",
]

