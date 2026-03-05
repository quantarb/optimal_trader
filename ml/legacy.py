"""Bridge the legacy ML package into the Django app namespace.

Import from `ml.legacy` to access the existing functionality currently
implemented under `modules.models`.
"""

from importlib import import_module
from types import ModuleType
from typing import Any

_LEGACY_MODULE = "modules.models"

__all__ = [
    "FitSpec",
    "Model",
    "SequenceSpec",
    "RLConfig",
    "run_a2c_workflow",
    "run_ppo_workflow",
    "backtest_buy_and_hold_equal_weight",
    "backtest_strategy_per_stock_discrete",
    "make_rebalance_mask",
    "apply_buy_cap",
    "trade_cost_from_bps",
    "summarize_returns",
]


def _legacy_module() -> ModuleType:
    return import_module(_LEGACY_MODULE)


def __getattr__(name: str) -> Any:
    try:
        return getattr(_legacy_module(), name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_legacy_module())))
