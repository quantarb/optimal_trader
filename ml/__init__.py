from __future__ import annotations

from importlib import import_module

from ml.base import FitSpec, Model, SequenceSpec

__all__ = [
    "FitSpec",
    "Model",
    "SequenceSpec",
    "AutoEncoderConfig",
    "TorchAutoEncoder",
    "AutoGluonClassifier",
    "AutoGluonRegressor",
    "SklearnRFClassifier",
    "SklearnRFRegressor",
    "RLConfig",
    "run_a2c_workflow",
    "run_a2c_per_symbol_workflow",
    "run_ppo_workflow",
    "backtest_buy_and_hold_equal_weight",
    "backtest_strategy_per_stock_discrete",
    "make_rebalance_mask",
    "apply_buy_cap",
    "trade_cost_from_bps",
    "summarize_returns",
]


_LAZY_EXPORTS = {
    "AutoEncoderConfig": ("ml.autoencoder", "AutoEncoderConfig"),
    "TorchAutoEncoder": ("ml.autoencoder", "TorchAutoEncoder"),
    "AutoGluonClassifier": ("ml.frameworks", "AutoGluonClassifier"),
    "AutoGluonRegressor": ("ml.frameworks", "AutoGluonRegressor"),
    "SklearnRFClassifier": ("ml.frameworks", "SklearnRFClassifier"),
    "SklearnRFRegressor": ("ml.frameworks", "SklearnRFRegressor"),
    "RLConfig": ("ml.rl", "RLConfig"),
    "run_a2c_workflow": ("ml.rl", "run_a2c_workflow"),
    "run_a2c_per_symbol_workflow": ("ml.rl", "run_a2c_per_symbol_workflow"),
    "run_ppo_workflow": ("ml.rl", "run_ppo_workflow"),
    "backtest_buy_and_hold_equal_weight": ("ml.rl", "backtest_buy_and_hold_equal_weight"),
    "backtest_strategy_per_stock_discrete": ("ml.rl", "backtest_strategy_per_stock_discrete"),
    "make_rebalance_mask": ("ml.rl", "make_rebalance_mask"),
    "apply_buy_cap": ("ml.rl", "apply_buy_cap"),
    "trade_cost_from_bps": ("ml.rl", "trade_cost_from_bps"),
    "summarize_returns": ("ml.rl", "summarize_returns"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(str(name))
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
