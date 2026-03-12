from __future__ import annotations

from importlib import import_module

__all__ = ["AutoGluonClassifier", "AutoGluonRegressor"]


_LAZY_EXPORTS = {
    "AutoGluonClassifier": ("ml.frameworks.autogluon.classifier", "AutoGluonClassifier"),
    "AutoGluonRegressor": ("ml.frameworks.autogluon.regressor", "AutoGluonRegressor"),
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
