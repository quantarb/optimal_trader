from __future__ import annotations

from importlib import import_module

__all__ = ["CumlRFClassifier", "SklearnMoERFClassifier", "SklearnRFClassifier", "SklearnRFRegressor"]


_LAZY_EXPORTS = {
    "CumlRFClassifier": ("ml.frameworks.sklearn.cuml_classifier", "CumlRFClassifier"),
    "SklearnMoERFClassifier": ("ml.frameworks.sklearn.moe_classifier", "SklearnMoERFClassifier"),
    "SklearnRFClassifier": ("ml.frameworks.sklearn.classifier", "SklearnRFClassifier"),
    "SklearnRFRegressor": ("ml.frameworks.sklearn.regressor", "SklearnRFRegressor"),
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
