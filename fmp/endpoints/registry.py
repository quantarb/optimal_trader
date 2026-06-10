from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType
from typing import Callable, Any

from .base import EndpointDefinition

# Auto-discover endpoint modules in this package (all .py except special ones).
# Each endpoint module is expected to define a callable named "build" that
# takes a symbol-like object and returns an EndpointDefinition.
# This eliminates the manual from-import + _BUILDERS tuple maintenance smell.
_EXCLUDE = {"__init__", "base", "registry", "helpers"}


def _discover_builders() -> list[Callable[[Any], EndpointDefinition]]:
    builders: list[Callable[[Any], EndpointDefinition]] = []
    package_name = __package__ or "fmp.endpoints"
    package = importlib.import_module(package_name)

    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name in _EXCLUDE or module_info.ispkg:
            continue
        try:
            mod: ModuleType = importlib.import_module(f"{package_name}.{module_info.name}")
            b = getattr(mod, "build", None)
            if callable(b):
                builders.append(b)
        except Exception:
            # Discovery is best-effort; a bad endpoint module will surface on first use.
            continue
    return builders


_BUILDERS: list[Callable[[Any], EndpointDefinition]] = _discover_builders()


def get_symbol_endpoint_definitions(symbol_obj: Any) -> list[EndpointDefinition]:
    return [builder(symbol_obj) for builder in _BUILDERS]
