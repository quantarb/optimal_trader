from __future__ import annotations

import math
from typing import Any, Sequence

from .models import ModelArtifact


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    return str(value)


def next_model_version(name: str) -> int:
    latest = (
        ModelArtifact.objects.filter(name=name)
        .order_by("-version")
        .values_list("version", flat=True)
        .first()
    )
    return 1 if latest is None else int(latest) + 1


def save_model_artifact(
    *,
    name: str,
    model_obj: Any,
    framework: str = "",
    task_type: str = "",
    target_col: str = "",
    feature_cols: Sequence[str] | None = None,
    metrics: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    version: int | None = None,
    is_active: bool = True,
) -> ModelArtifact:
    record = ModelArtifact(
        name=name,
        version=next_model_version(name) if version is None else int(version),
        framework=framework,
        task_type=task_type,
        target_col=target_col,
        feature_cols=list(feature_cols or []),
        metrics=_json_safe(dict(metrics or {})),
        params=_json_safe(dict(params or {})),
        metadata=_json_safe(dict(metadata or {})),
        is_active=is_active,
    )
    record.set_artifact(model_obj)
    record.save()
    return record


def load_model_artifact(*, name: str, version: int | None = None) -> Any:
    queryset = ModelArtifact.objects.filter(name=name)
    if version is None:
        record = queryset.order_by("-version").first()
    else:
        record = queryset.filter(version=version).first()
    if record is None:
        raise ModelArtifact.DoesNotExist(f"No model artifact found for {name!r}.")
    return record.get_artifact()
