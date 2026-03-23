from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from django.db import DatabaseError

from fmp.models import Symbol, WorkflowState

if TYPE_CHECKING:
    from django.http import HttpRequest


@dataclass(frozen=True)
class UniverseArtifactRecord:
    artifact_id: int
    name: str
    symbols: list[str]


def workflow_symbols_from_request(request: HttpRequest) -> list[str]:
    artifact_id = selected_universe_artifact_id(request)
    if artifact_id > 0:
        selected = universe_symbols_from_artifact_id(artifact_id)
        if selected:
            return selected

    session_symbols = request.session.get("universe_screener_symbols") or []
    if isinstance(session_symbols, list) and session_symbols:
        return _normalize_symbols(session_symbols)

    state = WorkflowState.objects.filter(key="default").only("universe_symbols").first()
    if state and isinstance(state.universe_symbols, list):
        return _normalize_symbols(state.universe_symbols)
    return []


def selected_universe_artifact_id(request: HttpRequest) -> int:
    artifact_id_raw = (
        request.POST.get("universe_artifact_id")
        or request.GET.get("universe_artifact_id")
        or request.POST.get("universeArtifactId")
        or request.GET.get("universeArtifactId")
    )
    return _parse_int(artifact_id_raw)


def latest_universe_artifact_id() -> int:
    artifact = _latest_universe_artifact()
    return _parse_int(getattr(artifact, "pk", 0))


def universe_artifact_choices() -> list[tuple[str, str]]:
    artifacts = _list_universe_artifacts(limit=100)
    choices: list[tuple[str, str]] = []
    for artifact in artifacts:
        record = _build_universe_artifact_record(artifact)
        label = f"#{record.artifact_id}"
        if record.name:
            label += f" - {record.name}"
        if record.symbols:
            label += f" ({len(record.symbols)} symbols)"
        choices.append((str(record.artifact_id), label))
    return choices


def universe_artifact_name(artifact_id: int) -> str:
    if artifact_id <= 0:
        return ""
    artifact = _universe_artifact_by_id(artifact_id)
    if artifact is None:
        return ""
    return _build_universe_artifact_record(artifact).name


def universe_symbols_from_artifact_id(artifact_id: int) -> list[str]:
    if artifact_id <= 0:
        return []
    artifact = _universe_artifact_by_id(artifact_id)
    if artifact is None:
        return []
    return _build_universe_artifact_record(artifact).symbols


def build_symbol_choices(preferred_symbols: list[str] | None = None) -> list[tuple[str, str]]:
    rows = Symbol.objects.order_by("symbol").values_list("symbol", "company_name")
    label_map: dict[str, str] = {}
    for symbol, company_name in rows:
        code = str(symbol or "").strip().upper()
        if not code:
            continue
        label_map[code] = f"{code} - {company_name}" if company_name else code

    if preferred_symbols:
        preferred = []
        seen: set[str] = set()
        for raw in preferred_symbols:
            code = str(raw or "").strip().upper()
            if not code or code in seen:
                continue
            seen.add(code)
            preferred.append((code, label_map.get(code, code)))
        if preferred:
            return preferred

    return sorted(label_map.items(), key=lambda item: item[0])


def default_feature_symbol(request: HttpRequest, fallback: str = "AAPL") -> str:
    symbols = workflow_symbols_from_request(request)
    if symbols:
        return symbols[0]
    value = Symbol.objects.order_by("symbol").values_list("symbol", flat=True).first()
    symbol = str(value or "").strip().upper()
    return symbol or fallback


def _normalize_symbols(values: list[str]) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()
    for value in values:
        code = str(value or "").strip().upper()
        if not code or code in seen:
            continue
        seen.add(code)
        symbols.append(code)
    return symbols


def _parse_int(value: object) -> int:
    try:
        return int(str(value or "").strip())
    except (TypeError, ValueError):
        return 0


def _artifact_model():
    from pipeline.models import Artifact

    return Artifact


def _latest_universe_artifact():
    try:
        Artifact = _artifact_model()
        return Artifact.objects.filter(artifact_type="UNIVERSE").only("id").first()
    except (ImportError, DatabaseError):
        return None


def _list_universe_artifacts(*, limit: int) -> list[Any]:
    try:
        Artifact = _artifact_model()
        rows = (
            Artifact.objects.filter(artifact_type="UNIVERSE")
            .select_related("pipeline_run")
            .order_by("-created_at", "-id")[:limit]
        )
        return list(rows)
    except (ImportError, DatabaseError):
        return []


def _universe_artifact_by_id(artifact_id: int):
    if artifact_id <= 0:
        return None
    try:
        Artifact = _artifact_model()
        return Artifact.objects.filter(pk=artifact_id, artifact_type="UNIVERSE").select_related("pipeline_run").first()
    except (ImportError, DatabaseError):
        return None


def _build_universe_artifact_record(artifact: Any) -> UniverseArtifactRecord:
    artifact_id = _parse_int(getattr(artifact, "pk", 0))
    payload = _artifact_payload(artifact)
    return UniverseArtifactRecord(
        artifact_id=artifact_id,
        name=_artifact_name(artifact, payload),
        symbols=_normalize_symbols(list(payload.get("symbols") or [])),
    )


def _artifact_payload(artifact: Any) -> dict[str, Any]:
    payload = dict(getattr(artifact, "content", {}) or {})
    file_payload = _artifact_file_payload(str(getattr(artifact, "uri", "") or "").strip())
    return file_payload or payload


def _artifact_file_payload(uri: str) -> dict[str, Any] | None:
    if not uri:
        return None
    path = Path(uri)
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _artifact_name(artifact: Any, payload: dict[str, Any]) -> str:
    metadata = dict(getattr(artifact, "metadata", {}) or {})
    name = str(metadata.get("name") or payload.get("name") or "").strip()
    if name:
        return name
    pipeline_run = getattr(artifact, "pipeline_run", None)
    run_name = str(getattr(pipeline_run, "name", "") or "").strip()
    if run_name:
        return run_name
    artifact_id = _parse_int(getattr(artifact, "pk", 0))
    return f"Universe #{artifact_id}" if artifact_id > 0 else ""
