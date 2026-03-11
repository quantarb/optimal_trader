from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from fmp.models import Symbol, WorkflowState

if TYPE_CHECKING:
    from django.http import HttpRequest


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
    try:
        return int(str(artifact_id_raw or "").strip())
    except Exception:
        return 0


def latest_universe_artifact_id() -> int:
    try:
        from pipeline.models import Artifact

        artifact = Artifact.objects.filter(artifact_type="UNIVERSE").only("id").first()
        return int(artifact.pk) if artifact is not None else 0
    except Exception:
        return 0


def universe_artifact_choices() -> list[tuple[str, str]]:
    try:
        from pipeline.models import Artifact

        rows = (
            Artifact.objects.filter(artifact_type="UNIVERSE")
            .select_related("pipeline_run")
            .order_by("-created_at", "-id")[:100]
        )
        choices: list[tuple[str, str]] = []
        for artifact in rows:
            name = universe_artifact_name(int(artifact.pk))
            symbol_count = len(_normalize_symbols(list((artifact.content or {}).get("symbols") or [])))
            label = f"#{artifact.pk}"
            if name:
                label += f" - {name}"
            if symbol_count:
                label += f" ({symbol_count} symbols)"
            choices.append((str(artifact.pk), label))
        return choices
    except Exception:
        return []


def universe_artifact_name(artifact_id: int) -> str:
    if artifact_id <= 0:
        return ""
    try:
        from pipeline.models import Artifact

        artifact = Artifact.objects.filter(pk=artifact_id, artifact_type="UNIVERSE").select_related("pipeline_run").first()
        if artifact is None:
            return ""
        name = str((artifact.metadata or {}).get("name") or (artifact.content or {}).get("name") or "").strip()
        if name:
            return name
        run_name = str((artifact.pipeline_run.name if artifact.pipeline_run else "") or "").strip()
        if run_name:
            return run_name
        return f"Universe #{artifact.pk}"
    except Exception:
        return ""


def universe_symbols_from_artifact_id(artifact_id: int) -> list[str]:
    if artifact_id <= 0:
        return []

    try:
        from pipeline.models import Artifact

        artifact = Artifact.objects.filter(pk=artifact_id, artifact_type="UNIVERSE").first()
        if artifact is None:
            return []
        payload = dict(artifact.content or {})
        uri = str(artifact.uri or "").strip()
        if uri:
            path = Path(uri)
            if path.exists() and path.is_file():
                try:
                    file_payload = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(file_payload, dict):
                        payload = file_payload
                except Exception:
                    pass
        return _normalize_symbols(payload.get("symbols") or [])
    except Exception:
        return []


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
