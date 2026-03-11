from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path
from typing import Any, Callable

from django.utils import timezone

from .research_suite import RESEARCH_REPORT_SCHEMA_VERSION
from .services import ARTIFACT_DIR


def load_cohort_summary_files(limit: int = 25, *, artifact_dir: str | Path | None = None) -> list[dict[str, Any]]:
    def build_row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
        leaderboard_rows = list(payload.get("leaderboard_rows") or [])
        rejected_rows = list(payload.get("rejected_rows") or [])
        report_summary = dict(payload.get("report_summary") or {})
        research_profile = dict(payload.get("research_profile") or {})
        base_artifacts = dict(payload.get("base_artifacts") or {})
        strategy_definition = dict(payload.get("strategy_definition") or {})
        suite_outputs = list(payload.get("suite_outputs") or [])
        fit_job = str(payload.get("fit_job") or "")
        score_job = str(payload.get("score_job") or "")
        if not fit_job and suite_outputs:
            fit_job = ", ".join(sorted({str(item.get("fit_job") or "") for item in suite_outputs if str(item.get("fit_job") or "")}))
        return {
            "path": str(path),
            "name": path.name,
            "leaderboard_rows": leaderboard_rows,
            "rejected_rows": rejected_rows,
            "variant_count": len(leaderboard_rows),
            "fit_job": fit_job,
            "score_job": score_job,
            "strategy_definition": strategy_definition,
            "base_artifacts": base_artifacts,
            "kind": "research_report",
            "report_summary": report_summary,
            "research_profile": research_profile,
            "payload": payload,
            "updated_at": timezone.datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
        }

    return _load_report_files(
        limit=limit,
        artifact_dir=artifact_dir,
        schema_version=RESEARCH_REPORT_SCHEMA_VERSION,
        required_keys=("leaderboard_rows", "report_summary", "research_profile"),
        row_builder=build_row,
    )


def load_diagnostic_report_files(limit: int = 25, *, artifact_dir: str | Path | None = None) -> list[dict[str, Any]]:
    def build_row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "path": str(path),
            "name": path.name,
            "payload": payload,
            "recommendations": list(payload.get("recommendations") or []),
            "observations": list(payload.get("observations") or []),
            "candidate_rule": dict(payload.get("candidate_rule") or {}),
            "updated_at": timezone.datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
        }

    return _load_report_files(
        limit=limit,
        artifact_dir=artifact_dir,
        kind="diagnostic_report",
        row_builder=build_row,
    )


def load_oracle_report_files(limit: int = 25, *, artifact_dir: str | Path | None = None) -> list[dict[str, Any]]:
    def build_row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "path": str(path),
            "name": path.name,
            "payload": payload,
            "oracle_summary": dict(payload.get("oracle_summary") or {}),
            "updated_at": timezone.datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
        }

    return _load_report_files(limit=limit, artifact_dir=artifact_dir, kind="oracle_trade_report", row_builder=build_row)


def load_feature_attribution_files(limit: int = 25, *, artifact_dir: str | Path | None = None) -> list[dict[str, Any]]:
    def build_row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "path": str(path),
            "name": path.name,
            "payload": payload,
            "summary": dict(payload.get("summary") or {}),
            "updated_at": timezone.datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
        }

    return _load_report_files(limit=limit, artifact_dir=artifact_dir, kind="feature_attribution_report", row_builder=build_row)


def load_json_payload_from_uri(uri: str) -> dict[str, Any]:
    path = Path(str(uri or "").strip())
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _load_report_files(
    *,
    limit: int,
    row_builder: Callable[[Path, dict[str, Any]], dict[str, Any]],
    artifact_dir: str | Path | None = None,
    kind: str = "",
    schema_version: int | None = None,
    required_keys: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    summary_dir = Path(artifact_dir or ARTIFACT_DIR)
    if not summary_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(summary_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if kind and str(payload.get("kind") or "") != kind:
            continue
        if schema_version is not None and int(payload.get("schema_version") or 0) != int(schema_version):
            continue
        if required_keys and not all(payload.get(key) for key in required_keys):
            continue
        rows.append(row_builder(path, payload))
        if len(rows) >= limit:
            break
    return rows
