from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from django.db import DatabaseError
from django.urls import reverse

from ml.models import ModelArtifact as SavedModelArtifact

from .artifact_backtest_support import build_equity_curve_context
from .models import Artifact
from .service_runtime import (
    ARTIFACT_STORAGE_FORMAT_JSON,
    infer_artifact_storage_format,
    read_frame_rows,
)
from .view_support import _safe_float, _to_int


JSON_PREVIEW_MAX_CHARS = 12000
STRATEGY_FEATURE_PREVIEW_LIMIT = 20
BACKTEST_DAILY_ROW_PREVIEW_LIMIT = 5
MAX_SYMBOL_SUMMARY_ROWS = 50
BACKTEST_PREVIEW_ROWS_LIMIT = 5000
SUMMARY_DECIMALS = 8
RATE_DECIMALS = 4
PERCENT_MULTIPLIER = 100.0
PREDICTION_ARTIFACT_TYPES = {
    "PREDICTIONS",
    "CLASSIFIER_PREDICTIONS",
    "REGRESSOR_PREDICTIONS",
    "AUTOENCODER_SCORES",
    "MTL_PREDICTIONS",
}


def _normalized_date(value: Any) -> str:
    return str(value or "")[:10]


def _json_dict_rows(blob: Any) -> list[dict[str, Any]]:
    if isinstance(blob, list):
        rows = blob
    elif isinstance(blob, dict):
        rows = blob.get("rows")
    else:
        rows = []
    if not isinstance(rows, list):
        return []
    return [dict(value) for value in rows if isinstance(value, dict)]


def _read_rows_from_storage(
    artifact: Artifact,
    *,
    path: Path,
    limit: int,
    include_all_rows: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    storage_format = infer_artifact_storage_format(artifact, default=path.suffix.lower().lstrip("."))
    if storage_format == ARTIFACT_STORAGE_FORMAT_JSON:
        rows = _json_dict_rows(json.loads(path.read_text(encoding="utf-8")))
        return rows[:limit], rows if include_all_rows else []
    preview_rows = read_frame_rows(artifact, limit=limit)
    all_rows = read_frame_rows(artifact) if include_all_rows else []
    return preview_rows, all_rows


def _persist_label_statistics(
    artifact: Artifact,
    *,
    content_payload: dict[str, Any],
    rows_for_stats: list[dict[str, Any]],
) -> None:
    if not rows_for_stats:
        return
    try:
        from .services import _build_label_statistics

        content_payload["statistics"] = _build_label_statistics(rows_for_stats)
        artifact.content = content_payload
        artifact.save(update_fields=["content"])
    except (AttributeError, DatabaseError, ImportError, TypeError, ValueError):
        return


def _load_artifact_preview_rows(artifact: Artifact, limit: int = 100) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    content_payload = dict(artifact.content or {})
    need_stats_fallback = (not dict(content_payload.get("statistics") or {})) and str(artifact.artifact_type) == "LABELS"
    all_rows_for_stats: list[dict[str, Any]] = []
    uri = str(artifact.uri or "").strip()
    if uri:
        path = Path(uri)
        if path.exists() and path.is_file():
            try:
                rows, all_rows_for_stats = _read_rows_from_storage(
                    artifact,
                    path=path,
                    limit=limit,
                    include_all_rows=need_stats_fallback,
                )
            except (OSError, TypeError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
                rows = []
                all_rows_for_stats = []
    if need_stats_fallback:
        _persist_label_statistics(
            artifact,
            content_payload=content_payload,
            rows_for_stats=all_rows_for_stats,
        )
    return rows, content_payload


def _metric_row_value(row: dict[str, Any], key: str) -> float | None:
    raw = row.get(key)
    if raw in (None, "") and key == "prediction_score":
        raw = row.get("signal_score")
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _artifact_metric_value(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [value for row in rows if (value := _metric_row_value(row, key)) is not None]
    if not values:
        return None
    return sum(values) / float(len(values))


def _artifact_symbol_summary(rows: list[dict[str, Any]], artifact: Artifact) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        grouped.setdefault(symbol, []).append(row)
    research_query = _research_query_for_symbol(artifact)
    summary_rows: list[dict[str, Any]] = []
    for symbol, symbol_rows in sorted(grouped.items()):
        dates = [_normalized_date(row.get("date")) for row in symbol_rows if _normalized_date(row.get("date"))]
        summary_rows.append(
            {
                "symbol": symbol,
                "rows": len(symbol_rows),
                "min_date": min(dates) if dates else "",
                "max_date": max(dates) if dates else "",
                "avg_prediction_score": _artifact_metric_value(symbol_rows, "prediction_score"),
                "avg_strategy_score": _artifact_metric_value(symbol_rows, "strategy_score"),
                "avg_realized_return": _artifact_metric_value(symbol_rows, "realized_return"),
                "research_query": research_query,
            }
        )
    return summary_rows[:MAX_SYMBOL_SUMMARY_ROWS]


def _truncate_json_preview(payload: Any, *, max_chars: int = JSON_PREVIEW_MAX_CHARS) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n... [truncated {len(text) - max_chars} chars]"


def _summarize_strategy_content_payload(content_payload: dict[str, Any]) -> dict[str, Any]:
    summary = dict(content_payload)
    feature_cols = list(summary.get("feature_cols") or [])
    summary["feature_col_count"] = len(feature_cols)
    summary["feature_cols_preview"] = feature_cols[:STRATEGY_FEATURE_PREVIEW_LIMIT]
    if len(feature_cols) > STRATEGY_FEATURE_PREVIEW_LIMIT:
        summary["feature_cols_hidden_count"] = len(feature_cols) - STRATEGY_FEATURE_PREVIEW_LIMIT
    summary.pop("feature_cols", None)
    return summary


def _summarize_backtest_content_payload(content_payload: dict[str, Any]) -> dict[str, Any]:
    summary = dict(content_payload)
    daily_rows = list(summary.get("daily_rows") or [])
    summary["daily_row_count"] = len(daily_rows)
    summary["daily_rows_preview"] = daily_rows[:BACKTEST_DAILY_ROW_PREVIEW_LIMIT]
    summary.pop("daily_rows", None)
    return summary


def _round_or_none(value: float | None, digits: int) -> float | None:
    return round(value, digits) if value is not None else None


def _percent_or_none(value: float | None, digits: int) -> float | None:
    return round(value * PERCENT_MULTIPLIER, digits) if value is not None else None


def _strategy_source_params(metadata: dict[str, Any]) -> list[tuple[str, int]]:
    params: list[tuple[str, int]] = []
    feature_id = int(metadata.get("source_features_artifact_id") or 0)
    label_id = int(metadata.get("source_label_artifact_id") or 0)
    prediction_ids = [int(value) for value in list(metadata.get("source_prediction_artifact_ids") or []) if int(value or 0) > 0]
    if feature_id > 0:
        params.append(("feature_artifact_id", feature_id))
    if label_id > 0:
        params.append(("label_artifact_id", label_id))
    params.extend(("prediction_artifact_id", prediction_id) for prediction_id in prediction_ids)
    return params


def _strategy_symbol_detail_link(artifact: Artifact, symbol: str) -> str:
    metadata = dict(artifact.metadata or {})
    params = [("strategy_artifact_id", int(artifact.id)), *_strategy_source_params(metadata)]
    return f"{reverse('pipeline-symbol-research', args=[symbol])}?{urlencode(params, doseq=True)}"


def _backtest_symbol_detail_link(backtest_artifact: Artifact, strategy_artifact: Artifact | None, symbol: str) -> str:
    params: list[tuple[str, Any]] = [("backtest_artifact_id", int(backtest_artifact.id))]
    if strategy_artifact is not None:
        strategy_link = _strategy_symbol_detail_link(strategy_artifact, symbol)
        separator = "&" if "?" in strategy_link else "?"
        return f"{strategy_link}{separator}backtest_artifact_id={int(backtest_artifact.id)}"
    return f"{reverse('pipeline-symbol-research', args=[symbol])}?{urlencode(params)}"


def _build_equity_curve_context(backtest_artifact: Artifact) -> dict[str, Any]:
    return build_equity_curve_context(
        backtest_artifact,
        load_artifact_preview_rows=_load_artifact_preview_rows,
        normalized_date=_normalized_date,
        safe_float=_safe_float,
        to_int=_to_int,
    )


def _prediction_source_params(artifact: Artifact, metadata: dict[str, Any]) -> dict[str, int]:
    params = {"prediction_artifact_id": int(artifact.id)}
    source_feature_id = int(metadata.get("source_features_artifact_id") or 0)
    source_label_id = int(metadata.get("source_label_artifact_id") or 0)
    if source_feature_id > 0:
        params["feature_artifact_id"] = source_feature_id
    if source_label_id > 0:
        params["label_artifact_id"] = source_label_id
    return params


def _strategy_query_params(metadata: dict[str, Any]) -> dict[str, int]:
    strategy_params = _strategy_source_params(metadata)
    params = {
        key: value
        for key, value in strategy_params
        if key != "prediction_artifact_id"
    }
    prediction_ids = [value for key, value in strategy_params if key == "prediction_artifact_id"]
    if prediction_ids:
        params["prediction_artifact_id"] = prediction_ids[0]
    return params


def _research_query_params(artifact: Artifact, metadata: dict[str, Any]) -> dict[str, int] | None:
    artifact_type = str(artifact.artifact_type or "").upper()
    if artifact_type == "LABELS":
        return {"label_artifact_id": int(artifact.id)}
    if artifact_type == "FEATURES":
        return {"feature_artifact_id": int(artifact.id)}
    if artifact_type in PREDICTION_ARTIFACT_TYPES:
        return _prediction_source_params(artifact, metadata)
    if artifact_type == "STRATEGY_DATASET":
        return _strategy_query_params(metadata)
    return None


def _research_query_for_symbol(artifact: Artifact) -> str:
    metadata = dict(artifact.metadata or {})
    if str(artifact.artifact_type or "").upper() == "BACKTEST_RESULT":
        source_strategy_id = int(metadata.get("source_strategy_dataset_artifact_id") or 0)
        if source_strategy_id > 0:
            strategy_artifact = Artifact.objects.filter(pk=source_strategy_id).first()
            if strategy_artifact is not None:
                return _research_query_for_symbol(strategy_artifact)
    params = _research_query_params(artifact, metadata) or {}
    encoded = urlencode(params)
    return f"?{encoded}" if encoded else ""


def _saved_model_for_pipeline_artifact(artifact: Artifact) -> SavedModelArtifact | None:
    content_payload = dict(artifact.content or {})
    metadata = dict(artifact.metadata or {})
    saved_model_id = _to_int(content_payload.get("model_artifact_id") or metadata.get("saved_model_artifact_id"))
    if saved_model_id <= 0:
        return None
    return SavedModelArtifact.objects.filter(pk=saved_model_id).first()


__all__ = [
    "_artifact_symbol_summary",
    "_backtest_symbol_detail_link",
    "_build_equity_curve_context",
    "_load_artifact_preview_rows",
    "_research_query_for_symbol",
    "_saved_model_for_pipeline_artifact",
    "_strategy_symbol_detail_link",
    "_summarize_backtest_content_payload",
    "_summarize_strategy_content_payload",
    "_truncate_json_preview",
]
