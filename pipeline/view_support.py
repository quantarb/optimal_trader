from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from django.urls import reverse

from analysis.research import recent_artifact_choices, resolve_artifact
from analysis.situation_similarity import load_market_situation_cluster_artifact, resolve_market_situation_artifact

from .models import Artifact


def _cohort_summary_detail(payload: dict[str, Any]) -> dict[str, Any]:
    base_artifacts = dict(payload.get("base_artifacts") or {})
    label_artifact_id = _to_int(base_artifacts.get("labels"))
    feature_artifact_id = _to_int(base_artifacts.get("features"))
    primary_rows = list(payload.get("leaderboard_rows") or [])
    rejected_rows = list(payload.get("rejected_rows") or [])
    referenced_ids: set[int] = set()
    if label_artifact_id > 0:
        referenced_ids.add(label_artifact_id)
    if feature_artifact_id > 0:
        referenced_ids.add(feature_artifact_id)
    for row in primary_rows + rejected_rows:
        for key in ("prediction_artifact_id", "strategy_artifact_id", "backtest_artifact_id"):
            value = _to_int(row.get(key))
            if value > 0:
                referenced_ids.add(value)
    existing_ids = set(Artifact.objects.filter(id__in=list(referenced_ids)).values_list("id", flat=True))

    def decorate_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        detail_rows: list[dict[str, Any]] = []
        for row in source_rows:
            item = dict(row)
            prediction_artifact_id = _to_int(item.get("prediction_artifact_id"))
            strategy_artifact_id = _to_int(item.get("strategy_artifact_id"))
            backtest_artifact_id = _to_int(item.get("backtest_artifact_id"))
            query_items: list[tuple[str, Any]] = []
            if label_artifact_id in existing_ids:
                query_items.append(("label_artifact_id", label_artifact_id))
            if feature_artifact_id in existing_ids:
                query_items.append(("feature_artifact_id", feature_artifact_id))
            if prediction_artifact_id in existing_ids:
                query_items.append(("prediction_artifact_id", prediction_artifact_id))
            if strategy_artifact_id in existing_ids:
                query_items.append(("strategy_artifact_id", strategy_artifact_id))
            if backtest_artifact_id in existing_ids:
                query_items.append(("backtest_artifact_id", backtest_artifact_id))
            item["strategy_url"] = reverse("pipeline-strategy-detail", args=[strategy_artifact_id]) if strategy_artifact_id in existing_ids else ""
            item["backtest_url"] = reverse("pipeline-backtest-detail", args=[backtest_artifact_id]) if backtest_artifact_id in existing_ids else ""
            item["research_url"] = (
                f"{reverse('pipeline-symbol-research', args=['AAPL'])}?{urlencode(query_items, doseq=True)}"
                if query_items
                else ""
            )
            missing_refs: list[str] = []
            if prediction_artifact_id > 0 and prediction_artifact_id not in existing_ids:
                missing_refs.append(f"prediction #{prediction_artifact_id}")
            if strategy_artifact_id > 0 and strategy_artifact_id not in existing_ids:
                missing_refs.append(f"strategy #{strategy_artifact_id}")
            if backtest_artifact_id > 0 and backtest_artifact_id not in existing_ids:
                missing_refs.append(f"backtest #{backtest_artifact_id}")
            item["stale_references"] = missing_refs
            fit_seconds = _safe_float(item.get("fit_seconds") if "fit_seconds" in item else item.get("avg_fit_seconds")) or 0.0
            score_seconds = _safe_float(item.get("score_seconds")) or 0.0
            backtest_seconds = _safe_float(item.get("backtest_seconds") if "backtest_seconds" in item else item.get("avg_backtest_seconds")) or 0.0
            max_drawdown = _safe_float(item.get("max_drawdown") if "max_drawdown" in item else item.get("walk_forward_max_drawdown"))
            cumulative_return = _safe_float(item.get("cumulative_return") if "cumulative_return" in item else item.get("walk_forward_cumulative_return"))
            item["total_runtime_seconds"] = round(fit_seconds + score_seconds + backtest_seconds, 6)
            item["return_to_drawdown"] = (
                round((cumulative_return or 0.0) / abs(max_drawdown), 6)
                if max_drawdown not in (None, 0.0)
                else None
            )
            detail_rows.append(item)
        return detail_rows

    detail_rows = decorate_rows(primary_rows)
    decorated_rejected_rows = decorate_rows(rejected_rows)
    detail_rows.sort(
        key=lambda row: float(row.get("walk_forward_excess_cumulative_return") or row.get("final_equity") or 0.0),
        reverse=True,
    )
    leaderboard = {
        "best_equity": max(
            detail_rows,
            key=lambda row: float(row.get("walk_forward_final_equity") or row.get("final_equity") or 0.0),
            default=None,
        ),
        "best_excess": max(
            detail_rows,
            key=lambda row: float(row.get("walk_forward_excess_cumulative_return") or row.get("excess_cumulative_return") or -999.0),
            default=None,
        ),
        "best_drawdown": max(
            (
                row
                for row in detail_rows
                if _safe_float(row.get("max_drawdown") if "max_drawdown" in row else row.get("walk_forward_max_drawdown")) is not None
            ),
            key=lambda row: float(row.get("max_drawdown") if "max_drawdown" in row else row.get("walk_forward_max_drawdown") or -999.0),
            default=None,
        ),
        "fastest_fit": min(
            detail_rows,
            key=lambda row: float(row.get("fit_seconds") if "fit_seconds" in row else row.get("avg_fit_seconds") or 10**9),
            default=None,
        ),
        "best_stability": max(
            detail_rows,
            key=lambda row: float(row.get("valid_fold_rate") or 0.0) - float(row.get("fold_excess_cumulative_return_std") or 0.0),
            default=None,
        ),
        "best_return_to_drawdown": max(
            (row for row in detail_rows if row.get("return_to_drawdown") is not None),
            key=lambda row: float(row.get("return_to_drawdown") or -999.0),
            default=None,
        ),
    }
    return {
        "rows": detail_rows,
        "rejected_rows": decorated_rejected_rows,
        "label_artifact_id": label_artifact_id if label_artifact_id in existing_ids else 0,
        "feature_artifact_id": feature_artifact_id if feature_artifact_id in existing_ids else 0,
        "leaderboard": leaderboard,
        "report_summary": dict(payload.get("report_summary") or {}),
        "research_profile": dict(payload.get("research_profile") or {}),
    }


def _clean_ids(raw: Any) -> list[int]:
    out: list[int] = []
    for value in list(raw or []):
        try:
            out.append(int(value))
        except Exception:
            continue
    return out


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _annotate_rows_with_bar_pct(rows: list[dict[str, Any]], key: str, output_key: str, *, absolute: bool = True) -> list[dict[str, Any]]:
    values: list[float] = []
    for row in rows:
        parsed = _to_float(row.get(key))
        if parsed is None:
            continue
        values.append(abs(parsed) if absolute else parsed)
    max_value = max(values) if values else 0.0
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        parsed = _to_float(item.get(key))
        if parsed is None or max_value <= 0:
            item[output_key] = 0
        else:
            baseline = abs(parsed) if absolute else parsed
            item[output_key] = int(round(max(0.0, min(100.0, (baseline / max_value) * 100.0))))
        out.append(item)
    return out


def _json_safe_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_data(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe_data(value.item())
        except Exception:
            pass
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return value


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _artifact_choice_label(artifact: Artifact) -> str:
    run_name = str((artifact.pipeline_run.name if artifact.pipeline_run else "") or "").strip() or "unnamed"
    return f"#{int(artifact.id)} | run #{int(artifact.pipeline_run_id)} | {artifact.artifact_type} | {run_name}"


UI_PREDICTION_ARTIFACT_TYPES = ["CLASSIFIER_PREDICTIONS", "REGRESSOR_PREDICTIONS", "MTL_PREDICTIONS"]
UI_STATE_PANEL_ARTIFACT_TYPES = UI_PREDICTION_ARTIFACT_TYPES + ["MARKET_SITUATION_CLUSTER"]
UI_MODEL_ARTIFACT_TYPES = ["CLASSIFIER_MODEL", "REGRESSOR_MODEL"]
INSIGHT_PREDICTION_ARTIFACT_TYPES = ["PREDICTIONS", "CLASSIFIER_PREDICTIONS", "REGRESSOR_PREDICTIONS", "AUTOENCODER_SCORES", "MTL_PREDICTIONS"]


def _artifact_choices(artifact_types: Any, limit: int = 50) -> list[tuple[int, str]]:
    rows = recent_artifact_choices(artifact_types, limit=limit)
    return [(int(row.id), _artifact_choice_label(row)) for row in rows]


def _load_market_situation_artifacts(limit: int = 30) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    artifacts = (
        Artifact.objects.filter(artifact_type="MARKET_SITUATION_CLUSTER")
        .select_related("pipeline_run")
        .order_by("-created_at", "-id")[: max(1, limit)]
    )
    for artifact in artifacts:
        try:
            bundle = load_market_situation_cluster_artifact(artifact)
        except Exception:
            continue
        rows.append(
            {
                "artifact": artifact,
                "summary": dict(bundle.summary.get("summary") or artifact.content or {}),
                "payload": bundle.summary,
                "clusters": list(bundle.summary.get("clusters") or []),
            }
        )
    return rows


def _parse_symbols_csv(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for chunk in str(raw or "").replace("\n", ",").split(","):
        symbol = chunk.strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            out.append(symbol)
    return out


def _selected_prediction_ids_from_request(request) -> list[int]:
    ids = _clean_ids(request.GET.getlist("prediction_artifact_id"))
    if ids:
        return ids
    raw = str(request.GET.get("prediction_artifact_ids") or "").strip()
    if not raw:
        return []
    return _clean_ids(raw.split(","))


def _insight_selection_context(request) -> dict[str, Any]:
    strategy_artifact_id = _to_int(request.GET.get("strategy_artifact_id"))
    feature_artifact_id = _to_int(request.GET.get("feature_artifact_id"))
    label_artifact_id = _to_int(request.GET.get("label_artifact_id"))
    market_situation_artifact_id = _to_int(request.GET.get("market_situation_artifact_id"))
    search_method = str(request.GET.get("search_method") or "hybrid").strip().lower() or "hybrid"
    if search_method not in {"numeric", "text_embedding", "hybrid"}:
        search_method = "hybrid"
    prediction_artifact_ids = _selected_prediction_ids_from_request(request)
    return {
        "strategy_artifact_id": strategy_artifact_id,
        "feature_artifact_id": feature_artifact_id,
        "label_artifact_id": label_artifact_id,
        "market_situation_artifact_id": market_situation_artifact_id,
        "search_method": search_method,
        "search_method_choices": [
            {"value": "hybrid", "label": "Hybrid Search"},
            {"value": "numeric", "label": "Numeric Search"},
            {"value": "text_embedding", "label": "Embedding Search"},
        ],
        "prediction_artifact_ids": prediction_artifact_ids,
        "strategy_artifact_choices": recent_artifact_choices("STRATEGY_DATASET", limit=20),
        "feature_artifact_choices": recent_artifact_choices("FEATURES", limit=20),
        "label_artifact_choices": recent_artifact_choices("LABELS", limit=20),
        "market_situation_artifact_choices": recent_artifact_choices("MARKET_SITUATION_CLUSTER", limit=20),
        "prediction_artifact_choices": recent_artifact_choices(INSIGHT_PREDICTION_ARTIFACT_TYPES, limit=30),
        "selected_strategy_artifact": resolve_artifact(artifact_type="STRATEGY_DATASET", artifact_id=strategy_artifact_id) if strategy_artifact_id > 0 else None,
        "selected_feature_artifact": resolve_artifact(artifact_type="FEATURES", artifact_id=feature_artifact_id) if feature_artifact_id > 0 else None,
        "selected_label_artifact": resolve_artifact(artifact_type="LABELS", artifact_id=label_artifact_id) if label_artifact_id > 0 else None,
        "selected_market_situation_artifact": resolve_market_situation_artifact(artifact_id=market_situation_artifact_id) if market_situation_artifact_id > 0 else resolve_market_situation_artifact(),
    }


__all__ = [
    "INSIGHT_PREDICTION_ARTIFACT_TYPES",
    "UI_MODEL_ARTIFACT_TYPES",
    "UI_PREDICTION_ARTIFACT_TYPES",
    "UI_STATE_PANEL_ARTIFACT_TYPES",
    "_annotate_rows_with_bar_pct",
    "_artifact_choice_label",
    "_artifact_choices",
    "_clean_ids",
    "_cohort_summary_detail",
    "_insight_selection_context",
    "_json_safe_data",
    "_load_market_situation_artifacts",
    "_parse_symbols_csv",
    "_safe_float",
    "_to_float",
    "_to_int",
]
