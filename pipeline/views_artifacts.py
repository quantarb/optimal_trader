from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from django.http import Http404, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_GET

from .artifact_support import (
    _artifact_symbol_summary,
    _backtest_symbol_detail_link,
    _build_equity_curve_context,
    _load_artifact_preview_rows,
    _research_query_for_symbol,
    _saved_model_for_pipeline_artifact,
    _strategy_symbol_detail_link,
    _summarize_backtest_content_payload,
    _summarize_strategy_content_payload,
    _truncate_json_preview,
)
from .feature_presentation import format_feature_value, get_feature_definition
from .models import Artifact
from .view_support import (
    _annotate_rows_with_bar_pct,
    _safe_float,
    _to_float,
)


@require_GET
def strategy_detail(request, artifact_id: int):
    artifact = get_object_or_404(
        Artifact.objects.select_related("pipeline_run"),
        pk=int(artifact_id),
        artifact_type="STRATEGY_DATASET",
    )
    preview_rows, content_payload = _load_artifact_preview_rows(artifact, limit=300)
    symbol_summary = _artifact_symbol_summary(preview_rows, artifact)
    for row in symbol_summary:
        row["detail_url"] = _strategy_symbol_detail_link(artifact, str(row["symbol"]))
    metadata = dict(artifact.metadata or {})
    strategy_config = dict(metadata.get("strategy_config") or {})
    selected_preview_rows = [row for row in preview_rows if _safe_float(row.get("selected_on_rebalance")) == 1.0][:20]
    symbol_summary_display = symbol_summary[:12]
    top_symbol_rows = sorted(
        [dict(row) for row in symbol_summary],
        key=lambda row: float(row.get("avg_strategy_score") or 0.0),
        reverse=True,
    )[:8]
    top_symbol_rows = _annotate_rows_with_bar_pct(top_symbol_rows, "avg_strategy_score", "score_bar_pct")
    content_summary = _summarize_strategy_content_payload(content_payload)
    return render(
        request,
        "pipeline/strategy_detail.html",
        {
            "artifact": artifact,
            "content_payload": content_payload,
            "strategy_config": strategy_config,
            "symbol_summary": symbol_summary_display,
            "symbol_summary_count": len(symbol_summary),
            "symbol_summary_hidden_count": max(len(symbol_summary) - len(symbol_summary_display), 0),
            "top_symbol_rows": top_symbol_rows,
            "selected_preview_rows": selected_preview_rows,
            "selected_preview_row_count": len(selected_preview_rows),
            "preview_rows": preview_rows[:12],
            "preview_row_count": len(preview_rows),
            "content_json": _truncate_json_preview(content_summary),
            "metadata_json": _truncate_json_preview(metadata),
            "preview_rows_json": _truncate_json_preview(preview_rows[:8], max_chars=8000),
        },
    )


@require_GET
def backtest_detail(request, artifact_id: int):
    artifact = get_object_or_404(
        Artifact.objects.select_related("pipeline_run"),
        pk=int(artifact_id),
        artifact_type="BACKTEST_RESULT",
    )
    preview_rows, content_payload = _load_artifact_preview_rows(artifact, limit=400)
    metadata = dict(artifact.metadata or {})
    strategy_artifact_id = int(metadata.get("source_strategy_dataset_artifact_id") or 0)
    strategy_artifact = Artifact.objects.filter(pk=strategy_artifact_id, artifact_type="STRATEGY_DATASET").first() if strategy_artifact_id > 0 else None
    symbol_summary = _artifact_symbol_summary(preview_rows, artifact)
    for row in symbol_summary:
        row["detail_url"] = _backtest_symbol_detail_link(artifact, strategy_artifact, str(row["symbol"]))
    equity_context = _build_equity_curve_context(artifact)
    daily_rows_full = list(content_payload.get("daily_rows") or equity_context.get("backtest_daily_rows_full") or [])
    contribution_rows_full = list(equity_context.get("contribution_rows") or [])
    monthly_return_rows = list(equity_context.get("monthly_return_rows") or [])
    contribution_symbol_filter = str(request.GET.get("symbol") or "").strip().upper()
    start_date = str(request.GET.get("start") or "").strip()
    end_date = str(request.GET.get("end") or "").strip()
    contribution_sort = str(request.GET.get("contrib_sort") or "realized_return_desc").strip().lower()

    daily_rows = [
        row
        for row in daily_rows_full
        if (not start_date or str(row.get("date") or "")[:10] >= start_date)
        and (not end_date or str(row.get("date") or "")[:10] <= end_date)
    ]
    contribution_rows = contribution_rows_full
    if contribution_symbol_filter:
        contribution_rows = [row for row in contribution_rows if str(row.get("symbol") or "").strip().upper() == contribution_symbol_filter]
    if contribution_sort == "realized_return_asc":
        contribution_rows = sorted(contribution_rows, key=lambda row: float(row.get("realized_return") or 0.0))
    elif contribution_sort == "rows_desc":
        contribution_rows = sorted(contribution_rows, key=lambda row: int(row.get("rows") or 0), reverse=True)
    else:
        contribution_rows = sorted(contribution_rows, key=lambda row: float(row.get("realized_return") or 0.0), reverse=True)
    recent_daily_rows = (daily_rows or [])[-20:]
    top_contribution_rows = _annotate_rows_with_bar_pct(contribution_rows[:8], "realized_return", "return_bar_pct")
    bottom_contribution_rows = _annotate_rows_with_bar_pct(list(reversed(contribution_rows[-8:])) if contribution_rows else [], "realized_return", "return_bar_pct")
    monthly_return_rows_display = _annotate_rows_with_bar_pct(monthly_return_rows[:12], "return", "return_bar_pct")
    recent_daily_rows = _annotate_rows_with_bar_pct(recent_daily_rows, "net_daily_return", "return_bar_pct")
    cumulative_return = _safe_float(content_payload.get("cumulative_return"))
    max_drawdown = _safe_float(content_payload.get("max_drawdown"))
    benchmark_cumulative_return = _to_float((equity_context.get("report_summary") or {}).get("benchmark_cumulative_return"))
    content_summary = _summarize_backtest_content_payload(content_payload)
    return render(
        request,
        "pipeline/backtest_detail.html",
        {
            **equity_context,
            "artifact": artifact,
            "strategy_artifact": strategy_artifact,
            "content_payload": content_payload,
            "symbol_summary": symbol_summary,
            "preview_rows": preview_rows[:80],
            "daily_rows": recent_daily_rows,
            "daily_row_count": len(daily_rows or []),
            "daily_hidden_count": max(len(daily_rows or []) - len(recent_daily_rows), 0),
            "contribution_rows": top_contribution_rows,
            "bottom_contribution_rows": bottom_contribution_rows,
            "contribution_row_count": len(contribution_rows),
            "contribution_hidden_count": max(len(contribution_rows) - len(top_contribution_rows), 0),
            "monthly_return_rows": monthly_return_rows_display,
            "monthly_return_row_count": len(monthly_return_rows),
            "monthly_return_hidden_count": max(len(monthly_return_rows) - len(monthly_return_rows_display), 0),
            "available_symbols": sorted({str(row.get('symbol') or '').strip().upper() for row in contribution_rows_full if str(row.get('symbol') or '').strip()}),
            "contribution_symbol_filter": contribution_symbol_filter,
            "start_date": start_date,
            "end_date": end_date,
            "contribution_sort": contribution_sort,
            "cumulative_return_pct": round((cumulative_return or 0.0) * 100.0, 4) if cumulative_return is not None else None,
            "max_drawdown_pct": round((max_drawdown or 0.0) * 100.0, 4) if max_drawdown is not None else None,
            "excess_cumulative_return": ((cumulative_return or 0.0) - (benchmark_cumulative_return or 0.0))
            if cumulative_return is not None or benchmark_cumulative_return is not None
            else None,
            "content_json": _truncate_json_preview(content_summary),
            "metadata_json": _truncate_json_preview(metadata),
        },
    )


@require_GET
def artifact_preview(request, artifact_id: int):
    artifact = Artifact.objects.filter(pk=int(artifact_id)).select_related("pipeline_run").first()
    if artifact is None:
        raise Http404("Artifact not found.")

    try:
        limit = int(request.GET.get("limit") or 100)
    except Exception:
        limit = 100
    limit = max(1, min(1000, limit))

    rows, content_payload = _load_artifact_preview_rows(artifact, limit=limit)

    return JsonResponse(
        {
            "artifact": {
                "artifact_id": int(artifact.id),
                "artifact_type": str(artifact.artifact_type),
                "pipeline_run_id": int(artifact.pipeline_run_id),
                "pipeline_run_name": str((artifact.pipeline_run.name if artifact.pipeline_run else "") or ""),
                "pipeline_run_status": str((artifact.pipeline_run.status if artifact.pipeline_run else "") or ""),
                "uri": str(artifact.uri or ""),
                "content": content_payload,
                "metadata": dict(artifact.metadata or {}),
                "created_at": artifact.created_at.isoformat() if artifact.created_at else None,
            },
            "rows": rows,
        }
    )


@require_GET
def artifact_detail(request, artifact_id: int):
    artifact = Artifact.objects.filter(pk=int(artifact_id)).select_related("pipeline_run", "producer_job").first()
    if artifact is None:
        raise Http404("Artifact not found.")

    try:
        limit = int(request.GET.get("limit") or 100)
    except Exception:
        limit = 100
    limit = max(1, min(1000, limit))

    preview_rows, content_payload = _load_artifact_preview_rows(artifact, limit=limit)
    columns = list(preview_rows[0].keys()) if preview_rows else []
    preview_values = [[row.get(col, "") for col in columns] for row in preview_rows]
    preview_display_columns = [get_feature_definition(col).display_name for col in columns]
    preview_rendered_values = [[format_feature_value(col, row.get(col, "")) for col in columns] for row in preview_rows]
    symbol_summary = _artifact_symbol_summary(preview_rows, artifact)
    saved_model = _saved_model_for_pipeline_artifact(artifact)
    saved_model_metrics = dict(saved_model.metrics or {}) if saved_model is not None else {}
    saved_model_params = dict(saved_model.params or {}) if saved_model is not None else {}
    saved_model_metadata = dict(saved_model.metadata or {}) if saved_model is not None else {}

    return render(
        request,
        "pipeline/artifact_detail.html",
        {
            "artifact": artifact,
            "content_payload": content_payload,
            "metadata_payload": dict(artifact.metadata or {}),
            "preview_rows": preview_rows,
            "preview_columns": columns,
            "preview_display_columns": preview_display_columns,
            "preview_values": preview_values,
            "preview_rendered_values": preview_rendered_values,
            "preview_limit": limit,
            "symbol_summary": symbol_summary,
            "saved_model": saved_model,
            "saved_model_metrics_json": json.dumps(saved_model_metrics, indent=2, sort_keys=True),
            "saved_model_params_json": json.dumps(saved_model_params, indent=2, sort_keys=True),
            "saved_model_metadata_json": json.dumps(saved_model_metadata, indent=2, sort_keys=True),
            "content_json": json.dumps(content_payload, indent=2, sort_keys=True),
            "metadata_json": json.dumps(dict(artifact.metadata or {}), indent=2, sort_keys=True),
            "research_query": _research_query_for_symbol(artifact),
            "is_model_artifact": str(artifact.artifact_type) in {"MODEL", "CLASSIFIER_MODEL", "REGRESSOR_MODEL", "AUTOENCODER_MODEL"},
            "is_strategy_artifact": str(artifact.artifact_type) == "STRATEGY_DATASET",
            "is_backtest_artifact": str(artifact.artifact_type) == "BACKTEST_RESULT",
        },
    )


@require_GET
def artifact_symbol_breakdown(request, artifact_id: int):
    artifact = Artifact.objects.filter(pk=int(artifact_id)).first()
    if artifact is None:
        raise Http404("Artifact not found.")
    symbol = str(request.GET.get("symbol") or "").strip().upper()
    if not symbol:
        return JsonResponse({"error": "Missing required query parameter: symbol"}, status=400)

    uri = str(artifact.uri or "").strip()
    if not uri:
        return JsonResponse({"error": "Artifact has no URI."}, status=400)
    path = Path(uri)
    if not path.exists() or not path.is_file():
        return JsonResponse({"error": "Artifact file not found."}, status=404)
    if path.suffix.lower() != ".csv":
        return JsonResponse({"error": "Symbol breakdown currently supports CSV artifacts only."}, status=400)

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if str(row.get("symbol") or "").strip().upper() != symbol:
                continue
            rows.append(dict(row))

    if not rows:
        return JsonResponse(
            {
                "artifact_id": int(artifact.id),
                "symbol": symbol,
                "summary": {"trades": 0},
                "grouped": [],
                "rows": [],
            }
        )

    def _f(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default

    returns = [_f(r.get("trade_return"), 0.0) for r in rows]
    holds = [_f(r.get("hold_days"), 0.0) for r in rows]
    wins = sum(1 for v in returns if v > 0)
    losses = sum(1 for v in returns if v < 0)
    flats = sum(1 for v in returns if v == 0)

    summary = {
        "trades": int(len(rows)),
        "wins": int(wins),
        "losses": int(losses),
        "flats": int(flats),
        "win_rate_pct": (wins / float(len(rows)) * 100.0) if rows else 0.0,
        "avg_return_pct": (sum(returns) / float(len(returns)) * 100.0) if returns else 0.0,
        "avg_hold_days": (sum(holds) / float(len(holds))) if holds else 0.0,
    }

    grouped_map: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        side = str(row.get("side") or ("long" if str(row.get("label") or "") == "1" else "short")).strip().lower()
        freq = str(row.get("freq") or "").strip().upper() or "NA"
        try:
            k = int(row.get("k") or 0)
        except Exception:
            k = 0
        key = (side, freq, k)
        bucket = grouped_map.setdefault(key, {"returns": [], "holds": []})
        bucket["returns"].append(_f(row.get("trade_return"), 0.0))
        bucket["holds"].append(_f(row.get("hold_days"), 0.0))

    grouped: list[dict[str, Any]] = []
    for (side, freq, k), bucket in sorted(grouped_map.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        vals = list(bucket["returns"])
        hvals = list(bucket["holds"])
        count = len(vals)
        mean_v = (sum(vals) / float(count)) if count else 0.0
        if count > 1:
            var = sum((v - mean_v) ** 2 for v in vals) / float(count - 1)
            std_v = var ** 0.5
        else:
            std_v = None
        hold_mean = (sum(hvals) / float(len(hvals))) if hvals else 0.0
        grouped.append(
            {
                "side": side,
                "freq": freq,
                "k": int(k),
                "trades": int(count),
                "trade_return_mean_pct": mean_v * 100.0,
                "trade_return_std_pct": (std_v * 100.0) if std_v is not None else None,
                "trade_duration_mean": hold_mean,
            }
        )

    preview_rows = sorted(rows, key=lambda row: str(row.get("date") or ""), reverse=True)[:100]
    return JsonResponse(
        {
            "artifact_id": int(artifact.id),
            "symbol": symbol,
            "summary": summary,
            "grouped": grouped,
            "rows": preview_rows,
        }
    )


__all__ = [
    "artifact_detail",
    "artifact_preview",
    "artifact_symbol_breakdown",
    "backtest_detail",
    "strategy_detail",
]
