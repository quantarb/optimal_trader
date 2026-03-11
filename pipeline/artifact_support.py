from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from django.urls import reverse

from ml.models import ModelArtifact as SavedModelArtifact

from .contracts import build_backtest_daily_rows_from_trade_rows, build_equity_curve_from_daily_rows
from .models import Artifact
from .service_runtime import (
    ARTIFACT_STORAGE_FORMAT_JSON,
    infer_artifact_storage_format,
    read_frame_rows,
)
from .view_support import _safe_float, _to_int


def _load_artifact_preview_rows(artifact: Artifact, limit: int = 100) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    content_payload = dict(artifact.content or {})
    stats = dict(content_payload.get("statistics") or {})
    need_stats_fallback = (not stats) and str(artifact.artifact_type) == "LABELS"
    all_rows_for_stats: list[dict[str, Any]] = [] if need_stats_fallback else []
    uri = str(artifact.uri or "").strip()
    if uri:
        path = Path(uri)
        if path.exists() and path.is_file():
            storage_format = infer_artifact_storage_format(artifact, default=path.suffix.lower().lstrip("."))
            try:
                if storage_format == ARTIFACT_STORAGE_FORMAT_JSON:
                    blob = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(blob, list):
                        if need_stats_fallback:
                            all_rows_for_stats = [dict(v) for v in blob if isinstance(v, dict)]
                            rows = all_rows_for_stats[:limit]
                        else:
                            rows = [dict(v) for v in blob[:limit] if isinstance(v, dict)]
                    elif isinstance(blob, dict):
                        inner_rows = blob.get("rows")
                        if isinstance(inner_rows, list):
                            if need_stats_fallback:
                                all_rows_for_stats = [dict(v) for v in inner_rows if isinstance(v, dict)]
                                rows = all_rows_for_stats[:limit]
                            else:
                                rows = [dict(v) for v in inner_rows[:limit] if isinstance(v, dict)]
                else:
                    rows = read_frame_rows(artifact, limit=limit)
                    if need_stats_fallback:
                        all_rows_for_stats = read_frame_rows(artifact)
            except Exception:
                rows = []
                all_rows_for_stats = []

    if need_stats_fallback and all_rows_for_stats:
        try:
            from .services import _build_label_statistics

            stats = _build_label_statistics(all_rows_for_stats)
            content_payload["statistics"] = stats
            artifact.content = content_payload
            artifact.save(update_fields=["content"])
        except Exception:
            pass
    return rows, content_payload


def _artifact_metric_value(rows: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        try:
            raw = row.get(key)
            if raw in (None, "") and key == "prediction_score":
                raw = row.get("signal_score")
            if raw in (None, ""):
                continue
            values.append(float(raw))
        except Exception:
            continue
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
    summary_rows: list[dict[str, Any]] = []
    for symbol, symbol_rows in sorted(grouped.items()):
        dates = [str(row.get("date") or "")[:10] for row in symbol_rows if str(row.get("date") or "").strip()]
        summary_rows.append(
            {
                "symbol": symbol,
                "rows": len(symbol_rows),
                "min_date": min(dates) if dates else "",
                "max_date": max(dates) if dates else "",
                "avg_prediction_score": _artifact_metric_value(symbol_rows, "prediction_score"),
                "avg_strategy_score": _artifact_metric_value(symbol_rows, "strategy_score"),
                "avg_realized_return": _artifact_metric_value(symbol_rows, "realized_return"),
                "research_query": _research_query_for_symbol(artifact),
            }
        )
    return summary_rows[:50]


def _truncate_json_preview(payload: Any, *, max_chars: int = 12000) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n... [truncated {len(text) - max_chars} chars]"


def _summarize_strategy_content_payload(content_payload: dict[str, Any]) -> dict[str, Any]:
    summary = dict(content_payload)
    feature_cols = list(summary.get("feature_cols") or [])
    summary["feature_col_count"] = len(feature_cols)
    summary["feature_cols_preview"] = feature_cols[:20]
    if len(feature_cols) > 20:
        summary["feature_cols_hidden_count"] = len(feature_cols) - 20
    if "feature_cols" in summary:
        del summary["feature_cols"]
    return summary


def _summarize_backtest_content_payload(content_payload: dict[str, Any]) -> dict[str, Any]:
    summary = dict(content_payload)
    daily_rows = list(summary.get("daily_rows") or [])
    summary["daily_row_count"] = len(daily_rows)
    summary["daily_rows_preview"] = daily_rows[:5]
    if "daily_rows" in summary:
        del summary["daily_rows"]
    return summary


def _strategy_symbol_detail_link(artifact: Artifact, symbol: str) -> str:
    metadata = dict(artifact.metadata or {})
    params: list[tuple[str, Any]] = [("strategy_artifact_id", int(artifact.id))]
    feature_id = int(metadata.get("source_features_artifact_id") or 0)
    label_id = int(metadata.get("source_label_artifact_id") or 0)
    prediction_ids = [int(v) for v in list(metadata.get("source_prediction_artifact_ids") or []) if int(v or 0) > 0]
    if feature_id > 0:
        params.append(("feature_artifact_id", feature_id))
    if label_id > 0:
        params.append(("label_artifact_id", label_id))
    for prediction_id in prediction_ids:
        params.append(("prediction_artifact_id", prediction_id))
    return f"{reverse('pipeline-symbol-research', args=[symbol])}?{urlencode(params, doseq=True)}"


def _backtest_symbol_detail_link(backtest_artifact: Artifact, strategy_artifact: Artifact | None, symbol: str) -> str:
    params: list[tuple[str, Any]] = [("backtest_artifact_id", int(backtest_artifact.id))]
    if strategy_artifact is not None:
        strategy_link = _strategy_symbol_detail_link(strategy_artifact, symbol)
        separator = "&" if "?" in strategy_link else "?"
        return f"{strategy_link}{separator}backtest_artifact_id={int(backtest_artifact.id)}"
    return f"{reverse('pipeline-symbol-research', args=[symbol])}?{urlencode(params)}"


def _build_equity_curve_context(backtest_artifact: Artifact) -> dict[str, Any]:
    preview_rows, content_payload = _load_artifact_preview_rows(backtest_artifact, limit=5000)
    metadata = dict(backtest_artifact.metadata or {})
    strategy_artifact_id = _to_int(metadata.get("source_strategy_dataset_artifact_id"))
    daily_rows = list(content_payload.get("daily_rows") or [])
    if not daily_rows and preview_rows:
        daily_rows = build_backtest_daily_rows_from_trade_rows(preview_rows)
    curve_rows = list(metadata.get("equity_curve") or [])
    if not curve_rows and daily_rows:
        curve_rows = build_equity_curve_from_daily_rows(daily_rows)
    if not daily_rows and curve_rows:
        daily_rows = [
            {
                "date": row.get("date"),
                "equity": row.get("equity"),
                "net_daily_return": row.get("net_daily_return"),
            }
            for row in curve_rows
        ]
    points: list[dict[str, Any]] = []
    for row in curve_rows:
        date_value = str(row.get("date") or "")[:10]
        equity = _safe_float(row.get("equity"))
        daily = _safe_float(row.get("net_daily_return"))
        if not date_value or equity is None:
            continue
        points.append({"x": date_value, "equity": equity, "net_daily_return": daily})
    turnover_series = [
        {"x": str(row.get("date") or "")[:10], "turnover": _safe_float(row.get("turnover")) or 0.0}
        for row in daily_rows
        if str(row.get("date") or "").strip()
    ]
    positions_series = [
        {"x": str(row.get("date") or "")[:10], "positions": int(_safe_float(row.get("positions")) or 0)}
        for row in daily_rows
        if str(row.get("date") or "").strip()
    ]
    monthly_returns: dict[str, float] = {}
    avg_positions: list[int] = []
    turnover_values: list[float] = []
    for row in daily_rows:
        date_value = str(row.get("date") or "")[:10]
        month_key = date_value[:7]
        net_daily_return = _safe_float(row.get("net_daily_return")) or 0.0
        monthly_returns[month_key] = (1.0 + monthly_returns.get(month_key, 0.0)) * (1.0 + net_daily_return) - 1.0
        avg_positions.append(int(_safe_float(row.get("positions")) or 0))
        turnover_values.append(_safe_float(row.get("turnover")) or 0.0)
    monthly_return_rows = [
        {"month": month, "return": round(value, 8)}
        for month, value in sorted(monthly_returns.items(), key=lambda item: item[0], reverse=True)
    ]
    contribution_by_symbol: dict[str, dict[str, Any]] = {}
    for row in preview_rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        payload = contribution_by_symbol.setdefault(symbol, {"symbol": symbol, "realized_return": 0.0, "rows": 0})
        payload["realized_return"] += _safe_float(row.get("realized_return")) or 0.0
        payload["rows"] += 1
    contribution_rows = sorted(
        [
            {
                "symbol": symbol,
                "realized_return": round(values["realized_return"], 8),
                "rows": int(values["rows"]),
            }
            for symbol, values in contribution_by_symbol.items()
        ],
        key=lambda item: item["realized_return"],
        reverse=True,
    )
    daily_return_values = [_safe_float(row.get("net_daily_return")) or 0.0 for row in daily_rows]
    avg_daily_return = sum(daily_return_values) / len(daily_return_values) if daily_return_values else 0.0
    positive_days = sum(1 for value in daily_return_values if value > 0)
    negative_days = sum(1 for value in daily_return_values if value < 0)
    daily_volatility = 0.0
    if len(daily_return_values) > 1:
        variance = sum((value - avg_daily_return) ** 2 for value in daily_return_values) / float(len(daily_return_values) - 1)
        daily_volatility = variance ** 0.5
    annual_volatility = daily_volatility * (252.0 ** 0.5)
    annual_return = 0.0
    if points:
        start_equity = points[0]["equity"]
        end_equity = points[-1]["equity"]
        years = max(float(len(points)) / 252.0, 1.0 / 252.0)
        if start_equity and end_equity > 0:
            annual_return = (end_equity / start_equity) ** (1.0 / years) - 1.0
    sharpe = ((avg_daily_return / daily_volatility) * (252.0 ** 0.5)) if daily_volatility > 0 else None
    downside_values = [value for value in daily_return_values if value < 0]
    downside_volatility = 0.0
    if downside_values:
        downside_volatility = (sum(value ** 2 for value in downside_values) / float(len(downside_values))) ** 0.5
    sortino = ((avg_daily_return / downside_volatility) * (252.0 ** 0.5)) if downside_volatility > 0 else None
    max_drawdown = _safe_float(content_payload.get("max_drawdown"))
    calmar = (annual_return / abs(max_drawdown)) if max_drawdown not in (None, 0.0) else None
    gross_profit = sum(max(0.0, _safe_float(row.get("realized_return")) or 0.0) for row in preview_rows)
    gross_loss = sum(abs(min(0.0, _safe_float(row.get("realized_return")) or 0.0)) for row in preview_rows)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None
    daily_sorted = [
        {
            "date": str(row.get("date") or "")[:10],
            "return": round(_safe_float(row.get("net_daily_return")) or 0.0, 8),
        }
        for row in daily_rows
        if str(row.get("date") or "").strip()
    ]
    best_day = max(daily_sorted, key=lambda item: item["return"], default=None)
    worst_day = min(daily_sorted, key=lambda item: item["return"], default=None)
    benchmark_points: list[dict[str, Any]] = []
    benchmark_final_equity = None
    benchmark_cumulative_return = None
    if strategy_artifact_id > 0:
        strategy_artifact = Artifact.objects.filter(pk=strategy_artifact_id, artifact_type="STRATEGY_DATASET").first()
        if strategy_artifact is not None:
            strategy_rows, _strategy_payload = _load_artifact_preview_rows(strategy_artifact, limit=50000)
            benchmark_daily: dict[str, list[float]] = {}
            for row in strategy_rows:
                date_value = str(row.get("date") or "")[:10]
                ret_1 = _safe_float(row.get("ret_1"))
                if not date_value or ret_1 is None:
                    continue
                benchmark_daily.setdefault(date_value, []).append(ret_1)
            benchmark_equity = 1.0
            for date_value in sorted(benchmark_daily):
                values = benchmark_daily[date_value]
                daily_bench = sum(values) / float(len(values)) if values else 0.0
                benchmark_equity *= 1.0 + float(daily_bench)
                benchmark_points.append(
                    {
                        "x": date_value,
                        "equity": round(float(benchmark_equity), 8),
                        "daily_return": round(float(daily_bench), 8),
                    }
                )
            if benchmark_points:
                benchmark_final_equity = benchmark_points[-1]["equity"]
                benchmark_cumulative_return = round(float(benchmark_final_equity) - 1.0, 8)
    report_summary = {
        "avg_positions": round(sum(avg_positions) / len(avg_positions), 4) if avg_positions else 0.0,
        "avg_turnover": round(sum(turnover_values) / len(turnover_values), 8) if turnover_values else 0.0,
        "avg_daily_return": round(avg_daily_return, 8),
        "annual_return": round(annual_return, 8),
        "annual_return_pct": round(annual_return * 100.0, 4),
        "annual_volatility": round(annual_volatility, 8),
        "annual_volatility_pct": round(annual_volatility * 100.0, 4),
        "sharpe": round(sharpe, 8) if sharpe is not None else None,
        "sortino": round(sortino, 8) if sortino is not None else None,
        "calmar": round(calmar, 8) if calmar is not None else None,
        "profit_factor": round(profit_factor, 8) if profit_factor is not None else None,
        "positive_day_rate": round(positive_days / len(daily_return_values), 8) if daily_return_values else 0.0,
        "positive_day_rate_pct": round((positive_days / len(daily_return_values)) * 100.0, 4) if daily_return_values else 0.0,
        "negative_day_rate": round(negative_days / len(daily_return_values), 8) if daily_return_values else 0.0,
        "negative_day_rate_pct": round((negative_days / len(daily_return_values)) * 100.0, 4) if daily_return_values else 0.0,
        "best_month": monthly_return_rows[0] if monthly_return_rows else None,
        "worst_month": monthly_return_rows[-1] if monthly_return_rows else None,
        "best_day": best_day,
        "worst_day": worst_day,
        "benchmark_final_equity": benchmark_final_equity,
        "benchmark_cumulative_return": benchmark_cumulative_return,
        "benchmark_cumulative_return_pct": round((benchmark_cumulative_return or 0.0) * 100.0, 4) if benchmark_cumulative_return is not None else None,
        "top_contributor": contribution_rows[0] if contribution_rows else None,
        "bottom_contributor": contribution_rows[-1] if contribution_rows else None,
    }
    return {
        "equity_curve_points_json": json.dumps(points),
        "benchmark_curve_points_json": json.dumps(benchmark_points),
        "equity_curve_count": int(len(points)),
        "backtest_daily_rows_full": daily_rows,
        "turnover_series_json": json.dumps(turnover_series),
        "positions_series_json": json.dumps(positions_series),
        "monthly_return_rows": monthly_return_rows,
        "contribution_rows": contribution_rows,
        "report_summary": report_summary,
    }


def _research_query_for_symbol(artifact: Artifact) -> str:
    params: dict[str, int] = {}
    artifact_type = str(artifact.artifact_type or "").upper()
    metadata = dict(artifact.metadata or {})
    if artifact_type == "LABELS":
        params["label_artifact_id"] = int(artifact.id)
    elif artifact_type == "FEATURES":
        params["feature_artifact_id"] = int(artifact.id)
    elif artifact_type in {"PREDICTIONS", "CLASSIFIER_PREDICTIONS", "REGRESSOR_PREDICTIONS", "AUTOENCODER_SCORES", "MTL_PREDICTIONS"}:
        params["prediction_artifact_id"] = int(artifact.id)
        source_feature_id = int(metadata.get("source_features_artifact_id") or 0)
        source_label_id = int(metadata.get("source_label_artifact_id") or 0)
        if source_feature_id > 0:
            params["feature_artifact_id"] = source_feature_id
        if source_label_id > 0:
            params["label_artifact_id"] = source_label_id
    elif artifact_type == "STRATEGY_DATASET":
        source_feature_id = int(metadata.get("source_features_artifact_id") or 0)
        source_label_id = int(metadata.get("source_label_artifact_id") or 0)
        source_prediction_ids = [int(v) for v in list(metadata.get("source_prediction_artifact_ids") or []) if int(v or 0) > 0]
        if source_feature_id > 0:
            params["feature_artifact_id"] = source_feature_id
        if source_label_id > 0:
            params["label_artifact_id"] = source_label_id
        if source_prediction_ids:
            params["prediction_artifact_id"] = source_prediction_ids[0]
    elif artifact_type == "BACKTEST_RESULT":
        source_strategy_id = int(metadata.get("source_strategy_dataset_artifact_id") or 0)
        if source_strategy_id > 0:
            strategy_artifact = Artifact.objects.filter(pk=source_strategy_id).first()
            if strategy_artifact is not None:
                return _research_query_for_symbol(strategy_artifact)
    encoded = urlencode(params)
    return f"?{encoded}" if encoded else ""


def _saved_model_for_pipeline_artifact(artifact: Artifact) -> SavedModelArtifact | None:
    content_payload = dict(artifact.content or {})
    metadata = dict(artifact.metadata or {})
    try:
        saved_model_id = int(content_payload.get("model_artifact_id") or metadata.get("saved_model_artifact_id") or 0)
    except Exception:
        saved_model_id = 0
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
