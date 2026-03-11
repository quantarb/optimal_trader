from __future__ import annotations

from typing import Any

from django.http import Http404, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from analysis.insights import build_opportunity_dashboard, build_portfolio_analysis, build_stock_intelligence
from analysis.research import (
    artifact_rows_for_symbol,
    build_backtest_chart_context,
    build_feature_table_context,
    build_label_chart_context,
    build_prediction_chart_series,
    build_price_chart_context,
    build_strategy_chart_context,
    normalize_prediction_rows,
    recent_artifact_choices,
    resolve_artifact,
)

from .models import Artifact
from .view_support import (
    INSIGHT_PREDICTION_ARTIFACT_TYPES,
    _annotate_rows_with_bar_pct,
    _clean_ids,
    _insight_selection_context,
    _json_safe_data,
    _load_market_situation_artifacts,
    _parse_symbols_csv,
    _to_int,
)


@require_GET
def pipeline_market_state_api(request):
    symbol = str(request.GET.get("symbol") or "").strip().upper()
    if not symbol:
        return JsonResponse({"error": "Provide a symbol query parameter."}, status=400)
    selection = _insight_selection_context(request)
    reasoning_mode = str(request.GET.get("reasoning_mode") or "deterministic").strip() or "deterministic"
    try:
        payload = build_stock_intelligence(
            symbol=symbol,
            date=str(request.GET.get("date") or "").strip() or None,
            strategy_artifact_id=selection["strategy_artifact_id"],
            feature_artifact_id=selection["feature_artifact_id"],
            label_artifact_id=selection["label_artifact_id"],
            prediction_artifact_ids=selection["prediction_artifact_ids"],
            market_situation_artifact_id=selection["market_situation_artifact_id"],
            twin_count=max(3, _to_int(request.GET.get("k") or 10)),
            search_method=selection["search_method"],
            reasoning_mode=reasoning_mode,
        )
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    return JsonResponse(_json_safe_data(payload))


def pipeline_opportunities(request):
    selection = _insight_selection_context(request)
    limit = max(5, _to_int(request.GET.get("limit") or 20))
    error = ""
    dashboard: dict[str, Any] = {"rows": [], "as_of_date": "", "artifacts": {}}
    try:
        dashboard = build_opportunity_dashboard(
            strategy_artifact_id=selection["strategy_artifact_id"],
            feature_artifact_id=selection["feature_artifact_id"],
            label_artifact_id=selection["label_artifact_id"],
            prediction_artifact_ids=selection["prediction_artifact_ids"],
            market_situation_artifact_id=selection["market_situation_artifact_id"],
            search_method=selection["search_method"],
            limit=limit,
        )
    except Exception as exc:
        error = str(exc)
    rows = _annotate_rows_with_bar_pct(list(dashboard.get("rows") or []), "opportunity_score", "score_bar_pct")
    rows = _annotate_rows_with_bar_pct(rows, "confidence_score", "confidence_bar_pct")
    rows = _annotate_rows_with_bar_pct(rows, "market_familiarity_score", "familiarity_bar_pct")
    if str(request.GET.get("format") or "").strip().lower() == "json":
        return JsonResponse(
            _json_safe_data(
                {
                    **selection,
                    "dashboard": dashboard,
                    "rows": rows,
                    "error": error,
                }
            )
        )
    return render(
        request,
        "pipeline/opportunities.html",
        {
            **selection,
            "dashboard": dashboard,
            "rows": rows,
            "error": error,
            "page_title": "Opportunities",
            "header_metrics": [
                {"label": "As Of", "value": str(dashboard.get("as_of_date") or "-")},
                {"label": "Shown", "value": str(len(rows))},
                {"label": "Source", "value": str(((dashboard.get("frame_meta") or {}).get("source") or "-"))},
                {
                    "label": "Strategy Artifact",
                    "value": f"#{((dashboard.get('artifacts') or {}).get('strategy_artifact_id'))}"
                    if (dashboard.get("artifacts") or {}).get("strategy_artifact_id")
                    else "-",
                },
            ],
        },
    )


def pipeline_top_opportunities(request):
    mutable = request.GET.copy()
    if not mutable.get("limit"):
        mutable["limit"] = "10"
    request.GET = mutable
    return pipeline_opportunities(request)


def pipeline_stock_intelligence(request, symbol: str | None = None):
    selection = _insight_selection_context(request)
    symbol_value = str(symbol or request.GET.get("symbol") or "").strip().upper() or "AAPL"
    reasoning_mode = str(request.GET.get("reasoning_mode") or "deterministic").strip() or "deterministic"
    error = ""
    payload: dict[str, Any] = {}
    try:
        payload = build_stock_intelligence(
            symbol=symbol_value,
            date=str(request.GET.get("date") or "").strip() or None,
            strategy_artifact_id=selection["strategy_artifact_id"],
            feature_artifact_id=selection["feature_artifact_id"],
            label_artifact_id=selection["label_artifact_id"],
            prediction_artifact_ids=selection["prediction_artifact_ids"],
            market_situation_artifact_id=selection["market_situation_artifact_id"],
            twin_count=max(3, _to_int(request.GET.get("k") or 10)),
            search_method=selection["search_method"],
            reasoning_mode=reasoning_mode,
        )
    except Exception as exc:
        error = str(exc)
    twins = _annotate_rows_with_bar_pct(list(payload.get("historical_twins") or []), "similarity_score", "similarity_bar_pct")
    horizon_rows = _annotate_rows_with_bar_pct(list((payload.get("outcome_summary") or {}).get("horizon_rows") or []), "median_return", "return_bar_pct")
    driver_rows = _annotate_rows_with_bar_pct(list(payload.get("drivers") or []), "contribution", "contribution_bar_pct")
    if str(request.GET.get("format") or "").strip().lower() == "json":
        return JsonResponse(
            _json_safe_data(
                {
                    **selection,
                    "symbol": symbol_value,
                    "payload": payload,
                    "reasoning_mode": reasoning_mode,
                    "historical_twins": twins,
                    "horizon_rows": horizon_rows,
                    "driver_rows": driver_rows,
                    "error": error,
                }
            )
        )
    return render(
        request,
        "pipeline/stock_intelligence.html",
        {
            **selection,
            "symbol": symbol_value,
            "payload": payload,
            "reasoning_mode": reasoning_mode,
            "historical_twins": twins,
            "horizon_rows": horizon_rows,
            "driver_rows": driver_rows,
            "error": error,
            "header_pills": [
                *([f"State date {payload.get('date')}"] if payload.get("date") else []),
                *([f"Source {((payload.get('frame_meta') or {}).get('source'))}"] if ((payload.get("frame_meta") or {}).get("source")) else []),
                *([f"Search {str(selection.get('search_method') or '').replace('_', ' ').title()}"] if selection.get("search_method") else []),
            ],
            "header_metrics": [
                {"label": "Opportunity Score", "value": f"{float(((payload.get('opportunity') or {}).get('opportunity_score') or 0.0)):.0f}"},
                {"label": "Confidence", "value": str(((payload.get("opportunity") or {}).get("confidence_label") or "-"))},
                {"label": "Familiarity", "value": str(((payload.get("opportunity") or {}).get("market_familiarity_label") or "-"))},
                {"label": "Risk Indicator", "value": str(((payload.get("opportunity") or {}).get("risk_indicator") or "-"))},
            ],
        },
    )


def pipeline_portfolio_analysis(request):
    selection = _insight_selection_context(request)
    symbols = _parse_symbols_csv(str(request.GET.get("symbols") or ""))
    reasoning_mode = str(request.GET.get("reasoning_mode") or "deterministic").strip() or "deterministic"
    error = ""
    payload: dict[str, Any] = {
        "symbols": symbols,
        "rows": [],
        "strong_rows": [],
        "neutral_rows": [],
        "weak_rows": [],
        "portfolio_score": 0.0,
        "regime_similarity_score": 0.0,
        "risk_concentration_score": 0.0,
    }
    if symbols:
        try:
            payload = build_portfolio_analysis(
                symbols=symbols,
                strategy_artifact_id=selection["strategy_artifact_id"],
                feature_artifact_id=selection["feature_artifact_id"],
                label_artifact_id=selection["label_artifact_id"],
                prediction_artifact_ids=selection["prediction_artifact_ids"],
                market_situation_artifact_id=selection["market_situation_artifact_id"],
                search_method=selection["search_method"],
                reasoning_mode=reasoning_mode,
            )
        except Exception as exc:
            error = str(exc)
    rows = _annotate_rows_with_bar_pct(list(payload.get("rows") or []), "opportunity_score", "score_bar_pct")
    if str(request.GET.get("format") or "").strip().lower() == "json":
        return JsonResponse(
            _json_safe_data(
                {
                    **selection,
                    "payload": payload,
                    "reasoning_mode": reasoning_mode,
                    "rows": rows,
                    "error": error,
                    "symbols_csv": ", ".join(symbols),
                }
            )
        )
    return render(
        request,
        "pipeline/portfolio_analysis.html",
        {
            **selection,
            "payload": payload,
            "reasoning_mode": reasoning_mode,
            "rows": rows,
            "error": error,
            "symbols_csv": ", ".join(symbols),
            "header_metrics": [
                {"label": "Portfolio Score", "value": f"{float(payload.get('portfolio_score') or 0.0):.0f}"},
                {"label": "Regime Similarity", "value": f"{float(payload.get('regime_similarity_score') or 0.0):.0f}"},
                {"label": "Risk Concentration", "value": f"{float(payload.get('risk_concentration_score') or 0.0):.0f}"},
            ],
        },
    )


def pipeline_market_situations(request):
    rows = _load_market_situation_artifacts(limit=30)
    selected_id = _to_int(request.GET.get("artifact_id"))
    selected_row = next((row for row in rows if int(row["artifact"].id) == selected_id), rows[0] if rows else None)
    payload = dict((selected_row or {}).get("payload") or {})
    cluster_rows = _annotate_rows_with_bar_pct(
        [
            {
                **dict(row),
                "cluster_size": int(row.get("cluster_size") or 0),
                "median_return": float(((row.get("outcome_statistics") or {}).get("median_return") or 0.0)),
                "win_rate": float(((row.get("outcome_statistics") or {}).get("win_rate") or 0.0)),
                "avg_hold_days": float(((row.get("outcome_statistics") or {}).get("avg_hold_days") or 0.0)),
                "yearly_median_return_std": float(((row.get("outcome_statistics") or {}).get("yearly_median_return_std") or 0.0)),
            }
            for row in list(payload.get("clusters") or [])
        ],
        "cluster_size",
        "size_bar_pct",
    )
    cluster_rows = _annotate_rows_with_bar_pct(cluster_rows, "win_rate", "win_rate_bar_pct")
    if str(request.GET.get("format") or "").strip().lower() == "json":
        return JsonResponse(
            _json_safe_data(
                {
                    "artifacts": [
                        {
                            "artifact_id": int(row["artifact"].id),
                            "summary": dict(row.get("summary") or {}),
                        }
                        for row in rows
                    ],
                    "selected_artifact_id": int((selected_row or {}).get("artifact").id) if selected_row else 0,
                    "payload": payload,
                    "cluster_rows": cluster_rows,
                }
            )
        )
    return render(
        request,
        "pipeline/market_situations.html",
        {
            "artifacts": rows,
            "selected_row": selected_row,
            "payload": payload,
            "summary": dict(payload.get("summary") or {}),
            "cluster_rows": cluster_rows,
        },
    )


@require_GET
def symbol_research_view(request, symbol: str):
    symbol_value = str(symbol or "").strip().upper()
    if not symbol_value:
        raise Http404("Symbol is required.")

    label_artifact = resolve_artifact(
        artifact_type="LABELS",
        artifact_id=_to_int(request.GET.get("label_artifact_id")),
        pipeline_run_id=_to_int(request.GET.get("label_run_id")),
    )
    feature_artifact = resolve_artifact(
        artifact_type="FEATURES",
        artifact_id=_to_int(request.GET.get("feature_artifact_id")),
        pipeline_run_id=_to_int(request.GET.get("feature_run_id")),
    )
    strategy_artifact = resolve_artifact(
        artifact_type="STRATEGY_DATASET",
        artifact_id=_to_int(request.GET.get("strategy_artifact_id")),
        pipeline_run_id=_to_int(request.GET.get("strategy_run_id")),
    )
    backtest_artifact = resolve_artifact(
        artifact_type="BACKTEST_RESULT",
        artifact_id=_to_int(request.GET.get("backtest_artifact_id")),
        pipeline_run_id=_to_int(request.GET.get("backtest_run_id")),
    )
    prediction_artifact_ids = _clean_ids(request.GET.getlist("prediction_artifact_id"))
    prediction_artifacts: list[Artifact]
    if prediction_artifact_ids:
        by_id = {
            int(row.id): row
            for row in Artifact.objects.filter(id__in=prediction_artifact_ids, artifact_type__in=INSIGHT_PREDICTION_ARTIFACT_TYPES)
            .select_related("pipeline_run")
        }
        prediction_artifacts = [by_id[row_id] for row_id in prediction_artifact_ids if row_id in by_id]
    else:
        single_prediction_artifact = resolve_artifact(
            artifact_type=INSIGHT_PREDICTION_ARTIFACT_TYPES,
            artifact_id=_to_int(request.GET.get("prediction_artifact_id")),
            pipeline_run_id=_to_int(request.GET.get("prediction_run_id")),
        )
        prediction_artifacts = [single_prediction_artifact] if single_prediction_artifact is not None else []

    label_rows = artifact_rows_for_symbol(label_artifact, symbol_value) if label_artifact is not None else []
    feature_rows = artifact_rows_for_symbol(feature_artifact, symbol_value) if feature_artifact is not None else []
    strategy_rows = artifact_rows_for_symbol(strategy_artifact, symbol_value) if strategy_artifact is not None else []
    backtest_rows = artifact_rows_for_symbol(backtest_artifact, symbol_value) if backtest_artifact is not None else []
    prediction_artifact_rows: list[tuple[Artifact, list[dict[str, Any]]]] = []
    prediction_rows: list[dict[str, Any]] = []
    for artifact in prediction_artifacts:
        rows = normalize_prediction_rows(artifact_rows_for_symbol(artifact, symbol_value))
        prediction_artifact_rows.append((artifact, rows))
        for row in rows:
            merged = dict(row)
            merged["source_artifact_id"] = int(artifact.id)
            merged["source_artifact_type"] = str(artifact.artifact_type)
            prediction_rows.append(merged)
    prediction_artifact = prediction_artifacts[0] if prediction_artifacts else None

    context = {
        "symbol": symbol_value,
        "label_artifact": label_artifact,
        "feature_artifact": feature_artifact,
        "strategy_artifact": strategy_artifact,
        "backtest_artifact": backtest_artifact,
        "prediction_artifact": prediction_artifact,
        "prediction_artifacts": prediction_artifacts,
        "label_artifact_choices": recent_artifact_choices("LABELS"),
        "feature_artifact_choices": recent_artifact_choices("FEATURES"),
        "strategy_artifact_choices": recent_artifact_choices("STRATEGY_DATASET"),
        "backtest_artifact_choices": recent_artifact_choices("BACKTEST_RESULT"),
        "prediction_artifact_choices": recent_artifact_choices(INSIGHT_PREDICTION_ARTIFACT_TYPES),
        "selected_prediction_artifact_ids": [int(a.id) for a in prediction_artifacts],
        "label_rows": label_rows[:50],
        "prediction_rows": prediction_rows[:100],
        **build_price_chart_context(symbol_value),
        **build_label_chart_context(label_rows),
        **build_prediction_chart_series(prediction_artifact_rows),
        **build_strategy_chart_context(strategy_rows),
        **build_backtest_chart_context(backtest_rows),
        **build_feature_table_context(feature_rows),
    }
    return render(request, "pipeline/symbol_research.html", context)


__all__ = [
    "pipeline_market_state_api",
    "pipeline_market_situations",
    "pipeline_opportunities",
    "pipeline_portfolio_analysis",
    "pipeline_stock_intelligence",
    "pipeline_top_opportunities",
    "symbol_research_view",
]
