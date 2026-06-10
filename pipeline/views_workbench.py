from __future__ import annotations

from django.shortcuts import get_object_or_404, render

from analysis.research import recent_artifact_choices

from .forms import (
    BacktestPipelineForm,
    FitModelPipelineForm,
    OptimalTradeResearchForm,
    ScoreModelPipelineForm,
    StrategyDefinitionForm,
    StrategyDatasetPipelineForm,
)
from .models import Artifact, StrategyDefinition
from .run_support import _launch_pipeline_run
from .strategy_definitions import ensure_default_strategy_definitions, strategy_definition_choices
from .universe_selection import parse_exchange_values, resolve_symbol_universe
from .view_support import (
    UI_PREDICTION_ARTIFACT_TYPES,
    UI_STATE_PANEL_ARTIFACT_TYPES,
    _artifact_choices,
)


def pipeline_ui_view(request):
    return render(request, "pipeline/dashboard.html", {})


def pipeline_lab_view(
    request,
    *,
    run_optimal_trade_research_suite_fn,
    threading_module,
    mag7_symbols,
):
    feature_choices = _artifact_choices("FEATURES")
    label_choices = _artifact_choices("LABELS")
    prediction_choices = _artifact_choices(UI_STATE_PANEL_ARTIFACT_TYPES)
    classifier_model_choices = _artifact_choices("CLASSIFIER_MODEL")
    regressor_model_choices = _artifact_choices("REGRESSOR_MODEL")
    mtl_model_choices = _artifact_choices("MTL_MODEL")
    strategy_dataset_choices = _artifact_choices("STRATEGY_DATASET")
    strategy_def_choices = strategy_definition_choices()

    fit_form = FitModelPipelineForm(
        request.POST if request.method == "POST" and request.POST.get("lab_action") == "fit_model" else None,
        feature_choices=feature_choices,
        label_choices=label_choices,
        prediction_choices=prediction_choices,
    )
    score_form = ScoreModelPipelineForm(
        request.POST if request.method == "POST" and request.POST.get("lab_action") == "score_model" else None,
        model_choices=classifier_model_choices + regressor_model_choices + mtl_model_choices,
        feature_choices=feature_choices,
        label_choices=label_choices,
        prediction_choices=prediction_choices,
    )
    strategy_form = StrategyDatasetPipelineForm(
        request.POST if request.method == "POST" and request.POST.get("lab_action") == "build_strategy" else None,
        strategy_definition_choices=strategy_def_choices,
        feature_choices=feature_choices,
        label_choices=label_choices,
        prediction_choices=prediction_choices,
    )
    backtest_form = BacktestPipelineForm(
        request.POST if request.method == "POST" and request.POST.get("lab_action") == "backtest_strategy" else None,
        strategy_choices=strategy_dataset_choices,
    )
    research_form = OptimalTradeResearchForm(
        request.POST if request.method == "POST" and request.POST.get("lab_action") == "run_research_suite" else None,
        feature_choices=feature_choices,
        label_choices=label_choices,
    )

    started_run = None
    started_suite = None
    action_error = ""

    if request.method == "POST":
        action = str(request.POST.get("lab_action") or "").strip()
        try:
            if action == "fit_model" and fit_form.is_valid():
                data = fit_form.cleaned_data
                started_run = _launch_pipeline_run(
                    name=str(data["name"]),
                    target_job=str(data["job_type"]),
                    mode="strict",
                    config={
                        "algorithm": str(data.get("algorithm") or "random_forest_classifier"),
                        "target_col": str(data.get("target_col") or ""),
                        "split_ratio": float(data["split_ratio"]),
                        "research_scope": str(data.get("research_scope") or ""),
                        "min_abs_trade_return_pct": float(data["min_abs_trade_return_pct"]) if data.get("min_abs_trade_return_pct") not in (None, "") else None,
                        "max_hold_days": int(data["max_hold_days"]) if data.get("max_hold_days") not in (None, "") else None,
                        "sample_weight_mode": str(data.get("sample_weight_mode") or "uniform"),
                        "params": dict(data["params_json"]),
                        "prediction_artifact_ids": list(data["prediction_artifact_ids"]),
                        "model_name": str(data["name"]),
                    },
                    input_artifact_ids=[int(data["feature_artifact_id"]), int(data["label_artifact_id"])],
                )
            elif action == "score_model" and score_form.is_valid():
                data = score_form.cleaned_data
                started_run = _launch_pipeline_run(
                    name=str(data["name"]),
                    target_job=str(data["job_type"]),
                    mode="strict",
                    config={
                        "label_artifact_id": int(data.get("label_artifact_id") or 0),
                        "prediction_artifact_ids": list(data["prediction_artifact_ids"]),
                    },
                    input_artifact_ids=[int(data["model_artifact_id"]), int(data["feature_artifact_id"])],
                )
            elif action == "build_strategy" and strategy_form.is_valid():
                data = strategy_form.cleaned_data
                started_run = _launch_pipeline_run(
                    name=str(data["name"]),
                    target_job="build_strategy_dataset",
                    mode="strict",
                    config={
                        "strategy_definition_id": int(data["strategy_definition_id"]),
                        "label_artifact_id": int(data.get("label_artifact_id") or 0),
                        "prediction_artifact_ids": list(data["prediction_artifact_ids"]),
                    },
                    input_artifact_ids=[int(data["feature_artifact_id"])],
                )
            elif action == "backtest_strategy" and backtest_form.is_valid():
                data = backtest_form.cleaned_data
                started_run = _launch_pipeline_run(
                    name=str(data["name"]),
                    target_job="backtest_strategy",
                    mode="strict",
                    config={"transaction_cost_bps": float(data["transaction_cost_bps"])},
                    input_artifact_ids=[int(data["strategy_dataset_artifact_id"])],
                )
            elif action == "run_research_suite" and research_form.is_valid():
                data = research_form.cleaned_data
                folds = [
                    {
                        "name": f"wf_{year}",
                        "train_end_date": f"{year - 1}-12-31",
                        "backtest_start_date": f"{year}-01-01",
                        "backtest_end_date": f"{year}-12-31",
                    }
                    for year in range(int(data["test_start_year"]), int(data["test_end_year"]) + 1)
                ]
                feature_artifact = Artifact.objects.filter(pk=int(data.get("feature_artifact_id") or 0), artifact_type="FEATURES").first()
                label_artifact = Artifact.objects.filter(pk=int(data.get("label_artifact_id") or 0), artifact_type="LABELS").first()
                suite_name = str(data["name"]).strip()
                universe_mode = str(data.get("universe_mode") or "mag7").strip()
                if universe_mode == "us_market_cap_screen":
                    selected_symbols = resolve_symbol_universe(
                        min_market_cap=float(data["min_market_cap"]) if data.get("min_market_cap") not in (None, "") else None,
                        country=str(data.get("country") or "").strip() or None,
                        exchanges=parse_exchange_values(data.get("exchanges_csv")),
                        limit=int(data["max_symbols"]) if data.get("max_symbols") not in (None, "") else None,
                        exclude_pooled_vehicles=True,
                    )
                else:
                    selected_symbols = list(mag7_symbols)
                if not selected_symbols:
                    raise ValueError("No symbols matched the selected universe filters.")
                thread = threading_module.Thread(
                    target=lambda: run_optimal_trade_research_suite_fn(
                        symbols=selected_symbols,
                        folds=folds,
                        min_profit_pct=float(data["min_profit_pct"]),
                        transaction_cost_bps=float(data["transaction_cost_bps"]),
                        profile_name=str(data["profile_name"]),
                        feature_artifact=feature_artifact,
                        label_artifact=label_artifact,
                        output_basename=suite_name,
                        resume_existing=bool(data.get("resume_existing")),
                    ),
                    daemon=True,
                )
                thread.start()
                started_suite = {
                    "name": suite_name,
                    "profile_name": str(data["profile_name"]),
                    "fold_count": len(folds),
                    "feature_artifact_id": int(feature_artifact.id) if feature_artifact is not None else 0,
                    "label_artifact_id": int(label_artifact.id) if label_artifact is not None else 0,
                    "resume_existing": bool(data.get("resume_existing")),
                    "symbol_count": len(selected_symbols),
                    "universe_mode": universe_mode,
                }
            else:
                action_error = "Fix the form errors before starting the run."
        except Exception as exc:
            action_error = str(exc)

    return render(
        request,
        "pipeline/lab.html",
        {
            "fit_form": fit_form,
            "score_form": score_form,
            "strategy_form": strategy_form,
            "backtest_form": backtest_form,
            "research_form": research_form,
            "started_run": started_run,
            "started_suite": started_suite,
            "action_error": action_error,
        },
    )


def pipeline_strategies_view(request):
    ensure_default_strategy_definitions()
    feature_choices = _artifact_choices("FEATURES")
    label_choices = _artifact_choices("LABELS")
    prediction_choices = _artifact_choices(UI_PREDICTION_ARTIFACT_TYPES)
    strategy_dataset_choices = _artifact_choices("STRATEGY_DATASET")
    strategy_def_choices = strategy_definition_choices()

    strategy_form = StrategyDatasetPipelineForm(
        request.POST if request.method == "POST" and request.POST.get("strategy_action") == "build_strategy" else None,
        strategy_definition_choices=strategy_def_choices,
        feature_choices=feature_choices,
        label_choices=label_choices,
        prediction_choices=prediction_choices,
    )
    backtest_form = BacktestPipelineForm(
        request.POST if request.method == "POST" and request.POST.get("strategy_action") == "backtest_strategy" else None,
        strategy_choices=strategy_dataset_choices,
    )

    started_run = None
    action_error = ""
    if request.method == "POST":
        action = str(request.POST.get("strategy_action") or "").strip()
        try:
            if action == "build_strategy" and strategy_form.is_valid():
                data = strategy_form.cleaned_data
                started_run = _launch_pipeline_run(
                    name=str(data["name"]),
                    target_job="build_strategy_dataset",
                    mode="strict",
                    config={
                        "strategy_definition_id": int(data["strategy_definition_id"]),
                        "label_artifact_id": int(data.get("label_artifact_id") or 0),
                        "prediction_artifact_ids": list(data["prediction_artifact_ids"]),
                    },
                    input_artifact_ids=[int(data["feature_artifact_id"])],
                )
            elif action == "backtest_strategy" and backtest_form.is_valid():
                data = backtest_form.cleaned_data
                started_run = _launch_pipeline_run(
                    name=str(data["name"]),
                    target_job="backtest_strategy",
                    mode="strict",
                    config={"transaction_cost_bps": float(data["transaction_cost_bps"])},
                    input_artifact_ids=[int(data["strategy_dataset_artifact_id"])],
                )
            else:
                action_error = "Fix the strategy form errors before starting the run."
        except Exception as exc:
            action_error = str(exc)

    strategy_artifacts = recent_artifact_choices("STRATEGY_DATASET", limit=25)
    backtest_artifacts = recent_artifact_choices("BACKTEST_RESULT", limit=25)
    strategy_rows: list[dict[str, object]] = []
    for artifact in strategy_artifacts:
        content = dict(artifact.content or {})
        metadata = dict(artifact.metadata or {})
        strategy_rows.append(
            {
                "artifact": artifact,
                "rows": int(content.get("rows") or 0),
                "symbols": int(content.get("symbols") or 0),
                "selected_rows": int(content.get("selected_rows") or 0),
                "dates": int(content.get("dates") or 0),
                "avg_daily_positions": content.get("avg_daily_positions"),
                "source_features_artifact_id": int(metadata.get("source_features_artifact_id") or 0),
                "source_label_artifact_id": int(metadata.get("source_label_artifact_id") or 0),
                "source_prediction_artifact_ids": [int(v) for v in list(metadata.get("source_prediction_artifact_ids") or []) if int(v or 0) > 0],
                "strategy_definition_name": str(metadata.get("strategy_definition_name") or ""),
                "detail_url": f"/pipeline/strategies/{artifact.id}/",
            }
        )
    backtest_rows: list[dict[str, object]] = []
    for artifact in backtest_artifacts:
        content = dict(artifact.content or {})
        metadata = dict(artifact.metadata or {})
        strategy_artifact_id = int(metadata.get("source_strategy_dataset_artifact_id") or 0)
        strategy_artifact = Artifact.objects.filter(pk=strategy_artifact_id).first() if strategy_artifact_id > 0 else None
        backtest_rows.append(
            {
                "artifact": artifact,
                "trades": int(content.get("trades") or 0),
                "wins": int(content.get("wins") or 0),
                "losses": int(content.get("losses") or 0),
                "avg_return": content.get("avg_return"),
                "cumulative_return": content.get("cumulative_return"),
                "days": int(content.get("days") or 0),
                "final_equity": content.get("final_equity"),
                "max_drawdown": content.get("max_drawdown"),
                "strategy_artifact": strategy_artifact,
                "detail_url": f"/pipeline/backtests/{artifact.id}/",
            }
        )

    return render(
        request,
        "pipeline/strategies.html",
        {
            "strategy_form": strategy_form,
            "backtest_form": backtest_form,
            "started_run": started_run,
            "action_error": action_error,
            "strategy_rows": strategy_rows,
            "backtest_rows": backtest_rows,
        },
    )


def strategy_definition_list_view(request):
    ensure_default_strategy_definitions()
    form = StrategyDefinitionForm(request.POST or None)
    started_definition = None
    action_error = ""
    if request.method == "POST":
        if form.is_valid():
            started_definition = form.save()
            form = StrategyDefinitionForm()
        else:
            action_error = "Fix the form errors before saving the strategy definition."
    definitions = list(StrategyDefinition.objects.order_by("name", "id"))
    return render(
        request,
        "pipeline/strategy_definitions.html",
        {
            "definitions": definitions,
            "form": form,
            "started_definition": started_definition,
            "action_error": action_error,
        },
    )


def strategy_definition_edit_view(request, definition_id: int):
    ensure_default_strategy_definitions()
    definition = get_object_or_404(StrategyDefinition, pk=int(definition_id))
    form = StrategyDefinitionForm(request.POST or None, instance=definition)
    saved = False
    action_error = ""
    if request.method == "POST":
        if form.is_valid():
            form.save()
            saved = True
        else:
            action_error = "Fix the form errors before saving the strategy definition."
    return render(
        request,
        "pipeline/strategy_definition_edit.html",
        {
            "definition": definition,
            "form": form,
            "saved": saved,
            "action_error": action_error,
        },
    )


__all__ = [
    "pipeline_lab_view",
    "pipeline_strategies_view",
    "pipeline_ui_view",
    "strategy_definition_edit_view",
    "strategy_definition_list_view",
]
