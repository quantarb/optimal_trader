from __future__ import annotations

import json

from django.shortcuts import get_object_or_404, redirect, render

from .execution import build_symbol_choices, merge_job_params, run_training_job
from .forms import ModelTrainingForm
from .models import ModelArtifact, ModelTrainingJob


def train_model_view(request):
    saved_job = None
    run_result = None
    symbol_choices = build_symbol_choices()

    if request.method == "POST" and request.POST.get("delete_job_id"):
        form = ModelTrainingForm(symbol_choices=symbol_choices)
        job = ModelTrainingJob.objects.filter(pk=request.POST.get("delete_job_id")).first()
        if job is None:
            run_result = {"ok": False, "message": "Training job not found."}
        else:
            job.delete()
            run_result = {"ok": True, "message": "Training job deleted."}
    elif request.method == "POST" and request.POST.get("run_job_id"):
        form = ModelTrainingForm(symbol_choices=symbol_choices)
        job = ModelTrainingJob.objects.filter(pk=request.POST.get("run_job_id")).first()
        if job is None:
            run_result = {"ok": False, "message": "Training job not found."}
        else:
            try:
                artifact = run_training_job(job)
            except Exception as exc:
                run_result = {"ok": False, "message": str(exc)}
            else:
                run_result = {
                    "ok": True,
                    "message": f"Saved artifact {artifact.name} v{artifact.version}.",
                }
    elif request.method == "POST":
        form = ModelTrainingForm(request.POST, symbol_choices=symbol_choices)
        if form.is_valid():
            params = merge_job_params(
                form.cleaned_data["params_json"],
                symbol=form.cleaned_data["symbol"],
            )
            saved_job = ModelTrainingJob.objects.create(
                name=form.cleaned_data["name"],
                framework=form.cleaned_data["framework"],
                algorithm=form.cleaned_data["algorithm"],
                task_type=form.cleaned_data["task_type"],
                target_col=form.cleaned_data["target_col"],
                feature_cols=form.cleaned_data["feature_families"],
                split_ratio=form.cleaned_data["split_ratio"],
                params=params,
                notes=form.cleaned_data["notes"],
                status="pending",
            )
            form = ModelTrainingForm(symbol_choices=symbol_choices)
    else:
        form = ModelTrainingForm(symbol_choices=symbol_choices)

    recent_jobs = ModelTrainingJob.objects.select_related("latest_artifact")[:10]
    return render(
        request,
        "ml/train_form.html",
        {
            "form": form,
            "saved_job": saved_job,
            "run_result": run_result,
            "recent_jobs": recent_jobs,
        },
    )


def model_artifact_detail_view(request, artifact_id: int):
    artifact = get_object_or_404(ModelArtifact, pk=artifact_id)
    if request.method == "POST" and request.POST.get("delete_artifact") == "1":
        artifact.delete()
        return redirect("train_model")

    related_jobs = ModelTrainingJob.objects.filter(latest_artifact=artifact).order_by("-updated_at")[:10]
    model_summary = str((artifact.metadata or {}).get("model_summary") or "").strip()
    context = {
        "artifact": artifact,
        "related_jobs": related_jobs,
        "model_summary": model_summary,
        "metrics_json": json.dumps(artifact.metrics or {}, indent=2, sort_keys=True),
        "params_json": json.dumps(artifact.params or {}, indent=2, sort_keys=True),
        "metadata_json": json.dumps(artifact.metadata or {}, indent=2, sort_keys=True),
    }
    return render(request, "ml/model_detail.html", context)
