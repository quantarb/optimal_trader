from __future__ import annotations

import json

import pandas as pd
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect, render

from analysis.research import (
    build_prediction_chart_context,
    build_price_chart_context,
    normalize_prediction_rows,
)
from pipeline.service_runtime import read_frame_artifact

from .models import ModelArtifact


def _saved_prediction_frame_for_artifact(artifact: ModelArtifact) -> pd.DataFrame:
    metadata = dict(artifact.metadata or {})
    uri = str(metadata.get("predictions_uri") or "").strip()
    if not uri:
        return pd.DataFrame()
    try:
        prediction_df = read_frame_artifact(uri)
    except Exception:
        return pd.DataFrame()
    return prediction_df


def _prediction_symbol_rows(prediction_df: pd.DataFrame) -> list[dict[str, object]]:
    if prediction_df.empty or "symbol" not in prediction_df.columns:
        return []
    rows: list[dict[str, object]] = []
    for symbol, group in prediction_df.groupby("symbol", dropna=True):
        avg_prediction = None
        if "prediction" in group.columns and group["prediction"].notna().any():
            try:
                avg_prediction = float(pd.to_numeric(group["prediction"], errors="coerce").dropna().mean())
            except Exception:
                avg_prediction = None
        avg_score = None
        if "prediction_score" in group.columns and group["prediction_score"].notna().any():
            try:
                avg_score = float(pd.to_numeric(group["prediction_score"], errors="coerce").dropna().mean())
            except Exception:
                avg_score = None
        rows.append(
            {
                "symbol": str(symbol),
                "rows": int(len(group)),
                "avg_prediction": avg_prediction,
                "avg_score": avg_score,
                "min_date": group["date"].min(),
                "max_date": group["date"].max(),
            }
        )
    rows.sort(key=lambda row: (-int(row["rows"]), str(row["symbol"])))
    return rows


def _prediction_detail_rows(prediction_df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if prediction_df.empty:
        return rows
    normalized_rows = normalize_prediction_rows(prediction_df.to_dict(orient="records"))
    cols = [
        "date",
        "symbol",
        "prediction",
        "prediction_score",
        "signal_score",
        "market_position",
        "trade_return",
        "hold_days",
        "side",
        "freq",
        "k",
    ]
    for row in sorted(normalized_rows, key=lambda item: (str(item.get("date") or ""), str(item.get("symbol") or ""))):
        out: dict[str, object] = {}
        for col in cols:
            if col not in row:
                continue
            value = row.get(col)
            if hasattr(value, "isoformat"):
                try:
                    value = value.isoformat()
                except Exception:
                    pass
            out[col] = value
        rows.append(out)
    return rows


def model_artifact_detail_view(request, artifact_id: int):
    artifact = get_object_or_404(ModelArtifact, pk=artifact_id)
    if request.method == "POST" and request.POST.get("delete_artifact") == "1":
        artifact.delete()
        return redirect("pipeline-lab")

    model_summary = str((artifact.metadata or {}).get("model_summary") or "").strip()
    prediction_df = _saved_prediction_frame_for_artifact(artifact)
    context = {
        "artifact": artifact,
        "model_summary": model_summary,
        "metrics_json": json.dumps(artifact.metrics or {}, indent=2, sort_keys=True),
        "params_json": json.dumps(artifact.params or {}, indent=2, sort_keys=True),
        "metadata_json": json.dumps(artifact.metadata or {}, indent=2, sort_keys=True),
        "prediction_symbol_rows": _prediction_symbol_rows(prediction_df),
        "prediction_rows_count": int(len(prediction_df)),
    }
    return render(request, "ml/model_detail.html", context)


def model_artifact_symbol_predictions_view(request, artifact_id: int, symbol: str):
    artifact = get_object_or_404(ModelArtifact, pk=artifact_id)
    symbol_value = str(symbol or "").strip().upper()
    prediction_df = _saved_prediction_frame_for_artifact(artifact)
    if prediction_df.empty or "symbol" not in prediction_df.columns:
        raise Http404("Prediction rows unavailable for this model artifact.")
    prediction_df = prediction_df[prediction_df["symbol"] == symbol_value].copy()
    if prediction_df.empty:
        raise Http404("No prediction rows found for symbol.")
    prediction_rows = _prediction_detail_rows(prediction_df)
    context = {
        "artifact": artifact,
        "symbol": symbol_value,
        "prediction_rows": prediction_rows,
        **build_price_chart_context(symbol_value),
        **build_prediction_chart_context(prediction_rows),
    }
    return render(request, "ml/model_symbol_detail.html", context)
