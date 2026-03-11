from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from pipeline.cohort_runner import COHORT_SUMMARY_SCHEMA_VERSION, run_walk_forward_model_cohort_backtests
from pipeline.models import Artifact, StrategyDefinition
from .oracle_reports import summarize_prediction_artifact_set_oracle_coverage


FEATURE_ATTRIBUTION_SCHEMA_VERSION = 1


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_cached_payload(path: Path, required_keys: Sequence[str], *, schema_version: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("schema_version") or 0) != int(schema_version):
        return None
    if any(key not in payload for key in required_keys):
        return None
    return payload


def _feature_signature(feature_families: Sequence[str]) -> str:
    cleaned = [str(value).strip() for value in list(feature_families or []) if str(value).strip()]
    return " + ".join(cleaned) if cleaned else "unattributed"


def _enrich_rows_with_oracle_coverage(
    *,
    label_artifact: Artifact | None,
    summary_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    selection_quantile: float,
) -> list[dict[str, Any]]:
    if label_artifact is None:
        out = []
        for row in aggregate_rows:
            item = dict(row)
            item["feature_family_signature"] = _feature_signature(item.get("feature_families") or [])
            item["oracle_recall"] = 0.0
            item["oracle_cluster_coverage_rate"] = 0.0
            item["oracle_selected_avg_trade_return"] = 0.0
            item["oracle_selected_rows_mean"] = 0.0
            item["oracle_prediction_artifact_count"] = 0
            out.append(item)
        return out

    prediction_ids_by_variant: dict[str, list[int]] = {}
    for row in summary_rows:
        variant_name = str(row.get("variant_name") or "").strip()
        prediction_artifact_id = int(row.get("prediction_artifact_id") or 0)
        if not variant_name or prediction_artifact_id <= 0:
            continue
        ids = prediction_ids_by_variant.setdefault(variant_name, [])
        if prediction_artifact_id not in ids:
            ids.append(prediction_artifact_id)

    out: list[dict[str, Any]] = []
    for row in aggregate_rows:
        item = dict(row)
        prediction_ids = prediction_ids_by_variant.get(str(item.get("variant_name") or ""), [])
        prediction_artifacts = list(Artifact.objects.filter(id__in=prediction_ids).order_by("id")) if prediction_ids else []
        oracle = summarize_prediction_artifact_set_oracle_coverage(
            label_artifact=label_artifact,
            prediction_artifacts=prediction_artifacts,
            selection_quantile=float(selection_quantile),
        )
        item["feature_family_signature"] = _feature_signature(item.get("feature_families") or [])
        item["oracle_recall"] = float(oracle.get("oracle_recall") or 0.0)
        item["oracle_cluster_coverage_rate"] = float(oracle.get("oracle_cluster_coverage_rate") or 0.0)
        item["oracle_selected_avg_trade_return"] = float(oracle.get("oracle_selected_avg_trade_return") or 0.0)
        item["oracle_selected_rows_mean"] = float(oracle.get("oracle_selected_rows_mean") or 0.0)
        item["oracle_prediction_artifact_count"] = int(oracle.get("prediction_artifact_count") or 0)
        out.append(item)
    return out


def _build_marginal_rows(rows: list[dict[str, Any]], baseline_signature: str) -> list[dict[str, Any]]:
    baseline = next((dict(row) for row in rows if str(row.get("feature_family_signature") or "") == baseline_signature), None)
    if baseline is None and rows:
        baseline = dict(rows[0])
        baseline_signature = str(baseline.get("feature_family_signature") or baseline_signature)
    if baseline is None:
        return []

    marginal_rows: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["baseline_signature"] = baseline_signature
        item["delta_oracle_recall"] = round(float(item.get("oracle_recall") or 0.0) - float(baseline.get("oracle_recall") or 0.0), 6)
        item["delta_oracle_cluster_coverage_rate"] = round(
            float(item.get("oracle_cluster_coverage_rate") or 0.0) - float(baseline.get("oracle_cluster_coverage_rate") or 0.0),
            6,
        )
        item["delta_mean_fold_excess_cumulative_return"] = round(
            float(item.get("mean_fold_excess_cumulative_return") or 0.0) - float(baseline.get("mean_fold_excess_cumulative_return") or 0.0),
            6,
        )
        item["delta_walk_forward_excess_cumulative_return"] = round(
            float(item.get("walk_forward_excess_cumulative_return") or 0.0) - float(baseline.get("walk_forward_excess_cumulative_return") or 0.0),
            6,
        )
        marginal_rows.append(item)
    marginal_rows.sort(
        key=lambda row: (
            float(row.get("delta_oracle_cluster_coverage_rate") or 0.0),
            float(row.get("delta_oracle_recall") or 0.0),
            float(row.get("delta_mean_fold_excess_cumulative_return") or 0.0),
        ),
        reverse=True,
    )
    return marginal_rows


def run_feature_family_attribution_suite(
    *,
    symbols: Sequence[str],
    folds: Sequence[dict[str, Any]],
    fit_job: str,
    base_model_config: dict[str, Any],
    feature_family_groups: Sequence[Sequence[str]],
    feature_config: dict[str, Any] | None = None,
    transaction_cost_bps: float = 10.0,
    backtest_config: dict[str, Any] | None = None,
    selection_quantile: float = 0.8,
    baseline_families: Sequence[str] | None = None,
    universe_artifact: Artifact | None = None,
    label_artifact: Artifact | None = None,
    feature_artifact: Artifact | None = None,
    strategy_definition: StrategyDefinition | None = None,
    output_basename: str = "feature_family_attribution",
    resume_existing: bool = False,
) -> dict[str, Any]:
    output_dir = Path("data") / "pipeline_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{output_basename}.json"
    csv_path = output_dir / f"{output_basename}.csv"
    if resume_existing:
        cached_payload = _load_cached_payload(
            json_path,
            required_keys=("rows", "marginal_rows", "summary"),
            schema_version=FEATURE_ATTRIBUTION_SCHEMA_VERSION,
        )
        if cached_payload is not None:
            cached_payload["summary_json_path"] = str(json_path)
            cached_payload["summary_csv_path"] = str(csv_path)
            return cached_payload

    attribution_config = dict(base_model_config or {})
    attribution_config["feature_family_mode"] = "grouped_family"
    attribution_config["feature_family_groups"] = [list(group) for group in list(feature_family_groups or []) if list(group)]

    cohort_payload = run_walk_forward_model_cohort_backtests(
        symbols=symbols,
        fit_job=str(fit_job).strip(),
        base_model_config=attribution_config,
        folds=folds,
        universe_artifact=universe_artifact,
        label_artifact=label_artifact,
        feature_artifact=feature_artifact,
        feature_config=dict(feature_config or {}),
        strategy_definition=strategy_definition,
        strategy_definition_slug=f"{output_basename}-feature-attribution",
        strategy_definition_name=f"{output_basename} feature attribution",
        strategy_config={
            "gate_quantile": 0.55,
            "top_k": 3,
            "rebalance_freq": "W",
            "gross_exposure": 0.8,
            "selection_side": "long_only",
            "signal_combination": "mean",
        },
        transaction_cost_bps=float(transaction_cost_bps),
        backtest_config=dict(backtest_config or {}),
        output_basename=output_basename,
        resume_existing=resume_existing,
    )

    base_artifacts = dict(cohort_payload.get("base_artifacts") or {})
    resolved_label_artifact = label_artifact
    if resolved_label_artifact is None:
        resolved_label_artifact = Artifact.objects.filter(pk=int(base_artifacts.get("labels") or 0), artifact_type="LABELS").first()

    rows = _enrich_rows_with_oracle_coverage(
        label_artifact=resolved_label_artifact,
        summary_rows=[dict(row) for row in list(cohort_payload.get("summary_rows") or [])],
        aggregate_rows=[dict(row) for row in list(cohort_payload.get("aggregate_rows") or [])],
        selection_quantile=float(selection_quantile),
    )
    rows.sort(
        key=lambda row: (
            float(row.get("oracle_cluster_coverage_rate") or 0.0),
            float(row.get("oracle_recall") or 0.0),
            float(row.get("mean_fold_excess_cumulative_return") or 0.0),
        ),
        reverse=True,
    )
    baseline_signature = _feature_signature(list(baseline_families or (feature_family_groups[0] if feature_family_groups else [])))
    marginal_rows = _build_marginal_rows(rows, baseline_signature=baseline_signature)

    best_row = rows[0] if rows else {}
    summary = {
        "fit_job": str(fit_job).strip(),
        "baseline_signature": baseline_signature,
        "variant_count": int(len(rows)),
        "best_variant_name": str(best_row.get("variant_name") or ""),
        "best_feature_family_signature": str(best_row.get("feature_family_signature") or ""),
        "best_oracle_recall": float(best_row.get("oracle_recall") or 0.0),
        "best_oracle_cluster_coverage_rate": float(best_row.get("oracle_cluster_coverage_rate") or 0.0),
        "best_mean_fold_excess_cumulative_return": float(best_row.get("mean_fold_excess_cumulative_return") or 0.0),
        "selection_quantile": float(selection_quantile),
    }

    recommendations: list[str] = []
    if marginal_rows:
        top_lift = marginal_rows[0]
        if float(top_lift.get("delta_oracle_cluster_coverage_rate") or 0.0) > 0:
            recommendations.append(
                f"'{top_lift.get('feature_family_signature')}' adds the most oracle cluster coverage versus baseline '{baseline_signature}'."
            )
        if float(top_lift.get("delta_mean_fold_excess_cumulative_return") or 0.0) > 0:
            recommendations.append("The same bundle also improved walk-forward excess return. Promote it into the main research suite.")
    if rows and float(best_row.get("oracle_cluster_coverage_rate") or 0.0) < 0.25:
        recommendations.append("Even the best bundle still covers a narrow oracle subset. Add more diverse families or switch the next pass to MTL.")
    if not recommendations:
        recommendations.append("No clear attribution winner emerged. Expand the family bundles or change the label recipe.")

    payload = {
        "schema_version": FEATURE_ATTRIBUTION_SCHEMA_VERSION,
        "kind": "feature_attribution_report",
        "symbols": list(symbols),
        "folds": [dict(fold) for fold in folds],
        "fit_job": str(fit_job).strip(),
        "feature_family_groups": [list(group) for group in list(feature_family_groups or [])],
        "base_artifacts": base_artifacts,
        "cohort_summary_path": str(cohort_payload.get("summary_json_path") or ""),
        "selection_quantile": float(selection_quantile),
        "rows": rows,
        "marginal_rows": marginal_rows,
        "summary": summary,
        "recommendations": recommendations,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_rows_csv(csv_path, rows)
    payload["summary_json_path"] = str(json_path)
    payload["summary_csv_path"] = str(csv_path)
    return payload
