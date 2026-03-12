from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cohort_runner import _build_equal_weight_benchmark, _evaluate_variant_gates, _run_pipeline_job
from .direct_strategy_runner import DIRECT_STRATEGY_SUMMARY_SCHEMA_VERSION
from .factor_analysis import summarize_return_frame
from .models import Artifact, StrategyDefinition
from .strategy_definitions import upsert_strategy_definition


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_prediction_artifact_strategy_backtest(
    *,
    feature_artifact: Artifact,
    prediction_artifact: Artifact,
    train_end_date: str,
    backtest_start_date: str,
    backtest_end_date: str,
    strategy_definition: StrategyDefinition | None = None,
    strategy_definition_slug: str,
    strategy_definition_name: str,
    strategy_config: Mapping[str, Any],
    label_artifact: Artifact | None = None,
    validation_config: Mapping[str, Any] | None = None,
    backtest_config: Mapping[str, Any] | None = None,
    output_basename: str,
) -> dict[str, Any]:
    output_dir = Path("data") / "pipeline_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{output_basename}.json"
    csv_path = output_dir / f"{output_basename}.csv"

    resolved_strategy_definition = strategy_definition or upsert_strategy_definition(
        slug=str(strategy_definition_slug),
        name=str(strategy_definition_name),
        strategy_type="notebook_topk_v1",
        description="Strategy used by the prediction artifact runner.",
        config=dict(strategy_config or {}),
    )
    strategy_artifact = _run_pipeline_job(
        name=f"{output_basename}-strategy",
        requested_job="build_strategy_dataset",
        config={
            "strategy_definition_id": int(resolved_strategy_definition.id),
            "label_artifact_id": int(label_artifact.id) if label_artifact is not None else 0,
            "prediction_artifact_ids": [int(prediction_artifact.id)],
            "strategy_start_date": str(backtest_start_date or ""),
            "strategy_end_date": str(backtest_end_date or ""),
        },
        input_ids=[int(feature_artifact.id)],
    )
    backtest_artifact = _run_pipeline_job(
        name=f"{output_basename}-backtest",
        requested_job="backtest_strategy",
        config={
            "backtest_start_date": str(backtest_start_date or ""),
            "backtest_end_date": str(backtest_end_date or ""),
            **dict(backtest_config or {}),
        },
        input_ids=[int(strategy_artifact.id)],
    )

    backtest_content = dict(backtest_artifact.content or {})
    backtest_meta = dict(backtest_artifact.metadata or {})
    strategy_meta = dict(strategy_artifact.metadata or {})
    daily_rows = list(backtest_content.get("daily_rows") or [])
    return_summary = summarize_return_frame(
        daily_rows,
        series_name=str(strategy_definition_slug),
        series_kind="strategy",
    )
    benchmark = _build_equal_weight_benchmark(
        strategy_artifact,
        allowed_symbols=dict(backtest_config or {}).get("allowed_symbols"),
    )
    row = {
        "variant_name": str(strategy_definition_slug),
        "fit_job": "custom_prediction_artifact",
        "score_job": "custom_prediction_artifact",
        "feature_families": [],
        "label_ks": [],
        "dataset_build_seconds": 0.0,
        "fit_seconds": 0.0,
        "score_seconds": float(dict(prediction_artifact.metadata or {}).get("score_seconds") or 0.0),
        "strategy_build_seconds": float(strategy_meta.get("strategy_build_seconds") or 0.0),
        "backtest_seconds": float(backtest_meta.get("backtest_seconds") or 0.0),
        "coverage_start_date": str((feature_artifact.metadata or {}).get("feature_start_date") or ""),
        "coverage_end_date": str((feature_artifact.metadata or {}).get("feature_end_date") or ""),
        "coverage_rows": int((feature_artifact.content or {}).get("rows") or 0),
        "oracle_cluster_scope": "generalist",
        "oracle_cluster_keys": [],
        "oracle_cluster_rows": 0,
        "trained_rows": int(dict(prediction_artifact.content or {}).get("trained_rows") or 0),
        "rows_scored": int(dict(prediction_artifact.content or {}).get("rows") or 0),
        "selected_rows": int((strategy_artifact.content or {}).get("selected_rows") or 0),
        "final_equity": float(backtest_content.get("final_equity") or 0.0),
        "cumulative_return": float(backtest_content.get("cumulative_return") or 0.0),
        "max_drawdown": float(backtest_content.get("max_drawdown") or 0.0),
        "trades": int(backtest_content.get("trades") or 0),
        "sharpe": float(return_summary.get("sharpe") or 0.0),
        "avg_turnover": float(return_summary.get("avg_turnover") or 0.0),
        "total_turnover": float(return_summary.get("total_turnover") or 0.0),
        "positive_days": int(return_summary.get("positive_days") or 0),
        "negative_days": int(return_summary.get("negative_days") or 0),
        "benchmark_days": int(benchmark.get("benchmark_days") or 0),
        "benchmark_final_equity": float(benchmark.get("benchmark_final_equity") or 0.0),
        "benchmark_cumulative_return": float(benchmark.get("benchmark_cumulative_return") or 0.0),
        "benchmark_max_drawdown": float(benchmark.get("benchmark_max_drawdown") or 0.0),
        "backtest_fee_bps": float((backtest_meta.get("backtest_config") or {}).get("fee_bps") or dict(backtest_config or {}).get("fee_bps") or 0.0),
        "backtest_slippage_bps": float((backtest_meta.get("backtest_config") or {}).get("slippage_bps") or dict(backtest_config or {}).get("slippage_bps") or 0.0),
        "excess_cumulative_return": round(
            float(backtest_content.get("cumulative_return") or 0.0) - float(benchmark.get("benchmark_cumulative_return") or 0.0),
            8,
        ),
        "relative_final_equity": round(
            float(backtest_content.get("final_equity") or 0.0) - float(benchmark.get("benchmark_final_equity") or 0.0),
            8,
        ),
        "model_artifact_id": 0,
        "prediction_artifact_id": int(prediction_artifact.id),
        "strategy_artifact_id": int(strategy_artifact.id),
        "backtest_artifact_id": int(backtest_artifact.id),
    }
    row["total_runtime_seconds"] = round(
        float(row["dataset_build_seconds"])
        + float(row["fit_seconds"])
        + float(row["score_seconds"])
        + float(row["strategy_build_seconds"])
        + float(row["backtest_seconds"]),
        6,
    )
    row.update(_evaluate_variant_gates(row, validation_config=validation_config))

    payload = {
        "schema_version": DIRECT_STRATEGY_SUMMARY_SCHEMA_VERSION,
        "mode": "prediction_artifact_strategy",
        "base_artifacts": {
            "features": int(feature_artifact.id),
            "predictions": int(prediction_artifact.id),
            "labels": int(label_artifact.id) if label_artifact is not None else 0,
        },
        "strategy_definition": {
            "id": int(resolved_strategy_definition.id),
            "name": str(resolved_strategy_definition.name),
            "slug": str(resolved_strategy_definition.slug),
            "strategy_type": str(resolved_strategy_definition.strategy_type),
        },
        "validation_config": dict(validation_config or {}),
        "backtest_config": dict(backtest_config or {}),
        "summary_rows": [row],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_rows_csv(csv_path, [row])
    payload["summary_json_path"] = str(json_path)
    payload["summary_csv_path"] = str(csv_path)
    return payload


__all__ = [
    "run_prediction_artifact_strategy_backtest",
]
