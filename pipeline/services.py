from __future__ import annotations

import json
import time
import uuid
from typing import Any

import pandas as pd
from django.db import transaction
from django.utils import timezone

from ml.execution import load_artifact_csv_frame
from ml.rl import RLConfig, run_a2c_workflow, run_ppo_workflow

from . import service_runtime as runtime
from .contracts import STATE_PANEL_ARTIFACT_TYPES
from .models import Artifact, JobRun, PipelineRun
from .service_jobs_data import execute_features, execute_labels, execute_universe
from .service_jobs_modeling import (
    execute_fit_model,
    execute_predict,
    execute_score_model,
    execute_train,
)
from .service_jobs_strategy import execute_backtest_strategy, execute_build_strategy_dataset

PipelineExecutionError = runtime.PipelineExecutionError
BuiltOutput = runtime.BuiltOutput
ARTIFACT_DIR = runtime.ARTIFACT_DIR
_stable_payload_hash = runtime.stable_payload_hash

JOB_TYPES = (
    "universe",
    "labels",
    "features",
    "train",
    "predict",
    "fit_classifier",
    "fit_regressor",
    "fit_autoencoder",
    "fit_mtl",
    "score_classifier",
    "score_regressor",
    "score_autoencoder",
    "score_mtl",
    "build_strategy_dataset",
    "backtest_strategy",
    "rl_train",
)

UI_JOB_TYPES = (
    "universe",
    "labels",
    "features",
    "fit_classifier",
    "fit_regressor",
    "fit_mtl",
    "score_classifier",
    "score_regressor",
    "score_mtl",
    "build_strategy_dataset",
    "backtest_strategy",
)

JOB_OUTPUT_ARTIFACT: dict[str, str] = {
    "universe": "UNIVERSE",
    "labels": "LABELS",
    "features": "FEATURES",
    "train": "MODEL",
    "predict": "PREDICTIONS",
    "fit_classifier": "CLASSIFIER_MODEL",
    "fit_regressor": "REGRESSOR_MODEL",
    "fit_autoencoder": "AUTOENCODER_MODEL",
    "fit_mtl": "MTL_MODEL",
    "score_classifier": "CLASSIFIER_PREDICTIONS",
    "score_regressor": "REGRESSOR_PREDICTIONS",
    "score_autoencoder": "AUTOENCODER_SCORES",
    "score_mtl": "MTL_PREDICTIONS",
    "build_strategy_dataset": "STRATEGY_DATASET",
    "backtest_strategy": "BACKTEST_RESULT",
    "rl_train": "RL_POLICY_RESULT",
}

JOB_REQUIRED_INPUTS: dict[str, list[str]] = {
    "universe": [],
    "labels": ["UNIVERSE"],
    "features": ["UNIVERSE"],
    "train": ["FEATURES", "LABELS"],
    "predict": ["MODEL", "FEATURES"],
    "fit_classifier": ["FEATURES", "LABELS"],
    "fit_regressor": ["FEATURES", "LABELS"],
    "fit_autoencoder": ["FEATURES", "LABELS"],
    "fit_mtl": ["FEATURES", "LABELS"],
    "score_classifier": ["CLASSIFIER_MODEL", "FEATURES"],
    "score_regressor": ["REGRESSOR_MODEL", "FEATURES"],
    "score_autoencoder": ["AUTOENCODER_MODEL", "FEATURES"],
    "score_mtl": ["MTL_MODEL", "FEATURES"],
    "build_strategy_dataset": ["FEATURES"],
    "backtest_strategy": ["STRATEGY_DATASET"],
    "rl_train": ["STRATEGY_DATASET"],
}

ARTIFACT_PRODUCER_JOB = {value: key for key, value in JOB_OUTPUT_ARTIFACT.items()}


def _sync_runtime_artifact_dir() -> None:
    runtime.ARTIFACT_DIR = ARTIFACT_DIR


def _ensure_supported_job(job_type: str) -> str:
    value = str(job_type or "").strip().lower()
    if value not in JOB_TYPES:
        raise PipelineExecutionError(f"Unsupported job type: {job_type!r}")
    return value


def _execute_rl_train(config: dict[str, Any], strategy_dataset_artifact: Artifact) -> BuiltOutput:
    started = time.perf_counter()
    strategy_df = load_artifact_csv_frame(strategy_dataset_artifact)
    if strategy_df.empty:
        raise PipelineExecutionError("Strategy dataset is empty.")

    required_cols = {"date", "symbol", "prob_buy", "ae_familiarity", "close"}
    ranking_col = "ranking" if "ranking" in strategy_df.columns else ("pred_rf_reg" if "pred_rf_reg" in strategy_df.columns else "")
    if ranking_col:
        required_cols.add(ranking_col)
    missing = [col for col in required_cols if col not in strategy_df.columns]
    if missing:
        raise PipelineExecutionError(f"RL opportunity-state training requires strategy columns: {', '.join(sorted(missing))}.")

    panel = strategy_df[["date", "symbol", "prob_buy", ranking_col, "ae_familiarity", "close"]].copy()
    panel = panel.rename(columns={ranking_col: "pred_rf_reg"})
    for col in ["prob_buy", "pred_rf_reg", "ae_familiarity", "close"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel = panel.dropna(subset=["date", "symbol", "prob_buy", "pred_rf_reg", "ae_familiarity", "close"]).copy()
    if panel.empty:
        raise PipelineExecutionError("No usable opportunity-state rows were available for RL training.")
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"]).sort_values(["date", "symbol"]).set_index(["date", "symbol"])

    train_split_date = str(config.get("train_split_date") or "").strip()
    if not train_split_date:
        max_date = pd.Timestamp(panel.index.get_level_values("date").max())
        inferred_year = max(int(max_date.year) - 1, 2000)
        train_split_date = f"{inferred_year}-12-31"
    eval_years_raw = list(config.get("eval_years") or [])
    eval_years: list[int] = []
    for value in eval_years_raw:
        try:
            parsed = int(value)
        except Exception:
            continue
        if parsed not in eval_years:
            eval_years.append(parsed)
    if not eval_years:
        split_ts = pd.Timestamp(train_split_date)
        eval_years = sorted({int(ts.year) for ts in panel.index.get_level_values("date") if pd.Timestamp(ts) > split_ts})
    if not eval_years:
        raise PipelineExecutionError("RL training requires at least one evaluation year after train_split_date.")

    algorithm = str(config.get("algorithm") or "ppo").strip().lower()
    rl_cfg = RLConfig(
        lookback_window=max(5, int(config.get("lookback_window") or 20)),
        eligibility_quantile=float(config.get("eligibility_quantile") or 0.5),
        rebalance_freq=str(config.get("rebalance_freq") or "W").strip() or "W",
        max_stocks_per_day=(int(config.get("max_stocks_per_day")) if config.get("max_stocks_per_day") not in (None, "") else 5),
        initial_balance=float(config.get("initial_balance") or 100000.0),
        fee_bps=float(config.get("fee_bps") or 5.0),
        slippage_bps=float(config.get("slippage_bps") or 5.0),
        ppo_episodes=max(1, int(config.get("episodes") or config.get("ppo_episodes") or 20)),
        drawdown_penalty_lambda=float(config.get("drawdown_penalty_lambda") or 0.10),
        seed=int(config.get("seed") or 42),
    )
    runner = run_ppo_workflow if algorithm == "ppo" else run_a2c_workflow
    try:
        result = runner(
            bt_panel=panel,
            cfg=rl_cfg,
            train_split_date=pd.Timestamp(train_split_date),
            years=eval_years,
        )
    except Exception as exc:
        raise PipelineExecutionError(str(exc)) from exc

    summary_df = result["rl_summary_df"].copy()
    yearly_df = result["rl_yearly_df"].copy()
    trade_log = result.get("trade_log", pd.DataFrame())

    def _json_safe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return json.loads(json.dumps(rows, default=str))

    summary_rows = _json_safe_rows(summary_df.to_dict(orient="records"))
    yearly_rows = _json_safe_rows(yearly_df.to_dict(orient="records"))
    trade_log_preview = _json_safe_rows(trade_log.head(100).to_dict(orient="records")) if isinstance(trade_log, pd.DataFrame) else []

    def _normalize_action_counts(value: Any) -> dict[str, int]:
        if isinstance(value, pd.Series):
            raw = value.to_dict()
        elif isinstance(value, dict):
            raw = dict(value)
        else:
            try:
                raw = dict(value or {})
            except Exception:
                raw = {}
        out: dict[str, int] = {}
        for key, item in raw.items():
            try:
                out[str(key)] = int(item)
            except Exception:
                continue
        return out

    key = f"rl_policy_{uuid.uuid4().hex}"
    payload = {
        "created_at": timezone.now().isoformat(),
        "algorithm": algorithm,
        "train_split_date": train_split_date,
        "eval_years": eval_years,
        "summary_rows": summary_rows,
        "yearly_rows": yearly_rows,
        "executed_action_counts": _normalize_action_counts(result.get("executed_action_counts")),
        "action_counts": _normalize_action_counts(result.get("action_counts")),
        "trade_log_preview": trade_log_preview,
    }
    uri = runtime.write_json(key, payload)
    best_summary = summary_rows[0] if summary_rows else {}
    duration = round(float(time.perf_counter() - started), 6)
    return BuiltOutput(
        artifact_type="RL_POLICY_RESULT",
        content={
            "algorithm": algorithm,
            "train_split_date": train_split_date,
            "eval_years": eval_years,
            "combined_total_return_pct": float(best_summary.get("combined_total_return_pct") or 0.0),
            "combined_sharpe": float(best_summary.get("combined_sharpe") or 0.0),
            "combined_max_drawdown_pct": float(best_summary.get("combined_max_drawdown_pct") or 0.0),
            "executed_buys": int((payload.get("executed_action_counts") or {}).get("buy") or 0),
            "executed_sells": int((payload.get("executed_action_counts") or {}).get("sell") or 0),
            "rl_seconds": duration,
        },
        metadata={
            "source_strategy_dataset_artifact_id": int(strategy_dataset_artifact.id),
            "rl_config": {
                "algorithm": algorithm,
                "lookback_window": int(rl_cfg.lookback_window),
                "eligibility_quantile": float(rl_cfg.eligibility_quantile),
                "rebalance_freq": str(rl_cfg.rebalance_freq),
                "max_stocks_per_day": int(rl_cfg.max_stocks_per_day or 0),
                "initial_balance": float(rl_cfg.initial_balance),
                "fee_bps": float(rl_cfg.fee_bps),
                "slippage_bps": float(rl_cfg.slippage_bps),
                "ppo_episodes": int(rl_cfg.ppo_episodes),
                "drawdown_penalty_lambda": float(rl_cfg.drawdown_penalty_lambda),
                "seed": int(rl_cfg.seed),
            },
            "rl_seconds": duration,
        },
        uri=uri,
    )


def _run_job_executor(
    job_type: str,
    config: dict[str, Any],
    inputs_by_type: dict[str, Artifact],
    *,
    pipeline_run: PipelineRun | None = None,
    job_run: JobRun | None = None,
    performance_tracer=None,
) -> BuiltOutput:
    _sync_runtime_artifact_dir()
    if job_type == "universe":
        return execute_universe(config, performance_tracer=performance_tracer)
    if job_type == "labels":
        return execute_labels(
            config,
            inputs_by_type["UNIVERSE"],
            pipeline_run=pipeline_run,
            job_run=job_run,
            performance_tracer=performance_tracer,
        )
    if job_type == "features":
        return execute_features(
            config,
            inputs_by_type["UNIVERSE"],
            pipeline_run=pipeline_run,
            job_run=job_run,
            performance_tracer=performance_tracer,
        )
    if job_type == "train":
        return execute_train(
            config,
            inputs_by_type["LABELS"],
            inputs_by_type["FEATURES"],
            pipeline_run=pipeline_run,
            job_run=job_run,
            performance_tracer=performance_tracer,
        )
    if job_type == "predict":
        return execute_predict(
            config,
            inputs_by_type["MODEL"],
            inputs_by_type["FEATURES"],
            pipeline_run=pipeline_run,
            job_run=job_run,
            performance_tracer=performance_tracer,
        )
    if job_type == "fit_classifier":
        return execute_fit_model(
            config,
            inputs_by_type["LABELS"],
            inputs_by_type["FEATURES"],
            algorithm="random_forest_classifier",
            task_type="classification",
            artifact_type="CLASSIFIER_MODEL",
            pipeline_run=pipeline_run,
            job_run=job_run,
            performance_tracer=performance_tracer,
        )
    if job_type == "fit_regressor":
        return execute_fit_model(
            config,
            inputs_by_type["LABELS"],
            inputs_by_type["FEATURES"],
            algorithm="random_forest_regressor",
            task_type="regression",
            artifact_type="REGRESSOR_MODEL",
            pipeline_run=pipeline_run,
            job_run=job_run,
            performance_tracer=performance_tracer,
        )
    if job_type == "fit_autoencoder":
        return execute_fit_model(
            config,
            inputs_by_type["LABELS"],
            inputs_by_type["FEATURES"],
            algorithm="autoencoder",
            task_type="reconstruction",
            artifact_type="AUTOENCODER_MODEL",
            pipeline_run=pipeline_run,
            job_run=job_run,
            performance_tracer=performance_tracer,
        )
    if job_type == "fit_mtl":
        return execute_fit_model(
            config,
            inputs_by_type["LABELS"],
            inputs_by_type["FEATURES"],
            algorithm="multi_task_forest",
            task_type="multitask",
            artifact_type="MTL_MODEL",
            pipeline_run=pipeline_run,
            job_run=job_run,
            performance_tracer=performance_tracer,
        )
    if job_type == "score_classifier":
        return execute_score_model(
            config,
            inputs_by_type["CLASSIFIER_MODEL"],
            inputs_by_type["FEATURES"],
            expected_pipeline_artifact_type="CLASSIFIER_MODEL",
            output_artifact_type="CLASSIFIER_PREDICTIONS",
            pipeline_run=pipeline_run,
            job_run=job_run,
            performance_tracer=performance_tracer,
        )
    if job_type == "score_regressor":
        return execute_score_model(
            config,
            inputs_by_type["REGRESSOR_MODEL"],
            inputs_by_type["FEATURES"],
            expected_pipeline_artifact_type="REGRESSOR_MODEL",
            output_artifact_type="REGRESSOR_PREDICTIONS",
            pipeline_run=pipeline_run,
            job_run=job_run,
            performance_tracer=performance_tracer,
        )
    if job_type == "score_autoencoder":
        return execute_score_model(
            config,
            inputs_by_type["AUTOENCODER_MODEL"],
            inputs_by_type["FEATURES"],
            expected_pipeline_artifact_type="AUTOENCODER_MODEL",
            output_artifact_type="AUTOENCODER_SCORES",
            pipeline_run=pipeline_run,
            job_run=job_run,
            performance_tracer=performance_tracer,
        )
    if job_type == "score_mtl":
        return execute_score_model(
            config,
            inputs_by_type["MTL_MODEL"],
            inputs_by_type["FEATURES"],
            expected_pipeline_artifact_type="MTL_MODEL",
            output_artifact_type="MTL_PREDICTIONS",
            pipeline_run=pipeline_run,
            job_run=job_run,
            performance_tracer=performance_tracer,
        )
    if job_type == "build_strategy_dataset":
        return execute_build_strategy_dataset(
            config,
            inputs_by_type["FEATURES"],
            pipeline_run=pipeline_run,
            job_run=job_run,
            performance_tracer=performance_tracer,
        )
    if job_type == "backtest_strategy":
        return execute_backtest_strategy(
            config,
            inputs_by_type["STRATEGY_DATASET"],
            pipeline_run=pipeline_run,
            job_run=job_run,
            performance_tracer=performance_tracer,
        )
    if job_type == "rl_train":
        return _execute_rl_train(config, inputs_by_type["STRATEGY_DATASET"])
    raise PipelineExecutionError(f"Unsupported job type: {job_type!r}")


def _latest_artifact_by_type(artifact_type: str) -> Artifact | None:
    return Artifact.objects.filter(artifact_type=artifact_type).order_by("-created_at", "-id").first()


def execute_pipeline_run(
    *,
    pipeline_run: PipelineRun,
    target_job: str,
    mode: str,
    config: dict[str, Any] | None = None,
    input_artifact_ids: list[int] | None = None,
    performance_tracer=None,
) -> Artifact:
    _sync_runtime_artifact_dir()
    target_job = _ensure_supported_job(target_job)
    mode_value = str(mode or PipelineRun.Mode.STRICT).strip().lower()
    if mode_value not in {PipelineRun.Mode.STRICT, PipelineRun.Mode.AUTO_BUILD_MISSING}:
        raise PipelineExecutionError(f"Unsupported run mode: {mode!r}")
    cfg = dict(config or {})

    provided_ids = [int(v) for v in list(input_artifact_ids or [])]
    provided_artifacts = list(Artifact.objects.filter(id__in=provided_ids))

    artifacts_by_type: dict[str, Artifact] = {}
    for artifact in provided_artifacts:
        artifacts_by_type[artifact.artifact_type] = artifact

    executed_jobs: dict[str, Artifact] = {}

    with transaction.atomic():
        pipeline_run.status = PipelineRun.Status.RUNNING
        if pipeline_run.started_at is None:
            pipeline_run.started_at = timezone.now()
        pipeline_run.error = ""
        pipeline_run.config = cfg
        pipeline_run.save(update_fields=["status", "started_at", "error", "config", "updated_at"])

    def ensure_job(job_type: str) -> Artifact:
        if job_type in executed_jobs:
            return executed_jobs[job_type]

        required_types = list(JOB_REQUIRED_INPUTS.get(job_type, []))
        resolved_inputs: dict[str, Artifact] = {}

        for artifact_type in required_types:
            existing = artifacts_by_type.get(artifact_type)
            if existing is not None:
                resolved_inputs[artifact_type] = existing
                continue

            if mode_value == PipelineRun.Mode.AUTO_BUILD_MISSING:
                reuse_existing = bool(cfg.get("reuse_existing_artifacts", True))
                if reuse_existing:
                    latest = _latest_artifact_by_type(artifact_type)
                    if latest is not None:
                        artifacts_by_type[artifact_type] = latest
                        resolved_inputs[artifact_type] = latest
                        continue

                upstream_job = ARTIFACT_PRODUCER_JOB.get(artifact_type)
                if not upstream_job:
                    raise PipelineExecutionError(
                        f"No upstream producer declared for required artifact type {artifact_type!r}."
                    )
                upstream_artifact = ensure_job(upstream_job)
                artifacts_by_type[artifact_type] = upstream_artifact
                resolved_inputs[artifact_type] = upstream_artifact
                continue

            raise PipelineExecutionError(
                f"Missing required input artifact type {artifact_type!r} for job {job_type!r} in STRICT mode."
            )

        job_run = JobRun.objects.create(
            pipeline_run=pipeline_run,
            job_type=job_type,
            status=JobRun.Status.RUNNING,
            config=cfg,
            started_at=timezone.now(),
        )
        if resolved_inputs:
            job_run.input_artifacts.set(list(resolved_inputs.values()))

        try:
            built = _run_job_executor(
                job_type,
                cfg,
                resolved_inputs,
                pipeline_run=pipeline_run,
                job_run=job_run,
                performance_tracer=performance_tracer,
            )
            safe_content = runtime.json_safe_value(built.content)
            safe_metadata = runtime.json_safe_value(built.metadata)
            payload_hash = runtime.artifact_payload_hash(safe_content, built.uri)
            artifact_stage = (
                performance_tracer.stage(
                    "pipeline.create_artifact_record",
                    category="artifact_creation",
                    workload_type="batched",
                    metadata={"artifact_type": built.artifact_type},
                )
                if performance_tracer is not None
                else None
            )
            if artifact_stage is None:
                artifact = Artifact.objects.create(
                    pipeline_run=pipeline_run,
                    producer_job=job_run,
                    artifact_type=built.artifact_type,
                    key=uuid.uuid4().hex,
                    uri=built.uri,
                    content=safe_content,
                    metadata=safe_metadata,
                    payload_hash=payload_hash,
                )
            else:
                with artifact_stage:
                    artifact = Artifact.objects.create(
                        pipeline_run=pipeline_run,
                        producer_job=job_run,
                        artifact_type=built.artifact_type,
                        key=uuid.uuid4().hex,
                        uri=built.uri,
                        content=safe_content,
                        metadata=safe_metadata,
                        payload_hash=payload_hash,
                    )
            job_run.status = JobRun.Status.SUCCEEDED
            job_run.finished_at = timezone.now()
            job_run.save(update_fields=["status", "finished_at", "updated_at"])
            duration_seconds = None
            if job_run.started_at and job_run.finished_at:
                try:
                    duration_seconds = round(float((job_run.finished_at - job_run.started_at).total_seconds()), 6)
                except Exception:
                    duration_seconds = None
            artifact.content = dict(artifact.content or {})
            artifact.metadata = dict(artifact.metadata or {})
            if duration_seconds is not None:
                artifact.content["job_duration_seconds"] = duration_seconds
                artifact.metadata["job_duration_seconds"] = duration_seconds
                artifact.save(update_fields=["content", "metadata"])
        except Exception as exc:
            job_run.status = JobRun.Status.FAILED
            job_run.finished_at = timezone.now()
            job_run.error = str(exc)
            job_run.save(update_fields=["status", "finished_at", "error", "updated_at"])
            raise

        artifacts_by_type[artifact.artifact_type] = artifact
        executed_jobs[job_type] = artifact
        return artifact

    try:
        final_artifact = ensure_job(target_job)
    except Exception as exc:
        with transaction.atomic():
            pipeline_run.status = PipelineRun.Status.FAILED
            pipeline_run.finished_at = timezone.now()
            pipeline_run.error = str(exc)
            pipeline_run.save(update_fields=["status", "finished_at", "error", "updated_at"])
        raise

    with transaction.atomic():
        pipeline_run.status = PipelineRun.Status.SUCCEEDED
        pipeline_run.finished_at = timezone.now()
        pipeline_run.save(update_fields=["status", "finished_at", "updated_at"])

    return final_artifact


__all__ = [
    "ARTIFACT_DIR",
    "ARTIFACT_PRODUCER_JOB",
    "BuiltOutput",
    "JOB_OUTPUT_ARTIFACT",
    "JOB_REQUIRED_INPUTS",
    "JOB_TYPES",
    "PipelineExecutionError",
    "UI_JOB_TYPES",
    "_stable_payload_hash",
    "execute_pipeline_run",
]
