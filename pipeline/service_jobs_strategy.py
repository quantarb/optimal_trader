from __future__ import annotations

import time
import uuid
from typing import Any

import pandas as pd

from domain.backtests import StrategyBacktestSpec, StrategyDatasetSpec
from workflows.strategy import build_strategy_dataset_frame, run_strategy_backtest

from .contracts import (
    BACKTEST_REQUIRED_COLUMNS,
    STRATEGY_REQUIRED_COLUMNS,
    build_equity_curve_from_daily_rows,
    build_schema_metadata,
    validate_frame_columns,
)
from .service_runtime import (
    BuiltOutput,
    write_frame_artifact,
)
from .progress import ProgressReporter
 

def _artifact_storage_format(config: dict[str, Any]) -> str:
    return str(config.get("artifact_storage_format") or "csv").strip().lower() or "csv"


def _write_frame_output(
    *,
    key_prefix: str,
    frame: pd.DataFrame,
    fieldnames: list[str],
    storage_format: str,
    performance_tracer=None,
    stage_name: str,
    metadata: dict[str, Any] | None = None,
):
    key = f"{key_prefix}_{uuid.uuid4().hex}"
    if performance_tracer is not None:
        with performance_tracer.stage(
            stage_name,
            category="serialization",
            workload_type="batched",
            metadata=metadata or {"rows": int(len(frame)), "storage_format": storage_format},
        ):
            return write_frame_artifact(
                key,
                frame=frame,
                fieldnames=fieldnames,
                storage_format=storage_format,
            )
    return write_frame_artifact(
        key,
        frame=frame,
        fieldnames=fieldnames,
        storage_format=storage_format,
    )


def execute_build_strategy_dataset(
    config: dict[str, Any],
    features_artifact,
    *,
    pipeline_run=None,
    job_run=None,
    performance_tracer=None,
) -> BuiltOutput:
    progress = ProgressReporter(pipeline_run=pipeline_run, job_run=job_run)
    build_started = time.perf_counter()
    progress.update(
        phase="load_strategy_inputs",
        phase_label="Load strategy inputs",
        phase_index=1,
        phase_total=3,
        force=True,
    )
    result = build_strategy_dataset_frame(
        spec=StrategyDatasetSpec.from_mapping(config),
        features_artifact=features_artifact,
        performance_tracer=performance_tracer,
    )
    feature_df = result.frame
    progress.update(
        phase="compute_strategy_signals",
        phase_label="Compute strategy signals",
        phase_index=2,
        phase_total=3,
        message=f"{len(feature_df):,} rows",
        force=True,
    )
    validate_frame_columns(feature_df, STRATEGY_REQUIRED_COLUMNS, artifact_type="STRATEGY_DATASET")
    progress.update(
        phase="write_strategy_dataset",
        phase_label="Write strategy dataset",
        phase_index=3,
        phase_total=3,
        total_units=1,
        completed_units=0,
        force=True,
    )
    storage_format = _artifact_storage_format(config)
    stored = _write_frame_output(
        key_prefix="strategy_dataset",
        frame=feature_df,
        fieldnames=list(feature_df.columns),
        storage_format=storage_format,
        performance_tracer=performance_tracer,
        stage_name="strategy.serialize_dataset",
        metadata={"rows": int(len(feature_df)), "storage_format": storage_format},
    )
    duration = round(float(time.perf_counter() - build_started), 6)
    progress.update(
        phase="write_strategy_dataset",
        phase_label="Write strategy dataset",
        phase_index=3,
        phase_total=3,
        total_units=1,
        completed_units=1,
        message="Completed",
        force=True,
    )
    return BuiltOutput(
        artifact_type="STRATEGY_DATASET",
        content={
            "rows": int(len(feature_df)),
            "symbols": int(feature_df["symbol"].nunique()) if "symbol" in feature_df.columns else 0,
            "feature_cols": list(result.feature_cols),
            "selected_rows": int((feature_df["strategy_signal"] != 0).sum()) if "strategy_signal" in feature_df.columns else 0,
            "dates": int(feature_df["date"].nunique()) if "date" in feature_df.columns else 0,
            "avg_daily_positions": (
                round(float(sum(result.daily_position_counts.values()) / len(result.daily_position_counts)), 4)
                if result.daily_position_counts
                else 0.0
            ),
            "strategy_build_seconds": duration,
        },
        metadata={
            "source_features_artifact_id": int(features_artifact.id),
            "source_prediction_artifact_ids": list(result.source_prediction_artifact_ids),
            "source_label_artifact_id": int(result.source_label_artifact_id),
            "extra_panel_sources": list(result.panel_meta.get("extra_panel_sources") or []),
            "strategy_definition_id": int(result.strategy_definition.definition_id),
            "strategy_definition_name": str(result.strategy_definition.name),
            "strategy_definition_slug": str(result.strategy_definition.slug),
            "strategy_type": str(result.strategy_definition.strategy_type),
            "strategy_config": dict(result.strategy_meta.get("strategy_config") or result.strategy_definition.config),
            "score_logic": dict(result.score_meta),
            "strategy_definition": {
                "id": int(result.strategy_definition.definition_id),
                "name": str(result.strategy_definition.name),
                "slug": str(result.strategy_definition.slug),
                "strategy_type": str(result.strategy_definition.strategy_type),
                "config": dict(result.strategy_definition.config),
            },
            "daily_position_counts": dict(result.daily_position_counts),
            "daily_gross_exposure": dict(result.daily_gross_exposure),
            "strategy_start_date": str(result.start_date),
            "strategy_end_date": str(result.end_date),
            "strategy_build_seconds": duration,
            "schema": build_schema_metadata(
                artifact_type="STRATEGY_DATASET",
                required_columns=STRATEGY_REQUIRED_COLUMNS,
                actual_columns=list(feature_df.columns),
            ),
            **stored.storage_metadata(),
        },
        uri=stored.uri,
    )


def execute_backtest_strategy(
    config: dict[str, Any],
    strategy_dataset_artifact,
    *,
    pipeline_run=None,
    job_run=None,
    performance_tracer=None,
) -> BuiltOutput:
    progress = ProgressReporter(pipeline_run=pipeline_run, job_run=job_run)
    backtest_started = time.perf_counter()
    progress.update(
        phase="load_backtest_inputs",
        phase_label="Load backtest inputs",
        phase_index=1,
        phase_total=3,
        force=True,
    )
    spec = StrategyBacktestSpec.from_mapping(config)
    result = run_strategy_backtest(
        spec=spec,
        strategy_dataset_artifact=strategy_dataset_artifact,
        performance_tracer=performance_tracer,
    )
    progress.update(
        phase="compute_backtest",
        phase_label="Compute backtest",
        phase_index=2,
        phase_total=3,
        message=f"{len(result.daily_rows):,} dates | {int(result.trade_frame['symbol'].nunique()) if not result.trade_frame.empty else 0:,} symbols",
        force=True,
    )
    validate_frame_columns(result.trade_frame, BACKTEST_REQUIRED_COLUMNS, artifact_type="BACKTEST_RESULT")
    progress.update(
        phase="write_backtest_output",
        phase_label="Write backtest output",
        phase_index=3,
        phase_total=3,
        total_units=1,
        completed_units=0,
        force=True,
    )
    storage_format = _artifact_storage_format(config)
    stored = _write_frame_output(
        key_prefix="backtest",
        frame=result.trade_frame,
        fieldnames=[
            "date",
            "symbol",
            "strategy_signal",
            "strategy_score",
            "target_weight",
            "effective_weight",
            "asset_return",
            "gross_exposure",
            "realized_return",
            "turnover",
            "turnover_cost",
        ],
        storage_format=storage_format,
        performance_tracer=performance_tracer,
        stage_name="backtest.serialize_artifact",
        metadata={"rows": int(len(result.trade_frame)), "storage_format": storage_format},
    )
    duration = round(float(time.perf_counter() - backtest_started), 6)
    progress.update(
        phase="write_backtest_output",
        phase_label="Write backtest output",
        phase_index=3,
        phase_total=3,
        total_units=1,
        completed_units=1,
        message="Completed",
        force=True,
    )
    return BuiltOutput(
        artifact_type="BACKTEST_RESULT",
        content={
            "trades": int(result.trades),
            "wins": int(result.wins),
            "losses": int(result.losses),
            "avg_return": round(float(result.avg_return), 8),
            "cumulative_return": round(float(result.cumulative_return), 8),
            "days": int(len(result.daily_rows)),
            "final_equity": round(float(result.final_equity), 8),
            "max_drawdown": round(float(result.max_drawdown), 8),
            "daily_rows": list(result.daily_rows),
            "backtest_start_date": str(result.start_date),
            "backtest_end_date": str(result.end_date),
            "backtest_seconds": duration,
        },
        metadata={
            "source_strategy_dataset_artifact_id": int(strategy_dataset_artifact.id),
            "backtest_config": {
                "transaction_cost_bps": float(spec.transaction_cost_bps),
                "fee_bps": float(spec.fee_bps),
                "slippage_bps": float(spec.effective_slippage_bps()),
                "short_borrow_bps_annual": float(spec.short_borrow_bps_annual),
                "min_price": float(spec.min_price),
                "min_dollar_volume": float(spec.min_dollar_volume),
                "liquidity_filter_applied": bool(spec.min_dollar_volume > 0.0 and result.has_liquidity_data),
                "max_position_weight": float(spec.max_position_weight),
                "execution_delay_days": int(spec.execution_delay_days),
                "use_lagged_weights": bool(spec.use_lagged_weights),
                "turnover_half_l1": bool(spec.turnover_half_l1),
            },
            "equity_curve": build_equity_curve_from_daily_rows(result.daily_rows),
            "schema": build_schema_metadata(
                artifact_type="BACKTEST_RESULT",
                required_columns=BACKTEST_REQUIRED_COLUMNS,
                actual_columns=[
                    "date",
                    "symbol",
                    "strategy_signal",
                    "strategy_score",
                    "target_weight",
                    "effective_weight",
                    "asset_return",
                    "gross_exposure",
                    "realized_return",
                    "turnover",
                    "turnover_cost",
                ],
            ),
            "backtest_start_date": str(result.start_date),
            "backtest_end_date": str(result.end_date),
            "backtest_seconds": duration,
            **stored.storage_metadata(),
        },
        uri=stored.uri,
    )


__all__ = ["execute_backtest_strategy", "execute_build_strategy_dataset"]
