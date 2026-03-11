from __future__ import annotations

import time
import uuid
from typing import Any

import pandas as pd

from ml.execution import _dedupe_label_frame, build_feature_frame_from_artifacts, load_artifact_csv_frame

from .contracts import (
    BACKTEST_REQUIRED_COLUMNS,
    STRATEGY_REQUIRED_COLUMNS,
    build_equity_curve_from_daily_rows,
    build_schema_metadata,
    validate_frame_columns,
)
from .service_runtime import (
    BuiltOutput,
    PipelineExecutionError,
    as_bool,
    read_frame_artifact,
    safe_numeric_series,
    write_frame_artifact,
)
from .progress import ProgressReporter
from .strategy_definitions import apply_strategy_definition, resolve_strategy_definition
from .models import Artifact


def _compute_strategy_scores(
    feature_df: pd.DataFrame,
    *,
    strategy_type: str,
    strategy_config: dict[str, Any],
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    signal_combination = str(strategy_config.get("signal_combination") or "multiply").strip().lower() or "multiply"
    combined_score_expr = str(strategy_config.get("combined_score_expr") or "").strip()
    action_source_field = str(strategy_config.get("action_source_field") or "").strip()

    prob_buy = safe_numeric_series(feature_df, "prob_buy", default=0.0)
    ranking = safe_numeric_series(feature_df, "ranking", default=0.0)
    ae_familiarity = safe_numeric_series(feature_df, "ae_familiarity", default=1.0)

    if str(strategy_type) == "rl_policy_v1" or signal_combination == "direct":
        direct_field = action_source_field or "signal_score"
        if direct_field not in feature_df.columns:
            direct_field = "ranking" if "ranking" in feature_df.columns else "prob_buy"
        direct_score = safe_numeric_series(feature_df, direct_field, default=0.0)
        return direct_score, direct_score, {
            "signal_combination": "direct",
            "score_expression_used": direct_field,
            "score_source_field": direct_field,
        }

    if combined_score_expr:
        try:
            combined_score = pd.to_numeric(feature_df.eval(combined_score_expr, engine="python"), errors="coerce").fillna(0.0)
            return combined_score, combined_score, {
                "signal_combination": signal_combination,
                "score_expression_used": combined_score_expr,
                "score_source_field": "",
            }
        except Exception:
            pass

    if signal_combination == "mean":
        combined_score = pd.concat([prob_buy, ranking, ae_familiarity], axis=1).mean(axis=1, skipna=True).fillna(0.0)
    else:
        combined_score = (prob_buy * ranking * ae_familiarity).fillna(0.0)
        signal_combination = "multiply"

    return combined_score, combined_score, {
        "signal_combination": signal_combination,
        "score_expression_used": combined_score_expr or signal_combination,
        "score_source_field": "",
    }


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
    extra_prediction_ids = [int(v) for v in list(config.get("prediction_artifact_ids") or []) if int(v or 0) > 0]
    extra_prediction_artifacts = list(
        Artifact.objects.filter(id__in=extra_prediction_ids, artifact_type__in=(
            "CLASSIFIER_PREDICTIONS",
            "REGRESSOR_PREDICTIONS",
            "AUTOENCODER_SCORES",
            "MTL_PREDICTIONS",
            "PREDICTIONS",
        )).order_by("id")
    )
    stage_ctx = (
        performance_tracer.stage(
            "strategy.build_dataset",
            category="joins_merges",
            workload_type="batched",
            metadata={"extra_prediction_artifacts": len(extra_prediction_artifacts)},
        )
        if performance_tracer is not None
        else None
    )
    if stage_ctx is None:
        feature_df, feature_cols, panel_meta = build_feature_frame_from_artifacts(
            base_feature_artifact=features_artifact,
            extra_panel_artifacts=extra_prediction_artifacts,
        )
    else:
        with stage_ctx:
            feature_df, feature_cols, panel_meta = build_feature_frame_from_artifacts(
                base_feature_artifact=features_artifact,
                extra_panel_artifacts=extra_prediction_artifacts,
            )
    strategy_start_date = str(config.get("strategy_start_date") or config.get("start_date") or "").strip() or None
    strategy_end_date = str(config.get("strategy_end_date") or config.get("end_date") or "").strip() or None
    if strategy_start_date:
        feature_df = feature_df[pd.to_datetime(feature_df["date"], errors="coerce") >= pd.Timestamp(strategy_start_date)].copy()
    if strategy_end_date:
        feature_df = feature_df[pd.to_datetime(feature_df["date"], errors="coerce") <= pd.Timestamp(strategy_end_date)].copy()
    label_artifact_id = int(config.get("label_artifact_id") or 0)
    if label_artifact_id > 0:
        label_artifact = Artifact.objects.filter(pk=label_artifact_id, artifact_type="LABELS").first()
        if label_artifact is not None:
            label_df = load_artifact_csv_frame(label_artifact)
            label_df = _dedupe_label_frame(label_df)
            merge_cols = [
                col for col in ["date", "symbol", "label", "market_position", "trade_return", "hold_days", "side", "freq", "k"]
                if col in label_df.columns
            ]
            if "date" in merge_cols and "symbol" in merge_cols:
                feature_df = feature_df.merge(label_df[merge_cols], on=["date", "symbol"], how="left")

    strategy_definition = resolve_strategy_definition(config.get("strategy_definition_id"))
    progress.update(
        phase="compute_strategy_signals",
        phase_label="Compute strategy signals",
        phase_index=2,
        phase_total=3,
        message=f"{len(feature_df):,} rows",
        force=True,
    )
    component_frames: dict[str, list[pd.Series]] = {"prob_buy": [], "ranking": [], "ae_familiarity": []}
    for source in list(panel_meta.get("extra_panel_sources") or []):
        artifact_type = str(source.get("artifact_type") or "").strip().upper()
        columns = list(source.get("columns") or [])
        if artifact_type == "CLASSIFIER_PREDICTIONS":
            for col in columns:
                if col.endswith("__prediction_score"):
                    component_frames["prob_buy"].append(pd.to_numeric(feature_df[col], errors="coerce"))
        elif artifact_type == "REGRESSOR_PREDICTIONS":
            for col in columns:
                if col.endswith("__prediction") or col.endswith("__prediction_score"):
                    component_frames["ranking"].append(pd.to_numeric(feature_df[col], errors="coerce"))
        elif artifact_type == "AUTOENCODER_SCORES":
            for col in columns:
                if col.endswith("__prediction_score"):
                    component_frames["ae_familiarity"].append(pd.to_numeric(feature_df[col], errors="coerce"))
                elif col.endswith("__prediction"):
                    feature_df["ae_reconstruction_error"] = pd.to_numeric(feature_df[col], errors="coerce")
        elif artifact_type == "MTL_PREDICTIONS":
            for col in columns:
                if col.endswith("__mtl_prob_buy") or col.endswith("__prediction_score"):
                    component_frames["prob_buy"].append(pd.to_numeric(feature_df[col], errors="coerce"))
                elif col.endswith("__mtl_trade_return") or col.endswith("__prediction"):
                    component_frames["ranking"].append(pd.to_numeric(feature_df[col], errors="coerce"))
                elif col.endswith("__mtl_cluster_confidence"):
                    component_frames["ae_familiarity"].append(pd.to_numeric(feature_df[col], errors="coerce"))

    feature_df["prob_buy"] = (
        pd.concat(component_frames["prob_buy"], axis=1).mean(axis=1, skipna=True)
        if component_frames["prob_buy"]
        else pd.Series(1.0, index=feature_df.index, dtype=float)
    )
    feature_df["ranking"] = (
        pd.concat(component_frames["ranking"], axis=1).mean(axis=1, skipna=True)
        if component_frames["ranking"]
        else pd.to_numeric(feature_df.get("ret_1"), errors="coerce")
    )
    feature_df["ae_familiarity"] = (
        pd.concat(component_frames["ae_familiarity"], axis=1).mean(axis=1, skipna=True)
        if component_frames["ae_familiarity"]
        else pd.Series(1.0, index=feature_df.index, dtype=float)
    )
    feature_df["prob_buy"] = pd.to_numeric(feature_df["prob_buy"], errors="coerce").fillna(0.0)
    feature_df["ranking"] = pd.to_numeric(feature_df["ranking"], errors="coerce").fillna(0.0)
    feature_df["ae_familiarity"] = pd.to_numeric(feature_df["ae_familiarity"], errors="coerce").fillna(1.0)
    if "ae_reconstruction_error" in feature_df.columns:
        feature_df["ae_reconstruction_error"] = pd.to_numeric(feature_df["ae_reconstruction_error"], errors="coerce")
    combined_score, strategy_score, score_meta = _compute_strategy_scores(
        feature_df,
        strategy_type=str(strategy_definition.strategy_type),
        strategy_config=dict(strategy_definition.config or {}),
    )
    feature_df["combined_score"] = combined_score
    feature_df["strategy_score"] = strategy_score
    feature_df, strategy_meta = apply_strategy_definition(feature_df, strategy_definition)

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
    key = f"strategy_dataset_{uuid.uuid4().hex}"
    output_frame = feature_df.sort_values(["symbol", "date"]).reset_index(drop=True)
    storage_format = str(config.get("artifact_storage_format") or "csv").strip().lower() or "csv"
    if performance_tracer is not None:
        with performance_tracer.stage(
            "strategy.serialize_dataset",
            category="serialization",
            workload_type="batched",
            metadata={"rows": int(len(output_frame)), "storage_format": storage_format},
        ):
            stored = write_frame_artifact(
                key,
                frame=output_frame,
                fieldnames=list(feature_df.columns),
                storage_format=storage_format,
            )
    else:
        stored = write_frame_artifact(
            key,
            frame=output_frame,
            fieldnames=list(feature_df.columns),
            storage_format=storage_format,
        )
    selected_rows = feature_df[feature_df["strategy_signal"] != 0].copy()
    daily_counts: dict[str, int] = {}
    daily_gross: dict[str, float] = {}
    if not selected_rows.empty and "date" in selected_rows.columns:
        counts = selected_rows.groupby("date")["symbol"].count()
        gross = selected_rows.groupby("date")["target_weight"].apply(lambda s: pd.to_numeric(s, errors="coerce").abs().sum())
        daily_counts = {str(k)[:10]: int(v) for k, v in counts.items()}
        daily_gross = {str(k)[:10]: round(float(v), 8) for k, v in gross.items()}
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
            "feature_cols": list(feature_cols),
            "selected_rows": int(len(selected_rows)),
            "dates": int(feature_df["date"].nunique()) if "date" in feature_df.columns else 0,
            "avg_daily_positions": round(float(sum(daily_counts.values()) / len(daily_counts)), 4) if daily_counts else 0.0,
            "strategy_build_seconds": duration,
        },
        metadata={
            "source_features_artifact_id": int(features_artifact.id),
            "source_prediction_artifact_ids": [int(a.id) for a in extra_prediction_artifacts],
            "source_label_artifact_id": int(label_artifact_id) if label_artifact_id > 0 else 0,
            "extra_panel_sources": list(panel_meta.get("extra_panel_sources") or []),
            "strategy_definition_id": int(strategy_definition.definition_id),
            "strategy_definition_name": str(strategy_definition.name),
            "strategy_definition_slug": str(strategy_definition.slug),
            "strategy_type": str(strategy_definition.strategy_type),
            "strategy_config": dict(strategy_meta.get("strategy_config") or strategy_definition.config),
            "score_logic": dict(score_meta),
            "strategy_definition": {
                "id": int(strategy_definition.definition_id),
                "name": str(strategy_definition.name),
                "slug": str(strategy_definition.slug),
                "strategy_type": str(strategy_definition.strategy_type),
                "config": dict(strategy_definition.config),
            },
            "daily_position_counts": daily_counts,
            "daily_gross_exposure": daily_gross,
            "strategy_start_date": str(strategy_start_date or ""),
            "strategy_end_date": str(strategy_end_date or ""),
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
    if performance_tracer is not None:
        with performance_tracer.stage(
            "backtest.load_inputs",
            category="data_loading",
            workload_type="batched",
            metadata={},
        ):
            strategy_df = read_frame_artifact(
                strategy_dataset_artifact,
                parse_dates=False,
                normalize_symbols=False,
            )
    else:
        strategy_df = read_frame_artifact(
            strategy_dataset_artifact,
            parse_dates=False,
            normalize_symbols=False,
        )
    if strategy_df.empty:
        raise PipelineExecutionError("No strategy dataset rows available for backtest.")
    strategy_df["date"] = strategy_df["date"].astype(str).str[:10]
    backtest_start_date = str(config.get("backtest_start_date") or config.get("start_date") or "").strip() or None
    backtest_end_date = str(config.get("backtest_end_date") or config.get("end_date") or "").strip() or None
    if backtest_start_date:
        strategy_df = strategy_df[pd.to_datetime(strategy_df["date"], errors="coerce") >= pd.Timestamp(backtest_start_date)].copy()
    if backtest_end_date:
        strategy_df = strategy_df[pd.to_datetime(strategy_df["date"], errors="coerce") <= pd.Timestamp(backtest_end_date)].copy()
    strategy_df["date"] = pd.to_datetime(strategy_df["date"], errors="coerce")
    strategy_df = strategy_df.dropna(subset=["date"]).copy()
    strategy_df["symbol"] = strategy_df["symbol"].astype(str).str.strip().str.upper()
    strategy_df["strategy_signal"] = pd.to_numeric(strategy_df.get("strategy_signal"), errors="coerce").fillna(0).astype(int)
    strategy_df["target_weight"] = pd.to_numeric(strategy_df.get("target_weight"), errors="coerce").fillna(0.0)
    strategy_df["strategy_score"] = pd.to_numeric(strategy_df.get("strategy_score"), errors="coerce")
    strategy_df["asset_return"] = pd.to_numeric(strategy_df.get("ret_1"), errors="coerce").fillna(0.0)

    close_series = safe_numeric_series(strategy_df, "close", default=float("nan"))
    if close_series.isna().all() and "px__close" in strategy_df.columns:
        close_series = pd.to_numeric(strategy_df.get("px__close"), errors="coerce")
    strategy_df["close"] = close_series

    volume_series = safe_numeric_series(strategy_df, "volume", default=float("nan"))
    if volume_series.isna().all() and "px__volume" in strategy_df.columns:
        volume_series = pd.to_numeric(strategy_df.get("px__volume"), errors="coerce")
    strategy_df["volume"] = volume_series

    if "dollar_vol" in strategy_df.columns:
        dollar_volume_series = pd.to_numeric(strategy_df.get("dollar_vol"), errors="coerce")
    elif "px__dollar_vol" in strategy_df.columns:
        dollar_volume_series = pd.to_numeric(strategy_df.get("px__dollar_vol"), errors="coerce")
    else:
        dollar_volume_series = strategy_df["close"] * strategy_df["volume"]
    strategy_df["dollar_volume"] = dollar_volume_series
    has_liquidity_data = bool(strategy_df["dollar_volume"].notna().any())

    if strategy_df.empty:
        raise PipelineExecutionError("Strategy dataset produced no rows for backtest.")

    fee_bps = float(config.get("fee_bps") or 0.0)
    slippage_bps = float(config.get("slippage_bps") or 0.0)
    transaction_cost_bps = max(0.0, float(config.get("transaction_cost_bps") or 0.0))
    if fee_bps <= 0.0 and slippage_bps <= 0.0:
        slippage_bps = transaction_cost_bps
    max_position_weight = float(config.get("max_position_weight") or 0.0)
    min_price = float(config.get("min_price") or 0.0)
    min_dollar_volume = float(config.get("min_dollar_volume") or 0.0)
    short_borrow_bps_annual = max(0.0, float(config.get("short_borrow_bps_annual") or 0.0))
    execution_delay_days = max(0, int(config.get("execution_delay_days") or 1))
    turnover_half_l1 = as_bool(config.get("turnover_half_l1"), default=True)
    use_lagged_weights = as_bool(config.get("use_lagged_weights"), default=True)

    if max_position_weight > 0.0:
        strategy_df["target_weight"] = strategy_df["target_weight"].clip(lower=-max_position_weight, upper=max_position_weight)
    if min_price > 0.0:
        strategy_df.loc[strategy_df["close"].fillna(0.0) < min_price, "target_weight"] = 0.0
    if min_dollar_volume > 0.0 and has_liquidity_data:
        strategy_df.loc[strategy_df["dollar_volume"].fillna(0.0) < min_dollar_volume, "target_weight"] = 0.0

    symbol_order = sorted(strategy_df["symbol"].dropna().unique().tolist())
    date_index = pd.DatetimeIndex(sorted(strategy_df["date"].dropna().unique().tolist()))
    weights = (
        strategy_df.pivot_table(index="date", columns="symbol", values="target_weight", aggfunc="last")
        .reindex(index=date_index, columns=symbol_order)
        .fillna(0.0)
    )
    returns = (
        strategy_df.pivot_table(index="date", columns="symbol", values="asset_return", aggfunc="last")
        .reindex(index=date_index, columns=symbol_order)
        .fillna(0.0)
    )
    strategy_scores = (
        strategy_df.pivot_table(index="date", columns="symbol", values="strategy_score", aggfunc="last")
        .reindex(index=date_index, columns=symbol_order)
    )
    strategy_signals = (
        strategy_df.pivot_table(index="date", columns="symbol", values="strategy_signal", aggfunc="last")
        .reindex(index=date_index, columns=symbol_order)
        .fillna(0)
        .astype(int)
    )
    progress.update(
        phase="compute_backtest",
        phase_label="Compute backtest",
        phase_index=2,
        phase_total=3,
        message=f"{len(date_index):,} dates | {len(symbol_order):,} symbols",
        force=True,
    )
    compute_stage = (
        performance_tracer.stage(
            "backtest.compute",
            category="backtesting",
            workload_type="vectorized",
            metadata={"dates": len(date_index), "symbols": len(symbol_order)},
        )
        if performance_tracer is not None
        else None
    )
    if compute_stage is None:
        effective_weights = weights.copy()
        if use_lagged_weights:
            effective_weights = effective_weights.shift(execution_delay_days).fillna(0.0)
        if (effective_weights.abs().sum(axis=1) <= 0).all():
            raise PipelineExecutionError("Strategy dataset produced no active portfolio rows.")

        weight_changes = effective_weights.diff().abs().fillna(effective_weights.abs())
        turnover = weight_changes.sum(axis=1)
        if turnover_half_l1:
            turnover = turnover * 0.5
        turnover_cost = turnover * ((fee_bps + slippage_bps) / 10000.0)
        short_borrow_cost = effective_weights.clip(upper=0.0).abs().sum(axis=1) * (short_borrow_bps_annual / 10000.0 / 252.0)
        realized_matrix = effective_weights * returns
        daily_return = realized_matrix.sum(axis=1)
        net_daily_return = daily_return - turnover_cost - short_borrow_cost
        equity_curve_series = (1.0 + net_daily_return).cumprod()
    else:
        with compute_stage:
            effective_weights = weights.copy()
            if use_lagged_weights:
                effective_weights = effective_weights.shift(execution_delay_days).fillna(0.0)
            if (effective_weights.abs().sum(axis=1) <= 0).all():
                raise PipelineExecutionError("Strategy dataset produced no active portfolio rows.")

            weight_changes = effective_weights.diff().abs().fillna(effective_weights.abs())
            turnover = weight_changes.sum(axis=1)
            if turnover_half_l1:
                turnover = turnover * 0.5
            turnover_cost = turnover * ((fee_bps + slippage_bps) / 10000.0)
            short_borrow_cost = effective_weights.clip(upper=0.0).abs().sum(axis=1) * (short_borrow_bps_annual / 10000.0 / 252.0)
            realized_matrix = effective_weights * returns
            daily_return = realized_matrix.sum(axis=1)
            net_daily_return = daily_return - turnover_cost - short_borrow_cost
            equity_curve_series = (1.0 + net_daily_return).cumprod()

    trade_frame = pd.DataFrame(
        {
            "target_weight": weights.stack(future_stack=True),
            "effective_weight": effective_weights.stack(future_stack=True),
            "asset_return": returns.stack(future_stack=True),
            "strategy_score": strategy_scores.stack(future_stack=True),
            "strategy_signal": strategy_signals.stack(future_stack=True),
            "realized_return": realized_matrix.stack(future_stack=True),
        }
    ).reset_index()
    trade_frame = trade_frame.rename(columns={"level_0": "date", "level_1": "symbol"})
    trade_frame["gross_exposure"] = trade_frame["effective_weight"].abs()
    trade_frame["turnover"] = trade_frame["date"].map(turnover.to_dict())
    trade_frame["turnover_cost"] = trade_frame["date"].map(turnover_cost.to_dict())
    trade_frame["short_borrow_cost"] = trade_frame["date"].map(short_borrow_cost.to_dict())
    trade_frame = trade_frame[
        (trade_frame["target_weight"].abs() > 1e-12) | (trade_frame["effective_weight"].abs() > 1e-12)
    ].copy()
    trade_frame["date"] = pd.to_datetime(trade_frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    trade_rows = trade_frame.sort_values(["date", "symbol"]).to_dict(orient="records")

    daily_rows: list[dict[str, Any]] = []
    for date_value in date_index:
        daily_rows.append(
            {
                "date": str(date_value.date()),
                "positions": int((effective_weights.loc[date_value].abs() > 1e-12).sum()),
                "gross_exposure": round(float(effective_weights.loc[date_value].abs().sum()), 8),
                "turnover": round(float(turnover.loc[date_value]), 8),
                "turnover_cost": round(float(turnover_cost.loc[date_value]), 8),
                "short_borrow_cost": round(float(short_borrow_cost.loc[date_value]), 8),
                "daily_return": round(float(daily_return.loc[date_value]), 8),
                "net_daily_return": round(float(net_daily_return.loc[date_value]), 8),
                "equity": round(float(equity_curve_series.loc[date_value]), 8),
            }
        )

    realized_returns = [float(row["realized_return"]) for row in trade_rows]
    net_daily_returns = [float(row["net_daily_return"]) for row in daily_rows]
    total = len(trade_rows)
    wins = sum(1 for value in realized_returns if value > 0)
    losses = sum(1 for value in realized_returns if value < 0)
    avg_return = (sum(net_daily_returns) / float(len(net_daily_returns))) if net_daily_returns else 0.0
    cumulative_return = float(equity_curve_series.iloc[-1] - 1.0)
    rolling_max = equity_curve_series.cummax()
    drawdown_series = (equity_curve_series / rolling_max) - 1.0
    max_drawdown = float(drawdown_series.min()) if not drawdown_series.empty else 0.0
    validate_frame_columns(pd.DataFrame(trade_rows), BACKTEST_REQUIRED_COLUMNS, artifact_type="BACKTEST_RESULT")
    progress.update(
        phase="write_backtest_output",
        phase_label="Write backtest output",
        phase_index=3,
        phase_total=3,
        total_units=1,
        completed_units=0,
        force=True,
    )
    key = f"backtest_{uuid.uuid4().hex}"
    storage_format = str(config.get("artifact_storage_format") or "csv").strip().lower() or "csv"
    trade_frame_output = pd.DataFrame(trade_rows)
    if performance_tracer is not None:
        with performance_tracer.stage(
            "backtest.serialize_artifact",
            category="serialization",
            workload_type="batched",
            metadata={"rows": int(len(trade_rows)), "storage_format": storage_format},
        ):
            stored = write_frame_artifact(
                key,
                frame=trade_frame_output,
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
            )
    else:
        stored = write_frame_artifact(
            key,
            frame=trade_frame_output,
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
            "trades": int(total),
            "wins": int(wins),
            "losses": int(losses),
            "avg_return": round(float(avg_return), 8),
            "cumulative_return": round(float(cumulative_return), 8),
            "days": int(len(daily_rows)),
            "final_equity": round(float(equity_curve_series.iloc[-1]), 8),
            "max_drawdown": round(float(max_drawdown), 8),
            "daily_rows": daily_rows,
            "backtest_start_date": str(backtest_start_date or ""),
            "backtest_end_date": str(backtest_end_date or ""),
            "backtest_seconds": duration,
        },
        metadata={
            "source_strategy_dataset_artifact_id": int(strategy_dataset_artifact.id),
            "backtest_config": {
                "transaction_cost_bps": float(transaction_cost_bps),
                "fee_bps": float(fee_bps),
                "slippage_bps": float(slippage_bps),
                "short_borrow_bps_annual": float(short_borrow_bps_annual),
                "min_price": float(min_price),
                "min_dollar_volume": float(min_dollar_volume),
                "liquidity_filter_applied": bool(min_dollar_volume > 0.0 and has_liquidity_data),
                "max_position_weight": float(max_position_weight),
                "execution_delay_days": int(execution_delay_days),
                "use_lagged_weights": bool(use_lagged_weights),
                "turnover_half_l1": bool(turnover_half_l1),
            },
            "equity_curve": build_equity_curve_from_daily_rows(daily_rows),
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
            "backtest_start_date": str(backtest_start_date or ""),
            "backtest_end_date": str(backtest_end_date or ""),
            "backtest_seconds": duration,
            **stored.storage_metadata(),
        },
        uri=stored.uri,
    )


__all__ = ["execute_backtest_strategy", "execute_build_strategy_dataset"]
