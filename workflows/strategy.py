from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import pandas as pd

from domain.backtests import StrategyBacktestSpec, StrategyDatasetSpec
from domain.models.datasets import filter_frame_by_date
from infra.repositories import DjangoArtifactRepository
from ml.execution import _dedupe_label_frame, build_feature_frame_from_artifacts, load_artifact_csv_frame
from pipeline.contracts import PREDICTION_ARTIFACT_TYPES
from pipeline.service_runtime import PipelineExecutionError, read_frame_artifact
from pipeline.strategy_definitions import ResolvedStrategyDefinition, apply_strategy_definition, resolve_strategy_definition
from .strategy_signal_support import (
    _collect_prediction_components,
    _compute_strategy_scores,
)
ACTIVE_WEIGHT_EPSILON = 1e-12
BPS_TO_DECIMAL = 10000.0
TRADING_DAYS_PER_YEAR = 252.0
SUMMARY_DECIMALS = 8


@dataclass(frozen=True)
class StrategyDatasetWorkflowResult:
    """Dataset build result for strategy artifact creation."""

    frame: pd.DataFrame
    feature_cols: list[str]
    panel_meta: dict[str, Any]
    strategy_definition: ResolvedStrategyDefinition
    strategy_meta: dict[str, Any]
    score_meta: dict[str, Any]
    source_prediction_artifact_ids: list[int]
    source_label_artifact_id: int
    daily_position_counts: dict[str, int]
    daily_gross_exposure: dict[str, float]
    start_date: str
    end_date: str


@dataclass(frozen=True)
class BacktestMatrices:
    """Pivoted strategy inputs aligned on shared dates and symbols."""

    date_index: pd.DatetimeIndex
    symbol_order: list[str]
    weights: pd.DataFrame
    returns: pd.DataFrame
    strategy_scores: pd.DataFrame
    strategy_signals: pd.DataFrame


@dataclass(frozen=True)
class BacktestComputation:
    """Vectorized backtest outputs derived from aligned matrices."""

    effective_weights: pd.DataFrame
    turnover: pd.Series
    turnover_cost: pd.Series
    short_borrow_cost: pd.Series
    realized_matrix: pd.DataFrame
    daily_return: pd.Series
    net_daily_return: pd.Series
    equity_curve: pd.Series


@dataclass(frozen=True)
class StrategyBacktestWorkflowResult:
    """Backtest result for artifact serialization and reporting."""

    trade_frame: pd.DataFrame
    daily_rows: list[dict[str, Any]]
    trades: int
    wins: int
    losses: int
    avg_return: float
    cumulative_return: float
    final_equity: float
    max_drawdown: float
    has_liquidity_data: bool
    start_date: str
    end_date: str


def _stage(performance_tracer, name: str, *, category: str, workload_type: str, metadata: dict[str, Any] | None = None):
    if performance_tracer is None:
        return nullcontext()
    return performance_tracer.stage(name, category=category, workload_type=workload_type, metadata=metadata)


def _numeric_series(frame: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def _merge_label_columns(feature_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    label_df = _dedupe_label_frame(label_df)
    merge_cols = [
        column
        for column in ["date", "symbol", "label", "market_position", "trade_return", "hold_days", "side", "freq", "k"]
        if column in label_df.columns
    ]
    if "date" not in merge_cols or "symbol" not in merge_cols:
        return feature_df
    return feature_df.merge(label_df[merge_cols], on=["date", "symbol"], how="left")
def _selected_position_summaries(feature_df: pd.DataFrame) -> tuple[dict[str, int], dict[str, float]]:
    selected_rows = feature_df[feature_df["strategy_signal"] != 0].copy()
    if selected_rows.empty or "date" not in selected_rows.columns:
        return {}, {}
    counts = selected_rows.groupby("date")["symbol"].count()
    gross = selected_rows.groupby("date")["target_weight"].apply(lambda series: pd.to_numeric(series, errors="coerce").abs().sum())
    daily_counts = {str(key)[:10]: int(value) for key, value in counts.items()}
    daily_gross = {str(key)[:10]: round(float(value), 8) for key, value in gross.items()}
    return daily_counts, daily_gross


def _strategy_source_artifacts(
    repo: DjangoArtifactRepository,
    *,
    spec: StrategyDatasetSpec,
) -> tuple[list[Any], int, Any | None]:
    extra_prediction_artifacts = repo.list_pipeline_artifacts(
        spec.prediction_artifact_ids,
        artifact_types=tuple(sorted(PREDICTION_ARTIFACT_TYPES)),
    )
    source_label_artifact_id = int(spec.label_artifact_id or 0)
    label_artifact = None
    if spec.label_artifact_id is not None:
        label_artifact = repo.get_pipeline_artifact(spec.label_artifact_id, artifact_type="LABELS")
    return extra_prediction_artifacts, source_label_artifact_id, label_artifact


def _strategy_feature_frame(
    *,
    spec: StrategyDatasetSpec,
    features_artifact,
    extra_prediction_artifacts: list[Any],
    label_artifact,
    performance_tracer=None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    with _stage(
        performance_tracer,
        "strategy.build_dataset",
        category="joins_merges",
        workload_type="batched",
        metadata={"extra_prediction_artifacts": len(extra_prediction_artifacts)},
    ):
        feature_df, feature_cols, panel_meta = build_feature_frame_from_artifacts(
            base_feature_artifact=features_artifact,
            extra_panel_artifacts=extra_prediction_artifacts,
        )
    feature_df = filter_frame_by_date(feature_df, start_date=spec.start_date, end_date=spec.end_date)
    if label_artifact is not None:
        feature_df = _merge_label_columns(feature_df, load_artifact_csv_frame(label_artifact))
    return feature_df, list(feature_cols), panel_meta


def _apply_strategy_scores_and_definition(
    feature_df: pd.DataFrame,
    *,
    strategy_definition: ResolvedStrategyDefinition,
    panel_meta: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    feature_df = _collect_prediction_components(feature_df, panel_meta)
    combined_score, strategy_score, score_meta = _compute_strategy_scores(
        feature_df,
        strategy_type=str(strategy_definition.strategy_type),
        strategy_config=dict(strategy_definition.config or {}),
    )
    feature_df["combined_score"] = combined_score
    feature_df["strategy_score"] = strategy_score
    feature_df, strategy_meta = apply_strategy_definition(feature_df, strategy_definition)
    return feature_df, score_meta, strategy_meta


def build_strategy_dataset_frame(
    *,
    spec: StrategyDatasetSpec,
    features_artifact,
    artifact_repo: DjangoArtifactRepository | None = None,
    performance_tracer=None,
) -> StrategyDatasetWorkflowResult:
    """Build the strategy dataset frame from features, predictions, and optional labels."""

    repo = artifact_repo or DjangoArtifactRepository()
    extra_prediction_artifacts, source_label_artifact_id, label_artifact = _strategy_source_artifacts(
        repo,
        spec=spec,
    )
    feature_df, feature_cols, panel_meta = _strategy_feature_frame(
        spec=spec,
        features_artifact=features_artifact,
        extra_prediction_artifacts=extra_prediction_artifacts,
        label_artifact=label_artifact,
        performance_tracer=performance_tracer,
    )
    strategy_definition = resolve_strategy_definition(spec.strategy_definition_id)
    feature_df, score_meta, strategy_meta = _apply_strategy_scores_and_definition(
        feature_df,
        strategy_definition=strategy_definition,
        panel_meta=panel_meta,
    )
    output_frame = feature_df.sort_values(["symbol", "date"]).reset_index(drop=True)
    daily_counts, daily_gross = _selected_position_summaries(output_frame)
    return StrategyDatasetWorkflowResult(
        frame=output_frame,
        feature_cols=list(feature_cols),
        panel_meta=panel_meta,
        strategy_definition=strategy_definition,
        strategy_meta=strategy_meta,
        score_meta=score_meta,
        source_prediction_artifact_ids=[int(artifact.id) for artifact in extra_prediction_artifacts],
        source_label_artifact_id=source_label_artifact_id,
        daily_position_counts=daily_counts,
        daily_gross_exposure=daily_gross,
        start_date=str(spec.start_date or ""),
        end_date=str(spec.end_date or ""),
    )


def _prepare_backtest_frame(strategy_dataset_artifact, *, spec: StrategyBacktestSpec, performance_tracer=None) -> tuple[pd.DataFrame, bool]:
    with _stage(
        performance_tracer,
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
    if strategy_df.empty:
        raise PipelineExecutionError("No strategy dataset rows available for backtest.")
    strategy_df["date"] = strategy_df["date"].astype(str).str[:10]
    strategy_df = filter_frame_by_date(strategy_df, start_date=spec.start_date, end_date=spec.end_date)
    strategy_df["date"] = pd.to_datetime(strategy_df["date"], errors="coerce")
    strategy_df = strategy_df.dropna(subset=["date"]).copy()
    if strategy_df.empty:
        raise PipelineExecutionError("Strategy dataset produced no rows for backtest.")
    strategy_df["symbol"] = strategy_df["symbol"].astype(str).str.strip().str.upper()
    strategy_df["strategy_signal"] = pd.to_numeric(strategy_df.get("strategy_signal"), errors="coerce").fillna(0).astype(int)
    strategy_df["target_weight"] = pd.to_numeric(strategy_df.get("target_weight"), errors="coerce").fillna(0.0)
    strategy_df["strategy_score"] = pd.to_numeric(strategy_df.get("strategy_score"), errors="coerce")
    strategy_df["asset_return"] = pd.to_numeric(strategy_df.get("ret_1"), errors="coerce").fillna(0.0)
    close_series = _numeric_series(strategy_df, "close", default=float("nan"))
    if close_series.isna().all() and "px__close" in strategy_df.columns:
        close_series = pd.to_numeric(strategy_df.get("px__close"), errors="coerce")
    strategy_df["close"] = close_series
    volume_series = _numeric_series(strategy_df, "volume", default=float("nan"))
    if volume_series.isna().all() and "px__volume" in strategy_df.columns:
        volume_series = pd.to_numeric(strategy_df.get("px__volume"), errors="coerce")
    strategy_df["volume"] = volume_series
    if "dollar_vol" in strategy_df.columns:
        strategy_df["dollar_volume"] = pd.to_numeric(strategy_df.get("dollar_vol"), errors="coerce")
    elif "px__dollar_vol" in strategy_df.columns:
        strategy_df["dollar_volume"] = pd.to_numeric(strategy_df.get("px__dollar_vol"), errors="coerce")
    else:
        strategy_df["dollar_volume"] = strategy_df["close"] * strategy_df["volume"]
    has_liquidity_data = bool(strategy_df["dollar_volume"].notna().any())
    return strategy_df, has_liquidity_data


def _apply_backtest_filters(strategy_df: pd.DataFrame, *, spec: StrategyBacktestSpec, has_liquidity_data: bool) -> pd.DataFrame:
    out = strategy_df.copy()
    if spec.allowed_symbols:
        out = out[out["symbol"].isin(set(spec.allowed_symbols))].copy()
    if spec.max_position_weight > 0.0:
        out["target_weight"] = out["target_weight"].clip(
            lower=-float(spec.max_position_weight),
            upper=float(spec.max_position_weight),
        )
    if spec.min_price > 0.0:
        out.loc[out["close"].fillna(0.0) < float(spec.min_price), "target_weight"] = 0.0
    if spec.min_dollar_volume > 0.0 and has_liquidity_data:
        out.loc[out["dollar_volume"].fillna(0.0) < float(spec.min_dollar_volume), "target_weight"] = 0.0
    return out


def _build_backtest_matrices(strategy_df: pd.DataFrame) -> BacktestMatrices:
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
    return BacktestMatrices(
        date_index=date_index,
        symbol_order=symbol_order,
        weights=weights,
        returns=returns,
        strategy_scores=strategy_scores,
        strategy_signals=strategy_signals,
    )


def _compute_backtest(matrices: BacktestMatrices, *, spec: StrategyBacktestSpec, performance_tracer=None) -> BacktestComputation:
    with _stage(
        performance_tracer,
        "backtest.compute",
        category="backtesting",
        workload_type="vectorized",
        metadata={"dates": len(matrices.date_index), "symbols": len(matrices.symbol_order)},
    ):
        effective_weights = matrices.weights.copy()
        if spec.use_lagged_weights:
            effective_weights = effective_weights.shift(spec.execution_delay_days).fillna(0.0)
        if (effective_weights.abs().sum(axis=1) <= 0).all():
            raise PipelineExecutionError("Strategy dataset produced no active portfolio rows.")
        turnover = effective_weights.diff().abs().fillna(effective_weights.abs()).sum(axis=1)
        if spec.turnover_half_l1:
            turnover = turnover * 0.5
        turnover_cost = turnover * ((float(spec.fee_bps) + float(spec.effective_slippage_bps())) / BPS_TO_DECIMAL)
        short_borrow_cost = effective_weights.clip(upper=0.0).abs().sum(axis=1) * (
            float(spec.short_borrow_bps_annual) / BPS_TO_DECIMAL / TRADING_DAYS_PER_YEAR
        )
        realized_matrix = effective_weights * matrices.returns
        daily_return = realized_matrix.sum(axis=1)
        net_daily_return = daily_return - turnover_cost - short_borrow_cost
        equity_curve = (1.0 + net_daily_return).cumprod()
    return BacktestComputation(
        effective_weights=effective_weights,
        turnover=turnover,
        turnover_cost=turnover_cost,
        short_borrow_cost=short_borrow_cost,
        realized_matrix=realized_matrix,
        daily_return=daily_return,
        net_daily_return=net_daily_return,
        equity_curve=equity_curve,
    )


def _build_trade_frame(matrices: BacktestMatrices, computation: BacktestComputation) -> pd.DataFrame:
    trade_frame = pd.DataFrame(
        {
            "target_weight": matrices.weights.stack(future_stack=True),
            "effective_weight": computation.effective_weights.stack(future_stack=True),
            "asset_return": matrices.returns.stack(future_stack=True),
            "strategy_score": matrices.strategy_scores.stack(future_stack=True),
            "strategy_signal": matrices.strategy_signals.stack(future_stack=True),
            "realized_return": computation.realized_matrix.stack(future_stack=True),
        }
    ).reset_index()
    trade_frame = trade_frame.rename(columns={"level_0": "date", "level_1": "symbol"})
    trade_frame["gross_exposure"] = trade_frame["effective_weight"].abs()
    trade_frame["turnover"] = trade_frame["date"].map(computation.turnover.to_dict())
    trade_frame["turnover_cost"] = trade_frame["date"].map(computation.turnover_cost.to_dict())
    trade_frame["short_borrow_cost"] = trade_frame["date"].map(computation.short_borrow_cost.to_dict())
    trade_frame = trade_frame[
        (trade_frame["target_weight"].abs() > ACTIVE_WEIGHT_EPSILON)
        | (trade_frame["effective_weight"].abs() > ACTIVE_WEIGHT_EPSILON)
    ].copy()
    trade_frame["date"] = pd.to_datetime(trade_frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return trade_frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def _build_daily_rows(matrices: BacktestMatrices, computation: BacktestComputation) -> list[dict[str, Any]]:
    return [
        {
            "date": str(date_value.date()),
            "positions": int((computation.effective_weights.loc[date_value].abs() > ACTIVE_WEIGHT_EPSILON).sum()),
            "gross_exposure": round(float(computation.effective_weights.loc[date_value].abs().sum()), SUMMARY_DECIMALS),
            "turnover": round(float(computation.turnover.loc[date_value]), SUMMARY_DECIMALS),
            "turnover_cost": round(float(computation.turnover_cost.loc[date_value]), SUMMARY_DECIMALS),
            "short_borrow_cost": round(float(computation.short_borrow_cost.loc[date_value]), SUMMARY_DECIMALS),
            "daily_return": round(float(computation.daily_return.loc[date_value]), SUMMARY_DECIMALS),
            "net_daily_return": round(float(computation.net_daily_return.loc[date_value]), SUMMARY_DECIMALS),
            "equity": round(float(computation.equity_curve.loc[date_value]), SUMMARY_DECIMALS),
        }
        for date_value in matrices.date_index
    ]


def _summarize_backtest(trade_frame: pd.DataFrame, daily_rows: list[dict[str, Any]], equity_curve: pd.Series) -> tuple[int, int, int, float, float, float, float]:
    realized_returns = pd.to_numeric(trade_frame.get("realized_return"), errors="coerce").fillna(0.0).tolist()
    net_daily_returns = [float(row["net_daily_return"]) for row in daily_rows]
    trades = int(len(trade_frame))
    wins = sum(1 for value in realized_returns if value > 0)
    losses = sum(1 for value in realized_returns if value < 0)
    avg_return = (sum(net_daily_returns) / float(len(net_daily_returns))) if net_daily_returns else 0.0
    cumulative_return = float(equity_curve.iloc[-1] - 1.0)
    rolling_max = equity_curve.cummax()
    drawdown_series = (equity_curve / rolling_max) - 1.0
    max_drawdown = float(drawdown_series.min()) if not drawdown_series.empty else 0.0
    final_equity = float(equity_curve.iloc[-1])
    return trades, wins, losses, avg_return, cumulative_return, final_equity, max_drawdown


def run_strategy_backtest(
    *,
    spec: StrategyBacktestSpec,
    strategy_dataset_artifact,
    performance_tracer=None,
) -> StrategyBacktestWorkflowResult:
    """Run the vectorized backtest for a strategy dataset artifact."""

    strategy_df, has_liquidity_data = _prepare_backtest_frame(
        strategy_dataset_artifact,
        spec=spec,
        performance_tracer=performance_tracer,
    )
    strategy_df = _apply_backtest_filters(strategy_df, spec=spec, has_liquidity_data=has_liquidity_data)
    matrices = _build_backtest_matrices(strategy_df)
    computation = _compute_backtest(matrices, spec=spec, performance_tracer=performance_tracer)
    trade_frame = _build_trade_frame(matrices, computation)
    daily_rows = _build_daily_rows(matrices, computation)
    trades, wins, losses, avg_return, cumulative_return, final_equity, max_drawdown = _summarize_backtest(
        trade_frame,
        daily_rows,
        computation.equity_curve,
    )
    return StrategyBacktestWorkflowResult(
        trade_frame=trade_frame,
        daily_rows=daily_rows,
        trades=trades,
        wins=wins,
        losses=losses,
        avg_return=avg_return,
        cumulative_return=cumulative_return,
        final_equity=final_equity,
        max_drawdown=max_drawdown,
        has_liquidity_data=has_liquidity_data,
        start_date=str(spec.start_date or ""),
        end_date=str(spec.end_date or ""),
    )
