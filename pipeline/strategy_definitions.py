from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .factor_signals import evaluate_signal_expression
from .models import StrategyDefinition
from .portfolio_optimization import (
    build_expected_return_series,
    build_neutrality_matrix,
    build_portfolio_optimization_config,
    build_return_panel,
    ensure_constraint_columns,
    estimate_risk_model,
    optimize_mean_variance_portfolio,
)


DEFAULT_NOTEBOOK_TOPK_SLUG = "notebook-topk-v1"


@dataclass(frozen=True)
class ResolvedStrategyDefinition:
    definition_id: int
    name: str
    slug: str
    strategy_type: str
    config: dict[str, Any]


def _default_definition_config() -> dict[str, Any]:
    return {
        "gate_quantile": 0.5,
        "top_k": 20,
        "rebalance_freq": "W",
        "gross_exposure": 0.8,
        "prob_buy_field": "prob_buy",
        "ranking_field": "ranking",
        "ae_familiarity_field": "ae_familiarity",
        "combined_score_expr": "prob_buy * ranking * ae_familiarity",
        "selection_side": "long_only",
    }


def _coalesce_config(definition: StrategyDefinition) -> dict[str, Any]:
    config = dict(definition.config or {})
    config.setdefault("gate_quantile", float(definition.gate_quantile))
    config.setdefault("top_k", int(definition.top_k))
    config.setdefault("rebalance_freq", str(definition.rebalance_freq))
    config.setdefault("gross_exposure", float(definition.gross_exposure))
    config.setdefault("selection_side", str(definition.selection_side))
    config.setdefault("signal_combination", str(definition.signal_combination))
    if definition.action_source_field:
        config.setdefault("action_source_field", str(definition.action_source_field))
    config.setdefault("action_threshold", float(definition.action_threshold))
    return config


def ensure_default_strategy_definitions() -> list[StrategyDefinition]:
    definition, _created = StrategyDefinition.objects.get_or_create(
        slug=DEFAULT_NOTEBOOK_TOPK_SLUG,
        defaults={
            "name": "Notebook Top-K Weekly",
            "strategy_type": StrategyDefinition.StrategyType.NOTEBOOK_TOPK_V1,
            "gate_quantile": 0.5,
            "top_k": 20,
            "rebalance_freq": StrategyDefinition.RebalanceFreq.WEEKLY,
            "gross_exposure": 0.8,
            "selection_side": StrategyDefinition.SelectionSide.LONG_ONLY,
            "signal_combination": StrategyDefinition.SignalCombination.MULTIPLY,
            "action_source_field": "",
            "action_threshold": 0.0,
            "config": _default_definition_config(),
            "description": "Weekly percentile gate plus combined classifier/regressor/autoencoder score, then top-k selection.",
            "is_active": True,
        },
    )
    desired = {
        "strategy_type": StrategyDefinition.StrategyType.NOTEBOOK_TOPK_V1,
        "gate_quantile": 0.5,
        "top_k": 20,
        "rebalance_freq": StrategyDefinition.RebalanceFreq.WEEKLY,
        "gross_exposure": 0.8,
        "selection_side": StrategyDefinition.SelectionSide.LONG_ONLY,
        "signal_combination": StrategyDefinition.SignalCombination.MULTIPLY,
        "action_source_field": "",
        "action_threshold": 0.0,
    }
    changed = False
    for key, value in desired.items():
        if getattr(definition, key) != value:
            setattr(definition, key, value)
            changed = True
    merged_config = _default_definition_config()
    if dict(definition.config or {}) != merged_config:
        definition.config = merged_config
        changed = True
    if changed:
        definition.save()
    return [definition]


def strategy_definition_choices() -> list[tuple[int, str]]:
    ensure_default_strategy_definitions()
    rows = StrategyDefinition.objects.filter(is_active=True).order_by("name", "id")
    return [(int(row.id), f"#{int(row.id)} | {row.name} | {row.strategy_type}") for row in rows]


def upsert_strategy_definition(
    *,
    slug: str,
    name: str,
    strategy_type: str,
    config: dict[str, Any],
    description: str = "",
) -> StrategyDefinition:
    definition, _created = StrategyDefinition.objects.update_or_create(
        slug=str(slug),
        defaults={
            "name": str(name),
            "strategy_type": str(strategy_type),
            "gate_quantile": float(config.get("gate_quantile") or 0.5),
            "top_k": int(config.get("top_k") or 20),
            "rebalance_freq": str(config.get("rebalance_freq") or StrategyDefinition.RebalanceFreq.WEEKLY),
            "gross_exposure": float(config.get("gross_exposure") or 0.8),
            "selection_side": str(config.get("selection_side") or StrategyDefinition.SelectionSide.LONG_ONLY),
            "signal_combination": str(config.get("signal_combination") or StrategyDefinition.SignalCombination.MULTIPLY),
            "action_source_field": str(config.get("action_source_field") or ""),
            "action_threshold": float(config.get("action_threshold") or 0.0),
            "config": dict(config or {}),
            "description": str(description or ""),
            "is_active": True,
        },
    )
    return definition


def resolve_strategy_definition(strategy_definition_id: int | None = None) -> ResolvedStrategyDefinition:
    ensure_default_strategy_definitions()
    definition: StrategyDefinition | None = None
    if int(strategy_definition_id or 0) > 0:
        definition = StrategyDefinition.objects.filter(pk=int(strategy_definition_id), is_active=True).first()
    if definition is None:
        definition = StrategyDefinition.objects.filter(slug=DEFAULT_NOTEBOOK_TOPK_SLUG, is_active=True).first()
    if definition is None:
        raise ValueError("No active strategy definition is available.")
    return ResolvedStrategyDefinition(
        definition_id=int(definition.id),
        name=str(definition.name),
        slug=str(definition.slug),
        strategy_type=str(definition.strategy_type),
        config=_coalesce_config(definition),
    )


def _rebalance_dates(unique_dates: pd.DatetimeIndex, rebalance_freq: str) -> set[pd.Timestamp]:
    if rebalance_freq == "D":
        return set(unique_dates)
    if rebalance_freq == "M":
        return set(pd.Series(unique_dates, index=unique_dates).groupby(unique_dates.to_period("M")).head(1).tolist())
    return set(pd.Series(unique_dates, index=unique_dates).groupby(unique_dates.to_period("W")).head(1).tolist())


def _normalized_direct_weights(
    signals: pd.Series,
    *,
    gross_exposure: float,
    selection_side: str,
    threshold: float,
    transform: str,
) -> pd.Series:
    base = pd.to_numeric(signals, errors="coerce").fillna(0.0)
    if selection_side == "long_only":
        base = base.clip(lower=0.0)
    else:
        base = base.clip(lower=-1.0, upper=1.0)
    if threshold > 0:
        base = base.where(base.abs() >= threshold, 0.0)
    transform_value = str(transform or "identity").strip().lower() or "identity"
    if transform_value == "sign":
        base = base.gt(0.0).astype(float) - base.lt(0.0).astype(float)
    gross = float(base.abs().sum())
    if gross <= 0:
        return pd.Series(0.0, index=base.index, dtype=float)
    scaled = base * (gross_exposure / gross)
    return scaled.astype(float)


def _resolve_score_field(group: pd.DataFrame, preferred_field: str) -> str:
    candidates = [
        str(preferred_field or "").strip(),
        "strategy_score",
        "signal_score",
        "combined_score",
    ]
    for candidate in candidates:
        if candidate and candidate in group.columns:
            return candidate
    return str(preferred_field or "strategy_score").strip() or "strategy_score"


def _resolve_bucket(raw_value: Any, *, bucket_count: int, default: int) -> int:
    text = str(raw_value or "").strip().lower()
    if text == "top":
        return int(bucket_count)
    if text == "bottom":
        return 1
    try:
        bucket = int(raw_value)
    except Exception:
        bucket = int(default)
    return min(max(bucket, 1), int(bucket_count))


def _resolve_quantile_share(raw_value: Any, *, default: float) -> float:
    try:
        quantile = float(raw_value) if raw_value not in (None, "") else float(default)
    except Exception:
        quantile = float(default)
    return min(max(quantile, 0.0), 1.0)


def _quantile_bucket_weights(
    group: pd.DataFrame,
    *,
    score_field: str,
    bucket_count: int,
    long_bucket: int,
    short_bucket: int,
    gross_exposure: float,
    selection_side: str,
    higher_score_is_better: bool,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    scores = pd.to_numeric(group.get(score_field), errors="coerce")
    valid_scores = scores.dropna()
    weights = pd.Series(0.0, index=group.index, dtype=float)
    best_rank = pd.Series("", index=group.index, dtype=object)
    buckets = pd.Series(0, index=group.index, dtype=int)
    eligible = pd.Series(0, index=group.index, dtype=int)
    if valid_scores.empty:
        return weights, best_rank, buckets, eligible

    ranking_signal = valid_scores if higher_score_is_better else (-1.0 * valid_scores)
    rank_positions = ranking_signal.rank(method="first", ascending=True)
    bucket_labels = (((rank_positions - 1.0) * float(bucket_count)) // float(len(rank_positions))).astype(int) + 1
    display_rank = ranking_signal.rank(method="first", ascending=False).astype(int)

    eligible.loc[valid_scores.index] = 1
    best_rank.loc[display_rank.index] = display_rank.astype(str)
    buckets.loc[bucket_labels.index] = bucket_labels.astype(int)

    long_index = bucket_labels[bucket_labels == int(long_bucket)].index.tolist()
    short_index = bucket_labels[bucket_labels == int(short_bucket)].index.tolist()

    if selection_side == "long_only":
        if long_index:
            weights.loc[long_index] = float(gross_exposure) / float(len(long_index))
        return weights, best_rank, buckets, eligible

    if long_index and short_index and int(long_bucket) != int(short_bucket):
        half_gross = float(gross_exposure) / 2.0
        weights.loc[long_index] = half_gross / float(len(long_index))
        weights.loc[short_index] = -half_gross / float(len(short_index))
    return weights, best_rank, buckets, eligible


def _factor_quantile_weights(
    group: pd.DataFrame,
    *,
    score_field: str,
    long_quantile: float,
    short_quantile: float,
    gross_exposure: float,
    selection_side: str,
    higher_score_is_better: bool,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    scores = pd.to_numeric(group.get(score_field), errors="coerce")
    valid_scores = scores.dropna()
    weights = pd.Series(0.0, index=group.index, dtype=float)
    best_rank = pd.Series("", index=group.index, dtype=object)
    buckets = pd.Series(0, index=group.index, dtype=int)
    eligible = pd.Series(0, index=group.index, dtype=int)
    if valid_scores.empty:
        return weights, best_rank, buckets, eligible

    ranking_signal = valid_scores if higher_score_is_better else (-1.0 * valid_scores)
    ordered = ranking_signal.sort_values(ascending=False, kind="mergesort")
    ordered_index = ordered.index.tolist()
    universe_size = len(ordered_index)
    long_count = min(universe_size, max(0, int(math.ceil(float(universe_size) * float(long_quantile) - 1e-12))))
    short_count = min(universe_size, max(0, int(math.ceil(float(universe_size) * float(short_quantile) - 1e-12))))
    if float(long_quantile) > 0.0 and long_count == 0:
        long_count = 1
    if float(short_quantile) > 0.0 and short_count == 0:
        short_count = 1

    long_index = ordered_index[:long_count]
    short_start = max(long_count, universe_size - short_count)
    short_index = ordered_index[short_start:] if short_count > 0 else []
    display_rank = ranking_signal.rank(method="first", ascending=False).astype(int)

    eligible.loc[valid_scores.index] = 1
    best_rank.loc[display_rank.index] = display_rank.astype(str)
    if long_index:
        buckets.loc[long_index] = 2
    if short_index:
        buckets.loc[short_index] = 1

    if selection_side == "long_only":
        if long_index:
            weights.loc[long_index] = float(gross_exposure) / float(len(long_index))
        return weights, best_rank, buckets, eligible

    if long_index and short_index:
        half_gross = float(gross_exposure) / 2.0
        weights.loc[long_index] = half_gross / float(len(long_index))
        weights.loc[short_index] = -half_gross / float(len(short_index))
    return weights, best_rank, buckets, eligible


def _combine_cross_sectional_sleeves(active_sleeves: list[dict[str, Any]]) -> dict[str, float]:
    if not active_sleeves:
        return {}
    combined: dict[str, float] = {}
    for sleeve in active_sleeves:
        for symbol, weight in dict(sleeve.get("weights") or {}).items():
            normalized = str(symbol).strip().upper()
            if not normalized:
                continue
            combined[normalized] = combined.get(normalized, 0.0) + float(weight)
    sleeve_count = float(len(active_sleeves))
    if sleeve_count <= 0:
        return {}
    return {
        symbol: float(weight) / sleeve_count
        for symbol, weight in combined.items()
        if abs(float(weight) / sleeve_count) > 1e-12
    }


def _optimized_portfolio_weights(
    group: pd.DataFrame,
    *,
    date_value: pd.Timestamp,
    score_field: str,
    current_weights: dict[str, float],
    portfolio_optimization_config,
    higher_score_is_better: bool,
    return_panel: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, dict[str, Any]]:
    normalized_symbols = group["symbol"].astype(str).str.strip().str.upper()
    expected_returns, display_rank, bucket_labels = build_expected_return_series(
        group,
        score_field=score_field,
        config=portfolio_optimization_config,
        higher_score_is_better=higher_score_is_better,
    )
    eligible = pd.to_numeric(group.get(score_field), errors="coerce").notna().astype(int)
    expected_by_symbol = pd.Series(expected_returns.to_numpy(dtype=float), index=normalized_symbols, dtype=float)
    neutrality_matrix, neutrality_labels = build_neutrality_matrix(
        group,
        columns=portfolio_optimization_config.constraints.neutrality_columns,
    )
    risk_model = estimate_risk_model(
        return_panel,
        as_of_date=pd.Timestamp(date_value),
        symbols=normalized_symbols.tolist(),
        config=portfolio_optimization_config.risk_model,
    )
    optimization_result = optimize_mean_variance_portfolio(
        expected_by_symbol,
        risk_model=risk_model,
        previous_weights=current_weights,
        config=portfolio_optimization_config,
        neutrality_matrix=neutrality_matrix,
        neutrality_labels=neutrality_labels,
    )
    weights = pd.Series(0.0, index=group.index, dtype=float)
    previous_weight = pd.Series(0.0, index=group.index, dtype=float)
    expected_estimate = pd.Series(0.0, index=group.index, dtype=float)
    for idx, symbol in zip(group.index.tolist(), normalized_symbols.tolist()):
        weights.loc[idx] = float(optimization_result.weights.get(symbol, 0.0))
        previous_weight.loc[idx] = float(current_weights.get(symbol, 0.0))
        expected_estimate.loc[idx] = float(optimization_result.expected_returns.get(symbol, 0.0))
    return weights, display_rank, bucket_labels, eligible, {
        "expected_return_estimate": expected_estimate,
        "previous_weight": previous_weight,
        "optimization_status": str(optimization_result.status),
        "optimization_success": int(bool(optimization_result.success)),
        "optimization_objective": float(optimization_result.objective_value),
        "optimization_variance": float(optimization_result.portfolio_variance),
        "optimization_expected_portfolio_return": float(optimization_result.expected_portfolio_return),
        "optimization_turnover": float(optimization_result.turnover),
        "optimization_gross_exposure": float(optimization_result.gross_exposure),
        "optimization_net_exposure": float(optimization_result.net_exposure),
        "optimization_max_abs_weight": float(optimization_result.max_abs_weight),
        "optimization_constraint_violation": float(optimization_result.constraint_violation),
        "optimization_iterations": int(optimization_result.iterations),
        "risk_model_type": str(optimization_result.risk_model.model_type),
        "risk_model_observations": int(optimization_result.risk_model.observations),
        "risk_model_condition_number": float(optimization_result.risk_model.condition_number),
        "risk_model_min_eigenvalue": float(optimization_result.risk_model.min_eigenvalue),
        "risk_model_max_eigenvalue": float(optimization_result.risk_model.max_eigenvalue),
        "risk_model_shrinkage": float(optimization_result.risk_model.shrinkage),
        "risk_model_variance_floor": float(optimization_result.risk_model.variance_floor),
        "neutrality_exposure_summary": ", ".join(
            f"{label}={value:.6f}"
            for label, value in sorted(optimization_result.neutrality_exposures.items())
        ),
    }


def apply_strategy_definition(feature_df: pd.DataFrame, definition: ResolvedStrategyDefinition) -> tuple[pd.DataFrame, dict[str, Any]]:
    if feature_df.empty:
        return feature_df.copy(), {"strategy_config": dict(definition.config)}

    config = dict(definition.config or {})
    gate_quantile = min(1.0, max(0.0, float(config.get("gate_quantile") or 0.5)))
    top_k = max(1, int(config.get("top_k") or 20))
    rebalance_freq = str(config.get("rebalance_freq") or "W").strip().upper()
    gross_exposure = max(0.0, float(config.get("gross_exposure") or 1.0))
    portfolio_side = str(config.get("selection_side") or "long_only").strip().lower() or "long_only"
    signal_combination = str(config.get("signal_combination") or "multiply").strip().lower() or "multiply"
    action_source_field = str(config.get("action_source_field") or "").strip()
    action_threshold = max(0.0, float(config.get("action_threshold") or 0.0))
    action_transform = str(config.get("action_transform") or "identity").strip().lower() or "identity"
    portfolio_construction = str(config.get("portfolio_construction") or config.get("factor_construction") or "").strip().lower()
    cross_sectional_score_field = str(config.get("cross_sectional_score_field") or "").strip()
    cross_sectional_bucket_count = max(2, int(config.get("cross_sectional_bucket_count") or 10))
    holding_period_rebalances = max(1, int(config.get("holding_period_rebalances") or 1))
    ranking_lag_days = max(0, int(config.get("ranking_lag_days") or 0))
    higher_score_is_better = bool(config.get("higher_score_is_better", True))
    factor_signal = str(config.get("factor_signal") or "").strip()
    long_quantile = _resolve_quantile_share(config.get("long_quantile"), default=0.5)
    short_quantile = _resolve_quantile_share(config.get("short_quantile"), default=0.5)
    long_bucket = _resolve_bucket(
        config.get("long_bucket"),
        bucket_count=cross_sectional_bucket_count,
        default=cross_sectional_bucket_count,
    )
    short_bucket = _resolve_bucket(
        config.get("short_bucket"),
        bucket_count=cross_sectional_bucket_count,
        default=1,
    )
    optimized_portfolio_constructions = {"optimized_mean_variance", "optimized_portfolio", "mean_variance"}
    portfolio_optimization_config = None

    out = feature_df.copy()
    out["strategy_signal"] = 0
    out["target_weight"] = 0.0
    out["rank"] = ""
    out["eligible"] = 0
    out["selected_on_rebalance"] = 0
    out["rebalance_date"] = 0
    out["cross_sectional_bucket"] = 0
    out["signal_score"] = pd.to_numeric(out["strategy_score"], errors="coerce")
    out["portfolio_side"] = portfolio_side

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    unique_dates = pd.DatetimeIndex(sorted(out["date"].dropna().unique()))
    rebalance_dates = _rebalance_dates(unique_dates, rebalance_freq)
    ordered_rebalance_dates = [date_value for date_value in unique_dates if date_value in rebalance_dates]
    rebalance_order = {date_value: idx for idx, date_value in enumerate(ordered_rebalance_dates)}

    current_symbols: list[str] = []
    current_weights: dict[str, float] = {}
    active_sleeves: list[dict[str, Any]] = []
    return_panel = pd.DataFrame()

    if portfolio_construction == "cross_sectional_quantiles":
        score_field = _resolve_score_field(out, cross_sectional_score_field or action_source_field)
        out["_cross_sectional_score"] = pd.to_numeric(out.get(score_field), errors="coerce")
        if ranking_lag_days > 0:
            out["_cross_sectional_score"] = out.groupby("symbol", sort=False)["_cross_sectional_score"].shift(ranking_lag_days)
    elif portfolio_construction == "long_short_factor":
        resolved_factor_signal = factor_signal or cross_sectional_score_field or action_source_field or "strategy_score"
        out["_cross_sectional_score"] = evaluate_signal_expression(
            out,
            expression=resolved_factor_signal,
            strict=True,
        )
        if ranking_lag_days > 0:
            out["_cross_sectional_score"] = out.groupby("symbol", sort=False)["_cross_sectional_score"].shift(ranking_lag_days)
    elif portfolio_construction in optimized_portfolio_constructions:
        portfolio_optimization_config = build_portfolio_optimization_config(
            config,
            gross_exposure=gross_exposure,
            selection_side=portfolio_side,
        )
        resolved_signal = factor_signal or cross_sectional_score_field or action_source_field or "strategy_score"
        if resolved_signal in out.columns:
            out["_cross_sectional_score"] = pd.to_numeric(out.get(resolved_signal), errors="coerce")
        else:
            out["_cross_sectional_score"] = evaluate_signal_expression(
                out,
                expression=resolved_signal,
                strict=True,
            )
        if ranking_lag_days > 0:
            out["_cross_sectional_score"] = out.groupby("symbol", sort=False)["_cross_sectional_score"].shift(ranking_lag_days)
        out = ensure_constraint_columns(
            out,
            columns=portfolio_optimization_config.constraints.neutrality_columns,
        )
        return_panel, _return_source = build_return_panel(
            out,
            return_col_candidates=portfolio_optimization_config.return_col_candidates,
            price_col_candidates=portfolio_optimization_config.price_col_candidates,
        )
        out["expected_return_estimate"] = 0.0
        out["previous_weight"] = 0.0
        out["weight_change"] = 0.0
        out["optimization_status"] = ""
        out["optimization_success"] = 0
        out["optimization_objective"] = 0.0
        out["optimization_variance"] = 0.0
        out["optimization_expected_portfolio_return"] = 0.0
        out["optimization_turnover"] = 0.0
        out["optimization_gross_exposure"] = 0.0
        out["optimization_net_exposure"] = 0.0
        out["optimization_max_abs_weight"] = 0.0
        out["optimization_constraint_violation"] = 0.0
        out["optimization_iterations"] = 0
        out["risk_model_type"] = ""
        out["risk_model_observations"] = 0
        out["risk_model_condition_number"] = 0.0
        out["risk_model_min_eigenvalue"] = 0.0
        out["risk_model_max_eigenvalue"] = 0.0
        out["risk_model_shrinkage"] = 0.0
        out["risk_model_variance_floor"] = 0.0
        out["neutrality_exposure_summary"] = ""

    for date_value, group in out.groupby("date", sort=True):
        idxs = group.index.tolist()
        if date_value in rebalance_dates:
            out.loc[idxs, "rebalance_date"] = 1
            if portfolio_construction in {"cross_sectional_quantiles", "long_short_factor"}:
                current_rebalance_index = int(rebalance_order.get(date_value, 0))
                active_sleeves = [
                    sleeve
                    for sleeve in active_sleeves
                    if int(sleeve.get("end_rebalance_index") or -1) >= current_rebalance_index
                ]
                if portfolio_construction == "long_short_factor":
                    selected_weights, display_rank, bucket_labels, eligible = _factor_quantile_weights(
                        group,
                        score_field="_cross_sectional_score",
                        long_quantile=long_quantile,
                        short_quantile=short_quantile,
                        gross_exposure=gross_exposure,
                        selection_side=portfolio_side,
                        higher_score_is_better=higher_score_is_better,
                    )
                else:
                    selected_weights, display_rank, bucket_labels, eligible = _quantile_bucket_weights(
                        group,
                        score_field="_cross_sectional_score",
                        bucket_count=cross_sectional_bucket_count,
                        long_bucket=long_bucket,
                        short_bucket=short_bucket,
                        gross_exposure=gross_exposure,
                        selection_side=portfolio_side,
                        higher_score_is_better=higher_score_is_better,
                    )
                selected = selected_weights[selected_weights != 0.0]
                out.loc[eligible[eligible == 1].index.tolist(), "eligible"] = 1
                out.loc[display_rank.index.tolist(), "rank"] = display_rank
                out.loc[bucket_labels.index.tolist(), "cross_sectional_bucket"] = bucket_labels
                out.loc[selected.index.tolist(), "selected_on_rebalance"] = 1
                if not selected.empty:
                    active_sleeves.append(
                        {
                            "end_rebalance_index": current_rebalance_index + holding_period_rebalances - 1,
                            "weights": {
                                str(group.loc[idx, "symbol"]).strip().upper(): float(weight)
                                for idx, weight in selected.items()
                            },
                        }
                    )
                current_weights = _combine_cross_sectional_sleeves(active_sleeves)
                current_symbols = sorted(current_weights.keys())
            elif portfolio_construction in optimized_portfolio_constructions and portfolio_optimization_config is not None:
                selected_weights, display_rank, bucket_labels, eligible, optimization_meta = _optimized_portfolio_weights(
                    group,
                    date_value=date_value,
                    score_field="_cross_sectional_score",
                    current_weights=current_weights,
                    portfolio_optimization_config=portfolio_optimization_config,
                    higher_score_is_better=higher_score_is_better,
                    return_panel=return_panel,
                )
                selected = selected_weights[selected_weights.abs() > 1e-12]
                out.loc[eligible[eligible == 1].index.tolist(), "eligible"] = 1
                out.loc[display_rank.index.tolist(), "rank"] = display_rank
                out.loc[bucket_labels.index.tolist(), "cross_sectional_bucket"] = bucket_labels
                out.loc[selected.index.tolist(), "selected_on_rebalance"] = 1
                out.loc[idxs, "expected_return_estimate"] = optimization_meta["expected_return_estimate"]
                out.loc[idxs, "previous_weight"] = optimization_meta["previous_weight"]
                out.loc[idxs, "weight_change"] = selected_weights - optimization_meta["previous_weight"]
                for field in [
                    "optimization_status",
                    "optimization_success",
                    "optimization_objective",
                    "optimization_variance",
                    "optimization_expected_portfolio_return",
                    "optimization_turnover",
                    "optimization_gross_exposure",
                    "optimization_net_exposure",
                    "optimization_max_abs_weight",
                    "optimization_constraint_violation",
                    "optimization_iterations",
                    "risk_model_type",
                    "risk_model_observations",
                    "risk_model_condition_number",
                    "risk_model_min_eigenvalue",
                    "risk_model_max_eigenvalue",
                    "risk_model_shrinkage",
                    "risk_model_variance_floor",
                    "neutrality_exposure_summary",
                ]:
                    out.loc[idxs, field] = optimization_meta[field]
                current_weights = {
                    str(group.loc[idx, "symbol"]).strip().upper(): float(weight)
                    for idx, weight in selected.items()
                }
                current_symbols = sorted(current_weights.keys())
            elif str(definition.strategy_type) == StrategyDefinition.StrategyType.RL_POLICY_V1 or signal_combination == "direct":
                signal_field = action_source_field or "signal_score"
                if signal_field not in group.columns:
                    signal_field = "strategy_score" if "strategy_score" in group.columns else signal_field
                direct_weights = _normalized_direct_weights(
                    pd.to_numeric(group.get(signal_field), errors="coerce"),
                    gross_exposure=gross_exposure,
                    selection_side=portfolio_side,
                    threshold=action_threshold,
                    transform=action_transform,
                )
                active = direct_weights[direct_weights != 0]
                current_weights = {
                    str(group.loc[idx, "symbol"]).strip().upper(): float(weight)
                    for idx, weight in active.items()
                }
                current_symbols = sorted(current_weights.keys())
                out.loc[active.index.tolist(), "eligible"] = 1
                out.loc[active.index.tolist(), "selected_on_rebalance"] = 1
                ranked_index = active.abs().sort_values(ascending=False).index.tolist()
                for rank, idx in enumerate(ranked_index, start=1):
                    out.loc[idx, "rank"] = str(int(rank))
            else:
                buy_thr = group["prob_buy"].quantile(gate_quantile) if group["prob_buy"].notna().any() else None
                rank_thr = group["ranking"].quantile(gate_quantile) if group["ranking"].notna().any() else None
                fam_thr = group["ae_familiarity"].quantile(gate_quantile) if group["ae_familiarity"].notna().any() else None

                eligible_mask = pd.Series(True, index=group.index)
                if buy_thr is not None:
                    eligible_mask &= group["prob_buy"] >= float(buy_thr)
                if rank_thr is not None:
                    eligible_mask &= group["ranking"] >= float(rank_thr)
                if fam_thr is not None:
                    eligible_mask &= group["ae_familiarity"] >= float(fam_thr)

                eligible_group = group.loc[eligible_mask].sort_values(["combined_score", "symbol"], ascending=[False, True])
                selected = eligible_group.head(top_k)
                out.loc[eligible_group.index.tolist(), "eligible"] = 1
                out.loc[selected.index.tolist(), "eligible"] = 1
                out.loc[selected.index.tolist(), "selected_on_rebalance"] = 1
                current_symbols = [str(symbol).strip().upper() for symbol in selected["symbol"].astype(str).tolist()]
                current_weights = {}
                for rank, idx in enumerate(selected.index.tolist(), start=1):
                    out.loc[idx, "rank"] = str(int(rank))
        if current_symbols:
            held_mask = out.loc[idxs, "symbol"].astype(str).str.upper().isin(current_symbols)
            held_rows = out.loc[idxs].loc[held_mask]
            held_index = held_rows.index.tolist()
            if current_weights:
                for idx in held_index:
                    symbol = str(out.loc[idx, "symbol"]).strip().upper()
                    weight = float(current_weights.get(symbol, 0.0))
                    out.loc[idx, "strategy_signal"] = 1 if weight > 0 else (-1 if weight < 0 else 0)
                    out.loc[idx, "target_weight"] = round(weight, 8)
            else:
                per_name_weight = gross_exposure / float(len(current_symbols)) if current_symbols else 0.0
                out.loc[held_index, "strategy_signal"] = 1
                out.loc[held_index, "target_weight"] = round(float(per_name_weight), 8)

    if "_cross_sectional_score" in out.columns:
        out = out.drop(columns=["_cross_sectional_score"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out, {
        "strategy_config": {
            "gate_quantile": float(gate_quantile),
            "top_k": int(top_k),
            "rebalance_freq": rebalance_freq,
            "gross_exposure": float(gross_exposure),
            "selection_side": portfolio_side,
            "signal_combination": signal_combination,
            "action_source_field": action_source_field,
            "action_threshold": float(action_threshold),
            "action_transform": action_transform,
            "portfolio_construction": portfolio_construction,
            "cross_sectional_score_field": cross_sectional_score_field,
            "cross_sectional_bucket_count": int(cross_sectional_bucket_count),
            "long_bucket": int(long_bucket),
            "short_bucket": int(short_bucket),
            "factor_signal": factor_signal,
            "long_quantile": float(long_quantile),
            "short_quantile": float(short_quantile),
            "holding_period_rebalances": int(holding_period_rebalances),
            "ranking_lag_days": int(ranking_lag_days),
            "higher_score_is_better": bool(higher_score_is_better),
            "portfolio_optimization": (
                portfolio_optimization_config.as_dict()
                if portfolio_optimization_config is not None
                else {}
            ),
        }
    }
