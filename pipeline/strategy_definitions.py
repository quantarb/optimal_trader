from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .models import StrategyDefinition


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


def _normalized_direct_weights(signals: pd.Series, *, gross_exposure: float, selection_side: str, threshold: float) -> pd.Series:
    base = pd.to_numeric(signals, errors="coerce").fillna(0.0)
    if selection_side == "long_only":
        base = base.clip(lower=0.0)
    else:
        base = base.clip(lower=-1.0, upper=1.0)
    if threshold > 0:
        base = base.where(base.abs() >= threshold, 0.0)
    gross = float(base.abs().sum())
    if gross <= 0:
        return pd.Series(0.0, index=base.index, dtype=float)
    scaled = base * (gross_exposure / gross)
    return scaled.astype(float)


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

    out = feature_df.copy()
    out["strategy_signal"] = 0
    out["target_weight"] = 0.0
    out["rank"] = ""
    out["eligible"] = 0
    out["selected_on_rebalance"] = 0
    out["rebalance_date"] = 0
    out["signal_score"] = pd.to_numeric(out["strategy_score"], errors="coerce")
    out["portfolio_side"] = portfolio_side

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    unique_dates = pd.DatetimeIndex(sorted(out["date"].dropna().unique()))
    rebalance_dates = _rebalance_dates(unique_dates, rebalance_freq)

    current_symbols: list[str] = []
    current_weights: dict[str, float] = {}
    for date_value, group in out.groupby("date", sort=True):
        idxs = group.index.tolist()
        if date_value in rebalance_dates:
            out.loc[idxs, "rebalance_date"] = 1
            if str(definition.strategy_type) == StrategyDefinition.StrategyType.RL_POLICY_V1 or signal_combination == "direct":
                signal_field = action_source_field or "signal_score"
                if signal_field not in group.columns:
                    signal_field = "strategy_score" if "strategy_score" in group.columns else signal_field
                direct_weights = _normalized_direct_weights(
                    pd.to_numeric(group.get(signal_field), errors="coerce"),
                    gross_exposure=gross_exposure,
                    selection_side=portfolio_side,
                    threshold=action_threshold,
                )
                active = direct_weights[direct_weights != 0]
                current_symbols = [str(group.loc[idx, "symbol"]) for idx in active.index.tolist()]
                current_weights = {str(group.loc[idx, "symbol"]): float(weight) for idx, weight in active.items()}
                out.loc[active.index.tolist(), "eligible"] = 1
                out.loc[active.index.tolist(), "selected_on_rebalance"] = 1
                ranked_index = active.abs().sort_values(ascending=False).index.tolist()
                for rank, idx in enumerate(ranked_index, start=1):
                    out.loc[idx, "rank"] = int(rank)
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
                current_symbols = [str(symbol) for symbol in selected["symbol"].astype(str).tolist()]
                current_weights = {}
                for rank, idx in enumerate(selected.index.tolist(), start=1):
                    out.loc[idx, "rank"] = int(rank)
        if current_symbols:
            held_mask = out.loc[idxs, "symbol"].astype(str).isin(current_symbols)
            held_rows = out.loc[idxs].loc[held_mask]
            held_index = held_rows.index.tolist()
            if current_weights:
                for idx in held_index:
                    symbol = str(out.loc[idx, "symbol"])
                    weight = float(current_weights.get(symbol, 0.0))
                    out.loc[idx, "strategy_signal"] = 1 if weight > 0 else (-1 if weight < 0 else 0)
                    out.loc[idx, "target_weight"] = round(weight, 8)
            else:
                per_name_weight = gross_exposure / float(len(current_symbols)) if current_symbols else 0.0
                out.loc[held_index, "strategy_signal"] = 1
                out.loc[held_index, "target_weight"] = round(float(per_name_weight), 8)

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
        }
    }
