from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from pipeline.factor_signals import build_multi_factor_score_frame


DIRECT_SIGNAL_COMBINATION = "direct"
MEAN_SIGNAL_COMBINATION = "mean"
MULTIPLY_SIGNAL_COMBINATION = "multiply"
FACTOR_COMPONENT_SIGNAL_COMBINATION = "factor_components"
PREDICTION_COMPONENT_DEFAULTS = {
    "prob_buy": 1.0,
    "ae_familiarity": 1.0,
}
PREDICTION_COMPONENT_SUFFIX_RULES = {
    "CLASSIFIER_PREDICTIONS": {"prob_buy": ("__prediction_score",)},
    "REGRESSOR_PREDICTIONS": {"ranking": ("__prediction", "__prediction_score")},
    "AUTOENCODER_SCORES": {"ae_familiarity": ("__prediction_score",)},
    "MTL_PREDICTIONS": {
        "prob_buy": ("__mtl_prob_buy", "__prediction_score"),
        "ranking": ("__mtl_trade_return", "__prediction"),
        "ae_familiarity": ("__mtl_cluster_confidence",),
    },
}
PREDICTION_DERIVED_COLUMN_RULES = {
    "AUTOENCODER_SCORES": {"ae_reconstruction_error": ("__prediction",)},
}


@dataclass(frozen=True)
class StrategyScoreInputs:
    signal_combination: str
    combined_score_expr: str
    action_source_field: str
    factor_components: list[Any]


def _numeric_series(frame: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def _mean_or_default(series_list: list[pd.Series], *, default: pd.Series) -> pd.Series:
    if series_list:
        return pd.concat(series_list, axis=1).mean(axis=1, skipna=True)
    return default


def _matching_rule_target(rules: dict[str, tuple[str, ...]], column: str) -> str | None:
    for target, suffixes in rules.items():
        if any(str(column).endswith(suffix) for suffix in suffixes):
            return target
    return None


def _source_component_series(
    out: pd.DataFrame,
    *,
    artifact_type: str,
    columns: list[str],
) -> tuple[dict[str, list[pd.Series]], dict[str, pd.Series]]:
    component_series = {"prob_buy": [], "ranking": [], "ae_familiarity": []}
    derived_series: dict[str, pd.Series] = {}
    numeric_columns = out.reindex(columns=columns).apply(pd.to_numeric, errors="coerce") if columns else pd.DataFrame(index=out.index)
    component_rules = PREDICTION_COMPONENT_SUFFIX_RULES.get(artifact_type, {})
    derived_rules = PREDICTION_DERIVED_COLUMN_RULES.get(artifact_type, {})
    for column in columns:
        numeric = numeric_columns[column]
        component_name = _matching_rule_target(component_rules, column)
        if component_name is not None:
            component_series[component_name].append(numeric)
            continue
        derived_column = _matching_rule_target(derived_rules, column)
        if derived_column is not None:
            derived_series[derived_column] = numeric
    return component_series, derived_series


def _merge_component_series(
    components: dict[str, list[pd.Series]],
    source_components: dict[str, list[pd.Series]],
) -> None:
    for component_name, series_list in source_components.items():
        components[component_name].extend(series_list)


def _apply_derived_columns(out: pd.DataFrame, derived_columns: dict[str, pd.Series]) -> None:
    for column_name, series in derived_columns.items():
        out[column_name] = series


def _collect_prediction_components(feature_df: pd.DataFrame, panel_meta: dict[str, Any]) -> pd.DataFrame:
    out = feature_df.copy()
    components: dict[str, list[pd.Series]] = {"prob_buy": [], "ranking": [], "ae_familiarity": []}
    for source in list(panel_meta.get("extra_panel_sources") or []):
        artifact_type = str(source.get("artifact_type") or "").strip().upper()
        source_components, derived_columns = _source_component_series(
            out,
            artifact_type=artifact_type,
            columns=list(source.get("columns") or []),
        )
        _merge_component_series(components, source_components)
        _apply_derived_columns(out, derived_columns)
    out["prob_buy"] = _mean_or_default(
        components["prob_buy"],
        default=pd.Series(PREDICTION_COMPONENT_DEFAULTS["prob_buy"], index=out.index, dtype=float),
    )
    out["ranking"] = _mean_or_default(components["ranking"], default=pd.to_numeric(out.get("ret_1"), errors="coerce"))
    out["ae_familiarity"] = _mean_or_default(
        components["ae_familiarity"],
        default=pd.Series(PREDICTION_COMPONENT_DEFAULTS["ae_familiarity"], index=out.index, dtype=float),
    )
    out["prob_buy"] = pd.to_numeric(out["prob_buy"], errors="coerce").fillna(0.0)
    out["ranking"] = pd.to_numeric(out["ranking"], errors="coerce").fillna(0.0)
    out["ae_familiarity"] = pd.to_numeric(out["ae_familiarity"], errors="coerce").fillna(
        PREDICTION_COMPONENT_DEFAULTS["ae_familiarity"]
    )
    if "ae_reconstruction_error" in out.columns:
        out["ae_reconstruction_error"] = pd.to_numeric(out["ae_reconstruction_error"], errors="coerce")
    return out


def _factor_component_scores(
    feature_df: pd.DataFrame,
    *,
    factor_components: list[Any],
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    factor_frame, factor_meta = build_multi_factor_score_frame(
        feature_df,
        factor_components=factor_components,
        output_col="factor_model_score",
    )
    for column in list(factor_meta.get("component_columns") or []) + [str(factor_meta.get("output_col") or "factor_model_score")]:
        if column in factor_frame.columns:
            feature_df[column] = pd.to_numeric(factor_frame[column], errors="coerce")
    combined_score = _numeric_series(feature_df, "factor_model_score", default=0.0)
    return combined_score, combined_score, {
        "signal_combination": FACTOR_COMPONENT_SIGNAL_COMBINATION,
        "score_expression_used": FACTOR_COMPONENT_SIGNAL_COMBINATION,
        "score_source_field": "factor_model_score",
        "factor_components": list(factor_meta.get("components") or []),
    }


def _evaluate_score_expression(feature_df: pd.DataFrame, expression: str) -> pd.Series | None:
    if not expression:
        return None
    try:
        return pd.to_numeric(feature_df.eval(expression, engine="python"), errors="coerce").fillna(0.0)
    except (AttributeError, KeyError, NameError, SyntaxError, TypeError, ValueError, pd.errors.UndefinedVariableError):
        return None


def _direct_strategy_scores(
    feature_df: pd.DataFrame,
    *,
    action_source_field: str,
    combined_score_expr: str,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    direct_field = action_source_field or "signal_score"
    if direct_field in feature_df.columns:
        direct_score = _numeric_series(feature_df, direct_field, default=0.0)
        return direct_score, direct_score, {
            "signal_combination": DIRECT_SIGNAL_COMBINATION,
            "score_expression_used": direct_field,
            "score_source_field": direct_field,
        }
    expression_score = _evaluate_score_expression(feature_df, combined_score_expr)
    if expression_score is not None:
        return expression_score, expression_score, {
            "signal_combination": DIRECT_SIGNAL_COMBINATION,
            "score_expression_used": combined_score_expr,
            "score_source_field": "",
        }
    fallback_field = "ranking" if "ranking" in feature_df.columns else "prob_buy"
    direct_score = _numeric_series(feature_df, fallback_field, default=0.0)
    return direct_score, direct_score, {
        "signal_combination": DIRECT_SIGNAL_COMBINATION,
        "score_expression_used": fallback_field,
        "score_source_field": fallback_field,
    }


def _combined_signal_scores(
    *,
    signal_combination: str,
    prob_buy: pd.Series,
    ranking: pd.Series,
    ae_familiarity: pd.Series,
) -> tuple[pd.Series, str]:
    if signal_combination == MEAN_SIGNAL_COMBINATION:
        combined_score = pd.concat([prob_buy, ranking, ae_familiarity], axis=1).mean(axis=1, skipna=True).fillna(0.0)
        return combined_score, MEAN_SIGNAL_COMBINATION
    return (prob_buy * ranking * ae_familiarity).fillna(0.0), MULTIPLY_SIGNAL_COMBINATION


def _strategy_score_inputs(strategy_config: dict[str, Any]) -> StrategyScoreInputs:
    return StrategyScoreInputs(
        signal_combination=str(strategy_config.get("signal_combination") or MULTIPLY_SIGNAL_COMBINATION).strip().lower() or MULTIPLY_SIGNAL_COMBINATION,
        combined_score_expr=str(strategy_config.get("combined_score_expr") or "").strip(),
        action_source_field=str(strategy_config.get("action_source_field") or "").strip(),
        factor_components=list(
            strategy_config.get("factor_components")
            or strategy_config.get("cross_sectional_factor_components")
            or []
        ),
    )


def _compute_strategy_scores(
    feature_df: pd.DataFrame,
    *,
    strategy_type: str,
    strategy_config: dict[str, Any],
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    score_inputs = _strategy_score_inputs(strategy_config)

    prob_buy = _numeric_series(feature_df, "prob_buy", default=0.0)
    ranking = _numeric_series(feature_df, "ranking", default=0.0)
    ae_familiarity = _numeric_series(feature_df, "ae_familiarity", default=1.0)

    if score_inputs.factor_components:
        return _factor_component_scores(feature_df, factor_components=score_inputs.factor_components)

    if str(strategy_type) == "rl_policy_v1" or score_inputs.signal_combination == DIRECT_SIGNAL_COMBINATION:
        return _direct_strategy_scores(
            feature_df,
            action_source_field=score_inputs.action_source_field,
            combined_score_expr=score_inputs.combined_score_expr,
        )

    expression_score = _evaluate_score_expression(feature_df, score_inputs.combined_score_expr)
    if expression_score is not None:
        return expression_score, expression_score, {
            "signal_combination": score_inputs.signal_combination,
            "score_expression_used": score_inputs.combined_score_expr,
            "score_source_field": "",
        }

    combined_score, signal_combination = _combined_signal_scores(
        signal_combination=score_inputs.signal_combination,
        prob_buy=prob_buy,
        ranking=ranking,
        ae_familiarity=ae_familiarity,
    )
    return combined_score, combined_score, {
        "signal_combination": signal_combination,
        "score_expression_used": score_inputs.combined_score_expr or signal_combination,
        "score_source_field": "",
    }
