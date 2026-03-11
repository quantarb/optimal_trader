from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from .cohort_runner import COHORT_SUMMARY_SCHEMA_VERSION, DEFAULT_VALIDATION_CONFIG, run_walk_forward_model_cohort_backtests
from .models import Artifact, StrategyDefinition
from analysis.oracle_reports import summarize_prediction_artifact_set_oracle_coverage

RESEARCH_REPORT_SCHEMA_VERSION = 3


RESEARCH_PROFILES: dict[str, dict[str, Any]] = {
    "small_universe_fast": {
        "description": "Fast smoke profile for narrow universes and short history.",
        "feature_config": {
            "include_price_technicals": True,
            "include_fundamental_change": False,
            "include_statement_quality": True,
            "include_event_features": True,
            "include_ownership_features": False,
            "include_economic_indicators": False,
            "include_treasury_rates": False,
        },
        "feature_family_groups": [
            ["prices_div_adj"],
            ["prices_div_adj", "analyst_estimates"],
            ["prices_div_adj", "income_statement", "income_statement_growth"],
        ],
        "validation_config": {
            "min_trained_rows": 30,
            "min_rows_scored": 15,
            "min_selected_rows": 5,
            "min_trades": 5,
            "min_benchmark_days": 10,
            "min_valid_fold_rate": 0.5,
            "max_fold_excess_std": 0.75,
        },
        "backtest_config": {
            "fee_bps": 2.0,
            "slippage_bps": 4.0,
            "execution_delay_days": 1,
            "turnover_half_l1": True,
            "min_price": 5.0,
            "min_dollar_volume": 1_000_000.0,
        },
        "oracle_cluster_mode": "top_clusters",
        "oracle_cluster_top_n": 2,
        "oracle_cluster_min_rows": 5,
        "include_cluster_generalist": True,
    },
    "broad_universe": {
        "description": "Rich cross-sectional profile for larger universes.",
        "feature_config": {
            "include_price_technicals": True,
            "include_fundamental_change": True,
            "include_statement_quality": True,
            "include_event_features": True,
            "include_ownership_features": False,
            "include_economic_indicators": False,
            "include_treasury_rates": False,
        },
        "feature_family_groups": [
            ["prices_div_adj", "income_statement", "income_statement_growth"],
            ["prices_div_adj", "key_metrics", "ratios", "income_statement", "income_statement_growth"],
            ["prices_div_adj", "income_statement", "income_statement_growth", "analyst_estimates"],
        ],
        "validation_config": {
            "min_trained_rows": 150,
            "min_rows_scored": 75,
            "min_selected_rows": 20,
            "min_trades": 20,
            "min_benchmark_days": 40,
            "min_valid_fold_rate": 0.66,
            "max_fold_excess_std": 0.5,
        },
        "backtest_config": {
            "fee_bps": 3.0,
            "slippage_bps": 7.0,
            "execution_delay_days": 1,
            "turnover_half_l1": True,
            "min_price": 5.0,
            "min_dollar_volume": 2_500_000.0,
        },
        "oracle_cluster_mode": "top_clusters",
        "oracle_cluster_top_n": 3,
        "oracle_cluster_min_rows": 12,
        "include_cluster_generalist": True,
    },
    "long_history": {
        "description": "Long-horizon profile with macro and rates enabled.",
        "feature_config": {
            "include_price_technicals": True,
            "include_fundamental_change": True,
            "include_statement_quality": True,
            "include_event_features": True,
            "include_ownership_features": False,
            "include_economic_indicators": True,
            "include_treasury_rates": True,
        },
        "feature_family_groups": [
            ["prices_div_adj", "income_statement", "income_statement_growth", "economic_indicators", "treasury_rates"],
            ["prices_div_adj", "key_metrics", "ratios", "economic_indicators", "treasury_rates"],
            ["prices_div_adj", "income_statement", "income_statement_growth", "analyst_estimates", "economic_indicators", "treasury_rates"],
            ["prices_div_adj", "key_metrics", "ratios", "income_statement", "income_statement_growth", "analyst_estimates", "economic_indicators", "treasury_rates"],
        ],
        "validation_config": {
            "min_trained_rows": 250,
            "min_rows_scored": 100,
            "min_selected_rows": 30,
            "min_trades": 30,
            "min_benchmark_days": 60,
            "min_valid_fold_rate": 0.66,
            "max_fold_excess_std": 0.35,
        },
        "backtest_config": {
            "fee_bps": 4.0,
            "slippage_bps": 8.0,
            "execution_delay_days": 1,
            "turnover_half_l1": True,
            "min_price": 5.0,
            "min_dollar_volume": 5_000_000.0,
        },
        "oracle_cluster_mode": "top_clusters",
        "oracle_cluster_top_n": 3,
        "oracle_cluster_min_rows": 15,
        "include_cluster_generalist": True,
    },
    "broad_universe_long_history": {
        "description": "Institutional research profile for large universes and long history.",
        "feature_config": {
            "include_price_technicals": True,
            "include_fundamental_change": True,
            "include_statement_quality": True,
            "include_event_features": True,
            "include_ownership_features": False,
            "include_economic_indicators": True,
            "include_treasury_rates": True,
        },
        "feature_family_groups": [
            ["prices_div_adj", "income_statement", "income_statement_growth", "analyst_estimates"],
            ["prices_div_adj", "income_statement", "income_statement_growth", "economic_indicators", "treasury_rates"],
            ["prices_div_adj", "key_metrics", "ratios", "income_statement", "income_statement_growth", "economic_indicators", "treasury_rates"],
            ["prices_div_adj", "key_metrics", "ratios", "income_statement", "income_statement_growth", "analyst_estimates", "economic_indicators", "treasury_rates"],
            ["prices_div_adj", "key_metrics", "ratios", "analyst_estimates", "economic_indicators", "treasury_rates"],
            ["prices_div_adj", "income_statement", "income_statement_growth", "analyst_estimates", "economic_indicators", "treasury_rates"],
        ],
        "validation_config": {
            "min_trained_rows": 300,
            "min_rows_scored": 120,
            "min_selected_rows": 40,
            "min_trades": 40,
            "min_benchmark_days": 80,
            "min_valid_fold_rate": 0.75,
            "max_fold_excess_std": 0.3,
        },
        "backtest_config": {
            "fee_bps": 5.0,
            "slippage_bps": 10.0,
            "execution_delay_days": 1,
            "turnover_half_l1": True,
            "min_price": 5.0,
            "min_dollar_volume": 10_000_000.0,
        },
        "oracle_cluster_mode": "top_clusters",
        "oracle_cluster_top_n": 4,
        "oracle_cluster_min_rows": 20,
        "include_cluster_generalist": True,
    },
}


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
    if any(key not in payload for key in required_keys):
        return None
    if int(payload.get("schema_version") or 0) != int(schema_version):
        return None
    return payload


def research_profile_names() -> list[str]:
    return sorted(RESEARCH_PROFILES.keys())


def resolve_research_profile(profile_name: str) -> dict[str, Any]:
    key = str(profile_name or "").strip()
    if key not in RESEARCH_PROFILES:
        raise ValueError(f"Unknown research profile: {profile_name!r}. Available: {', '.join(research_profile_names())}")
    profile = dict(RESEARCH_PROFILES[key])
    profile["name"] = key
    return profile


def _default_suite_specs(*, min_profit_pct: float, profile: dict[str, Any]) -> list[dict[str, Any]]:
    feature_family_groups = [list(group) for group in list(profile.get("feature_family_groups") or [])]
    common_model_config = {
        "split_ratio": 1.0,
        "min_profit_pct": float(min_profit_pct),
        "feature_family_mode": "grouped_family",
        "feature_family_groups": feature_family_groups,
        "label_horizon_mode": "grouped_k",
        "label_k_groups": [[1], [2, 4]],
        "min_abs_trade_return_pct": 8.0,
        "max_hold_days": 90,
        "sample_weight_mode": "trade_return_abs",
        "oracle_cluster_mode": str(profile.get("oracle_cluster_mode") or ""),
        "oracle_cluster_top_n": int(profile.get("oracle_cluster_top_n") or 0),
        "oracle_cluster_min_rows": int(profile.get("oracle_cluster_min_rows") or 0),
        "include_cluster_generalist": bool(profile.get("include_cluster_generalist", True)),
    }
    return [
        {
            "suite_name": "regression_multiply",
            "fit_job": "fit_regressor",
            "base_model_config": {
                **common_model_config,
                "model_name": "optimal_trade_reg",
            },
            "strategy_config": {
                "gate_quantile": 0.6,
                "top_k": 3,
                "rebalance_freq": "W",
                "gross_exposure": 0.8,
                "selection_side": "long_only",
                "signal_combination": "multiply",
            },
        },
        {
            "suite_name": "regression_mean",
            "fit_job": "fit_regressor",
            "base_model_config": {
                **common_model_config,
                "model_name": "optimal_trade_reg_mean",
            },
            "strategy_config": {
                "gate_quantile": 0.5,
                "top_k": 3,
                "rebalance_freq": "W",
                "gross_exposure": 0.8,
                "selection_side": "long_only",
                "signal_combination": "mean",
            },
        },
        {
            "suite_name": "classifier_multiply",
            "fit_job": "fit_classifier",
            "base_model_config": {
                **common_model_config,
                "model_name": "optimal_trade_clf",
            },
            "strategy_config": {
                "gate_quantile": 0.65,
                "top_k": 2,
                "rebalance_freq": "W",
                "gross_exposure": 0.8,
                "selection_side": "long_only",
                "signal_combination": "multiply",
            },
        },
        {
            "suite_name": "mtl_mean",
            "fit_job": "fit_mtl",
            "base_model_config": {
                **common_model_config,
                "model_name": "optimal_trade_mtl",
            },
            "strategy_config": {
                "gate_quantile": 0.55,
                "top_k": 3,
                "rebalance_freq": "W",
                "gross_exposure": 0.8,
                "selection_side": "long_only",
                "signal_combination": "mean",
            },
        },
    ]


def _rank_leaderboard_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = [dict(row) for row in rows if bool(row.get("passed_stability_gates"))]
    ranked.sort(
        key=lambda row: (
            float(row.get("mean_fold_excess_cumulative_return") or 0.0),
            float(row.get("oracle_cluster_coverage_rate") or 0.0),
            float(row.get("oracle_recall") or 0.0),
            -float(row.get("fold_excess_cumulative_return_std") or 0.0),
            float(row.get("valid_fold_rate") or 0.0),
            -abs(float(row.get("walk_forward_max_drawdown") or 0.0)),
            float(row.get("walk_forward_excess_cumulative_return") or 0.0),
            float(row.get("walk_forward_final_equity") or 0.0),
        ),
        reverse=True,
    )
    return ranked


def _feature_signature(feature_families: Sequence[str]) -> str:
    cleaned = [str(value).strip() for value in list(feature_families or []) if str(value).strip()]
    return " + ".join(cleaned) if cleaned else "unattributed"


def _enrich_aggregate_rows_with_oracle_coverage(
    *,
    aggregate_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    label_artifact_id: int,
    selection_quantile: float = 0.8,
) -> list[dict[str, Any]]:
    if label_artifact_id <= 0 or not aggregate_rows:
        return [dict(row) for row in aggregate_rows]
    label_artifact = Artifact.objects.filter(pk=int(label_artifact_id), artifact_type="LABELS").first()
    if label_artifact is None:
        return [dict(row) for row in aggregate_rows]

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
        coverage = summarize_prediction_artifact_set_oracle_coverage(
            label_artifact=label_artifact,
            prediction_artifacts=prediction_artifacts,
            selection_quantile=float(selection_quantile),
        )
        item.update(coverage)
        item["feature_family_signature"] = _feature_signature(item.get("feature_families") or [])
        out.append(item)
    return out


def _build_oracle_family_rows(aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in aggregate_rows:
        signature = str(row.get("feature_family_signature") or _feature_signature(row.get("feature_families") or []))
        state = grouped.setdefault(
            signature,
            {
                "feature_family_signature": signature,
                "variants": 0,
                "oracle_recall_sum": 0.0,
                "oracle_cluster_coverage_sum": 0.0,
                "mean_fold_excess_sum": 0.0,
                "walk_forward_excess_sum": 0.0,
            },
        )
        state["variants"] += 1
        state["oracle_recall_sum"] += float(row.get("oracle_recall") or 0.0)
        state["oracle_cluster_coverage_sum"] += float(row.get("oracle_cluster_coverage_rate") or 0.0)
        state["mean_fold_excess_sum"] += float(row.get("mean_fold_excess_cumulative_return") or 0.0)
        state["walk_forward_excess_sum"] += float(row.get("walk_forward_excess_cumulative_return") or 0.0)

    family_rows: list[dict[str, Any]] = []
    for state in grouped.values():
        variants = max(int(state["variants"]), 1)
        family_rows.append(
            {
                "feature_family_signature": str(state["feature_family_signature"]),
                "variants": int(variants),
                "avg_oracle_recall": round(float(state["oracle_recall_sum"] / variants), 6),
                "avg_oracle_cluster_coverage_rate": round(float(state["oracle_cluster_coverage_sum"] / variants), 6),
                "avg_mean_fold_excess_cumulative_return": round(float(state["mean_fold_excess_sum"] / variants), 6),
                "avg_walk_forward_excess_cumulative_return": round(float(state["walk_forward_excess_sum"] / variants), 6),
            }
        )
    family_rows.sort(
        key=lambda row: (
            float(row.get("avg_oracle_cluster_coverage_rate") or 0.0),
            float(row.get("avg_oracle_recall") or 0.0),
            float(row.get("avg_mean_fold_excess_cumulative_return") or 0.0),
        ),
        reverse=True,
    )
    return family_rows


def _build_report_summary(
    *,
    profile: dict[str, Any],
    suite_outputs: list[dict[str, Any]],
    leaderboard_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    failure_reasons: dict[str, int] = {}
    for suite in suite_outputs:
        if str(suite.get("status")) == "failed":
            reason = str(suite.get("error") or "suite_failed")
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    rejection_reasons: dict[str, int] = {}
    for row in rejected_rows:
        for reason in list(row.get("stability_gate_reasons") or []):
            key = str(reason)
            rejection_reasons[key] = rejection_reasons.get(key, 0) + 1

    best_variant = leaderboard_rows[0] if leaderboard_rows else None
    specialist_rows = [dict(row) for row in aggregate_rows if str(row.get("oracle_cluster_scope") or "generalist") == "specialist"]
    best_specialist = _rank_leaderboard_rows(specialist_rows)[0] if specialist_rows else None
    runtime_totals = {
        "dataset_build": sum(float(row.get("avg_dataset_build_seconds") or 0.0) for row in aggregate_rows),
        "fit": sum(float(row.get("avg_fit_seconds") or 0.0) for row in aggregate_rows),
        "score": sum(float(row.get("avg_score_seconds") or 0.0) for row in aggregate_rows),
        "strategy": sum(float(row.get("avg_strategy_build_seconds") or 0.0) for row in aggregate_rows),
        "backtest": sum(float(row.get("avg_backtest_seconds") or 0.0) for row in aggregate_rows),
    }
    runtime_totals["total"] = sum(runtime_totals.values())
    slowest_stage_name, slowest_stage_seconds = max(runtime_totals.items(), key=lambda item: item[1]) if runtime_totals else ("total", 0.0)
    recommendations: list[str] = []
    if leaderboard_rows and abs(float(best_variant.get("walk_forward_max_drawdown") or 0.0)) >= 0.35:
        recommendations.append("Top candidate still carries large drawdown. Tighten exposure, widen breadth, or reduce turnover before promotion.")
    if leaderboard_rows and float(best_variant.get("oracle_cluster_coverage_rate") or 0.0) < 0.25:
        recommendations.append("Top candidate still recovers a narrow oracle subset. Prefer broader feature bundles or MTL before trusting the policy layer.")
    if best_specialist is not None and best_variant is not None:
        if float(best_specialist.get("oracle_recall") or 0.0) > float(best_variant.get("oracle_recall") or 0.0):
            recommendations.append("Cluster-specialist variants recover more oracle rows than the global leader. Promote specialist heads or mixture-of-experts into the next pass.")
    if rejected_rows:
        most_common_reason = max(rejection_reasons.items(), key=lambda item: item[1])[0]
        recommendations.append(f"Most rejected variants failed on '{most_common_reason}'. Prioritize experiments that address that gate.")
    if slowest_stage_name == "backtest" and slowest_stage_seconds > 0:
        recommendations.append("Backtests remain the slowest stage. Reuse strategy datasets, increase vectorization, and tighten the candidate set before large sweeps.")
    if slowest_stage_name == "dataset_build" and slowest_stage_seconds > 0:
        recommendations.append("Dataset assembly dominates runtime. Materialize reusable label and feature artifacts before expanding the experiment grid.")
    if not recommendations:
        recommendations.append("Current leaderboard passed the configured gates. Next step is a broader walk-forward scope using the same profile.")
    return {
        "research_profile": str(profile.get("name") or ""),
        "profile_description": str(profile.get("description") or ""),
        "suite_count": int(len(suite_outputs)),
        "succeeded_suite_count": int(sum(1 for suite in suite_outputs if str(suite.get("status")) == "succeeded")),
        "failed_suite_count": int(sum(1 for suite in suite_outputs if str(suite.get("status")) == "failed")),
        "leaderboard_count": int(len(leaderboard_rows)),
        "rejected_count": int(len(rejected_rows)),
        "best_variant_name": str(best_variant.get("variant_name") or "") if best_variant else "",
        "best_excess_cumulative_return": float(best_variant.get("walk_forward_excess_cumulative_return") or 0.0) if best_variant else 0.0,
        "best_mean_fold_excess_cumulative_return": float(best_variant.get("mean_fold_excess_cumulative_return") or 0.0) if best_variant else 0.0,
        "best_fold_excess_cumulative_return_std": float(best_variant.get("fold_excess_cumulative_return_std") or 0.0) if best_variant else 0.0,
        "best_valid_fold_rate": float(best_variant.get("valid_fold_rate") or 0.0) if best_variant else 0.0,
        "best_oracle_recall": float(best_variant.get("oracle_recall") or 0.0) if best_variant else 0.0,
        "best_oracle_cluster_coverage_rate": float(best_variant.get("oracle_cluster_coverage_rate") or 0.0) if best_variant else 0.0,
        "cluster_specialist_variant_count": int(len(specialist_rows)),
        "best_specialist_variant_name": str(best_specialist.get("variant_name") or "") if best_specialist else "",
        "best_specialist_oracle_recall": float(best_specialist.get("oracle_recall") or 0.0) if best_specialist else 0.0,
        "best_specialist_cluster_coverage_rate": float(best_specialist.get("oracle_cluster_coverage_rate") or 0.0) if best_specialist else 0.0,
        "failure_reasons": failure_reasons,
        "rejection_reasons": rejection_reasons,
        "runtime_summary": {
            "stage_totals_seconds": {key: round(float(value), 6) for key, value in runtime_totals.items()},
            "slowest_stage": str(slowest_stage_name),
            "slowest_stage_seconds": round(float(slowest_stage_seconds), 6),
            "mean_variant_runtime_seconds": round(
                float(sum(float(row.get("avg_total_runtime_seconds") or 0.0) for row in aggregate_rows) / float(len(aggregate_rows))),
                6,
            ) if aggregate_rows else 0.0,
            "variant_count_with_runtime": int(len(aggregate_rows)),
        },
        "recommendations": recommendations,
    }


def run_optimal_trade_research_suite(
    *,
    symbols: Sequence[str],
    folds: Sequence[dict[str, Any]],
    min_profit_pct: float,
    transaction_cost_bps: float,
    profile_name: str = "broad_universe_long_history",
    validation_config_override: dict[str, Any] | None = None,
    universe_artifact: Artifact | None = None,
    label_artifact: Artifact | None = None,
    feature_artifact: Artifact | None = None,
    strategy_definition: StrategyDefinition | None = None,
    output_basename: str = "optimal_trade_research_suite",
    resume_existing: bool = False,
) -> dict[str, Any]:
    output_dir = Path("data") / "pipeline_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{output_basename}.json"
    csv_path = output_dir / f"{output_basename}.csv"
    if resume_existing:
        cached_payload = _load_cached_payload(
            json_path,
            required_keys=("leaderboard_rows", "suite_outputs", "report_summary"),
            schema_version=RESEARCH_REPORT_SCHEMA_VERSION,
        )
        if cached_payload is not None:
            cached_payload["summary_json_path"] = str(json_path)
            cached_payload["summary_csv_path"] = str(csv_path)
            return cached_payload

    profile = resolve_research_profile(profile_name)
    validation_config = dict(DEFAULT_VALIDATION_CONFIG)
    validation_config.update(dict(profile.get("validation_config") or {}))
    validation_config.update(dict(validation_config_override or {}))
    feature_config = dict(profile.get("feature_config") or {})
    backtest_config = dict(profile.get("backtest_config") or {})
    suite_specs = _default_suite_specs(min_profit_pct=float(min_profit_pct), profile=profile)

    suite_outputs: list[dict[str, Any]] = []
    all_aggregate_rows: list[dict[str, Any]] = []
    oracle_variant_rows: list[dict[str, Any]] = []
    for spec in suite_specs:
        suite_json_path = output_dir / f"{output_basename}__{spec['suite_name']}.json"
        try:
            payload = None
            if resume_existing:
                payload = _load_cached_payload(
                    suite_json_path,
                    required_keys=("aggregate_rows", "summary_rows", "folds"),
                    schema_version=COHORT_SUMMARY_SCHEMA_VERSION,
                )
                if payload is not None:
                    payload["summary_json_path"] = str(suite_json_path)
                    payload["summary_csv_path"] = str(output_dir / f"{output_basename}__{spec['suite_name']}.csv")
            if payload is None:
                payload = run_walk_forward_model_cohort_backtests(
                    symbols=symbols,
                    fit_job=str(spec["fit_job"]),
                    base_model_config=dict(spec["base_model_config"]),
                    folds=folds,
                    universe_artifact=universe_artifact,
                    label_artifact=label_artifact,
                    feature_artifact=feature_artifact,
                    feature_config=feature_config,
                    strategy_definition=strategy_definition,
                    strategy_definition_slug=f"{spec['suite_name']}-strategy",
                    strategy_definition_name=f"{spec['suite_name']} strategy",
                    strategy_config=dict(spec["strategy_config"]),
                    validation_config=validation_config,
                    transaction_cost_bps=float(transaction_cost_bps),
                    backtest_config=backtest_config,
                    output_basename=f"{output_basename}__{spec['suite_name']}",
                    resume_existing=resume_existing,
                )
            base_artifacts = dict(payload.get("base_artifacts") or {})
            resolved_label_artifact_id = int(base_artifacts.get("labels") or (label_artifact.id if label_artifact is not None else 0))
            aggregate_rows = _enrich_aggregate_rows_with_oracle_coverage(
                aggregate_rows=[dict(row) for row in list(payload.get("aggregate_rows") or [])],
                summary_rows=[dict(row) for row in list(payload.get("summary_rows") or [])],
                label_artifact_id=resolved_label_artifact_id,
                selection_quantile=0.8,
            )
            for row in aggregate_rows:
                row["suite_name"] = str(spec["suite_name"])
                row["fit_job"] = str(spec["fit_job"])
                row["signal_combination"] = str(spec["strategy_config"].get("signal_combination") or "")
                row["top_k"] = int(spec["strategy_config"].get("top_k") or 0)
                row["gate_quantile"] = float(spec["strategy_config"].get("gate_quantile") or 0.0)
                row["research_profile"] = str(profile["name"])
                all_aggregate_rows.append(row)
                oracle_variant_rows.append(
                    {
                        "variant_name": str(row.get("variant_name") or ""),
                        "suite_name": str(spec["suite_name"]),
                        "fit_job": str(spec["fit_job"]),
                        "feature_family_signature": str(row.get("feature_family_signature") or ""),
                        "oracle_recall": float(row.get("oracle_recall") or 0.0),
                        "oracle_cluster_coverage_rate": float(row.get("oracle_cluster_coverage_rate") or 0.0),
                        "oracle_selected_avg_trade_return": float(row.get("oracle_selected_avg_trade_return") or 0.0),
                        "mean_fold_excess_cumulative_return": float(row.get("mean_fold_excess_cumulative_return") or 0.0),
                        "walk_forward_excess_cumulative_return": float(row.get("walk_forward_excess_cumulative_return") or 0.0),
                        "oracle_cluster_scope": str(row.get("oracle_cluster_scope") or "generalist"),
                        "oracle_cluster_keys": list(row.get("oracle_cluster_keys") or []),
                        "passed_stability_gates": bool(row.get("passed_stability_gates")),
                    }
                )
            suite_outputs.append(
                {
                    "suite_name": str(spec["suite_name"]),
                    "fit_job": str(spec["fit_job"]),
                    "status": "succeeded",
                    "research_profile": str(profile["name"]),
                    "feature_config": feature_config,
                    "validation_config": validation_config,
                    "strategy_config": dict(spec["strategy_config"]),
                    "backtest_config": backtest_config,
                    "summary_json_path": str(payload.get("summary_json_path") or ""),
                    "summary_csv_path": str(payload.get("summary_csv_path") or ""),
                    "aggregate_rows": aggregate_rows,
                }
            )
        except Exception as exc:
            suite_outputs.append(
                {
                    "suite_name": str(spec["suite_name"]),
                    "fit_job": str(spec["fit_job"]),
                    "status": "failed",
                    "research_profile": str(profile["name"]),
                    "feature_config": feature_config,
                    "validation_config": validation_config,
                    "strategy_config": dict(spec["strategy_config"]),
                    "backtest_config": backtest_config,
                    "error": str(exc),
                    "aggregate_rows": [],
                }
            )

    leaderboard_rows = _rank_leaderboard_rows(all_aggregate_rows)
    rejected_rows = [dict(row) for row in all_aggregate_rows if not bool(row.get("passed_stability_gates"))]
    oracle_variant_rows.sort(
        key=lambda row: (
            float(row.get("oracle_cluster_coverage_rate") or 0.0),
            float(row.get("oracle_recall") or 0.0),
            float(row.get("mean_fold_excess_cumulative_return") or 0.0),
        ),
        reverse=True,
    )
    oracle_feature_family_rows = _build_oracle_family_rows(all_aggregate_rows)
    report_summary = _build_report_summary(
        profile=profile,
        suite_outputs=suite_outputs,
        leaderboard_rows=leaderboard_rows,
        rejected_rows=rejected_rows,
        aggregate_rows=all_aggregate_rows,
    )

    payload = {
        "schema_version": RESEARCH_REPORT_SCHEMA_VERSION,
        "symbols": list(symbols),
        "folds": [dict(fold) for fold in folds],
        "min_profit_pct": float(min_profit_pct),
        "transaction_cost_bps": float(transaction_cost_bps),
        "research_profile": {
            "name": str(profile["name"]),
            "description": str(profile.get("description") or ""),
            "feature_config": feature_config,
            "feature_family_groups": [list(group) for group in list(profile.get("feature_family_groups") or [])],
            "oracle_cluster_mode": str(profile.get("oracle_cluster_mode") or ""),
            "oracle_cluster_top_n": int(profile.get("oracle_cluster_top_n") or 0),
            "oracle_cluster_min_rows": int(profile.get("oracle_cluster_min_rows") or 0),
            "validation_config": validation_config,
            "backtest_config": backtest_config,
        },
        "suite_outputs": suite_outputs,
        "leaderboard_rows": leaderboard_rows,
        "rejected_rows": rejected_rows,
        "oracle_variant_rows": oracle_variant_rows,
        "oracle_feature_family_rows": oracle_feature_family_rows,
        "report_summary": report_summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_rows_csv(csv_path, leaderboard_rows)
    payload["summary_json_path"] = str(json_path)
    payload["summary_csv_path"] = str(csv_path)
    return payload
