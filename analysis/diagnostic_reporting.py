from __future__ import annotations

from typing import Any

import pandas as pd

from pipeline.models import Artifact


MAX_QUANTILE_BUCKETS = 5
RATE_DECIMALS = 4
RETURN_DECIMALS = 6
PERCENT_MULTIPLIER = 100.0
MAX_ACCEPTABLE_DRAWDOWN = 0.35
HIGH_TURNOVER_THRESHOLD = 0.75
MIN_POSITION_BREADTH = 3.0
MIN_TOP_BUCKET_WIN_RATE = 0.7
MIN_REGRESSOR_EDGE = 0.1
MIN_COMBINED_SIGNAL_WIN_RATE = 0.8
MIN_RULE_SAMPLE_ROWS = 25
AE_SIGNAL_MISMATCH_MULTIPLIER = 10.0
MAX_TRUSTED_RL_BUY_COUNT = 5
MAX_ACCEPTABLE_RL_DRAWDOWN_PCT = 40.0


def _quantile_bucket_report(df: pd.DataFrame, value_col: str, *, high_is_good: bool) -> list[dict[str, Any]]:
    work = df[[value_col, "label", "trade_return"]].dropna().copy()
    if work.empty:
        return []
    rank = work[value_col].rank(method="first", pct=True, ascending=high_is_good)
    buckets = min(MAX_QUANTILE_BUCKETS, int(rank.nunique()))
    if buckets <= 1:
        return []
    work["bucket"] = pd.qcut(rank, q=buckets, duplicates="drop")
    return [
        {
            "bucket": str(bucket),
            "rows": int(len(group)),
            "win_rate": round(float(group["label"].mean()), RATE_DECIMALS),
            "avg_trade_return": round(float(group["trade_return"].mean()), RETURN_DECIMALS),
            "median_trade_return": round(float(group["trade_return"].median()), RETURN_DECIMALS),
        }
        for bucket, group in work.groupby("bucket", observed=True)
    ]


def candidate_rule(
    panel: pd.DataFrame,
    *,
    classifier_quantile: float,
    regressor_quantile: float,
    autoencoder_quantile: float,
) -> dict[str, Any]:
    if panel.empty:
        return {}
    prob_thr = float(panel["prob_buy"].quantile(classifier_quantile))
    reg_thr = float(panel["pred_rf_reg"].quantile(regressor_quantile))
    ae_thr = float(panel["ae_familiarity"].quantile(autoencoder_quantile))
    selected = panel[
        (panel["prob_buy"] >= prob_thr)
        & (panel["pred_rf_reg"] >= reg_thr)
        & (panel["ae_familiarity"] >= ae_thr)
    ].copy()
    return {
        "description": "Long-only when classifier probability, regressor return, and AE familiarity all clear elevated thresholds.",
        "prob_buy_gte": prob_thr,
        "pred_rf_reg_gte": reg_thr,
        "ae_familiarity_gte": ae_thr,
        "rows": int(len(selected)),
        "win_rate": round(float(selected["label"].mean()), RATE_DECIMALS) if len(selected) else None,
        "avg_trade_return": round(float(selected["trade_return"].mean()), RETURN_DECIMALS) if len(selected) else None,
        "median_trade_return": round(float(selected["trade_return"].median()), RETURN_DECIMALS) if len(selected) else None,
    }


def prediction_quantiles(panel: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    return {
        "classifier_prob_buy": _quantile_bucket_report(panel, "prob_buy", high_is_good=True),
        "regressor_trade_return": _quantile_bucket_report(panel, "pred_rf_reg", high_is_good=True),
        "autoencoder_familiarity": _quantile_bucket_report(panel, "ae_familiarity", high_is_good=True),
        "autoencoder_raw_reconstruction_error": _quantile_bucket_report(panel, "ae_recon_error", high_is_good=False),
        "combined_rank_mean": _quantile_bucket_report(panel, "combined_rank_mean", high_is_good=True),
    }


def ae_signal_bug_check(strategy: pd.DataFrame, panel: pd.DataFrame) -> dict[str, Any]:
    strategy_median = None
    if not strategy.empty and "ae_familiarity" in strategy.columns:
        strategy_median = float(pd.to_numeric(strategy.get("ae_familiarity"), errors="coerce").median())
    return {
        "strategy_dataset_ae_familiarity_median": strategy_median,
        "raw_autoencoder_familiarity_median": float(panel["ae_familiarity"].median()) if not panel.empty else None,
        "raw_autoencoder_reconstruction_median": float(panel["ae_recon_error"].median()) if not panel.empty else None,
    }


def average_daily_metric(daily_rows: list[dict[str, Any]], key: str) -> float:
    if not daily_rows:
        return 0.0
    return float(sum(float(row.get(key) or 0.0) for row in daily_rows) / len(daily_rows))


def backtest_summary_context(backtest_artifact: Artifact | None) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = dict(backtest_artifact.content or {}) if backtest_artifact is not None else {}
    backtest_cfg = dict((backtest_artifact.metadata or {}).get("backtest_config") or {}) if backtest_artifact is not None else {}
    daily_rows = list(summary.get("daily_rows") or [])
    summary["avg_positions"] = average_daily_metric(daily_rows, "positions")
    summary["avg_turnover"] = average_daily_metric(daily_rows, "turnover")
    return summary, backtest_cfg


def artifact_reference_ids(
    *,
    label_artifact: Artifact,
    classifier_predictions_artifact: Artifact,
    regressor_predictions_artifact: Artifact,
    autoencoder_scores_artifact: Artifact,
    strategy_artifact: Artifact | None,
    backtest_artifact: Artifact | None,
) -> dict[str, int]:
    return {
        "labels": int(label_artifact.id),
        "classifier_predictions": int(classifier_predictions_artifact.id),
        "regressor_predictions": int(regressor_predictions_artifact.id),
        "autoencoder_scores": int(autoencoder_scores_artifact.id),
        "strategy_dataset": int(strategy_artifact.id) if strategy_artifact is not None else 0,
        "backtest_result": int(backtest_artifact.id) if backtest_artifact is not None else 0,
    }


def runtime_summary(
    *,
    label_artifact: Artifact,
    model_metrics: dict[str, dict[str, Any]],
    backtest_artifact: Artifact | None,
) -> dict[str, Any]:
    summary = {
        "labels": float((label_artifact.content or {}).get("job_duration_seconds") or 0.0),
        "fit_classifier": float(model_metrics["classifier"].get("job_duration_seconds") or 0.0),
        "fit_regressor": float(model_metrics["regressor"].get("job_duration_seconds") or 0.0),
        "fit_autoencoder": float(model_metrics["autoencoder"].get("job_duration_seconds") or 0.0),
        "backtest": float((backtest_artifact.content or {}).get("backtest_seconds") or 0.0) if backtest_artifact is not None else 0.0,
    }
    summary["slowest_stage"] = max(summary, key=summary.get) if summary else ""
    return summary


def base_diagnostic_report(
    *,
    label_artifact: Artifact,
    classifier_predictions_artifact: Artifact,
    regressor_predictions_artifact: Artifact,
    autoencoder_scores_artifact: Artifact,
    strategy_artifact: Artifact | None,
    backtest_artifact: Artifact | None,
    panel: pd.DataFrame,
    strategy: pd.DataFrame,
    model_metrics: dict[str, dict[str, Any]],
    backtest_summary: dict[str, Any],
    backtest_cfg: dict[str, Any],
    classifier_quantile: float,
    regressor_quantile: float,
    autoencoder_quantile: float,
) -> dict[str, Any]:
    return {
        "kind": "diagnostic_report",
        "artifacts": artifact_reference_ids(
            label_artifact=label_artifact,
            classifier_predictions_artifact=classifier_predictions_artifact,
            regressor_predictions_artifact=regressor_predictions_artifact,
            autoencoder_scores_artifact=autoencoder_scores_artifact,
            strategy_artifact=strategy_artifact,
            backtest_artifact=backtest_artifact,
        ),
        "label_summary": dict(label_artifact.content or {}),
        "model_metrics": model_metrics,
        "prediction_quantiles": prediction_quantiles(panel),
        "candidate_rule": candidate_rule(
            panel,
            classifier_quantile=classifier_quantile,
            regressor_quantile=regressor_quantile,
            autoencoder_quantile=autoencoder_quantile,
        ),
        "backtest_summary": backtest_summary,
        "backtest_config": backtest_cfg,
        "ae_signal_bug_check": ae_signal_bug_check(strategy, panel),
    }


def _top_quantile_row(quantiles: dict[str, Any], key: str) -> dict[str, Any]:
    rows = list(quantiles.get(key) or [])
    return dict(rows[-1]) if rows else {}


def _backtest_recommendations(backtest: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    if not backtest:
        return recommendations
    if abs(float(backtest.get("max_drawdown") or 0.0)) >= MAX_ACCEPTABLE_DRAWDOWN:
        recommendations.append("Drawdown is too high. Reduce gross exposure, cap single-name weights harder, and test slower rebalance frequencies.")
    if float(backtest.get("avg_turnover") or 0.0) >= HIGH_TURNOVER_THRESHOLD:
        recommendations.append("Turnover is high enough that cost stress should be widened. Re-run with larger slippage and tighter liquidity floors.")
    if float(backtest.get("avg_positions") or 0.0) < MIN_POSITION_BREADTH:
        recommendations.append("The system is concentrated. Increase breadth with higher top-k or gentler gates before trusting live deployability.")
    return recommendations


def _signal_recommendations(
    *,
    classifier_top: dict[str, Any],
    regressor_top: dict[str, Any],
    combined_top: dict[str, Any],
    rule: dict[str, Any],
    ae_check: dict[str, Any],
) -> list[str]:
    recommendations: list[str] = []
    if classifier_top and float(classifier_top.get("win_rate") or 0.0) < MIN_TOP_BUCKET_WIN_RATE:
        recommendations.append("Classifier calibration is weak in the top bucket. Revisit the label recipe or add stronger class-balance controls.")
    if regressor_top and float(regressor_top.get("avg_trade_return") or 0.0) <= MIN_REGRESSOR_EDGE:
        recommendations.append("Regressor spread is weak. Expand feature bundles or raise the minimum profit threshold to sharpen the target.")
    if (
        combined_top
        and float(combined_top.get("avg_trade_return") or 0.0) > 0.0
        and float(combined_top.get("win_rate") or 0.0) >= MIN_COMBINED_SIGNAL_WIN_RATE
    ):
        recommendations.append("The combined signal is materially cleaner than the single-model outputs. Promote it as the default rules baseline for future sweeps.")
    if rule and int(rule.get("rows") or 0) < MIN_RULE_SAMPLE_ROWS:
        recommendations.append("The candidate rule is too sparse. Lower at least one threshold or widen the universe before expecting consistent live frequency.")
    if ae_check:
        strategy_median = float(ae_check.get("strategy_dataset_ae_familiarity_median") or 0.0)
        raw_median = float(ae_check.get("raw_autoencoder_familiarity_median") or 0.0)
        if raw_median > 0.0 and strategy_median > raw_median * AE_SIGNAL_MISMATCH_MULTIPLIER:
            recommendations.append("AE familiarity in the strategy dataset does not match the raw scored signal. Fix score plumbing before trusting AE-based selection.")
    return recommendations


def _rl_recommendations(best_rl: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    if not best_rl:
        return recommendations
    if int(best_rl.get("executed_buys") or 0) <= MAX_TRUSTED_RL_BUY_COUNT:
        recommendations.append("The RL policy is too sparse to trust. Add action penalties for inactivity and compare against simpler threshold rules before promotion.")
    if abs(float(best_rl.get("combined_max_drawdown_pct") or 0.0)) >= MAX_ACCEPTABLE_RL_DRAWDOWN_PCT:
        recommendations.append("RL returns come with extreme drawdown. Increase drawdown penalty or add exposure constraints in the environment.")
    return recommendations


def _runtime_recommendations(runtime_summary_row: dict[str, Any]) -> list[str]:
    if not runtime_summary_row:
        return []
    bottleneck = str(runtime_summary_row.get("slowest_stage") or "")
    if bottleneck == "backtest":
        return ["Backtests dominate runtime. Reuse the same features and scores across strategy sweeps and keep the vectorized backtest path as the default."]
    if bottleneck == "fit_regressor":
        return ["Regressor fitting is the main runtime bottleneck. Narrow hyperparameter search before widening the universe."]
    return []


def build_recommendations(*, report: dict[str, Any]) -> list[str]:
    backtest = dict(report.get("backtest_summary") or {})
    rule = dict(report.get("candidate_rule") or {})
    quantiles = dict(report.get("prediction_quantiles") or {})
    classifier_top = _top_quantile_row(quantiles, "classifier_prob_buy")
    regressor_top = _top_quantile_row(quantiles, "regressor_trade_return")
    combined_top = _top_quantile_row(quantiles, "combined_rank_mean")
    ae_check = dict(report.get("ae_signal_bug_check") or {})
    best_rl = dict(report.get("best_rl_result") or {})
    runtime_summary_row = dict(report.get("runtime_summary") or {})
    recommendations = [
        *_backtest_recommendations(backtest),
        *_signal_recommendations(
            classifier_top=classifier_top,
            regressor_top=regressor_top,
            combined_top=combined_top,
            rule=rule,
            ae_check=ae_check,
        ),
        *_rl_recommendations(best_rl),
        *_runtime_recommendations(runtime_summary_row),
    ]
    if not recommendations:
        recommendations.append("No immediate red flags were detected. The next step is a broader universe walk-forward run with the same diagnostics enabled.")
    return recommendations


def build_observations(*, report: dict[str, Any]) -> list[str]:
    observations: list[str] = []
    labels = dict(report.get("label_summary") or {})
    label_stats = dict(labels.get("statistics", {}).get("trade_stats", {}))
    backtest = dict(report.get("backtest_summary") or {})
    best_rl = dict(report.get("best_rl_result") or {})
    if label_stats:
        observations.append(
            f"Oracle labels contain {int(label_stats.get('total_trades') or 0)} trades across {int(label_stats.get('symbols_count') or 0)} symbols with median trade return {float(label_stats.get('median_return_pct') or 0.0):.2f}%."
        )
    if backtest:
        observations.append(
            f"Rules backtest finished at {float(backtest.get('final_equity') or 0.0):.2f}x with max drawdown {float(backtest.get('max_drawdown') or 0.0) * PERCENT_MULTIPLIER:.2f}%."
        )
    if best_rl:
        observations.append(
            f"Best RL sweep result returned {float(best_rl.get('combined_total_return_pct') or 0.0):.2f}% on the 2024-2025 evaluation slice."
        )
    return observations


def finalize_diagnostic_report(
    *,
    report: dict[str, Any],
    label_artifact: Artifact,
    model_metrics: dict[str, dict[str, Any]],
    backtest_artifact: Artifact | None,
    rl_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    report["runtime_summary"] = runtime_summary(
        label_artifact=label_artifact,
        model_metrics=model_metrics,
        backtest_artifact=backtest_artifact,
    )
    report["rl_results"] = rl_rows
    report["best_rl_result"] = rl_rows[0] if rl_rows else None
    report["observations"] = build_observations(report=report)
    report["recommendations"] = build_recommendations(report=report)
    return report
