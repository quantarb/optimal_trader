from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml import RLConfig, run_a2c_workflow, run_ppo_workflow

from pipeline.models import Artifact
from pipeline.service_runtime import read_frame_artifact


def _model_payload_from_prediction_artifact(prediction_artifact: Artifact, expected_type: str) -> dict[str, Any]:
    source_model_id = int((prediction_artifact.metadata or {}).get("source_model_artifact_id") or 0)
    model_artifact = Artifact.objects.filter(pk=source_model_id, artifact_type=expected_type).first()
    return dict(model_artifact.content or {}) if model_artifact is not None else {}


def _read_csv_artifact(artifact: Artifact) -> pd.DataFrame:
    path = Path(str(artifact.uri or ""))
    if not path.exists():
        raise ValueError(f"Artifact #{artifact.id} file does not exist.")
    df = read_frame_artifact(artifact)
    return df


def _score_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing expected columns: {', '.join(candidates)}")


def _quantile_bucket_report(df: pd.DataFrame, value_col: str, *, high_is_good: bool) -> list[dict[str, Any]]:
    work = df[[value_col, "label", "trade_return"]].dropna().copy()
    if work.empty:
        return []
    rank = work[value_col].rank(method="first", pct=True, ascending=high_is_good)
    buckets = min(5, int(rank.nunique()))
    if buckets <= 1:
        return []
    work["bucket"] = pd.qcut(rank, q=buckets, duplicates="drop")
    rows: list[dict[str, Any]] = []
    for bucket, group in work.groupby("bucket", observed=True):
        rows.append(
            {
                "bucket": str(bucket),
                "rows": int(len(group)),
                "win_rate": round(float(group["label"].mean()), 4),
                "avg_trade_return": round(float(group["trade_return"].mean()), 6),
                "median_trade_return": round(float(group["trade_return"].median()), 6),
            }
        )
    return rows


def _candidate_rule(panel: pd.DataFrame) -> dict[str, Any]:
    if panel.empty:
        return {}
    prob_thr = float(panel["prob_buy"].quantile(0.8))
    reg_thr = float(panel["pred_rf_reg"].quantile(0.8))
    ae_thr = float(panel["ae_familiarity"].quantile(0.6))
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
        "win_rate": round(float(selected["label"].mean()), 4) if len(selected) else None,
        "avg_trade_return": round(float(selected["trade_return"].mean()), 6) if len(selected) else None,
        "median_trade_return": round(float(selected["trade_return"].median()), 6) if len(selected) else None,
    }


def _build_recommendations(*, report: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    backtest = dict(report.get("backtest_summary") or {})
    rule = dict(report.get("candidate_rule") or {})
    quantiles = dict(report.get("prediction_quantiles") or {})
    classifier_top = list(quantiles.get("classifier_prob_buy") or [])[-1:] or []
    regressor_top = list(quantiles.get("regressor_trade_return") or [])[-1:] or []
    combined_top = list(quantiles.get("combined_rank_mean") or [])[-1:] or []
    ae_check = dict(report.get("ae_signal_bug_check") or {})
    best_rl = dict(report.get("best_rl_result") or {})
    runtime_summary = dict(report.get("runtime_summary") or {})

    if backtest:
        if abs(float(backtest.get("max_drawdown") or 0.0)) >= 0.35:
            recommendations.append("Drawdown is too high. Reduce gross exposure, cap single-name weights harder, and test slower rebalance frequencies.")
        if float(backtest.get("avg_turnover") or 0.0) >= 0.75:
            recommendations.append("Turnover is high enough that cost stress should be widened. Re-run with larger slippage and tighter liquidity floors.")
        if float(backtest.get("avg_positions") or 0.0) < 3.0:
            recommendations.append("The system is concentrated. Increase breadth with higher top-k or gentler gates before trusting live deployability.")
    if classifier_top and float(classifier_top[0].get("win_rate") or 0.0) < 0.7:
        recommendations.append("Classifier calibration is weak in the top bucket. Revisit the label recipe or add stronger class-balance controls.")
    if regressor_top and float(regressor_top[0].get("avg_trade_return") or 0.0) <= 0.1:
        recommendations.append("Regressor spread is weak. Expand feature bundles or raise the minimum profit threshold to sharpen the target.")
    if combined_top and float(combined_top[0].get("avg_trade_return") or 0.0) > 0.0 and float(combined_top[0].get("win_rate") or 0.0) >= 0.8:
        recommendations.append("The combined signal is materially cleaner than the single-model outputs. Promote it as the default rules baseline for future sweeps.")
    if rule and int(rule.get("rows") or 0) < 25:
        recommendations.append("The candidate rule is too sparse. Lower at least one threshold or widen the universe before expecting consistent live frequency.")
    if ae_check:
        strategy_median = float(ae_check.get("strategy_dataset_ae_familiarity_median") or 0.0)
        raw_median = float(ae_check.get("raw_autoencoder_familiarity_median") or 0.0)
        if strategy_median > raw_median * 10.0 and raw_median > 0.0:
            recommendations.append("AE familiarity in the strategy dataset does not match the raw scored signal. Fix score plumbing before trusting AE-based selection.")
    if best_rl:
        if int(best_rl.get("executed_buys") or 0) <= 5:
            recommendations.append("The RL policy is too sparse to trust. Add action penalties for inactivity and compare against simpler threshold rules before promotion.")
        if abs(float(best_rl.get("combined_max_drawdown_pct") or 0.0)) >= 40.0:
            recommendations.append("RL returns come with extreme drawdown. Increase drawdown penalty or add exposure constraints in the environment.")
    if runtime_summary:
        bottleneck = str(runtime_summary.get("slowest_stage") or "")
        if bottleneck == "backtest":
            recommendations.append("Backtests dominate runtime. Reuse the same features and scores across strategy sweeps and keep the vectorized backtest path as the default.")
        elif bottleneck == "fit_regressor":
            recommendations.append("Regressor fitting is the main runtime bottleneck. Narrow hyperparameter search before widening the universe.")
    if not recommendations:
        recommendations.append("No immediate red flags were detected. The next step is a broader universe walk-forward run with the same diagnostics enabled.")
    return recommendations


def _build_observations(*, report: dict[str, Any]) -> list[str]:
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
            f"Rules backtest finished at {float(backtest.get('final_equity') or 0.0):.2f}x with max drawdown {float(backtest.get('max_drawdown') or 0.0) * 100.0:.2f}%."
        )
    if best_rl:
        observations.append(
            f"Best RL sweep result returned {float(best_rl.get('combined_total_return_pct') or 0.0):.2f}% on the 2024-2025 evaluation slice."
        )
    return observations


def build_diagnostic_report(
    *,
    label_artifact: Artifact,
    classifier_predictions_artifact: Artifact,
    regressor_predictions_artifact: Artifact,
    autoencoder_scores_artifact: Artifact,
    strategy_artifact: Artifact | None = None,
    backtest_artifact: Artifact | None = None,
    run_rl: bool = False,
    rl_train_split_date: str = "2023-12-31",
    rl_years: list[int] | None = None,
    rl_algorithms: list[str] | None = None,
    rl_eligibility_quantiles: list[float] | None = None,
    rl_max_stocks: list[int] | None = None,
    rl_episodes: int = 20,
) -> dict[str, Any]:
    labels = _read_csv_artifact(label_artifact)
    classifier = _read_csv_artifact(classifier_predictions_artifact)
    regressor = _read_csv_artifact(regressor_predictions_artifact)
    autoencoder = _read_csv_artifact(autoencoder_scores_artifact)
    strategy = _read_csv_artifact(strategy_artifact) if strategy_artifact is not None else pd.DataFrame()

    classifier_score_col = _score_column(classifier, ["prediction_score", "signal_score"])
    regressor_score_col = _score_column(regressor, ["prediction", "prediction_score", "signal_score"])
    ae_score_col = _score_column(autoencoder, ["prediction_score", "signal_score"])
    ae_raw_col = _score_column(autoencoder, ["prediction", "raw_prediction"])

    panel = classifier[["date", "symbol", classifier_score_col, "label", "trade_return"]].rename(columns={classifier_score_col: "prob_buy"})
    panel = panel.merge(
        regressor[["date", "symbol", regressor_score_col]].rename(columns={regressor_score_col: "pred_rf_reg"}),
        on=["date", "symbol"],
        how="inner",
    )
    panel = panel.merge(
        autoencoder[["date", "symbol", ae_score_col, ae_raw_col]].rename(columns={ae_score_col: "ae_familiarity", ae_raw_col: "ae_recon_error"}),
        on=["date", "symbol"],
        how="inner",
    )
    panel = panel.dropna(subset=["date", "symbol", "prob_buy", "pred_rf_reg", "ae_familiarity"]).copy()
    panel["label"] = pd.to_numeric(panel["label"], errors="coerce")
    panel["trade_return"] = pd.to_numeric(panel["trade_return"], errors="coerce")
    panel["prob_buy"] = pd.to_numeric(panel["prob_buy"], errors="coerce")
    panel["pred_rf_reg"] = pd.to_numeric(panel["pred_rf_reg"], errors="coerce")
    panel["ae_familiarity"] = pd.to_numeric(panel["ae_familiarity"], errors="coerce")
    panel["ae_recon_error"] = pd.to_numeric(panel["ae_recon_error"], errors="coerce")
    panel = panel.dropna(subset=["label", "trade_return"]).copy()
    panel["label"] = panel["label"].astype(int)
    panel["prob_rank"] = panel["prob_buy"].rank(method="first", pct=True)
    panel["reg_rank"] = panel["pred_rf_reg"].rank(method="first", pct=True)
    panel["ae_rank"] = panel["ae_familiarity"].rank(method="first", pct=True)
    panel["combined_rank_mean"] = panel[["prob_rank", "reg_rank", "ae_rank"]].mean(axis=1)

    backtest_summary = dict(backtest_artifact.content or {}) if backtest_artifact is not None else {}
    backtest_meta = dict(backtest_artifact.metadata or {}) if backtest_artifact is not None else {}
    backtest_cfg = dict(backtest_meta.get("backtest_config") or {})
    avg_positions = 0.0
    avg_turnover = 0.0
    daily_rows = list(backtest_summary.get("daily_rows") or [])
    if daily_rows:
        avg_positions = float(sum(float(row.get("positions") or 0.0) for row in daily_rows) / len(daily_rows))
        avg_turnover = float(sum(float(row.get("turnover") or 0.0) for row in daily_rows) / len(daily_rows))
    backtest_summary["avg_positions"] = avg_positions
    backtest_summary["avg_turnover"] = avg_turnover

    report: dict[str, Any] = {
        "kind": "diagnostic_report",
        "artifacts": {
            "labels": int(label_artifact.id),
            "classifier_predictions": int(classifier_predictions_artifact.id),
            "regressor_predictions": int(regressor_predictions_artifact.id),
            "autoencoder_scores": int(autoencoder_scores_artifact.id),
            "strategy_dataset": int(strategy_artifact.id) if strategy_artifact is not None else 0,
            "backtest_result": int(backtest_artifact.id) if backtest_artifact is not None else 0,
        },
        "label_summary": dict(label_artifact.content or {}),
        "model_metrics": {
            "classifier": _model_payload_from_prediction_artifact(classifier_predictions_artifact, "CLASSIFIER_MODEL"),
            "regressor": _model_payload_from_prediction_artifact(regressor_predictions_artifact, "REGRESSOR_MODEL"),
            "autoencoder": _model_payload_from_prediction_artifact(autoencoder_scores_artifact, "AUTOENCODER_MODEL"),
        },
        "prediction_quantiles": {
            "classifier_prob_buy": _quantile_bucket_report(panel, "prob_buy", high_is_good=True),
            "regressor_trade_return": _quantile_bucket_report(panel, "pred_rf_reg", high_is_good=True),
            "autoencoder_familiarity": _quantile_bucket_report(panel, "ae_familiarity", high_is_good=True),
            "autoencoder_raw_reconstruction_error": _quantile_bucket_report(panel, "ae_recon_error", high_is_good=False),
            "combined_rank_mean": _quantile_bucket_report(panel, "combined_rank_mean", high_is_good=True),
        },
        "candidate_rule": _candidate_rule(panel),
        "backtest_summary": backtest_summary,
        "backtest_config": backtest_cfg,
        "ae_signal_bug_check": {
            "strategy_dataset_ae_familiarity_median": float(pd.to_numeric(strategy.get("ae_familiarity"), errors="coerce").median()) if not strategy.empty and "ae_familiarity" in strategy.columns else None,
            "raw_autoencoder_familiarity_median": float(panel["ae_familiarity"].median()) if not panel.empty else None,
            "raw_autoencoder_reconstruction_median": float(panel["ae_recon_error"].median()) if not panel.empty else None,
        },
    }
    runtime_summary = {
        "labels": float((label_artifact.content or {}).get("job_duration_seconds") or 0.0),
        "fit_classifier": float(_model_payload_from_prediction_artifact(classifier_predictions_artifact, "CLASSIFIER_MODEL").get("job_duration_seconds") or 0.0),
        "fit_regressor": float(_model_payload_from_prediction_artifact(regressor_predictions_artifact, "REGRESSOR_MODEL").get("job_duration_seconds") or 0.0),
        "fit_autoencoder": float(_model_payload_from_prediction_artifact(autoencoder_scores_artifact, "AUTOENCODER_MODEL").get("job_duration_seconds") or 0.0),
        "backtest": float((backtest_artifact.content or {}).get("backtest_seconds") or 0.0) if backtest_artifact is not None else 0.0,
    }
    runtime_summary["slowest_stage"] = max(runtime_summary.items(), key=lambda item: float(item[1]))[0] if runtime_summary else ""
    report["runtime_summary"] = runtime_summary

    if run_rl:
        rl_rows: list[dict[str, Any]] = []
        rl_year_values = list(rl_years or [2024, 2025])
        rl_algo_values = list(rl_algorithms or ["ppo"])
        rl_quantile_values = list(rl_eligibility_quantiles or [0.5, 0.6])
        rl_max_stock_values = list(rl_max_stocks or [2, 3, 5])
        if strategy_artifact is None:
            raise ValueError("RL diagnostics require a strategy or scored panel reference.")
        feature_artifact_id = int((strategy_artifact.metadata or {}).get("source_features_artifact_id") or 0)
        feature_artifact = Artifact.objects.filter(pk=feature_artifact_id, artifact_type="FEATURES").first()
        if feature_artifact is None:
            raise ValueError("RL diagnostics require the source FEATURES artifact.")
        features = _read_csv_artifact(feature_artifact)
        if "close" not in features.columns:
            raise ValueError("RL diagnostics require 'close' in the FEATURES artifact.")
        bt_panel = features[["date", "symbol", "close"]].copy()
        bt_panel = bt_panel.merge(panel[["date", "symbol", "prob_buy", "pred_rf_reg", "ae_familiarity"]], on=["date", "symbol"], how="inner")
        bt_panel = bt_panel.dropna().set_index(["date", "symbol"]).sort_index()
        split_ts = pd.Timestamp(str(rl_train_split_date))
        for algo in rl_algo_values:
            for eligibility_quantile in rl_quantile_values:
                for max_stocks_per_day in rl_max_stock_values:
                    cfg = RLConfig(
                        lookback_window=20,
                        eligibility_quantile=float(eligibility_quantile),
                        rebalance_freq="W",
                        max_stocks_per_day=int(max_stocks_per_day),
                        initial_balance=100000.0,
                        fee_bps=float(backtest_cfg.get("fee_bps") or 5.0),
                        slippage_bps=float(backtest_cfg.get("slippage_bps") or 5.0),
                        ppo_episodes=int(rl_episodes),
                        drawdown_penalty_lambda=0.10,
                        seed=42,
                    )
                    runner = run_ppo_workflow if str(algo).lower() == "ppo" else run_a2c_workflow
                    result = runner(bt_panel=bt_panel, cfg=cfg, train_split_date=split_ts, years=rl_year_values)
                    summary = result["rl_summary_df"].iloc[0].to_dict()
                    rl_rows.append(
                        {
                            "algorithm": str(algo).lower(),
                            "eligibility_quantile": float(eligibility_quantile),
                            "max_stocks_per_day": int(max_stocks_per_day),
                            "combined_total_return_pct": float(summary.get("combined_total_return_pct") or 0.0),
                            "combined_sharpe": float(summary.get("combined_sharpe") or 0.0),
                            "combined_max_drawdown_pct": float(summary.get("combined_max_drawdown_pct") or 0.0),
                            "rebalance_days": int(summary.get("rebalance_days") or 0),
                            "executed_buys": int(result["executed_action_counts"].get("buy", 0)),
                            "executed_sells": int(result["executed_action_counts"].get("sell", 0)),
                        }
                    )
        rl_rows.sort(key=lambda row: float(row.get("combined_total_return_pct") or 0.0), reverse=True)
        report["rl_results"] = rl_rows
        report["best_rl_result"] = rl_rows[0] if rl_rows else None
    else:
        report["rl_results"] = []
        report["best_rl_result"] = None

    report["observations"] = _build_observations(report=report)
    report["recommendations"] = _build_recommendations(report=report)
    return report


def write_diagnostic_report(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path
