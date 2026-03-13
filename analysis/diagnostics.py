from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.models import Artifact
from pipeline.service_runtime import read_frame_artifact
from .diagnostic_reporting import (
    backtest_summary_context,
    base_diagnostic_report,
    finalize_diagnostic_report,
)
from .diagnostic_rl import run_rl_diagnostics


RATE_DECIMALS = 4
CANDIDATE_RULE_CLASSIFIER_QUANTILE = 0.8
CANDIDATE_RULE_REGRESSOR_QUANTILE = 0.8
CANDIDATE_RULE_AUTOENCODER_QUANTILE = 0.6
MODEL_ARTIFACT_TYPES = {
    "classifier": "CLASSIFIER_MODEL",
    "regressor": "REGRESSOR_MODEL",
    "autoencoder": "AUTOENCODER_MODEL",
}
PANEL_REQUIRED_COLUMNS = ("label", "trade_return", "prob_buy", "pred_rf_reg", "ae_familiarity", "ae_recon_error")


def _model_payload_from_prediction_artifact(prediction_artifact: Artifact, expected_type: str) -> dict[str, Any]:
    source_model_id = int((prediction_artifact.metadata or {}).get("source_model_artifact_id") or 0)
    model_artifact = Artifact.objects.filter(pk=source_model_id, artifact_type=expected_type).first()
    return dict(model_artifact.content or {}) if model_artifact is not None else {}


def _model_metric_payloads(
    *,
    classifier_predictions_artifact: Artifact,
    regressor_predictions_artifact: Artifact,
    autoencoder_scores_artifact: Artifact,
) -> dict[str, dict[str, Any]]:
    return {
        "classifier": _model_payload_from_prediction_artifact(
            classifier_predictions_artifact,
            MODEL_ARTIFACT_TYPES["classifier"],
        ),
        "regressor": _model_payload_from_prediction_artifact(
            regressor_predictions_artifact,
            MODEL_ARTIFACT_TYPES["regressor"],
        ),
        "autoencoder": _model_payload_from_prediction_artifact(
            autoencoder_scores_artifact,
            MODEL_ARTIFACT_TYPES["autoencoder"],
        ),
    }


def _read_csv_artifact(artifact: Artifact) -> pd.DataFrame:
    path = Path(str(artifact.uri or ""))
    if not path.exists():
        raise ValueError(f"Artifact #{artifact.id} file does not exist.")
    return read_frame_artifact(artifact)


def _load_diagnostic_frames(
    *,
    classifier_predictions_artifact: Artifact,
    regressor_predictions_artifact: Artifact,
    autoencoder_scores_artifact: Artifact,
    strategy_artifact: Artifact | None,
) -> dict[str, pd.DataFrame]:
    return {
        "classifier": _read_csv_artifact(classifier_predictions_artifact),
        "regressor": _read_csv_artifact(regressor_predictions_artifact),
        "autoencoder": _read_csv_artifact(autoencoder_scores_artifact),
        "strategy": _read_csv_artifact(strategy_artifact) if strategy_artifact is not None else pd.DataFrame(),
    }


def _score_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing expected columns: {', '.join(candidates)}")


def _coerce_numeric_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _build_diagnostic_panel(
    *,
    classifier: pd.DataFrame,
    regressor: pd.DataFrame,
    autoencoder: pd.DataFrame,
) -> pd.DataFrame:
    classifier_score_col = _score_column(classifier, ["prediction_score", "signal_score"])
    regressor_score_col = _score_column(regressor, ["prediction", "prediction_score", "signal_score"])
    ae_score_col = _score_column(autoencoder, ["prediction_score", "signal_score"])
    ae_raw_col = _score_column(autoencoder, ["prediction", "raw_prediction"])
    panel = classifier[["date", "symbol", classifier_score_col, "label", "trade_return"]].rename(
        columns={classifier_score_col: "prob_buy"}
    )
    panel = panel.merge(
        regressor[["date", "symbol", regressor_score_col]].rename(columns={regressor_score_col: "pred_rf_reg"}),
        on=["date", "symbol"],
        how="inner",
    )
    panel = panel.merge(
        autoencoder[["date", "symbol", ae_score_col, ae_raw_col]].rename(
            columns={ae_score_col: "ae_familiarity", ae_raw_col: "ae_recon_error"}
        ),
        on=["date", "symbol"],
        how="inner",
    )
    panel = panel.dropna(subset=["date", "symbol", "prob_buy", "pred_rf_reg", "ae_familiarity"]).copy()
    panel = _coerce_numeric_columns(panel, PANEL_REQUIRED_COLUMNS).dropna(subset=["label", "trade_return"]).copy()
    panel["label"] = panel["label"].astype(int)
    panel["prob_rank"] = panel["prob_buy"].rank(method="first", pct=True)
    panel["reg_rank"] = panel["pred_rf_reg"].rank(method="first", pct=True)
    panel["ae_rank"] = panel["ae_familiarity"].rank(method="first", pct=True)
    panel["combined_rank_mean"] = panel[["prob_rank", "reg_rank", "ae_rank"]].mean(axis=1)
    return panel


def _diagnostic_report_context(
    *,
    label_artifact: Artifact,
    classifier_predictions_artifact: Artifact,
    regressor_predictions_artifact: Artifact,
    autoencoder_scores_artifact: Artifact,
    strategy_artifact: Artifact | None,
    backtest_artifact: Artifact | None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    frames = _load_diagnostic_frames(
        classifier_predictions_artifact=classifier_predictions_artifact,
        regressor_predictions_artifact=regressor_predictions_artifact,
        autoencoder_scores_artifact=autoencoder_scores_artifact,
        strategy_artifact=strategy_artifact,
    )
    panel = _build_diagnostic_panel(
        classifier=frames["classifier"],
        regressor=frames["regressor"],
        autoencoder=frames["autoencoder"],
    )
    model_metrics = _model_metric_payloads(
        classifier_predictions_artifact=classifier_predictions_artifact,
        regressor_predictions_artifact=regressor_predictions_artifact,
        autoencoder_scores_artifact=autoencoder_scores_artifact,
    )
    backtest_summary, backtest_cfg = backtest_summary_context(backtest_artifact)
    report = base_diagnostic_report(
        label_artifact=label_artifact,
        classifier_predictions_artifact=classifier_predictions_artifact,
        regressor_predictions_artifact=regressor_predictions_artifact,
        autoencoder_scores_artifact=autoencoder_scores_artifact,
        strategy_artifact=strategy_artifact,
        backtest_artifact=backtest_artifact,
        panel=panel,
        strategy=frames["strategy"],
        model_metrics=model_metrics,
        backtest_summary=backtest_summary,
        backtest_cfg=backtest_cfg,
        classifier_quantile=CANDIDATE_RULE_CLASSIFIER_QUANTILE,
        regressor_quantile=CANDIDATE_RULE_REGRESSOR_QUANTILE,
        autoencoder_quantile=CANDIDATE_RULE_AUTOENCODER_QUANTILE,
    )
    return panel, model_metrics, backtest_cfg, report, frames


def _diagnostic_rl_rows(
    *,
    run_rl: bool,
    panel: pd.DataFrame,
    strategy_artifact: Artifact | None,
    backtest_cfg: dict[str, Any],
    rl_train_split_date: str,
    rl_years: list[int] | None,
    rl_algorithms: list[str] | None,
    rl_eligibility_quantiles: list[float] | None,
    rl_max_stocks: list[int] | None,
    rl_episodes: int,
) -> list[dict[str, Any]]:
    if not run_rl:
        return []
    return run_rl_diagnostics(
        panel=panel,
        strategy_artifact=strategy_artifact,
        backtest_cfg=backtest_cfg,
        rl_train_split_date=rl_train_split_date,
        rl_years=rl_years,
        rl_algorithms=rl_algorithms,
        rl_eligibility_quantiles=rl_eligibility_quantiles,
        rl_max_stocks=rl_max_stocks,
        rl_episodes=rl_episodes,
    )


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
    panel, model_metrics, backtest_cfg, report, _frames = _diagnostic_report_context(
        label_artifact=label_artifact,
        classifier_predictions_artifact=classifier_predictions_artifact,
        regressor_predictions_artifact=regressor_predictions_artifact,
        autoencoder_scores_artifact=autoencoder_scores_artifact,
        strategy_artifact=strategy_artifact,
        backtest_artifact=backtest_artifact,
    )
    rl_rows = _diagnostic_rl_rows(
        run_rl=run_rl,
        panel=panel,
        strategy_artifact=strategy_artifact,
        backtest_cfg=backtest_cfg,
        rl_train_split_date=rl_train_split_date,
        rl_years=rl_years,
        rl_algorithms=rl_algorithms,
        rl_eligibility_quantiles=rl_eligibility_quantiles,
        rl_max_stocks=rl_max_stocks,
        rl_episodes=rl_episodes,
    )
    return finalize_diagnostic_report(
        report=report,
        label_artifact=label_artifact,
        model_metrics=model_metrics,
        backtest_artifact=backtest_artifact,
        rl_rows=rl_rows,
    )


def write_diagnostic_report(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path
