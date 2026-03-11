from __future__ import annotations

import math
from typing import Any

import pandas as pd


PREDICTION_ARTIFACT_TYPES = {
    "PREDICTIONS",
    "CLASSIFIER_PREDICTIONS",
    "REGRESSOR_PREDICTIONS",
    "AUTOENCODER_SCORES",
    "MTL_PREDICTIONS",
}

STATE_PANEL_ARTIFACT_TYPES = set(PREDICTION_ARTIFACT_TYPES) | {"MARKET_SITUATION_CLUSTER"}

PREDICTION_REQUIRED_COLUMNS = ["date", "symbol", "raw_prediction", "signal_score"]
STRATEGY_REQUIRED_COLUMNS = [
    "date",
    "symbol",
    "prob_buy",
    "ranking",
    "ae_familiarity",
    "combined_score",
    "strategy_score",
    "strategy_signal",
    "target_weight",
]
BACKTEST_REQUIRED_COLUMNS = [
    "date",
    "symbol",
    "strategy_signal",
    "strategy_score",
    "target_weight",
    "effective_weight",
    "asset_return",
    "realized_return",
]


def _to_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except Exception:
        return None


def build_schema_metadata(*, artifact_type: str, required_columns: list[str], actual_columns: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": str(artifact_type),
        "row_key": ["date", "symbol"],
        "required_columns": list(required_columns),
        "actual_columns": list(actual_columns),
    }


def validate_frame_columns(df: pd.DataFrame, required_columns: list[str], *, artifact_type: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{artifact_type} is missing required columns: {', '.join(missing)}")


def normalize_prediction_output_frame(prediction_df: pd.DataFrame, *, artifact_type: str) -> pd.DataFrame:
    out = prediction_df.copy()
    if "prediction" not in out.columns:
        out["prediction"] = pd.Series(index=out.index, dtype="float64")
    if "prediction_score" not in out.columns:
        out["prediction_score"] = pd.Series(index=out.index, dtype="float64")

    out["raw_prediction"] = out["prediction"]
    out["signal_score"] = pd.to_numeric(out["prediction_score"], errors="coerce")
    raw_numeric = pd.to_numeric(out["raw_prediction"], errors="coerce")
    out.loc[out["signal_score"].isna(), "signal_score"] = raw_numeric[out["signal_score"].isna()]

    predicted_class = pd.Series(index=out.index, dtype="float64")
    artifact_type_upper = str(artifact_type or "").upper()
    if artifact_type_upper in {"PREDICTIONS", "CLASSIFIER_PREDICTIONS"}:
        prediction_numeric = pd.to_numeric(out["prediction"], errors="coerce")
        predicted_class = prediction_numeric
        if "prediction_score" in out.columns:
            missing_mask = predicted_class.isna()
            predicted_class.loc[missing_mask] = (out.loc[missing_mask, "signal_score"] >= 0.5).astype(float)
    out["predicted_class"] = predicted_class
    return out


def normalize_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if out.get("raw_prediction") in (None, "", "nan", "NaN"):
        out["raw_prediction"] = out.get("prediction")
    signal_score = _to_float(out.get("signal_score"))
    if signal_score is None:
        signal_score = _to_float(out.get("prediction_score"))
    if signal_score is None:
        signal_score = _to_float(out.get("raw_prediction"))
    if signal_score is not None:
        out["signal_score"] = signal_score
        if out.get("prediction_score") in (None, "", "nan", "NaN"):
            out["prediction_score"] = signal_score
    if out.get("predicted_class") in (None, "", "nan", "NaN"):
        raw_prediction = _to_float(out.get("raw_prediction"))
        if raw_prediction is not None and float(raw_prediction).is_integer():
            out["predicted_class"] = int(raw_prediction)
    return out


def build_backtest_daily_rows_from_trade_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        date_value = str(row.get("date") or "")[:10]
        if not date_value:
            continue
        daily = grouped.setdefault(
            date_value,
            {
                "date": date_value,
                "positions": 0,
                "gross_exposure": 0.0,
                "turnover": 0.0,
                "turnover_cost": 0.0,
                "daily_return": 0.0,
                "net_daily_return": 0.0,
                "equity": None,
            },
        )
        effective_weight = _to_float(row.get("effective_weight")) or 0.0
        gross_exposure = _to_float(row.get("gross_exposure"))
        realized_return = _to_float(row.get("realized_return")) or 0.0
        turnover = _to_float(row.get("turnover")) or 0.0
        turnover_cost = _to_float(row.get("turnover_cost")) or 0.0
        if abs(effective_weight) > 0:
            daily["positions"] += 1
        daily["gross_exposure"] += abs(gross_exposure if gross_exposure is not None else effective_weight)
        daily["daily_return"] += realized_return
        daily["turnover"] = turnover
        daily["turnover_cost"] = turnover_cost
    equity = 1.0
    out_rows: list[dict[str, Any]] = []
    for date_value in sorted(grouped):
        daily = grouped[date_value]
        daily["gross_exposure"] = round(float(daily["gross_exposure"]), 8)
        daily["daily_return"] = round(float(daily["daily_return"]), 8)
        daily["net_daily_return"] = round(float(daily["daily_return"] - daily["turnover_cost"]), 8)
        equity *= 1.0 + float(daily["net_daily_return"])
        daily["equity"] = round(float(equity), 8)
        out_rows.append(daily)
    return out_rows


def build_equity_curve_from_daily_rows(daily_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    curve: list[dict[str, Any]] = []
    for row in daily_rows:
        date_value = str(row.get("date") or "")[:10]
        equity = _to_float(row.get("equity"))
        if not date_value or equity is None:
            continue
        curve.append(
            {
                "date": date_value,
                "equity": round(equity, 8),
                "net_daily_return": round(_to_float(row.get("net_daily_return")) or 0.0, 8),
            }
        )
    return curve
