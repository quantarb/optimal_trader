from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ml.execution import build_feature_frame_from_artifacts, infer_feature_family_columns, load_artifact_csv_frame
from ml.multitask import derive_oracle_cluster_labels

from pipeline.models import Artifact


PREDICTION_ARTIFACT_TYPES = (
    "PREDICTIONS",
    "CLASSIFIER_PREDICTIONS",
    "REGRESSOR_PREDICTIONS",
    "AUTOENCODER_SCORES",
    "MTL_PREDICTIONS",
)

STATE_EXCLUDED_COLUMNS = {
    "date",
    "symbol",
    "label",
    "market_position",
    "trade_return",
    "hold_days",
    "side",
    "freq",
    "k",
    "oracle_cluster_key",
    "strategy_signal",
    "target_weight",
    "effective_weight",
    "asset_return",
    "gross_exposure",
    "realized_return",
    "turnover",
    "turnover_cost",
    "ret_fwd_30d",
    "ret_fwd_60d",
    "ret_fwd_90d",
    "ret_fwd_180d",
    "drawdown_fwd_30d",
    "drawdown_fwd_60d",
    "drawdown_fwd_90d",
    "drawdown_fwd_180d",
    "volatility_fwd_30d",
    "volatility_fwd_60d",
    "volatility_fwd_90d",
    "volatility_fwd_180d",
    "cluster_id",
    "cluster_code",
    "cluster_distance",
    "cluster_similarity",
    "cluster_description",
    "cluster_family_signature",
}

STATE_EXCLUDED_PREFIXES = (
    "ret_fwd_",
    "drawdown_fwd_",
    "volatility_fwd_",
)

STATE_EXCLUDED_SUFFIXES = (
    "__cluster_id",
    "__cluster_code",
    "__cluster_distance",
    "__cluster_similarity",
    "__cluster_description",
    "__cluster_family_signature",
)


@dataclass
class InsightArtifacts:
    strategy_artifact: Artifact | None
    feature_artifact: Artifact | None
    label_artifact: Artifact | None
    prediction_artifacts: list[Artifact]


def _latest_artifact(artifact_type: str) -> Artifact | None:
    return Artifact.objects.filter(artifact_type=artifact_type).select_related("pipeline_run").order_by("-created_at", "-id").first()


def _clean_prediction_ids(raw_ids: Sequence[int] | None) -> list[int]:
    out: list[int] = []
    for value in list(raw_ids or []):
        try:
            parsed = int(value)
        except Exception:
            continue
        if parsed > 0 and parsed not in out:
            out.append(parsed)
    return out


def resolve_insight_artifacts(
    *,
    strategy_artifact_id: int = 0,
    feature_artifact_id: int = 0,
    label_artifact_id: int = 0,
    prediction_artifact_ids: Sequence[int] | None = None,
) -> InsightArtifacts:
    explicit_non_strategy_inputs = int(feature_artifact_id or 0) > 0 or int(label_artifact_id or 0) > 0 or bool(_clean_prediction_ids(prediction_artifact_ids))
    strategy_artifact = (
        Artifact.objects.filter(pk=int(strategy_artifact_id), artifact_type="STRATEGY_DATASET").select_related("pipeline_run").first()
        if int(strategy_artifact_id or 0) > 0
        else (None if explicit_non_strategy_inputs else _latest_artifact("STRATEGY_DATASET"))
    )
    feature_artifact = Artifact.objects.filter(pk=int(feature_artifact_id), artifact_type="FEATURES").select_related("pipeline_run").first() if int(feature_artifact_id or 0) > 0 else None
    label_artifact = Artifact.objects.filter(pk=int(label_artifact_id), artifact_type="LABELS").select_related("pipeline_run").first() if int(label_artifact_id or 0) > 0 else None
    prediction_ids = _clean_prediction_ids(prediction_artifact_ids)
    prediction_artifacts = list(
        Artifact.objects.filter(id__in=prediction_ids, artifact_type__in=PREDICTION_ARTIFACT_TYPES)
        .select_related("pipeline_run")
        .order_by("id")
    ) if prediction_ids else []

    if strategy_artifact is not None:
        meta = dict(strategy_artifact.metadata or {})
        if feature_artifact is None:
            feature_id = int(meta.get("source_features_artifact_id") or 0)
            if feature_id > 0:
                feature_artifact = Artifact.objects.filter(pk=feature_id, artifact_type="FEATURES").select_related("pipeline_run").first()
        if label_artifact is None:
            label_id = int(meta.get("source_label_artifact_id") or 0)
            if label_id > 0:
                label_artifact = Artifact.objects.filter(pk=label_id, artifact_type="LABELS").select_related("pipeline_run").first()
        if not prediction_artifacts:
            meta_prediction_ids = _clean_prediction_ids(meta.get("source_prediction_artifact_ids") or [])
            if meta_prediction_ids:
                prediction_artifacts = list(
                    Artifact.objects.filter(id__in=meta_prediction_ids, artifact_type__in=PREDICTION_ARTIFACT_TYPES)
                    .select_related("pipeline_run")
                    .order_by("id")
                )

    if feature_artifact is None:
        feature_artifact = _latest_artifact("FEATURES")
    if label_artifact is None:
        label_artifact = _latest_artifact("LABELS")
    return InsightArtifacts(
        strategy_artifact=strategy_artifact,
        feature_artifact=feature_artifact,
        label_artifact=label_artifact,
        prediction_artifacts=prediction_artifacts,
    )


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out = out.dropna(subset=["date", "symbol"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    return out


def _augment_oracle_cluster_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "oracle_cluster_key" in out.columns:
        return out
    if {"trade_return", "hold_days"}.issubset(set(out.columns)):
        try:
            out["oracle_cluster_key"] = derive_oracle_cluster_labels(out)
        except Exception:
            out["oracle_cluster_key"] = ""
    else:
        out["oracle_cluster_key"] = ""
    return out


def resolve_price_column(df: pd.DataFrame) -> str:
    for column in ("close", "adj_close", "px__close"):
        if column in df.columns:
            return column
    for column in df.columns:
        lowered = str(column).lower()
        if lowered.endswith("__close") or lowered.endswith("_close"):
            return str(column)
    raise ValueError("No price column was found in the market-state frame.")


def _family_name_for_column(column: str) -> str:
    name = str(column).strip()
    if name.startswith("market_situation_cluster_") or name.endswith(STATE_EXCLUDED_SUFFIXES):
        return "market_situations"
    if name.startswith(("embedding_", "repr_emb_")):
        return "representation_embedding"
    if name in {"prob_buy", "ranking", "combined_score", "strategy_score", "signal_score", "prediction_score"}:
        return "model_signals"
    if name.startswith(("classifier_predictions_", "regressor_predictions_", "predictions_")):
        return "model_signals"
    if name.startswith(("ae_", "autoencoder_scores_")):
        return "novelty"
    if name.startswith("mtl_cluster"):
        return "oracle_cluster"
    if name.startswith("mtl_") or name.startswith("regressor_predictions_") or name.startswith("classifier_predictions_"):
        return "model_signals"
    if name.startswith(("econ__", "economic__", "fred__")):
        return "economic_indicators"
    if name.startswith(("tr__", "treasury__", "yield__", "rate__")):
        return "treasury_rates"
    if name.startswith("evt__ae_"):
        return "analyst_estimates"
    if name.startswith("evt__earn_"):
        return "earnings"
    if name.startswith("evt__"):
        return "event_features"
    if name.startswith(("is__", "isg__", "cf__", "cfg__", "bs__", "bsg__", "fg__", "km__", "ratio__")):
        raw_map = infer_feature_family_columns([name])
        if raw_map:
            return next(iter(raw_map.keys()))
    if "__" not in name or name.startswith(("sma_", "ema_", "vol_", "rsi_", "macd_", "bb_", "atr_", "stoch_", "adx_", "roc_", "mom_")):
        return "prices_div_adj"
    return "other"


def feature_family_map_from_columns(columns: Sequence[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for column in columns:
        family = _family_name_for_column(str(column))
        grouped.setdefault(family, []).append(str(column))
    return grouped


def embedding_columns_from_frame(df: pd.DataFrame) -> list[str]:
    numeric_cols: list[str] = []
    for column in df.columns:
        name = str(column)
        if name in STATE_EXCLUDED_COLUMNS:
            continue
        if any(name.startswith(prefix) for prefix in STATE_EXCLUDED_PREFIXES):
            continue
        if any(name.endswith(suffix) for suffix in STATE_EXCLUDED_SUFFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[name]):
            numeric_cols.append(name)
            continue
        parsed = pd.to_numeric(df[name], errors="coerce")
        if parsed.notna().any():
            numeric_cols.append(name)
    preferred = [name for name in numeric_cols if name in {"prob_buy", "ranking", "ae_familiarity", "strategy_score", "combined_score"}]
    remaining = [name for name in numeric_cols if name not in preferred]
    return preferred + remaining


def load_market_state_frame(
    *,
    strategy_artifact: Artifact | None,
    feature_artifact: Artifact | None,
    label_artifact: Artifact | None,
    prediction_artifacts: Sequence[Artifact] = (),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if strategy_artifact is not None:
        frame = load_artifact_csv_frame(strategy_artifact)
        frame = _normalize_frame(frame)
        frame = _augment_oracle_cluster_key(frame)
        embedding_cols = embedding_columns_from_frame(frame)
        return frame, {
            "source": "strategy_dataset",
            "strategy_artifact_id": int(strategy_artifact.id),
            "feature_artifact_id": int(feature_artifact.id) if feature_artifact is not None else 0,
            "label_artifact_id": int(label_artifact.id) if label_artifact is not None else 0,
            "prediction_artifact_ids": [int(artifact.id) for artifact in prediction_artifacts],
            "embedding_columns": embedding_cols,
            "feature_family_map": feature_family_map_from_columns(embedding_cols),
        }
    if feature_artifact is None:
        raise ValueError("A strategy or feature artifact is required to load market states.")
    joined, _feature_cols, panel_meta = build_feature_frame_from_artifacts(
        base_feature_artifact=feature_artifact,
        extra_panel_artifacts=prediction_artifacts,
    )
    frame = _normalize_frame(joined)
    if label_artifact is not None:
        label_df = load_artifact_csv_frame(label_artifact)
        merge_cols = [col for col in ("date", "symbol", "label", "market_position", "trade_return", "hold_days", "side", "freq", "k") if col in label_df.columns]
        if {"date", "symbol"}.issubset(set(merge_cols)):
            frame = frame.merge(label_df[merge_cols], on=["date", "symbol"], how="left")
    frame = _augment_oracle_cluster_key(frame)
    embedding_cols = embedding_columns_from_frame(frame)
    return frame, {
        "source": "feature_plus_predictions",
        "feature_artifact_id": int(feature_artifact.id),
        "label_artifact_id": int(label_artifact.id) if label_artifact is not None else 0,
        "prediction_artifact_ids": [int(artifact.id) for artifact in prediction_artifacts],
        "extra_panel_sources": list(panel_meta.get("extra_panel_sources") or []),
        "embedding_columns": embedding_cols,
        "feature_family_map": feature_family_map_from_columns(embedding_cols),
    }


def latest_rows_by_symbol(df: pd.DataFrame, *, as_of_date: str | None = None) -> pd.DataFrame:
    work = df.copy()
    if as_of_date:
        work = work[work["date"] <= pd.Timestamp(str(as_of_date))].copy()
    if work.empty:
        return work
    last_date = work["date"].max()
    latest = work[work["date"] == last_date].copy()
    latest = latest.sort_values(["symbol"]).reset_index(drop=True)
    return latest


def percentile_rank(series: pd.Series, value: float) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.5
    return float((numeric <= float(value)).mean())


def compute_market_state_embedding(
    *,
    symbol: str,
    date: str | None = None,
    strategy_artifact: Artifact | None,
    feature_artifact: Artifact | None,
    label_artifact: Artifact | None,
    prediction_artifacts: Sequence[Artifact] = (),
) -> dict[str, Any]:
    frame, meta = load_market_state_frame(
        strategy_artifact=strategy_artifact,
        feature_artifact=feature_artifact,
        label_artifact=label_artifact,
        prediction_artifacts=prediction_artifacts,
    )
    symbol_value = str(symbol).strip().upper()
    symbol_rows = frame[frame["symbol"] == symbol_value].copy()
    if symbol_rows.empty:
        raise ValueError(f"No market-state rows were found for symbol {symbol_value}.")
    if date:
        target_rows = symbol_rows[symbol_rows["date"] == pd.Timestamp(str(date))].copy()
    else:
        target_rows = symbol_rows[symbol_rows["date"] == symbol_rows["date"].max()].copy()
    if target_rows.empty:
        raise ValueError(f"No market-state row was found for {symbol_value} on {date}.")
    row = target_rows.sort_values("date").iloc[-1]
    embedding_cols = list(meta.get("embedding_columns") or [])
    numeric_frame = frame[embedding_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0) if embedding_cols else pd.DataFrame(index=frame.index)
    means = numeric_frame.mean(axis=0) if not numeric_frame.empty else pd.Series(dtype=float)
    stds = numeric_frame.std(axis=0, ddof=0).replace(0.0, 1.0) if not numeric_frame.empty else pd.Series(dtype=float)
    vector = ((pd.to_numeric(row[embedding_cols], errors="coerce").fillna(0.0) - means) / stds).fillna(0.0) if embedding_cols else pd.Series(dtype=float)
    return {
        "symbol": symbol_value,
        "date": str(pd.Timestamp(row["date"]).date()),
        "row": row.to_dict(),
        "embedding_columns": embedding_cols,
        "embedding_vector": vector.to_numpy(dtype=float),
        "frame": frame,
        "meta": meta,
    }
