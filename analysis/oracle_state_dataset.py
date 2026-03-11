from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .market_state import (
    feature_family_map_from_columns,
    load_market_state_frame,
)
from pipeline.models import Artifact


def _oracle_trade_mask(frame: pd.DataFrame) -> pd.Series:
    if "label" in frame.columns:
        return pd.to_numeric(frame["label"], errors="coerce").fillna(0.0) != 0.0
    if "market_position" in frame.columns:
        return pd.to_numeric(frame["market_position"], errors="coerce").fillna(0.0) != 0.0
    if {"trade_return", "hold_days"}.issubset(set(frame.columns)):
        return (
            pd.to_numeric(frame["trade_return"], errors="coerce").notna()
            & pd.to_numeric(frame["hold_days"], errors="coerce").notna()
        )
    raise ValueError("Market-state frame does not contain oracle trade columns.")


def _string_regime_from_series(series: pd.Series, *, high_label: str, low_label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    median_value = float(numeric.median()) if numeric.notna().any() else 0.0
    return pd.Series(
        np.where(numeric.fillna(median_value) >= median_value, high_label, low_label),
        index=series.index,
        dtype="object",
    )


def _derive_macro_regimes(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    econ_cols = [str(col) for col in out.columns if str(col).startswith(("econ__", "economic__", "fred__"))]
    rate_cols = [str(col) for col in out.columns if str(col).startswith(("tr__", "treasury__", "yield__", "rate__"))]

    if econ_cols:
        econ_proxy = out[econ_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        out["macro_liquidity_regime"] = _string_regime_from_series(
            econ_proxy,
            high_label="supportive",
            low_label="tight",
        )
    else:
        out["macro_liquidity_regime"] = "unknown"

    if rate_cols:
        rate_proxy = out[rate_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        out["macro_rate_regime"] = _string_regime_from_series(
            rate_proxy,
            high_label="high_rate",
            low_label="low_rate",
        )
    else:
        out["macro_rate_regime"] = "unknown"

    if "ret_1" in out.columns:
        momentum = pd.to_numeric(out["ret_1"], errors="coerce").fillna(0.0)
        out["price_momentum_regime"] = pd.Series(
            np.where(momentum >= 0.0, "positive", "negative"),
            index=out.index,
            dtype="object",
        )
    else:
        out["price_momentum_regime"] = "unknown"
    return out


def build_oracle_state_dataset(
    *,
    strategy_artifact: Artifact | None = None,
    feature_artifact: Artifact | None = None,
    label_artifact: Artifact | None = None,
    prediction_artifacts: Sequence[Artifact] = (),
    start_date: str | None = None,
    end_date: str | None = None,
    label_ks: Sequence[int] = (),
    min_abs_trade_return: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, meta = load_market_state_frame(
        strategy_artifact=strategy_artifact,
        feature_artifact=feature_artifact,
        label_artifact=label_artifact,
        prediction_artifacts=prediction_artifacts,
    )
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date", "symbol"]).copy()

    if start_date:
        work = work[work["date"] >= pd.Timestamp(str(start_date))].copy()
    if end_date:
        work = work[work["date"] <= pd.Timestamp(str(end_date))].copy()

    oracle_mask = _oracle_trade_mask(work)
    work = work[oracle_mask].copy()

    if label_ks and "k" in work.columns:
        ks = {int(value) for value in list(label_ks or []) if int(value or 0) > 0}
        if ks:
            work = work[pd.to_numeric(work["k"], errors="coerce").fillna(0).astype(int).isin(ks)].copy()

    if min_abs_trade_return is not None and "trade_return" in work.columns:
        threshold = abs(float(min_abs_trade_return))
        work = work[pd.to_numeric(work["trade_return"], errors="coerce").abs().fillna(0.0) >= threshold].copy()

    if work.empty:
        raise ValueError("No oracle trade entry states were available after filtering.")

    work = _derive_macro_regimes(work)
    work = work.sort_values(["date", "symbol"]).reset_index(drop=True)

    feature_columns = list(meta.get("embedding_columns") or [])
    family_map = feature_family_map_from_columns(feature_columns)
    sides = sorted({str(value).strip().lower() for value in work.get("side", pd.Series(dtype=str)).fillna("unknown").tolist() if str(value).strip()})

    dataset_meta = {
        "rows": int(len(work)),
        "symbols": int(work["symbol"].nunique()),
        "date_start": str(work["date"].min().date()),
        "date_end": str(work["date"].max().date()),
        "feature_columns": feature_columns,
        "feature_family_map": family_map,
        "feature_families": sorted(family_map.keys()),
        "sides": sides or ["unknown"],
        "source": dict(meta),
        "strategy_artifact_id": int(strategy_artifact.id) if strategy_artifact is not None else 0,
        "feature_artifact_id": int(feature_artifact.id) if feature_artifact is not None else 0,
        "label_artifact_id": int(label_artifact.id) if label_artifact is not None else 0,
        "prediction_artifact_ids": [int(artifact.id) for artifact in prediction_artifacts],
        "label_ks": [int(value) for value in list(label_ks or []) if int(value or 0) > 0],
        "min_abs_trade_return": float(min_abs_trade_return) if min_abs_trade_return is not None else None,
    }
    return work, dataset_meta
