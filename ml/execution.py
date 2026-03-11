from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import logging
import time
import uuid
from typing import Any, Sequence

import pandas as pd
from django.db import transaction

from domain.models.datasets import (
    dedupe_label_frame as _domain_dedupe_label_frame,
    feature_columns_from_frame as _domain_feature_columns_from_frame,
    filter_frame_by_date as _domain_filter_frame_by_date,
)
from domain.models.feature_families import infer_feature_family_columns as _domain_infer_feature_family_columns
from fmp.models import EconomicIndicatorSeries, Symbol, SymbolSectionHistorical, TreasuryRateSeries
from features.feature_builders import (
    build_event_features,
    build_fundamental_change_features,
    build_ownership_features,
    build_price_technical_features,
    build_statement_quality_features,
)
from features.macro import EconomicDataConfig, broadcast_series_to_daily, fetch_economic_data_series
from features.views import _load_adjusted_prices
from ml.base import FitSpec
from ml.frameworks.sklearn import SklearnRFClassifier, SklearnRFRegressor
from ml.raw_stack import train_ae
from pipeline.contracts import normalize_prediction_output_frame
from pipeline.models import Artifact
from pipeline.service_runtime import read_frame_artifact
from settings import BASE_DIR

from .models import ModelArtifact
from .multitask import derive_oracle_cluster_labels, train_multi_task_forest_bundle
from .store import save_model_artifact

logger = logging.getLogger(__name__)

JOB_CONTEXT_KEY = "__job_context__"

FUNDAMENTAL_PREFIXES = {
    "key_metrics": ("km__",),
    "ratios": ("ratio__",),
}

STATEMENT_PREFIXES = {
    "income_statement": ("is__",),
    "income_statement_growth": ("isg__",),
    "cash_flow": ("cf__",),
    "cash_flow_growth": ("cfg__",),
    "balance_sheet": ("bs__",),
    "balance_sheet_growth": ("bsg__",),
    "financial_growth": ("fg__",),
}

EVENT_PREFIXES = {
    "earnings": ("evt__earn_",),
    "analyst_estimates": ("evt__ae_",),
    "ratings_historical": ("evt__rating_",),
    "grades_historical": ("evt__grade_",),
}

TECHNICAL_PREFIXES = (
    "sma_",
    "ema_",
    "vol_",
    "rsi_",
    "macd_",
    "bb_",
    "atr_",
    "stoch_",
    "adx_",
    "roc_",
    "mom_",
)

PRICE_FAMILY_COLUMNS = {"close", "ret_1", "adj_close", "adj_open", "adj_high", "adj_low", "volume"}

PREDICTION_ARTIFACT_DIR = Path(BASE_DIR) / "data" / "pipeline_artifacts"


def build_symbol_choices(preferred_symbols: Sequence[str] | None = None) -> list[tuple[str, str]]:
    rows = Symbol.objects.order_by("symbol").values_list("symbol", "company_name")[:5000]
    label_map: dict[str, str] = {}
    for symbol, company_name in rows:
        code = str(symbol).strip().upper()
        if not code:
            continue
        label_map[code] = code if not company_name else f"{code} - {company_name}"

    if preferred_symbols:
        choices: list[tuple[str, str]] = []
        seen: set[str] = set()
        for symbol in preferred_symbols:
            code = str(symbol).strip().upper()
            if not code or code in seen:
                continue
            seen.add(code)
            choices.append((code, label_map.get(code, code)))
        if choices:
            return choices

    return sorted(label_map.items(), key=lambda x: x[0])[:500]


def merge_job_params(
    model_params: dict[str, Any],
    *,
    symbol: str | None = None,
    symbols: Sequence[str] | None = None,
) -> dict[str, Any]:
    payload = dict(model_params)
    context: dict[str, Any] = {}
    if symbols:
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in symbols:
            code = str(raw).strip().upper()
            if not code or code in seen:
                continue
            seen.add(code)
            cleaned.append(code)
        if cleaned:
            context["symbols"] = cleaned
            context["symbol"] = cleaned[0]
    elif symbol:
        context["symbol"] = str(symbol).strip().upper()
    payload[JOB_CONTEXT_KEY] = context
    return payload


def extract_model_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in dict(params or {}).items() if key != JOB_CONTEXT_KEY}


def extract_training_symbol(params: dict[str, Any]) -> str:
    symbols = extract_training_symbols(params)
    if symbols:
        return symbols[0]
    context = dict(params or {}).get(JOB_CONTEXT_KEY)
    if isinstance(context, dict):
        raw = context.get("symbol")
        if raw:
            return str(raw).strip().upper()
    return ""


def extract_training_symbols(params: dict[str, Any]) -> list[str]:
    context = dict(params or {}).get(JOB_CONTEXT_KEY)
    if not isinstance(context, dict):
        return []
    raw_symbols = context.get("symbols")
    out: list[str] = []
    seen: set[str] = set()
    if isinstance(raw_symbols, (list, tuple)):
        for raw in raw_symbols:
            code = str(raw).strip().upper()
            if not code or code in seen:
                continue
            seen.add(code)
            out.append(code)
    if out:
        return out
    raw_symbol = context.get("symbol")
    if raw_symbol:
        code = str(raw_symbol).strip().upper()
        if code:
            return [code]
    return []


def _artifact_id_from_params(params: dict[str, Any], key: str) -> int:
    context = dict(params or {}).get(JOB_CONTEXT_KEY)
    if not isinstance(context, dict):
        return 0
    try:
        return int(context.get(key) or 0)
    except Exception:
        return 0


def _load_artifact_csv_frame(artifact: Artifact) -> pd.DataFrame:
    if not str(artifact.uri or "").strip():
        raise ValueError(f"Artifact #{artifact.id} has no file path.")
    df = read_frame_artifact(artifact)
    if df.empty:
        return df
    if "date" not in df.columns or "symbol" not in df.columns:
        raise ValueError(f"Artifact #{artifact.id} must contain 'date' and 'symbol' columns.")
    df = df.dropna(subset=["date", "symbol"])
    return df


def _filter_frame_by_date(
    df: pd.DataFrame,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    return _domain_filter_frame_by_date(df, start_date=start_date, end_date=end_date)


def _dedupe_label_frame(label_df: pd.DataFrame) -> pd.DataFrame:
    return _domain_dedupe_label_frame(label_df)


def load_artifact_csv_frame(artifact: Artifact) -> pd.DataFrame:
    return _load_artifact_csv_frame(artifact)


def _feature_columns_from_artifact_df(feature_df: pd.DataFrame) -> list[str]:
    return _domain_feature_columns_from_frame(feature_df)


def infer_feature_family_columns(feature_cols: Sequence[str]) -> dict[str, list[str]]:
    return _domain_infer_feature_family_columns(feature_cols)


def _coverage_metadata(df: pd.DataFrame, feature_cols: Sequence[str]) -> dict[str, Any]:
    usable_cols = [str(col) for col in list(feature_cols) if str(col) in df.columns]
    if df.empty or not usable_cols or "date" not in df.columns:
        return {
            "coverage_start_date": "",
            "coverage_end_date": "",
            "coverage_rows": 0,
        }
    mask = df[usable_cols].notna().any(axis=1)
    if not mask.any():
        return {
            "coverage_start_date": "",
            "coverage_end_date": "",
            "coverage_rows": 0,
        }
    dates = pd.to_datetime(df.loc[mask, "date"], errors="coerce").dropna()
    if dates.empty:
        return {
            "coverage_start_date": "",
            "coverage_end_date": "",
            "coverage_rows": int(mask.sum()),
        }
    return {
        "coverage_start_date": str(dates.min().date().isoformat()),
        "coverage_end_date": str(dates.max().date().isoformat()),
        "coverage_rows": int(mask.sum()),
    }


def _rename_panel_columns(df: pd.DataFrame, *, prefix: str) -> tuple[pd.DataFrame, list[str]]:
    rename_map: dict[str, str] = {}
    feature_cols: list[str] = []
    for col in df.columns:
        if col in {"date", "symbol"}:
            continue
        renamed = f"{prefix}{col}"
        rename_map[col] = renamed
        feature_cols.append(renamed)
    if not rename_map:
        return df[["date", "symbol"]].copy(), []
    return df.rename(columns=rename_map), feature_cols


def _join_feature_panels(
    base_feature_artifact: Artifact,
    *,
    extra_artifacts: Sequence[Artifact] = (),
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    base_df = _load_artifact_csv_frame(base_feature_artifact)
    if base_df.empty:
        raise ValueError("Selected feature artifact has no rows.")
    joined = base_df.copy()
    feature_cols = _feature_columns_from_artifact_df(base_df)
    source_artifact_ids = [int(base_feature_artifact.id)]
    extra_sources: list[dict[str, Any]] = []

    for artifact in extra_artifacts:
        panel_df = _load_artifact_csv_frame(artifact)
        if panel_df.empty:
            continue
        prefix = f"{str(artifact.artifact_type or '').strip().lower()}_{int(artifact.id)}__"
        renamed_df, renamed_cols = _rename_panel_columns(panel_df, prefix=prefix)
        joined = joined.merge(renamed_df, on=["date", "symbol"], how="left")
        feature_cols.extend(renamed_cols)
        source_artifact_ids.append(int(artifact.id))
        extra_sources.append(
            {
                "artifact_id": int(artifact.id),
                "artifact_type": str(artifact.artifact_type),
                "prefix": prefix,
                "columns": renamed_cols,
            }
        )

    feature_cols = list(dict.fromkeys(feature_cols))
    return joined, feature_cols, {
        "base_feature_artifact_id": int(base_feature_artifact.id),
        "panel_artifact_ids": source_artifact_ids,
        "extra_panel_sources": extra_sources,
    }


def build_feature_frame_from_artifacts(
    *,
    base_feature_artifact: Artifact,
    extra_panel_artifacts: Sequence[Artifact] = (),
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    return _join_feature_panels(base_feature_artifact, extra_artifacts=extra_panel_artifacts)


def _normalize_oracle_cluster_keys(values: Sequence[Any] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in list(values or []):
        key = str(value).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _build_training_frame_from_artifacts(
    *,
    params: dict[str, Any],
    selected_families: Sequence[str],
    progress_callback=None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    feature_artifact_id = _artifact_id_from_params(params, "feature_artifact_id")
    label_artifact_id = _artifact_id_from_params(params, "label_artifact_id")
    if feature_artifact_id <= 0 or label_artifact_id <= 0:
        raise ValueError("Training job is missing selected feature or label artifact inputs.")

    feature_artifact = Artifact.objects.filter(pk=feature_artifact_id, artifact_type="FEATURES").first()
    if feature_artifact is None:
        raise ValueError(f"Feature artifact #{feature_artifact_id} was not found.")
    label_artifact = Artifact.objects.filter(pk=label_artifact_id, artifact_type="LABELS").first()
    if label_artifact is None:
        raise ValueError(f"Label artifact #{label_artifact_id} was not found.")

    if callable(progress_callback):
        progress_callback(completed=0, total=4, current_symbol="load_feature_artifact")

    started = time.perf_counter()
    feature_df = _load_artifact_csv_frame(feature_artifact)
    logger.info(
        "ml.train feature_artifact_loaded artifact=%s rows=%s cols=%s elapsed_s=%.3f",
        feature_artifact.id,
        len(feature_df),
        len(feature_df.columns),
        time.perf_counter() - started,
    )
    if callable(progress_callback):
        progress_callback(completed=1, total=4, current_symbol="load_label_artifact")

    started = time.perf_counter()
    label_df = _load_artifact_csv_frame(label_artifact)
    logger.info(
        "ml.train label_artifact_loaded artifact=%s rows=%s cols=%s elapsed_s=%.3f",
        label_artifact.id,
        len(label_df),
        len(label_df.columns),
        time.perf_counter() - started,
    )
    if feature_df.empty:
        raise ValueError("Selected feature artifact has no rows.")
    if label_df.empty:
        raise ValueError("Selected label artifact has no rows.")

    feature_cols = _feature_columns_from_artifact_df(feature_df)
    if not feature_cols:
        raise ValueError("Selected feature artifact has no usable feature columns.")

    if callable(progress_callback):
        progress_callback(completed=2, total=4, current_symbol="join_feature_label_rows")
    started = time.perf_counter()
    joined = feature_df.merge(label_df, on=["date", "symbol"], how="inner", suffixes=("", "_label"))
    logger.info(
        "ml.train artifacts_joined feature_artifact=%s label_artifact=%s joined_rows=%s elapsed_s=%.3f",
        feature_artifact.id,
        label_artifact.id,
        len(joined),
        time.perf_counter() - started,
    )
    if joined.empty:
        raise ValueError("Selected feature and label artifacts have no overlapping (date, symbol) rows.")
    if "sample_weight" not in joined.columns:
        joined["sample_weight"] = 1.0

    joined = joined.sort_values(["symbol", "date"]).reset_index(drop=True)
    symbols = sorted(set(joined["symbol"].astype(str).tolist()))

    if callable(progress_callback):
        progress_callback(completed=3, total=4, current_symbol="training_dataset_ready")

    return (
        joined,
        feature_cols,
        {
            "feature_artifact_id": int(feature_artifact.id),
            "label_artifact_id": int(label_artifact.id),
            "symbols": symbols,
            "symbols_count": len(symbols),
            "joined_rows": int(len(joined)),
            "selected_families": list(selected_families),
            "feature_df": feature_df,
            "label_df": label_df,
        },
    )


def build_training_frame_from_panel_artifacts(
    *,
    base_feature_artifact: Artifact,
    label_artifact: Artifact,
    extra_panel_artifacts: Sequence[Artifact] = (),
    start_date: str | None = None,
    end_date: str | None = None,
    feature_family: str | None = None,
    feature_families: Sequence[str] = (),
    label_k: int | None = None,
    label_ks: Sequence[int] = (),
    min_abs_trade_return: float | None = None,
    max_hold_days: int | None = None,
    sample_weight_mode: str = "uniform",
    oracle_cluster_keys: Sequence[str] = (),
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    joined_features, feature_cols, panel_meta = _join_feature_panels(
        base_feature_artifact,
        extra_artifacts=extra_panel_artifacts,
    )
    label_df = _load_artifact_csv_frame(label_artifact)
    family_map = infer_feature_family_columns(feature_cols)
    all_feature_cols = list(feature_cols)
    selected_feature_cols = list(feature_cols)
    selected_families = [str(value).strip() for value in list(feature_families or []) if str(value).strip()]
    if not selected_families:
        selected_families = [str(feature_family or "").strip()] if str(feature_family or "").strip() else []
    if selected_families:
        selected_feature_cols = []
        for family_name in selected_families:
            selected_feature_cols.extend(list(family_map.get(family_name) or []))
        selected_feature_cols = list(dict.fromkeys(selected_feature_cols))
        if not selected_feature_cols:
            raise ValueError(f"Feature artifact does not contain usable columns for families {selected_families!r}.")
    coverage_before = _coverage_metadata(joined_features, selected_feature_cols)
    joined_features = _filter_frame_by_date(joined_features, start_date=start_date, end_date=end_date)
    label_df = _filter_frame_by_date(label_df, start_date=start_date, end_date=end_date)
    selected_label_ks: list[int] = []
    for value in list(label_ks or []):
        try:
            parsed = int(value)
        except Exception:
            continue
        if parsed > 0 and parsed not in selected_label_ks:
            selected_label_ks.append(parsed)
    if not selected_label_ks and label_k is not None:
        selected_label_ks = [int(label_k)]
    if selected_label_ks and "k" in label_df.columns:
        label_df = label_df[pd.to_numeric(label_df["k"], errors="coerce").isin(selected_label_ks)].copy()
    label_rows_before_trade_filters = int(len(label_df))
    if min_abs_trade_return not in (None, "") and "trade_return" in label_df.columns:
        min_abs_value = max(0.0, float(min_abs_trade_return))
        label_df["trade_return"] = pd.to_numeric(label_df["trade_return"], errors="coerce")
        label_df = label_df[label_df["trade_return"].abs() >= min_abs_value].copy()
    if max_hold_days not in (None, "") and "hold_days" in label_df.columns:
        max_hold_value = max(1, int(max_hold_days))
        label_df["hold_days"] = pd.to_numeric(label_df["hold_days"], errors="coerce")
        label_df = label_df[label_df["hold_days"].fillna(max_hold_value + 1) <= max_hold_value].copy()
    label_df = _dedupe_label_frame(label_df)
    selected_cluster_keys = _normalize_oracle_cluster_keys(oracle_cluster_keys)
    cluster_rows_before_filter = int(len(label_df))
    if selected_cluster_keys:
        label_df["oracle_cluster_key"] = derive_oracle_cluster_labels(label_df)
        label_df = label_df[label_df["oracle_cluster_key"].isin(selected_cluster_keys)].copy()
        if label_df.empty:
            raise ValueError("Selected oracle cluster keys produced no label rows in the requested training window.")
    if label_df.empty:
        raise ValueError("Selected label artifact has no rows.")
    joined = joined_features.merge(label_df, on=["date", "symbol"], how="inner", suffixes=("", "_label"))
    if joined.empty:
        raise ValueError("Selected feature and label artifacts have no overlapping (date, symbol) rows.")
    if selected_feature_cols:
        usable_feature_cols = [col for col in selected_feature_cols if col in joined.columns]
        if usable_feature_cols:
            joined = joined[joined[usable_feature_cols].notna().any(axis=1)].copy()
            if joined.empty:
                raise ValueError("Selected training window has no rows with usable feature-family coverage.")
    if "sample_weight" not in joined.columns:
        joined["sample_weight"] = 1.0
    weight_mode = str(sample_weight_mode or "uniform").strip().lower() or "uniform"
    if weight_mode == "trade_return_abs" and "trade_return" in joined.columns:
        weights = pd.to_numeric(joined["trade_return"], errors="coerce").abs().fillna(0.0)
        joined["sample_weight"] = (1.0 + weights).astype(float)
    joined = joined.sort_values(["symbol", "date"]).reset_index(drop=True)
    symbols = sorted(set(joined["symbol"].astype(str).tolist()))
    coverage_after = _coverage_metadata(joined, selected_feature_cols)
    return (
        joined,
        selected_feature_cols,
        {
            "feature_artifact_id": int(base_feature_artifact.id),
            "label_artifact_id": int(label_artifact.id),
            "symbols": symbols,
            "symbols_count": len(symbols),
            "joined_rows": int(len(joined)),
            "feature_df": joined_features,
            "label_df": label_df,
            "panel_artifact_ids": list(panel_meta["panel_artifact_ids"]),
            "extra_panel_sources": list(panel_meta["extra_panel_sources"]),
            "start_date": str(start_date or ""),
            "end_date": str(end_date or ""),
            "feature_family": ",".join(selected_families),
            "feature_families": list(selected_families),
            "feature_family_columns": list(selected_feature_cols),
            "available_feature_families": sorted(family_map.keys()),
            "all_feature_columns": all_feature_cols,
            "label_k": int(label_k) if label_k is not None else None,
            "label_ks": list(selected_label_ks),
            "label_rows_before_trade_filters": int(label_rows_before_trade_filters),
            "coverage_before": coverage_before,
            "coverage_after": coverage_after,
            "label_rows_after_filters": int(len(label_df)),
            "min_abs_trade_return": None if min_abs_trade_return in (None, "") else float(min_abs_trade_return),
            "max_hold_days": None if max_hold_days in (None, "") else int(max_hold_days),
            "sample_weight_mode": weight_mode,
            "oracle_cluster_keys": list(selected_cluster_keys),
            "oracle_cluster_scope": "specialist" if selected_cluster_keys else "generalist",
            "cluster_rows_before_filter": int(cluster_rows_before_filter),
            "cluster_rows_after_filter": int(len(label_df)),
        },
    )


def _score_artifact_rows(
    *,
    model_obj: Any,
    feature_df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    prediction_df = feature_df.copy()
    usable_cols = [col for col in list(feature_cols) if col in prediction_df.columns]
    used_features = list(getattr(model_obj, "_used_features", []) or usable_cols)
    used_features = [col for col in used_features if col in prediction_df.columns]
    if not used_features:
        return pd.DataFrame()
    prediction_df = prediction_df.copy()
    for col in used_features:
        prediction_df[col] = pd.to_numeric(prediction_df[col], errors="coerce")
    if prediction_df[used_features].notna().any(axis=1).sum() == 0:
        return pd.DataFrame()
    prediction_df[used_features] = prediction_df[used_features].fillna(0.0)

    recon_error = getattr(model_obj, "recon_error", None)
    familiarity = getattr(model_obj, "familiarity", None)
    predict_frame = getattr(model_obj, "predict_frame", None)
    prediction_cols: dict[str, Any] = {}
    if callable(recon_error) and callable(familiarity):
        try:
            prediction_cols["prediction"] = recon_error(
                prediction_df,
                numeric_cols=used_features,
                categorical_cols=(),
            )
            prediction_cols["prediction_score"] = familiarity(
                prediction_df,
                numeric_cols=used_features,
                categorical_cols=(),
            )
        except Exception:
            return pd.DataFrame()
    elif callable(predict_frame):
        try:
            bundle_frame = predict_frame(prediction_df)
        except Exception:
            return pd.DataFrame()
        for col in list(bundle_frame.columns):
            prediction_cols[col] = bundle_frame[col]
    else:
        preds = model_obj.predict(prediction_df)
        prediction_cols["prediction"] = list(preds)

        predict_proba = getattr(getattr(model_obj, "model", None), "predict_proba", None)
        if callable(predict_proba):
            try:
                proba = predict_proba(prediction_df[used_features])
                if getattr(proba, "shape", None) is not None and len(proba.shape) == 2 and proba.shape[1] >= 2:
                    prediction_cols["prediction_score"] = proba[:, 1]
            except Exception:
                pass
    if prediction_cols:
        prediction_df = prediction_df.assign(**prediction_cols)

    if label_df is not None and not label_df.empty:
        label_df = _dedupe_label_frame(label_df)
        merge_cols = [
            col for col in ["date", "symbol", "label", "market_position", "trade_return", "hold_days", "side", "freq", "k"]
            if col in label_df.columns
        ]
        if "date" in merge_cols and "symbol" in merge_cols:
            prediction_df = prediction_df.merge(
                label_df[merge_cols].copy(),
                on=["date", "symbol"],
                how="left",
                suffixes=("", "_label"),
            )
    artifact_type = "PREDICTIONS"
    if callable(recon_error) and callable(familiarity):
        artifact_type = "AUTOENCODER_SCORES"
    elif "mtl_prob_buy" in prediction_df.columns or "mtl_trade_return" in prediction_df.columns:
        artifact_type = "MTL_PREDICTIONS"
    elif "prediction_score" in prediction_df.columns:
        artifact_type = "CLASSIFIER_PREDICTIONS"
    elif "prediction" in prediction_df.columns:
        artifact_type = "REGRESSOR_PREDICTIONS"
    prediction_df = normalize_prediction_output_frame(prediction_df, artifact_type=artifact_type)
    return prediction_df.sort_values(["symbol", "date"]).reset_index(drop=True)


def _write_prediction_rows_csv(name: str, prediction_df: pd.DataFrame) -> str:
    PREDICTION_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTION_ARTIFACT_DIR / f"ml_predictions_{uuid.uuid4().hex}.csv"
    prediction_df.to_csv(path, index=False)
    return str(path)


def _build_training_frame_single(*, symbol: str, selected_families: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    symbol_obj = Symbol.objects.filter(symbol__iexact=symbol).first()
    if symbol_obj is None:
        raise ValueError(f"Symbol {symbol!r} was not found.")

    df_prices = _load_adjusted_prices(symbol_obj, None, None)
    if df_prices.empty:
        raise ValueError(f"No adjusted price data found for {symbol}.")

    target_index = pd.MultiIndex.from_arrays(
        [df_prices.index, [symbol] * len(df_prices)],
        names=["date", "symbol"],
    )
    merged = pd.DataFrame(index=target_index)
    grouped: dict[str, list[str]] = {key: [] for key in selected_families}
    selected = set(selected_families)

    if "prices_div_adj" in selected:
        built = build_price_technical_features(symbol, df_prices)
        if not built.df.empty:
            merged = merged.join(built.df[built.feature_cols], how="left")
            grouped["prices_div_adj"] = list(built.feature_cols)

    if selected & set(FUNDAMENTAL_PREFIXES):
        built = build_fundamental_change_features(symbol_obj, target_index, df_prices=df_prices)
        if not built.df.empty:
            merged = merged.join(built.df[built.feature_cols], how="left")
            for family, prefixes in FUNDAMENTAL_PREFIXES.items():
                if family in selected:
                    grouped[family] = [col for col in built.feature_cols if col.startswith(prefixes)]

    if selected & set(STATEMENT_PREFIXES):
        built = build_statement_quality_features(symbol_obj, target_index)
        if not built.df.empty:
            merged = merged.join(built.df[built.feature_cols], how="left")
            for family, prefixes in STATEMENT_PREFIXES.items():
                if family in selected:
                    grouped[family] = [col for col in built.feature_cols if col.startswith(prefixes)]

    if selected & set(EVENT_PREFIXES):
        built = build_event_features(symbol_obj, target_index)
        if not built.df.empty:
            merged = merged.join(built.df[built.feature_cols], how="left")
            for family, prefixes in EVENT_PREFIXES.items():
                if family in selected:
                    grouped[family] = [col for col in built.feature_cols if col.startswith(prefixes)]

    if "insider_trading" in selected:
        built = build_ownership_features(symbol_obj, target_index)
        if not built.df.empty:
            merged = merged.join(built.df[built.feature_cols], how="left")
            grouped["insider_trading"] = [col for col in built.feature_cols if col.startswith("own__insider_")]

    if "economic_indicators" in selected:
        economic_series_codes = tuple(
            str(code) for code in EconomicIndicatorSeries.objects.order_by("code").values_list("code", flat=True)
        )
        economic_df = fetch_economic_data_series(
            api_key="",
            start_date=df_prices.index.min().date().isoformat(),
            end_date=df_prices.index.max().date().isoformat(),
            config=EconomicDataConfig(
                economic_indicator_series=economic_series_codes,
                include_treasury_rates=False,
            ),
        )
        if not economic_df.empty:
            daily = broadcast_series_to_daily(economic_df, target_index)
            cols = list(daily.columns)
            merged = merged.join(daily[cols], how="left")
            grouped["economic_indicators"] = cols

    if "treasury_rates" in selected:
        treasury_series_codes = tuple(
            str(code) for code in TreasuryRateSeries.objects.order_by("code").values_list("code", flat=True)
        )
        treasury_df = fetch_economic_data_series(
            api_key="",
            start_date=df_prices.index.min().date().isoformat(),
            end_date=df_prices.index.max().date().isoformat(),
            config=EconomicDataConfig(
                economic_indicator_series=treasury_series_codes,
                include_treasury_rates=False,
            ),
        )
        if not treasury_df.empty:
            daily = broadcast_series_to_daily(treasury_df, target_index)
            cols = list(daily.columns)
            merged = merged.join(daily[cols], how="left")
            grouped["treasury_rates"] = cols

    feature_cols: list[str] = []
    for family in selected_families:
        feature_cols.extend(grouped.get(family, []))
    feature_cols = list(dict.fromkeys(feature_cols))

    train_df = merged.reset_index()
    label_df = _load_generated_labels(symbol_obj)
    if not label_df.empty:
        train_df = train_df.merge(label_df, on=["date", "symbol"], how="left")
    train_df["close"] = df_prices.reindex(train_df["date"])["close"].to_numpy()
    train_df["sample_weight"] = 1.0
    train_df = train_df.dropna(subset=["date"])
    return train_df, feature_cols


def _load_generated_labels(symbol_obj: Symbol) -> pd.DataFrame:
    section_key = "labels_generated"
    qs = (
        SymbolSectionHistorical.objects.filter(symbol=symbol_obj, section_key=section_key)
        .only("record_date", "payload")
        .order_by("record_date", "-updated_at")
    )
    rows: list[dict[str, Any]] = []
    for item in qs.iterator():
        payload = item.payload if isinstance(item.payload, dict) else {}
        date_value = payload.get("date") or (item.record_date.isoformat() if item.record_date else "")
        date_str = str(date_value)[:10]
        if not date_str:
            continue
        rows.append(
            {
                "date": pd.to_datetime(date_str, errors="coerce"),
                "symbol": str(payload.get("symbol") or symbol_obj.symbol).strip().upper(),
                "label": payload.get("label"),
                "market_position": payload.get("market_position"),
                "trade_return": payload.get("trade_return"),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).dropna(subset=["date", "symbol"])
    # Keep one label per date/symbol.
    df = df.sort_values(["date"]).drop_duplicates(subset=["date", "symbol"], keep="first")
    return df


def _build_training_frame(
    *,
    symbols: Sequence[str],
    selected_families: Sequence[str],
    progress_callback=None,
) -> tuple[pd.DataFrame, list[str]]:
    if not symbols:
        raise ValueError("At least one training symbol is required.")
    frames: list[pd.DataFrame] = []
    feature_cols: list[str] = []
    total = len(symbols)
    completed = 0
    for symbol in symbols:
        if callable(progress_callback):
            progress_callback(completed=completed, total=total, current_symbol=symbol)
        symbol_df, symbol_cols = _build_training_frame_single(symbol=symbol, selected_families=selected_families)
        completed += 1
        if symbol_df.empty:
            if callable(progress_callback):
                progress_callback(completed=completed, total=total, current_symbol=symbol)
            continue
        frames.append(symbol_df)
        for col in symbol_cols:
            if col not in feature_cols:
                feature_cols.append(col)
        if callable(progress_callback):
            progress_callback(completed=completed, total=total, current_symbol=symbol)
    if not frames:
        raise ValueError("No adjusted price data found for selected symbols.")
    train_df = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    train_df = train_df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return train_df, feature_cols


def _attach_target(train_df: pd.DataFrame, *, target_col: str, task_type: str) -> pd.DataFrame:
    df = train_df.copy()
    if target_col in df.columns and df[target_col].notna().any():
        return df
    close = pd.to_numeric(df["close"], errors="coerce")
    if "symbol" in df.columns:
        next_return = close.groupby(df["symbol"]).pct_change()
        next_return = next_return.groupby(df["symbol"]).shift(-1)
    else:
        next_return = close.pct_change().shift(-1)
    if task_type == "classification":
        df[target_col] = (next_return > 0).astype(int)
    elif task_type == "regression":
        df[target_col] = next_return.astype(float)
    else:
        df[target_col] = next_return.astype(float)
    return df


def _train_classifier(
    *,
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    model_params: dict[str, Any],
    target_col: str,
    split_ratio: float,
) -> Any:
    df = _attach_target(train_df, target_col=target_col, task_type="classification")
    spec = FitSpec(
        feature_cols=list(feature_cols),
        target_col=target_col,
        weight_col="sample_weight",
        split_ratio=float(split_ratio),
    )
    model = SklearnRFClassifier(random_state=1337, **model_params)
    model.fit(df, spec, verbose=False)
    return model


def _train_regressor(
    *,
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    model_params: dict[str, Any],
    target_col: str,
    split_ratio: float,
) -> Any:
    df = _attach_target(train_df, target_col=target_col, task_type="regression")
    model = SklearnRFRegressor(
        test_size=max(0.0, 1.0 - float(split_ratio)),
        random_state=1337,
        **model_params,
    )
    spec = FitSpec(
        feature_cols=list(feature_cols),
        target_col=target_col,
        weight_col="sample_weight",
        split_ratio=float(split_ratio),
    )
    model.fit(df, spec, verbose=False)
    return model


def _train_autoencoder(train_df: pd.DataFrame, feature_cols: Sequence[str]) -> Any:
    ae_model, _numeric_cols = train_ae(train_df, feature_cols, verbose=False)
    setattr(ae_model, "_used_features", list(feature_cols))
    return ae_model


def _train_multi_task_bundle(
    *,
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    model_params: dict[str, Any],
    split_ratio: float,
) -> Any:
    bundle = train_multi_task_forest_bundle(
        train_df=train_df,
        feature_cols=list(feature_cols),
        split_ratio=float(split_ratio),
        model_params=model_params,
        include_cluster_head=True,
    )
    setattr(bundle, "_used_features", list(feature_cols))
    return bundle


def _metrics_for(model_obj: Any) -> dict[str, Any]:
    metrics_fn = getattr(model_obj, "metrics_report", None)
    if callable(metrics_fn):
        try:
            metrics = metrics_fn()
            return dict(metrics or {})
        except Exception:
            return {}
    return {}


def _model_summary(model_obj: Any) -> str:
    summarize_fn = getattr(model_obj, "summarize", None)
    if not callable(summarize_fn):
        return ""
    buf = StringIO()
    try:
        with redirect_stdout(buf):
            summarize_fn()
    except Exception:
        return ""
    return buf.getvalue().strip()


def train_model_from_artifact_inputs(
    *,
    name: str,
    algorithm: str,
    task_type: str,
    target_col: str,
    feature_artifact: Artifact,
    label_artifact: Artifact,
    extra_panel_artifacts: Sequence[Artifact] = (),
    framework: str = "sklearn",
    split_ratio: float = 0.8,
    params: dict[str, Any] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    feature_family: str | None = None,
    feature_families: Sequence[str] = (),
    label_k: int | None = None,
    label_ks: Sequence[int] = (),
    min_abs_trade_return: float | None = None,
    max_hold_days: int | None = None,
    sample_weight_mode: str = "uniform",
    oracle_cluster_keys: Sequence[str] = (),
    progress_callback=None,
) -> ModelArtifact:
    merged_params = merge_job_params(
        dict(params or {}),
        symbols=sorted(set(load_artifact_csv_frame(feature_artifact)["symbol"].astype(str).tolist())),
    )
    context = dict(merged_params.get(JOB_CONTEXT_KEY) or {})
    context["feature_artifact_id"] = int(feature_artifact.id)
    context["label_artifact_id"] = int(label_artifact.id)
    context["extra_panel_artifact_ids"] = [int(artifact.id) for artifact in extra_panel_artifacts]
    merged_params[JOB_CONTEXT_KEY] = context

    if callable(progress_callback):
        progress_callback(
            phase="prepare_training_dataset",
            phase_label="Prepare training dataset",
            phase_index=1,
            phase_total=3,
            force=True,
        )
    dataset_started = time.perf_counter()
    train_df, feature_cols, source_meta = build_training_frame_from_panel_artifacts(
        base_feature_artifact=feature_artifact,
        label_artifact=label_artifact,
        extra_panel_artifacts=extra_panel_artifacts,
        start_date=start_date,
        end_date=end_date,
        feature_family=feature_family,
        feature_families=feature_families,
        label_k=label_k,
        label_ks=label_ks,
        min_abs_trade_return=min_abs_trade_return,
        max_hold_days=max_hold_days,
        sample_weight_mode=sample_weight_mode,
        oracle_cluster_keys=oracle_cluster_keys,
    )
    dataset_build_seconds = time.perf_counter() - dataset_started
    model_params = extract_model_params(merged_params)
    algorithm_value = str(algorithm or "").strip().lower()
    task_type_value = str(task_type or "").strip().lower()

    if callable(progress_callback):
        progress_callback(
            phase="fit_model",
            phase_label="Fit model",
            phase_index=2,
            phase_total=3,
            message=f"{len(train_df):,} rows | {len(feature_cols):,} features",
            force=True,
        )
    fit_started = time.perf_counter()
    if algorithm_value == "random_forest_classifier":
        model_obj = _train_classifier(
            train_df=train_df,
            feature_cols=feature_cols,
            model_params=model_params,
            target_col=target_col,
            split_ratio=float(split_ratio),
        )
    elif algorithm_value == "random_forest_regressor":
        model_obj = _train_regressor(
            train_df=train_df,
            feature_cols=feature_cols,
            model_params=model_params,
            target_col=target_col,
            split_ratio=float(split_ratio),
        )
    elif algorithm_value == "autoencoder":
        model_obj = _train_autoencoder(train_df=train_df, feature_cols=feature_cols)
    elif algorithm_value == "multi_task_forest":
        model_obj = _train_multi_task_bundle(
            train_df=train_df,
            feature_cols=feature_cols,
            model_params=model_params,
            split_ratio=float(split_ratio),
        )
    else:
        raise ValueError(f"Unsupported pipeline training algorithm: {algorithm!r}")
    fit_seconds = time.perf_counter() - fit_started

    metadata = {
        "symbols": list(source_meta.get("symbols") or []),
        "symbols_count": int(source_meta.get("symbols_count") or 0),
        "source_feature_artifact_id": int(feature_artifact.id),
        "source_label_artifact_id": int(label_artifact.id),
        "source_panel_artifact_ids": [int(artifact.id) for artifact in extra_panel_artifacts],
        "extra_panel_sources": list(source_meta.get("extra_panel_sources") or []),
        "joined_rows": int(source_meta.get("joined_rows") or 0),
        "model_summary": _model_summary(model_obj),
        "train_start_date": str(start_date or ""),
        "train_end_date": str(end_date or ""),
        "dataset_build_seconds": round(float(dataset_build_seconds), 6),
        "fit_seconds": round(float(fit_seconds), 6),
        "feature_family": str(source_meta.get("feature_family") or ""),
        "feature_families": list(source_meta.get("feature_families") or []),
        "feature_family_columns": list(source_meta.get("feature_family_columns") or []),
        "available_feature_families": list(source_meta.get("available_feature_families") or []),
        "label_k": source_meta.get("label_k"),
        "label_ks": list(source_meta.get("label_ks") or []),
        "coverage_start_date": str((source_meta.get("coverage_after") or {}).get("coverage_start_date") or ""),
        "coverage_end_date": str((source_meta.get("coverage_after") or {}).get("coverage_end_date") or ""),
        "coverage_rows": int((source_meta.get("coverage_after") or {}).get("coverage_rows") or 0),
        "label_rows_before_trade_filters": int(source_meta.get("label_rows_before_trade_filters") or 0),
        "label_rows_after_filters": int(source_meta.get("label_rows_after_filters") or 0),
        "min_abs_trade_return": source_meta.get("min_abs_trade_return"),
        "max_hold_days": source_meta.get("max_hold_days"),
        "sample_weight_mode": str(source_meta.get("sample_weight_mode") or "uniform"),
        "oracle_cluster_keys": list(source_meta.get("oracle_cluster_keys") or []),
        "oracle_cluster_scope": str(source_meta.get("oracle_cluster_scope") or "generalist"),
        "cluster_rows_before_filter": int(source_meta.get("cluster_rows_before_filter") or 0),
        "cluster_rows_after_filter": int(source_meta.get("cluster_rows_after_filter") or 0),
    }
    artifact = save_model_artifact(
        name=name,
        model_obj=model_obj,
        framework=framework,
        task_type=task_type_value,
        target_col=target_col,
        feature_cols=feature_cols,
        metrics=_metrics_for(model_obj),
        params=model_params,
        metadata=metadata,
    )
    if callable(progress_callback):
        progress_callback(
            phase="generate_train_predictions",
            phase_label="Generate train diagnostics",
            phase_index=3,
            phase_total=3,
            message="Scoring training rows for saved diagnostics",
            force=True,
        )
    prediction_started = time.perf_counter()
    prediction_df = _score_artifact_rows(
        model_obj=model_obj,
        feature_df=source_meta["feature_df"],
        feature_cols=feature_cols,
        label_df=source_meta["label_df"],
    )
    train_prediction_seconds = time.perf_counter() - prediction_started
    predictions_uri = _write_prediction_rows_csv(name, prediction_df)
    artifact.metadata = dict(artifact.metadata or {})
    artifact.metadata["predictions_uri"] = predictions_uri
    artifact.metadata["prediction_rows"] = int(len(prediction_df))
    artifact.metadata["train_prediction_seconds"] = round(float(train_prediction_seconds), 6)
    artifact.save(update_fields=["metadata", "updated_at"])
    if callable(progress_callback):
        progress_callback(
            phase="generate_train_predictions",
            phase_label="Generate train diagnostics",
            phase_index=3,
            phase_total=3,
            total_units=1,
            completed_units=1,
            message="Completed",
            force=True,
        )
    return artifact


def score_model_from_artifact_inputs(
    *,
    model_record: ModelArtifact,
    feature_artifact: Artifact,
    extra_panel_artifacts: Sequence[Artifact] = (),
    label_artifact: Artifact | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    return_metadata: bool = False,
    progress_callback=None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    score_started = time.perf_counter()
    if callable(progress_callback):
        progress_callback(
            phase="load_scoring_inputs",
            phase_label="Load scoring inputs",
            phase_index=1,
            phase_total=2,
            force=True,
        )
    feature_df, feature_cols, _panel_meta = _join_feature_panels(
        feature_artifact,
        extra_artifacts=extra_panel_artifacts,
    )
    feature_df = _filter_frame_by_date(feature_df, start_date=start_date, end_date=end_date)
    label_df = _load_artifact_csv_frame(label_artifact) if label_artifact is not None else None
    if label_df is not None:
        label_df = _filter_frame_by_date(label_df, start_date=start_date, end_date=end_date)
    model_obj = model_record.get_artifact()
    if callable(progress_callback):
        progress_callback(
            phase="score_rows",
            phase_label="Score rows",
            phase_index=2,
            phase_total=2,
            message=f"{len(feature_df):,} candidate rows",
            force=True,
        )
    prediction_df = _score_artifact_rows(
        model_obj=model_obj,
        feature_df=feature_df,
        feature_cols=list(model_record.feature_cols or feature_cols),
        label_df=label_df,
    )
    if callable(progress_callback):
        progress_callback(
            phase="score_rows",
            phase_label="Score rows",
            phase_index=2,
            phase_total=2,
            total_units=1,
            completed_units=1,
            message=f"{len(prediction_df):,} rows scored",
            force=True,
        )
    metadata = {
        "score_seconds": round(float(time.perf_counter() - score_started), 6),
        "rows_scored": int(len(prediction_df)),
        "score_start_date": str(start_date or ""),
        "score_end_date": str(end_date or ""),
    }
    if return_metadata:
        return prediction_df, metadata
    return prediction_df
