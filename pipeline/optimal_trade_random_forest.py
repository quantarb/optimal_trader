from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from data.historical_prices import load_adjusted_price_frames
from data.preparation import MLDatasetConfig, prepare_ml_dataset
from domain.features import FeatureBuildSpec
from domain.labels.specs import LabelBuildSpec, parse_k_list
from fmp.models import Symbol
from fmp.refresh import ensure_symbol_price_history
from ml.base import FitSpec
from ml.frameworks.sklearn import SklearnRFClassifier
from ml.models import ModelArtifact
from ml.store import load_model_artifact, save_model_artifact
from pipeline.service_runtime import json_safe_value, write_frame_artifact, write_payload_artifact
from workflows.feature_runtime import (
    FeaturePanelDependencies,
    build_feature_panel_environment,
    build_feature_panel_frame,
)
from workflows.labels import build_oracle_labels

DEFAULT_OPTIMAL_TRADE_ETF_SYMBOLS: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "VNQ",
    "TLT",
    "IEF",
    "SHY",
    "LQD",
    "HYG",
    "GLD",
    "SLV",
    "DBC",
    "USO",
    "UNG",
    "FXE",
    "FXY",
)


@dataclass(frozen=True)
class OptimalTradeRandomForestConfig:
    symbols: tuple[str, ...]
    feature_start_date: str
    train_end_date: str
    ye_k_values: tuple[int, ...]
    min_profit_pct_points: float
    model_name_prefix: str
    output_basename: str
    n_estimators: int = 400
    random_state: int = 1337
    download_missing_prices: bool = True
    min_feature_coverage_pct: float = 10.0
    max_depth: int | None = None
    min_samples_leaf: int = 1
    min_samples_split: int = 2


def _normalize_symbols(symbols: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in list(symbols or []):
        symbol = str(raw or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def _feature_build_config(*, start_date: str, end_date: str) -> FeatureBuildSpec:
    return FeatureBuildSpec.from_mapping(
        {
            "start_date": start_date,
            "end_date": end_date,
            "include_price_technicals": True,
            "include_fundamental_change": True,
            "include_statement_quality": True,
            "include_event_features": True,
            "include_ownership_features": True,
            "include_economic_indicators": True,
            "include_treasury_rates": True,
            "include_representation_embedding": False,
        }
    )


def _ensure_price_frames(
    *,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    download_missing_prices: bool,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    normalized_symbols = _normalize_symbols(symbols)
    start_ts = pd.Timestamp(start_date).date() if start_date else None
    end_ts = pd.Timestamp(end_date).date() if end_date else None
    price_frames = load_adjusted_price_frames(normalized_symbols, start_date=start_date, end_date=end_date)
    diagnostics: list[dict[str, Any]] = []

    for symbol in normalized_symbols:
        symbol_obj = Symbol.objects.filter(symbol__iexact=symbol).only("id", "symbol").first()
        if symbol_obj is None:
            symbol_obj = Symbol.objects.create(symbol=symbol)

        frame = price_frames.get(symbol, pd.DataFrame())
        needs_refresh = frame.empty
        if not frame.empty and start_ts is not None and frame.index.min().date() > start_ts:
            needs_refresh = True
        if not frame.empty and end_ts is not None and frame.index.max().date() < end_ts:
            needs_refresh = True

        refresh_result: dict[str, Any] | None = None
        refresh_error = ""
        if needs_refresh and download_missing_prices:
            try:
                refresh_result = ensure_symbol_price_history(
                    symbol_obj,
                    target_start_date=start_ts,
                    target_end_date=end_ts,
                )
            except Exception as exc:
                refresh_error = str(exc)

        refreshed = load_adjusted_price_frames([symbol], start_date=start_date, end_date=end_date).get(symbol, pd.DataFrame())
        price_frames[symbol] = refreshed
        diagnostics.append(
            {
                "symbol": symbol,
                "attempted_refresh": bool(needs_refresh and download_missing_prices),
                "refresh_error": refresh_error,
                "refresh_fetch_mode": (refresh_result or {}).get("fetch_mode"),
                "records_fetched": int((refresh_result or {}).get("records_fetched") or 0),
                "price_rows": int(len(refreshed)),
                "start_date": refreshed.index.min().date().isoformat() if not refreshed.empty else "",
                "end_date": refreshed.index.max().date().isoformat() if not refreshed.empty else "",
                "status": (
                    "ok"
                    if not refreshed.empty
                    else ("refresh_failed" if refresh_error else "missing_prices")
                ),
            }
        )

    return price_frames, pd.DataFrame(diagnostics)


def _build_feature_frame(
    *,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    price_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    build_spec = _feature_build_config(start_date=start_date, end_date=end_date)
    dependencies = FeaturePanelDependencies(
        load_price_frames=lambda requested_symbols, start_date=None, end_date=None: {
            str(symbol).strip().upper(): price_frames.get(str(symbol).strip().upper(), pd.DataFrame())
            for symbol in list(requested_symbols or [])
        }
    )
    environment = build_feature_panel_environment(
        symbols=list(symbols),
        build_spec=build_spec,
        dependencies=dependencies,
    )
    feature_frame, fieldnames, metadata = build_feature_panel_frame(environment=environment)
    if feature_frame.empty:
        return pd.DataFrame(), fieldnames, metadata
    feature_frame = feature_frame.copy()
    feature_frame["date"] = pd.to_datetime(feature_frame["date"], errors="coerce")
    feature_frame["symbol"] = feature_frame["symbol"].astype(str).str.strip().str.upper()
    feature_frame = feature_frame.dropna(subset=["date", "symbol"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    return feature_frame, fieldnames, metadata


def build_optimal_trade_feature_frame(
    *,
    symbols: Sequence[str] = DEFAULT_OPTIMAL_TRADE_ETF_SYMBOLS,
    start_date: str = "2009-01-01",
    end_date: str = "2019-12-31",
    download_missing_prices: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    normalized_symbols = tuple(_normalize_symbols(symbols))
    if not normalized_symbols:
        raise ValueError("At least one symbol is required.")
    price_frames, price_diagnostics = _ensure_price_frames(
        symbols=normalized_symbols,
        start_date=str(start_date),
        end_date=str(end_date),
        download_missing_prices=bool(download_missing_prices),
    )
    feature_frame, _fieldnames, metadata = _build_feature_frame(
        symbols=normalized_symbols,
        start_date=str(start_date),
        end_date=str(end_date),
        price_frames=price_frames,
    )
    if not feature_frame.empty:
        if "px__dollar_vol" in feature_frame.columns and "dollar_volume" not in feature_frame.columns:
            feature_frame["dollar_volume"] = pd.to_numeric(feature_frame["px__dollar_vol"], errors="coerce")
        feature_frame["date"] = pd.to_datetime(feature_frame["date"], errors="coerce")
        feature_frame["symbol"] = feature_frame["symbol"].astype(str).str.strip().str.upper()
        feature_frame = feature_frame.dropna(subset=["date", "symbol"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    return feature_frame, price_diagnostics, metadata


def _build_label_frame(
    *,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    ye_k_values: Sequence[int],
    min_profit_pct_points: float,
    download_missing_prices: bool,
    price_frames: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = LabelBuildSpec(
        k_params={"YE": [int(value) for value in list(ye_k_values or [])]},
        min_profit_pct=max(0.0, float(min_profit_pct_points)) / 100.0,
        buy_execution="adj_high",
        sell_execution="adj_low",
        short_execution="adj_low",
        cover_execution="adj_high",
        start_date=start_date,
        end_date=end_date,
        download_missing_prices=bool(download_missing_prices),
    )
    label_result = build_oracle_labels(
        list(symbols),
        spec=spec,
        price_frames=price_frames,
    )
    label_frame = pd.DataFrame(label_result.label_rows)
    if label_frame.empty:
        return label_frame, label_result.statistics
    label_frame = label_frame.copy()
    label_frame["date"] = pd.to_datetime(label_frame["date"], errors="coerce")
    label_frame["symbol"] = label_frame["symbol"].astype(str).str.strip().str.upper()
    label_frame["trade_return"] = pd.to_numeric(label_frame.get("trade_return"), errors="coerce")
    label_frame["hold_days"] = pd.to_numeric(label_frame.get("hold_days"), errors="coerce")
    label_frame["k"] = pd.to_numeric(label_frame.get("k"), errors="coerce").astype("Int64")
    label_frame = label_frame.dropna(subset=["date", "symbol"]).sort_values(["date", "symbol", "k"]).reset_index(drop=True)
    return label_frame, label_result.statistics


def _prepare_training_dataset(
    *,
    feature_frame: pd.DataFrame,
    label_frame: pd.DataFrame,
    min_feature_coverage_pct: float = 0.0,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    features_indexed = feature_frame.set_index(["date", "symbol"]).sort_index()
    labels_indexed = label_frame.set_index(["date", "symbol"]).sort_index()
    dataset, feature_cols, _targets = prepare_ml_dataset(
        features_df=features_indexed,
        labels_df=labels_indexed,
        target_cols=["label", "trade_return", "hold_days"],
        weight_col=None,
        config=MLDatasetConfig(drop_nan_features=False),
        verbose=False,
    )
    if dataset.empty or not feature_cols:
        return dataset, feature_cols, pd.DataFrame(columns=["feature", "coverage_pct"])
    dataset = dataset.reset_index()
    for column in feature_cols:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
    active_feature_cols = [column for column in feature_cols if dataset[column].notna().any()]
    if not active_feature_cols:
        return pd.DataFrame(), [], pd.DataFrame(columns=["feature", "coverage_pct"])
    feature_coverage_pct = (
        dataset[active_feature_cols]
        .notna()
        .mean(axis=0)
        .mul(100.0)
        .sort_values(ascending=False)
        .rename("coverage_pct")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    coverage_threshold = max(0.0, float(min_feature_coverage_pct or 0.0))
    filtered_feature_cols = [
        str(column)
        for column in feature_coverage_pct.loc[feature_coverage_pct["coverage_pct"] >= coverage_threshold, "feature"].tolist()
    ]
    if not filtered_feature_cols:
        return pd.DataFrame(), [], feature_coverage_pct
    usable_mask = dataset[active_feature_cols].notna().any(axis=1)
    dataset = dataset.loc[usable_mask].copy()
    if dataset.empty:
        return dataset, filtered_feature_cols, feature_coverage_pct
    dataset = dataset[dataset[filtered_feature_cols].notna().any(axis=1)].copy()
    if dataset.empty:
        return dataset, filtered_feature_cols, feature_coverage_pct
    dataset[filtered_feature_cols] = dataset[filtered_feature_cols].fillna(0.0)
    dataset["label"] = pd.to_numeric(dataset["label"], errors="coerce").fillna(0).astype(int)
    dataset["trade_return"] = pd.to_numeric(dataset["trade_return"], errors="coerce")
    dataset["hold_days"] = pd.to_numeric(dataset["hold_days"], errors="coerce")
    dataset["sample_weight"] = 1.0
    return dataset.sort_values(["date", "symbol"]).reset_index(drop=True), filtered_feature_cols, feature_coverage_pct


def _top_feature_rows(feature_importance: dict[str, float], *, limit: int = 20) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, (feature, value) in enumerate(
        sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)[: max(0, int(limit))],
        start=1,
    ):
        rows.append(
            {
                "rank": int(rank),
                "feature": str(feature),
                "importance": float(value),
            }
        )
    return rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except Exception:
            pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _resolve_model_record(name: str, version: int | None = None) -> ModelArtifact:
    queryset = ModelArtifact.objects.filter(name=str(name))
    if version is None:
        record = queryset.order_by("-version").first()
    else:
        record = queryset.filter(version=int(version)).order_by("-version").first()
    if record is None:
        raise ModelArtifact.DoesNotExist(f"No model artifact found for {name!r}.")
    return record


def score_optimal_trade_random_forest_models(
    feature_frame: pd.DataFrame,
    *,
    model_name_prefix: str = "optimal_trade_rf_etf_pre2020",
    ye_list: Sequence[int] | str = (1, 2, 4, 8),
    complete_case: bool = False,
    classifier_version: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if feature_frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    ye_values = tuple(parse_k_list(ye_list))
    if not ye_values:
        raise ValueError("At least one YE horizon is required.")

    base_cols = [column for column in ["date", "symbol", "close", "dollar_volume", "px__dollar_vol"] if column in feature_frame.columns]
    scored = feature_frame[base_cols].copy()
    scored["date"] = pd.to_datetime(scored["date"], errors="coerce")
    scored["symbol"] = scored["symbol"].astype(str).str.strip().str.upper()
    diagnostics: list[dict[str, Any]] = []

    classifier_name = f"{str(model_name_prefix).strip()}_classifier"
    try:
        classifier_record = _resolve_model_record(classifier_name, version=classifier_version)
        classifier_model = load_model_artifact(name=classifier_name, version=int(classifier_record.version))
    except Exception as exc:
        diagnostics.append(
            {
                "status": "missing_model",
                "error": str(exc),
                "rows_scored": 0,
                "ye_list": [int(value) for value in ye_values],
            }
        )
        return scored, pd.DataFrame(diagnostics)

    used_features = list(dict.fromkeys(list(getattr(classifier_model, "_used_features", []) or [])))
    used_features = [column for column in used_features if column in feature_frame.columns]
    if not used_features:
        diagnostics.append(
            {
                "status": "no_matching_features",
                "error": "",
                "rows_scored": 0,
                "ye_list": [int(value) for value in ye_values],
            }
        )
        return scored, pd.DataFrame(diagnostics)

    work = feature_frame[["date", "symbol", *used_features]].copy()
    for column in used_features:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if bool(complete_case):
        work = work.dropna(subset=used_features).copy()
    else:
        work = work[work[used_features].notna().any(axis=1)].copy()
        work[used_features] = work[used_features].fillna(0.0)
    if work.empty:
        diagnostics.append(
            {
                "status": "no_rows_after_feature_filter",
                "error": "",
                "rows_scored": 0,
                "ye_list": [int(value) for value in ye_values],
            }
        )
        return scored, pd.DataFrame(diagnostics)

    classifier_feature_cols = list(getattr(classifier_model, "_used_features", []) or used_features)
    if not classifier_feature_cols:
        diagnostics.append(
            {
                "status": "model_feature_columns_missing_in_frame",
                "error": "",
                "rows_scored": 0,
                "ye_list": [int(value) for value in ye_values],
            }
        )
        return scored, pd.DataFrame(diagnostics)

    for column in classifier_feature_cols:
        if column not in work.columns:
            work[column] = 0.0

    clf_x = work[classifier_feature_cols].copy()
    if bool(complete_case):
        clf_x = clf_x.dropna()
    common_index = work.index.intersection(clf_x.index)
    if len(common_index) == 0:
        diagnostics.append(
            {
                "status": "no_rows_after_model_filter",
                "error": "",
                "rows_scored": 0,
                "ye_list": [int(value) for value in ye_values],
            }
        )
        return scored, pd.DataFrame(diagnostics)

    work = work.loc[common_index].copy()
    clf_x = clf_x.loc[common_index]
    if not bool(complete_case):
        clf_x = clf_x.fillna(0.0)

    proba_matrix = classifier_model.model.predict_proba(clf_x)
    if getattr(proba_matrix, "shape", None) is not None and proba_matrix.shape[1] >= 2:
        prob_long = pd.Series(proba_matrix[:, 1], index=work.index, dtype=float)
    else:
        prob_long = pd.Series(classifier_model.predict(clf_x), index=work.index, dtype=float)

    work["rf_signal"] = (2.0 * prob_long) - 1.0
    work["rf_prob_long"] = prob_long
    work["rf_signal_contributors"] = 1
    scored = scored.merge(
        work[["date", "symbol", "rf_signal", "rf_prob_long", "rf_signal_contributors"]],
        on=["date", "symbol"],
        how="left",
    )
    diagnostics.append(
        {
            "status": "ok",
            "error": "",
            "rows_scored": int(len(work)),
            "feature_count": int(len(used_features)),
            "classifier_name": classifier_name,
            "classifier_version": int(classifier_record.version),
            "signal_definition": "2 * P(long) - 1",
            "ye_list": [int(value) for value in ye_values],
        }
    )

    if "dollar_volume" not in scored.columns and "px__dollar_vol" in scored.columns:
        scored["dollar_volume"] = pd.to_numeric(scored["px__dollar_vol"], errors="coerce")
    scored = scored.sort_values(["date", "symbol"]).reset_index(drop=True)
    return scored, pd.DataFrame(diagnostics)


def _train_combined_models(
    *,
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    config: OptimalTradeRandomForestConfig,
) -> dict[str, Any]:
    rf_kwargs = {
        "n_estimators": int(config.n_estimators),
        "random_state": int(config.random_state),
        "n_jobs": -1,
        "min_samples_leaf": int(config.min_samples_leaf),
        "min_samples_split": int(config.min_samples_split),
    }
    if config.max_depth is not None:
        rf_kwargs["max_depth"] = int(config.max_depth)

    clf = SklearnRFClassifier(**rf_kwargs)
    clf.fit(
        train_df,
        FitSpec(
            feature_cols=list(feature_cols),
            target_col="label",
            weight_col="sample_weight",
            split_ratio=1.0,
        ),
        verbose=False,
    )

    classifier_name = f"{config.model_name_prefix}_classifier"
    classifier_artifact = save_model_artifact(
        name=classifier_name,
        model_obj=clf,
        framework="sklearn",
        task_type="classification",
        target_col="label",
        feature_cols=list(feature_cols),
        metrics=clf.metrics_report(),
        params=rf_kwargs,
        metadata={
            "train_end_date": str(config.train_end_date),
            "feature_start_date": str(config.feature_start_date),
            "label_freq": "YE",
            "label_ks": [int(value) for value in config.ye_k_values],
            "min_profit_pct_points": float(config.min_profit_pct_points),
            "min_feature_coverage_pct": float(config.min_feature_coverage_pct),
            "split_ratio": 1.0,
            "symbols": list(config.symbols),
            "training_rows": int(len(train_df)),
            "signal_definition": "2 * P(long) - 1",
        },
    )

    label_counts = train_df["label"].value_counts().to_dict()
    return {
        "training_rows": int(len(train_df)),
        "feature_count": int(len(feature_cols)),
        "long_rows": int(label_counts.get(1, 0)),
        "short_rows": int(label_counts.get(0, 0)),
        "avg_trade_return": float(pd.to_numeric(train_df["trade_return"], errors="coerce").mean()),
        "avg_hold_days": float(pd.to_numeric(train_df["hold_days"], errors="coerce").mean()),
        "ye_list": [int(value) for value in config.ye_k_values],
        "classifier_artifact_id": int(classifier_artifact.id),
        "classifier_name": classifier_name,
        "classifier_version": int(classifier_artifact.version),
        "classifier_metrics": json_safe_value(clf.metrics_report()),
        "classifier_top_features": _top_feature_rows(clf.feature_importance()),
        "signal_definition": "2 * P(long) - 1",
    }


def run_optimal_trade_random_forest_training(
    *,
    symbols: Sequence[str] = DEFAULT_OPTIMAL_TRADE_ETF_SYMBOLS,
    feature_start_date: str = "1900-01-01",
    train_end_date: str = "2019-12-31",
    ye_list: Sequence[int] | str = (1, 2, 4, 8),
    min_profit_pct: float = 5.0,
    min_feature_coverage_pct: float = 10.0,
    output_basename: str = "optimal_trade_rf_etf_pre2020_min5",
    model_name_prefix: str = "optimal_trade_rf_etf_pre2020_min5",
    n_estimators: int = 400,
    random_state: int = 1337,
    download_missing_prices: bool = True,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    min_samples_split: int = 2,
) -> dict[str, Any]:
    normalized_symbols = tuple(_normalize_symbols(symbols))
    if not normalized_symbols:
        raise ValueError("At least one symbol is required.")
    ye_values = tuple(parse_k_list(ye_list))
    if not ye_values:
        raise ValueError("At least one YE horizon is required.")

    config = OptimalTradeRandomForestConfig(
        symbols=normalized_symbols,
        feature_start_date=str(feature_start_date),
        train_end_date=str(train_end_date),
        ye_k_values=ye_values,
        min_profit_pct_points=float(min_profit_pct),
        model_name_prefix=str(model_name_prefix).strip() or "optimal_trade_rf_etf_pre2020",
        output_basename=str(output_basename).strip() or "optimal_trade_rf_etf_pre2020",
        n_estimators=int(n_estimators),
        random_state=int(random_state),
        download_missing_prices=bool(download_missing_prices),
        min_feature_coverage_pct=max(0.0, float(min_feature_coverage_pct)),
        max_depth=None if max_depth in (None, "") else int(max_depth),
        min_samples_leaf=max(1, int(min_samples_leaf)),
        min_samples_split=max(2, int(min_samples_split)),
    )

    price_frames, price_diagnostics = _ensure_price_frames(
        symbols=config.symbols,
        start_date=config.feature_start_date,
        end_date=config.train_end_date,
        download_missing_prices=config.download_missing_prices,
    )
    feature_frame, _fieldnames, feature_metadata = _build_feature_frame(
        symbols=config.symbols,
        start_date=config.feature_start_date,
        end_date=config.train_end_date,
        price_frames=price_frames,
    )
    if feature_frame.empty:
        raise ValueError("No feature rows were built for the requested symbol set.")
    label_frame, label_statistics = _build_label_frame(
        symbols=config.symbols,
        start_date=config.feature_start_date,
        end_date=config.train_end_date,
        ye_k_values=config.ye_k_values,
        min_profit_pct_points=config.min_profit_pct_points,
        download_missing_prices=config.download_missing_prices,
        price_frames=price_frames,
    )
    if label_frame.empty:
        raise ValueError("No optimal-trade labels were built for the requested symbol set.")

    feature_frame = feature_frame[feature_frame["date"] <= pd.Timestamp(config.train_end_date)].copy()
    label_frame = label_frame[label_frame["date"] <= pd.Timestamp(config.train_end_date)].copy()

    available_feature_symbols = sorted(feature_frame["symbol"].astype(str).str.upper().unique().tolist())
    available_label_symbols = sorted(label_frame["symbol"].astype(str).str.upper().unique().tolist())
    available_symbols = sorted(set(available_feature_symbols) & set(available_label_symbols))
    if not available_symbols:
        raise ValueError("Features and optimal-trade labels do not overlap on any requested symbols.")

    feature_frame = feature_frame[feature_frame["symbol"].isin(available_symbols)].copy()
    label_frame = label_frame[label_frame["symbol"].isin(available_symbols)].copy()

    combined_labels = label_frame[
        (label_frame["freq"].astype(str).str.upper() == "YE")
        & (pd.to_numeric(label_frame["k"], errors="coerce").isin([int(value) for value in config.ye_k_values]))
    ].copy()
    train_df, feature_cols, feature_coverage = _prepare_training_dataset(
        feature_frame=feature_frame,
        label_frame=combined_labels,
        min_feature_coverage_pct=config.min_feature_coverage_pct,
    )
    if train_df.empty:
        training_summary = {
            "training_rows": 0,
            "feature_count": 0,
            "status": "no_overlapping_training_rows",
            "ye_list": [int(value) for value in config.ye_k_values],
            "min_feature_coverage_pct": float(config.min_feature_coverage_pct),
        }
    else:
        training_summary = _train_combined_models(
            train_df=train_df,
            feature_cols=feature_cols,
            config=config,
        )
        training_summary["status"] = "ok"
        training_summary["min_feature_coverage_pct"] = float(config.min_feature_coverage_pct)
        training_summary["dense_feature_count_ge_threshold"] = int(len(feature_cols))
        training_summary["dropped_feature_count_below_threshold"] = int(
            max(0, len(feature_coverage) - len(feature_cols))
        )

    output_name = str(config.output_basename).strip()
    feature_file = write_frame_artifact(f"{output_name}__features", frame=feature_frame)
    label_file = write_frame_artifact(f"{output_name}__labels", frame=label_frame)
    price_diag_file = write_frame_artifact(f"{output_name}__price_diagnostics", frame=price_diagnostics)
    feature_coverage_file = write_frame_artifact(f"{output_name}__feature_coverage", frame=feature_coverage)
    model_summary_file = write_frame_artifact(f"{output_name}__model_summary", frame=pd.DataFrame([training_summary]))

    missing_symbols = sorted(set(config.symbols) - set(available_symbols))
    summary = {
        "schema_version": 1,
        "symbols_requested": list(config.symbols),
        "symbols_available_for_training": available_symbols,
        "symbols_missing_from_training_overlap": missing_symbols,
        "feature_start_date": str(config.feature_start_date),
        "train_end_date": str(config.train_end_date),
        "ye_list": [int(value) for value in config.ye_k_values],
        "min_profit_pct_points": float(config.min_profit_pct_points),
        "min_feature_coverage_pct": float(config.min_feature_coverage_pct),
        "download_missing_prices": bool(config.download_missing_prices),
        "feature_row_count": int(len(feature_frame)),
        "label_row_count": int(len(label_frame)),
        "feature_frame_uri": str(feature_file.uri),
        "label_frame_uri": str(label_file.uri),
        "price_diagnostics_uri": str(price_diag_file.uri),
        "feature_coverage_uri": str(feature_coverage_file.uri),
        "model_summary_uri": str(model_summary_file.uri),
        "feature_metadata": _json_safe(feature_metadata),
        "label_statistics": _json_safe(label_statistics),
        "model_training_summary": _json_safe(training_summary),
    }
    summary = _json_safe(summary)
    summary_file = write_payload_artifact(f"{output_name}__summary", summary)
    summary["summary_json_uri"] = str(summary_file.uri)
    return summary


__all__ = [
    "DEFAULT_OPTIMAL_TRADE_ETF_SYMBOLS",
    "run_optimal_trade_random_forest_training",
]
