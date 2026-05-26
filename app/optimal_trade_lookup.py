from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


DEFAULT_AE_ARTIFACT_DIR = Path("artifacts/raw_stack")
DEFAULT_QUERY_LOOKBACK_YEARS = 5
@dataclass(frozen=True)
class AutoEncoderBundle:
    model: Any
    numeric_cols: list[str]
    categorical_cols: list[str]
    metadata: dict[str, Any]
    source_label: str


@dataclass(frozen=True)
class RawStackArtifacts:
    artifact_dir: Path
    metadata: dict[str, Any]
    classifier: Any | None
    trade_return_regressor: Any | None
    duration_regressor: Any | None
    autoencoder: Any | None


@dataclass(frozen=True)
class OptimalTradeQuery:
    symbol: str
    as_of_date: str | None = None
    query_lookback_years: int = DEFAULT_QUERY_LOOKBACK_YEARS
    reference_symbols: tuple[str, ...] = ()
    reference_start_date: str = "2010-01-01"
    reference_end_date: str | None = None
    top_k: int = 10
    label_freq: str = "YE"
    label_k_values: tuple[int, ...] = (1, 2, 4, 8)
    min_profit_pct_points: float = 5.0
    download_missing_prices: bool = False
    artifact_dir: str = str(DEFAULT_AE_ARTIFACT_DIR)
    db_artifact_name: str = ""
    db_artifact_version: int | None = None


@dataclass(frozen=True)
class OptimalTradeSearchResult:
    query_summary: pd.DataFrame
    indicator_summary: pd.DataFrame
    nearest_trades: pd.DataFrame
    feature_attribution: pd.DataFrame
    model_predictions: dict[str, dict[str, Any]]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TradeVectorArtifacts:
    frame: pd.DataFrame
    latent_cols: list[str]
    backend: str
    faiss_index: Any | None
    normalized_latent: np.ndarray | None
    metadata: dict[str, Any]


NON_FEATURE_COLUMNS = {
    "date",
    "symbol",
    "side",
    "freq",
    "k",
    "entry_date",
    "exit_date",
    "entry_px",
    "exit_px",
    "ret_dec",
    "ret_pct",
    "hold_days",
    "trade_id",
    "horizon",
    "direction_label",
    "event",
    "label",
    "action_label",
    "market_position",
    "trade_return",
    "trade_duration_days",
}

TRADE_VECTOR_SUBDIR = "trade_vectors"
TRADE_VECTOR_FRAME_FILENAME = "entry_trade_vectors.pkl"
TRADE_VECTOR_META_FILENAME = "entry_trade_vectors_meta.json"
TRADE_VECTOR_FAISS_FILENAME = "entry_trade_vectors.faiss"
TRADE_VECTOR_MATRIX_FILENAME = "entry_trade_vectors.npy"


def bootstrap_django() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")
    # Jupyter/IPython commonly runs a background event loop, which makes
    # Django treat notebook cells as an async context even when we're doing
    # ordinary local exploratory reads. Allow sync ORM access for this local
    # research helper unless the user explicitly chose another setting.
    os.environ.setdefault("DJANGO_ALLOW_ASYNC_UNSAFE", "true")
    import django

    django.setup()


def list_available_autoencoder_artifacts(limit: int = 25) -> list[dict[str, Any]]:
    bootstrap_django()
    from ml.models import ModelArtifact

    queryset = (
        ModelArtifact.objects.filter(framework__iexact="torch")
        .filter(task_type__in=["embedding", "reconstruction"])
        .order_by("-id")
    )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for record in queryset[: max(int(limit), 1) * 4]:
        key = (str(record.name), int(record.version))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "name": str(record.name),
                "version": int(record.version),
                "id": int(record.id),
                "created_at": record.created_at.isoformat() if record.created_at else "",
                "task_type": str(record.task_type or ""),
            }
        )
        if len(rows) >= max(int(limit), 1):
            break
    return rows


def find_nearest_optimal_trades(query: OptimalTradeQuery) -> OptimalTradeSearchResult:
    bundle = load_autoencoder_bundle(
        artifact_dir=query.artifact_dir,
        db_artifact_name=query.db_artifact_name,
        db_artifact_version=query.db_artifact_version,
    )
    query_symbol = _normalize_symbol(query.symbol)
    as_of_ts = pd.Timestamp(query.as_of_date) if query.as_of_date else pd.Timestamp.today().normalize()
    reference_end_ts = pd.Timestamp(query.reference_end_date) if query.reference_end_date else (as_of_ts - pd.Timedelta(days=1))
    query_start_ts = _resolve_query_start_timestamp(query=query, as_of_ts=as_of_ts)
    reference_symbols = tuple(_normalize_symbols(query.reference_symbols or (query_symbol,)))

    query_row = build_latest_feature_row(
        symbol=query_symbol,
        start_date=query_start_ts.strftime("%Y-%m-%d"),
        end_date=as_of_ts.strftime("%Y-%m-%d"),
        download_missing_prices=bool(query.download_missing_prices),
    )
    prebuilt_vectors = None
    prebuilt_vector_error = ""
    try:
        prebuilt_vectors = load_trade_vector_artifacts(artifact_dir=query.artifact_dir)
    except Exception as exc:
        prebuilt_vector_error = f"{type(exc).__name__}: {exc}"
    if prebuilt_vectors is not None:
        reference_frame = prebuilt_vectors.frame.copy()
        symbol_filter = {str(symbol).strip().upper() for symbol in list(reference_symbols or []) if str(symbol).strip()}
        if symbol_filter:
            reference_frame = reference_frame[
                reference_frame["symbol"].astype(str).str.upper().isin(symbol_filter)
            ].copy()
        if not reference_frame.empty:
            query_embedding, query_familiarity = embed_rows(query_row, bundle)
            ranked = _query_trade_vector_artifacts(
                prebuilt_vectors,
                query_vector=query_embedding[0],
                top_k=max(int(query.top_k), 1),
                reference_symbols=tuple(_normalize_symbols(query.reference_symbols or ())),
            )
            if not ranked.empty:
                ranked = ranked.copy()
                ranked["trade_return_pct"] = pd.to_numeric(ranked.get("ret_dec"), errors="coerce") * 100.0
                ranked["entry_date"] = pd.to_datetime(ranked["entry_date"], errors="coerce")
                ranked["exit_date"] = pd.to_datetime(ranked["exit_date"], errors="coerce")

                query_summary = pd.DataFrame(
                    [
                        {
                            "symbol": query_symbol,
                            "as_of_date": str(pd.Timestamp(query_row["date"].iloc[0]).date()),
                            "ae_familiarity": round(float(query_familiarity[0]), 6),
                            "artifact_source": bundle.source_label,
                            "artifact_numeric_feature_count": int(len(bundle.numeric_cols)),
                            "reference_trade_count": int(len(reference_frame)),
                        }
                    ]
                )
                indicator_summary = build_indicator_summary(
                    query_row.iloc[0],
                    reference_frame,
                    feature_columns=bundle.numeric_cols,
                    max_per_family=6,
                )
                attribution_columns = resolve_numeric_feature_columns(
                    query_row=query_row,
                    reference_catalog=ranked,
                    feature_columns=bundle.numeric_cols,
                )
                feature_attribution = build_feature_attribution_summary(
                    query_row=query_row,
                    nearest_trades=ranked,
                    feature_columns=attribution_columns,
                    top_n=10,
                )
                model_predictions = build_optional_raw_stack_predictions(
                    query_row=query_row,
                    artifact_dir=query.artifact_dir,
                )
                nearest_trades = _format_nearest_trades(ranked)
                metadata = {
                    "artifact_source": bundle.source_label,
                    "artifact_metadata": dict(bundle.metadata or {}),
                    "query_symbol": query_symbol,
                    "query_date": str(pd.Timestamp(query_row["date"].iloc[0]).date()),
                    "reference_symbols": sorted(reference_frame["symbol"].astype(str).str.upper().unique().tolist()),
                    "reference_start_date": str(pd.Timestamp(query.reference_start_date).date()),
                    "reference_end_date": str(reference_end_ts.date()),
                    "reference_trade_count": int(len(reference_frame)),
                    "search_method": "prebuilt_trade_vector_index",
                    "trade_vector_backend": str(prebuilt_vectors.backend),
                    "trade_vector_metadata": dict(prebuilt_vectors.metadata or {}),
                }
                return OptimalTradeSearchResult(
                    query_summary=query_summary,
                    indicator_summary=indicator_summary,
                    nearest_trades=nearest_trades,
                    feature_attribution=feature_attribution,
                    model_predictions=model_predictions,
                    metadata=metadata,
                )
        if not prebuilt_vector_error:
            requested_symbols = tuple(_normalize_symbols(query.reference_symbols or (query_symbol,)))
            if requested_symbols:
                prebuilt_vector_error = (
                    "No prebuilt trade vectors were available for the requested symbol set: "
                    + ", ".join(requested_symbols)
                )
            else:
                prebuilt_vector_error = "No prebuilt trade vectors were available for the requested query."

    reference_catalog = build_reference_trade_catalog(
        symbols=reference_symbols,
        start_date=str(query.reference_start_date),
        end_date=reference_end_ts.strftime("%Y-%m-%d"),
        label_freq=str(query.label_freq or "YE"),
        label_k_values=tuple(int(value) for value in tuple(query.label_k_values or (1,))),
        min_profit_pct_points=float(query.min_profit_pct_points or 0.0),
        download_missing_prices=bool(query.download_missing_prices),
    )
    if reference_catalog.empty:
        raise RuntimeError("No historical optimal-trade reference rows were available for the requested search.")

    query_embedding, query_familiarity = embed_rows(query_row, bundle)
    reference_embeddings, reference_familiarity = embed_rows(reference_catalog, bundle)

    similarity_scores = cosine_similarity_matrix(reference_embeddings, query_embedding[0])
    ranked = reference_catalog.copy()
    ranked["similarity_score"] = similarity_scores
    ranked["candidate_ae_familiarity"] = reference_familiarity
    ranked["trade_return_pct"] = pd.to_numeric(ranked.get("ret_dec"), errors="coerce") * 100.0
    ranked["entry_date"] = pd.to_datetime(ranked["entry_date"], errors="coerce")
    ranked["exit_date"] = pd.to_datetime(ranked["exit_date"], errors="coerce")
    ranked = ranked.sort_values(["similarity_score", "trade_return_pct"], ascending=[False, False]).head(max(int(query.top_k), 1))

    query_summary = pd.DataFrame(
        [
            {
                "symbol": query_symbol,
                "as_of_date": str(pd.Timestamp(query_row["date"].iloc[0]).date()),
                "ae_familiarity": round(float(query_familiarity[0]), 6),
                "artifact_source": bundle.source_label,
                "artifact_numeric_feature_count": int(len(bundle.numeric_cols)),
                "reference_trade_count": int(len(reference_catalog)),
            }
        ]
    )
    indicator_summary = build_indicator_summary(
        query_row.iloc[0],
        reference_catalog,
        feature_columns=bundle.numeric_cols,
        max_per_family=6,
    )
    attribution_columns = resolve_numeric_feature_columns(
        query_row=query_row,
        reference_catalog=ranked,
        feature_columns=bundle.numeric_cols,
    )
    feature_attribution = build_feature_attribution_summary(
        query_row=query_row,
        nearest_trades=ranked,
        feature_columns=attribution_columns,
        top_n=10,
    )
    model_predictions = build_optional_raw_stack_predictions(
        query_row=query_row,
        artifact_dir=query.artifact_dir,
    )
    nearest_trades = _format_nearest_trades(ranked)
    metadata = {
        "artifact_source": bundle.source_label,
        "artifact_metadata": dict(bundle.metadata or {}),
        "query_symbol": query_symbol,
        "query_date": str(pd.Timestamp(query_row["date"].iloc[0]).date()),
        "reference_symbols": sorted(reference_catalog["symbol"].astype(str).str.upper().unique().tolist()),
        "reference_start_date": str(pd.Timestamp(query.reference_start_date).date()),
        "reference_end_date": str(reference_end_ts.date()),
        "reference_trade_count": int(len(reference_catalog)),
        "search_method": "rebuilt_reference_trade_catalog",
    }
    if prebuilt_vector_error:
        metadata["prebuilt_vector_fallback_reason"] = str(prebuilt_vector_error)
    return OptimalTradeSearchResult(
        query_summary=query_summary,
        indicator_summary=indicator_summary,
        nearest_trades=nearest_trades,
        feature_attribution=feature_attribution,
        model_predictions=model_predictions,
        metadata=metadata,
    )


def find_nearest_optimal_trades_by_features(
    query: OptimalTradeQuery,
    *,
    feature_columns: Sequence[str] | None = None,
) -> OptimalTradeSearchResult:
    query_symbol = _normalize_symbol(query.symbol)
    as_of_ts = pd.Timestamp(query.as_of_date) if query.as_of_date else pd.Timestamp.today().normalize()
    reference_end_ts = pd.Timestamp(query.reference_end_date) if query.reference_end_date else (as_of_ts - pd.Timedelta(days=1))
    query_start_ts = _resolve_query_start_timestamp(query=query, as_of_ts=as_of_ts)
    reference_symbols = tuple(_normalize_symbols(query.reference_symbols or (query_symbol,)))

    query_row = build_latest_feature_row(
        symbol=query_symbol,
        start_date=query_start_ts.strftime("%Y-%m-%d"),
        end_date=as_of_ts.strftime("%Y-%m-%d"),
        download_missing_prices=bool(query.download_missing_prices),
    )
    reference_catalog = build_reference_trade_catalog(
        symbols=reference_symbols,
        start_date=str(query.reference_start_date),
        end_date=reference_end_ts.strftime("%Y-%m-%d"),
        label_freq=str(query.label_freq or "YE"),
        label_k_values=tuple(int(value) for value in tuple(query.label_k_values or (1,))),
        min_profit_pct_points=float(query.min_profit_pct_points or 0.0),
        download_missing_prices=bool(query.download_missing_prices),
    )
    if reference_catalog.empty:
        raise RuntimeError("No historical optimal-trade reference rows were available for the requested search.")

    numeric_feature_columns = resolve_numeric_feature_columns(
        query_row=query_row,
        reference_catalog=reference_catalog,
        feature_columns=feature_columns,
    )
    if not numeric_feature_columns:
        raise RuntimeError("No shared numeric feature columns were available for feature-based nearest-trade search.")

    query_vector, reference_matrix, feature_means, feature_stds = build_feature_similarity_inputs(
        query_row=query_row,
        reference_catalog=reference_catalog,
        feature_columns=numeric_feature_columns,
    )
    similarity_scores = cosine_similarity_matrix(reference_matrix, query_vector)
    ranked = reference_catalog.copy()
    ranked["similarity_score"] = similarity_scores
    ranked["trade_return_pct"] = pd.to_numeric(ranked.get("ret_dec"), errors="coerce") * 100.0
    ranked["entry_date"] = pd.to_datetime(ranked["entry_date"], errors="coerce")
    ranked["exit_date"] = pd.to_datetime(ranked["exit_date"], errors="coerce")
    ranked = ranked.sort_values(["similarity_score", "trade_return_pct"], ascending=[False, False]).head(max(int(query.top_k), 1))

    query_summary = pd.DataFrame(
        [
            {
                "symbol": query_symbol,
                "as_of_date": str(pd.Timestamp(query_row["date"].iloc[0]).date()),
                "feature_count": int(len(numeric_feature_columns)),
                "reference_trade_count": int(len(reference_catalog)),
            }
        ]
    )
    indicator_summary = build_indicator_summary(
        query_row.iloc[0],
        reference_catalog,
        feature_columns=numeric_feature_columns,
        max_per_family=6,
    )
    feature_attribution = build_feature_attribution_summary(
        query_row=query_row,
        nearest_trades=ranked,
        feature_columns=numeric_feature_columns,
        top_n=10,
    )
    model_predictions = build_optional_raw_stack_predictions(
        query_row=query_row,
        artifact_dir=query.artifact_dir,
    )
    nearest_trades = _format_nearest_trades(ranked, include_familiarity=False)
    metadata = {
        "search_method": "numeric_feature_similarity",
        "query_symbol": query_symbol,
        "query_date": str(pd.Timestamp(query_row["date"].iloc[0]).date()),
        "reference_symbols": sorted(reference_catalog["symbol"].astype(str).str.upper().unique().tolist()),
        "reference_start_date": str(pd.Timestamp(query.reference_start_date).date()),
        "reference_end_date": str(reference_end_ts.date()),
        "reference_trade_count": int(len(reference_catalog)),
        "feature_count": int(len(numeric_feature_columns)),
        "feature_columns": list(numeric_feature_columns),
        "feature_means_preview": {col: round(float(feature_means[idx]), 6) for idx, col in enumerate(numeric_feature_columns[:10])},
        "feature_stds_preview": {col: round(float(feature_stds[idx]), 6) for idx, col in enumerate(numeric_feature_columns[:10])},
    }
    return OptimalTradeSearchResult(
        query_summary=query_summary,
        indicator_summary=indicator_summary,
        nearest_trades=nearest_trades,
        feature_attribution=feature_attribution,
        model_predictions=model_predictions,
        metadata=metadata,
    )


def load_autoencoder_bundle(
    *,
    artifact_dir: str = str(DEFAULT_AE_ARTIFACT_DIR),
    db_artifact_name: str = "",
    db_artifact_version: int | None = None,
) -> AutoEncoderBundle:
    artifact_dir_path = _resolve_artifact_dir_path(artifact_dir)
    ae_pickle_path = artifact_dir_path / "ae_raw.pkl"
    meta_path = artifact_dir_path / "meta.json"
    if ae_pickle_path.exists():
        with ae_pickle_path.open("rb") as handle:
            model = pickle.load(handle)
        metadata = _load_json(meta_path)
        numeric_cols = _resolve_numeric_cols(model, metadata)
        categorical_cols = _resolve_categorical_cols(model)
        return AutoEncoderBundle(
            model=model,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            metadata=metadata,
            source_label=str(ae_pickle_path),
        )

    if str(db_artifact_name or "").strip():
        bootstrap_django()
        from ml.models import ModelArtifact
        from ml.store import load_model_artifact

        model = load_model_artifact(name=str(db_artifact_name).strip(), version=db_artifact_version)
        record = (
            ModelArtifact.objects.filter(name=str(db_artifact_name).strip(), version=int(db_artifact_version)).first()
            if db_artifact_version is not None
            else ModelArtifact.objects.filter(name=str(db_artifact_name).strip()).order_by("-version").first()
        )
        metadata = dict(record.metadata or {}) if record is not None else {}
        numeric_cols = _resolve_numeric_cols(model, metadata)
        categorical_cols = _resolve_categorical_cols(model)
        version_label = int(record.version) if record is not None else int(db_artifact_version or 0)
        return AutoEncoderBundle(
            model=model,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            metadata=metadata,
            source_label=f"db:{db_artifact_name}:v{version_label}",
        )

    raise FileNotFoundError(
        "No autoencoder artifact was found. "
        f"Checked '{ae_pickle_path}'. "
        "Either export the traditional-ML raw stack to artifacts/raw_stack or set db_artifact_name to an existing ModelArtifact autoencoder."
    )


def load_available_raw_stack_artifacts(
    *,
    artifact_dir: str = str(DEFAULT_AE_ARTIFACT_DIR),
) -> RawStackArtifacts:
    artifact_dir_path = _resolve_artifact_dir_path(artifact_dir)
    metadata = _load_json(artifact_dir_path / "meta.json")
    classifier = _safe_pickle_load(artifact_dir_path / "clf_raw.pkl")
    trade_return_regressor = _safe_pickle_load(artifact_dir_path / "reg_trade_return_raw.pkl")
    if trade_return_regressor is None:
        trade_return_regressor = _safe_pickle_load(artifact_dir_path / "reg_raw.pkl")
    duration_regressor = _safe_pickle_load(artifact_dir_path / "reg_duration_raw.pkl")
    autoencoder = _safe_pickle_load(artifact_dir_path / "ae_raw.pkl")
    return RawStackArtifacts(
        artifact_dir=artifact_dir_path,
        metadata=metadata,
        classifier=classifier,
        trade_return_regressor=trade_return_regressor,
        duration_regressor=duration_regressor,
        autoencoder=autoencoder,
    )


def build_optional_raw_stack_predictions(
    *,
    query_row: pd.DataFrame,
    artifact_dir: str = str(DEFAULT_AE_ARTIFACT_DIR),
) -> dict[str, dict[str, Any]]:
    artifacts = load_available_raw_stack_artifacts(artifact_dir=artifact_dir)
    predictions: dict[str, dict[str, Any]] = {}

    if artifacts.classifier is not None:
        classifier_features = _resolve_runtime_feature_columns(
            artifacts.classifier,
            artifacts.metadata.get("feature_list"),
        )
        if classifier_features:
            classifier_input = _prepare_prediction_input(query_row, classifier_features)
            predicted_class_value = _predict_wrapper_value(artifacts.classifier, classifier_input, classifier_features)
            predicted_class = _map_classifier_output_label(artifacts.classifier, predicted_class_value)
            class_probabilities = _predict_wrapper_class_probabilities(
                artifacts.classifier,
                classifier_input,
                classifier_features,
            )
            predicted_probability = pd.to_numeric(
                pd.Series([class_probabilities.get(predicted_class)]),
                errors="coerce",
            ).iloc[0]
            if pd.isna(predicted_probability):
                predicted_probability = _predict_wrapper_probability(artifacts.classifier, classifier_input, classifier_features)
            predictions["classifier"] = {
                "predicted_class": predicted_class,
                "probability": None if pd.isna(predicted_probability) else float(predicted_probability),
                "class_probabilities": class_probabilities,
                "feature_count": int(len(classifier_features)),
            }

    if artifacts.trade_return_regressor is not None:
        regressor_features = _resolve_runtime_feature_columns(
            artifacts.trade_return_regressor,
            artifacts.metadata.get("feature_list"),
        )
        if regressor_features:
            regressor_input = _prepare_prediction_input(query_row, regressor_features)
            predicted_return = _predict_wrapper_value(artifacts.trade_return_regressor, regressor_input, regressor_features)
            if not pd.isna(predicted_return):
                predictions["regressor"] = {
                    "predicted_trade_return": float(predicted_return),
                    "predicted_trade_return_pct": float(predicted_return) * 100.0,
                    "feature_count": int(len(regressor_features)),
                }

    if artifacts.duration_regressor is not None:
        duration_features = _resolve_runtime_feature_columns(
            artifacts.duration_regressor,
            artifacts.metadata.get("feature_list"),
        )
        if duration_features:
            duration_input = _prepare_prediction_input(query_row, duration_features)
            predicted_days = _predict_wrapper_value(artifacts.duration_regressor, duration_input, duration_features)
            if not pd.isna(predicted_days):
                predictions["duration_regressor"] = {
                    "predicted_hold_days": float(predicted_days),
                    "feature_count": int(len(duration_features)),
                }

    if artifacts.autoencoder is not None:
        try:
            ae_numeric_cols = _resolve_numeric_cols(artifacts.autoencoder, artifacts.metadata)
        except Exception:
            ae_numeric_cols = [str(column) for column in list(artifacts.metadata.get("ae_numeric_cols") or [])]
        if ae_numeric_cols:
            ae_input = _prepare_model_frame(query_row, numeric_cols=ae_numeric_cols, categorical_cols=[])
            try:
                familiarity = np.asarray(
                    artifacts.autoencoder.familiarity(
                        ae_input,
                        numeric_cols=ae_numeric_cols,
                        categorical_cols=[],
                        quantile=99.9,
                        mode="latent_reciprocal_soft",
                    ),
                    dtype=float,
                ).reshape(-1)
            except Exception:
                familiarity = np.asarray([], dtype=float)
            if familiarity.size:
                predictions["autoencoder"] = {
                    "familiarity": float(familiarity[0]),
                    "feature_count": int(len(ae_numeric_cols)),
                }

    return predictions


def build_latest_feature_row(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    download_missing_prices: bool,
) -> pd.DataFrame:
    bootstrap_django()
    from pipeline.optimal_trade_random_forest import build_optimal_trade_feature_frame

    feature_frame, _diagnostics, _meta = build_optimal_trade_feature_frame(
        symbols=[symbol],
        start_date=str(start_date),
        end_date=str(end_date),
        download_missing_prices=bool(download_missing_prices),
    )
    if feature_frame.empty:
        raise RuntimeError(f"No feature rows were available for {symbol}.")
    feature_frame = feature_frame.copy()
    feature_frame["date"] = pd.to_datetime(feature_frame["date"], errors="coerce")
    feature_frame["symbol"] = feature_frame["symbol"].astype(str).str.strip().str.upper()
    feature_frame = feature_frame[
        (feature_frame["symbol"] == str(symbol).strip().upper())
        & feature_frame["date"].notna()
        & (feature_frame["date"] <= pd.Timestamp(end_date))
    ].copy()
    if feature_frame.empty:
        raise RuntimeError(f"No usable feature rows were found for {symbol} through {end_date}.")
    latest_date = feature_frame["date"].max()
    latest = feature_frame[feature_frame["date"] == latest_date].copy().sort_values(["date", "symbol"]).head(1)
    if latest.empty:
        raise RuntimeError(f"Unable to isolate the latest feature row for {symbol}.")
    return latest.reset_index(drop=True)


def build_reference_trade_catalog(
    *,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    label_freq: str,
    label_k_values: Sequence[int],
    min_profit_pct_points: float,
    download_missing_prices: bool,
) -> pd.DataFrame:
    normalized_symbols = tuple(_normalize_symbols(symbols))
    normalized_k_values = tuple(sorted({int(value) for value in list(label_k_values or []) if int(value) > 0}))
    if not normalized_k_values:
        normalized_k_values = (1,)
    return _build_reference_trade_catalog_cached(
        symbols=normalized_symbols,
        start_date=str(start_date),
        end_date=str(end_date),
        label_freq=str(label_freq or "YE").upper(),
        label_k_values=normalized_k_values,
        min_profit_pct_points=float(min_profit_pct_points or 0.0),
        download_missing_prices=bool(download_missing_prices),
    )


@lru_cache(maxsize=6)
def _build_reference_trade_catalog_cached(
    *,
    symbols: tuple[str, ...],
    start_date: str,
    end_date: str,
    label_freq: str,
    label_k_values: tuple[int, ...],
    min_profit_pct_points: float,
    download_missing_prices: bool,
) -> pd.DataFrame:
    bootstrap_django()
    from pipeline.api import build_label_dataframe
    from pipeline.optimal_trade_random_forest import build_optimal_trade_feature_frame
    from trading.live_trade import build_technical_dataframe_from_django

    normalized_symbols = list(symbols)
    if not normalized_symbols:
        raise RuntimeError("At least one reference symbol is required to build the historical label dataframe.")

    feature_frame, _diagnostics, _meta = build_optimal_trade_feature_frame(
        symbols=normalized_symbols,
        start_date=str(start_date),
        end_date=str(end_date),
        download_missing_prices=bool(download_missing_prices),
    )
    if feature_frame.empty:
        raise RuntimeError("No historical feature rows were available for the reference universe.")
    feature_frame = feature_frame.copy()
    feature_frame["date"] = pd.to_datetime(feature_frame["date"], errors="coerce")
    feature_frame["symbol"] = feature_frame["symbol"].astype(str).str.strip().str.upper()

    technical_df, _technical_cols = build_technical_dataframe_from_django(
        symbols=normalized_symbols,
        start_date=str(start_date),
        end_date=str(end_date),
    )
    if technical_df.empty:
        raise RuntimeError("No technical rows were available to build the traditional ML label dataframe.")
    symbols_in_panel = set(technical_df.index.get_level_values("symbol"))
    daily_map_all = {
        symbol: technical_df.xs(symbol, level="symbol").copy()
        for symbol in normalized_symbols
        if symbol in symbols_in_panel
    }

    def _resolve_price_on_or_before(symbol: str, target_date: pd.Timestamp) -> float | None:
        daily_df = daily_map_all.get(str(symbol).strip().upper())
        if daily_df is None or daily_df.empty or pd.isna(target_date):
            return None
        work = daily_df.copy()
        work.index = pd.to_datetime(work.index, errors="coerce")
        work = work[work.index.notna()].sort_index()
        if work.empty or "close" not in work.columns:
            return None
        target_ts = pd.Timestamp(target_date).normalize()
        candidates = work.loc[work.index <= target_ts]
        if candidates.empty:
            candidates = work.loc[work.index >= target_ts]
        if candidates.empty:
            return None
        value = pd.to_numeric(pd.Series([candidates.iloc[-1]["close"]]), errors="coerce").iloc[0]
        if pd.isna(value):
            return None
        return float(value)

    label_df = build_label_dataframe(
        daily_by_symbol=daily_map_all,
        k_params={str(label_freq or "YE").upper(): [int(value) for value in list(label_k_values or [1]) if int(value) > 0]},
        execution_params={
            "price_col": "close",
            "fee_bps": 5.0,
            "slippage_bps": 5.0,
        },
        weighting={
            "use_sample_weight": True,
            "alpha": 4.0,
            "r_clip": 0.10,
            "horizon_balance": True,
        },
        add_rank_labels=True,
        verbose=False,
    )
    if label_df.empty:
        raise RuntimeError("No historical labels were generated for the requested symbol history.")

    label_df = label_df.copy()
    if "symbol" not in label_df.columns or "date" not in label_df.columns:
        label_df = label_df.reset_index()
    action_label = label_df.get("label", pd.Series([""] * len(label_df), index=label_df.index)).astype(str).str.strip().str.lower()
    event_label = label_df.get("event", pd.Series([""] * len(label_df), index=label_df.index)).astype(str).str.strip().str.lower()
    if not event_label.eq("entry").any():
        inferred_event = action_label.map({
            "buy": "entry",
            "short": "entry",
            "sell": "exit",
            "cover": "exit",
        }).fillna("")
        event_label = inferred_event
        label_df["event"] = inferred_event
    label_df["symbol"] = label_df["symbol"].astype(str).str.strip().str.upper()
    label_df["date"] = pd.to_datetime(label_df["date"], errors="coerce")
    label_df["hold_days"] = pd.to_numeric(
        label_df.get("hold_days", label_df.get("trade_duration_days")),
        errors="coerce",
    )
    hold_delta = pd.to_timedelta(label_df["hold_days"].fillna(0.0), unit="D")
    action_label = label_df.get("label", pd.Series([""] * len(label_df), index=label_df.index)).astype(str).str.strip().str.lower()
    is_exit_event = action_label.isin(["sell", "cover"])
    stored_entry_date = pd.to_datetime(label_df.get("entry_date"), errors="coerce")
    stored_exit_date = pd.to_datetime(label_df.get("exit_date"), errors="coerce")
    label_df["entry_date"] = stored_entry_date
    label_df["exit_date"] = stored_exit_date
    missing_entry_date = label_df["entry_date"].isna()
    missing_exit_date = label_df["exit_date"].isna()
    label_df.loc[missing_entry_date, "entry_date"] = pd.to_datetime(label_df.loc[missing_entry_date, "date"], errors="coerce")
    label_df.loc[missing_entry_date & is_exit_event, "entry_date"] = pd.to_datetime(
        label_df.loc[missing_entry_date & is_exit_event, "date"],
        errors="coerce",
    ) - hold_delta.loc[missing_entry_date & is_exit_event]
    label_df["ret_dec"] = pd.to_numeric(
        label_df.get("ret_dec", label_df.get("trade_return")),
        errors="coerce",
    )
    if "k" in label_df.columns:
        label_df["k"] = pd.Series(pd.to_numeric(label_df["k"], errors="coerce"), index=label_df.index).astype("Int64")
    elif "horizon" in label_df.columns:
        extracted_k = label_df["horizon"].astype(str).str.extract(r"k(\d+)", expand=False)
        label_df["k"] = pd.Series(pd.to_numeric(extracted_k, errors="coerce"), index=label_df.index).astype("Int64")
    else:
        label_df["k"] = pd.Series([pd.NA] * len(label_df), index=label_df.index, dtype="Int64")
    if "freq" not in label_df.columns and "horizon" in label_df.columns:
        label_df["freq"] = label_df["horizon"].astype(str).str.split("_k").str[0]
    label_df.loc[missing_exit_date, "exit_date"] = pd.to_datetime(
        label_df.loc[missing_exit_date, "date"],
        errors="coerce",
    )
    label_df.loc[missing_exit_date & ~is_exit_event, "exit_date"] = label_df.loc[missing_exit_date & ~is_exit_event, "entry_date"] + hold_delta.loc[missing_exit_date & ~is_exit_event]

    stored_entry_px = pd.to_numeric(label_df.get("entry_px"), errors="coerce")
    stored_exit_px = pd.to_numeric(label_df.get("exit_px"), errors="coerce")
    label_df["entry_px"] = stored_entry_px
    label_df["exit_px"] = stored_exit_px
    missing_entry_px = label_df["entry_px"].isna()
    missing_exit_px = label_df["exit_px"].isna()
    label_df.loc[missing_entry_px, "entry_px"] = [
        _resolve_price_on_or_before(symbol, entry_date)
        for symbol, entry_date in zip(label_df.loc[missing_entry_px, "symbol"], label_df.loc[missing_entry_px, "entry_date"])
    ]
    label_df.loc[missing_exit_px, "exit_px"] = [
        _resolve_price_on_or_before(symbol, exit_date)
        for symbol, exit_date in zip(label_df.loc[missing_exit_px, "symbol"], label_df.loc[missing_exit_px, "exit_date"])
    ]
    label_df = label_df.dropna(subset=["symbol", "date", "entry_date"]).copy()
    search_action = label_df.get("action_label", label_df.get("label", pd.Series([""] * len(label_df), index=label_df.index)))
    search_action = search_action.astype(str).str.strip().str.lower()
    search_df = label_df.loc[search_action.isin(["buy", "short"])].copy()
    if search_df.empty:
        raise RuntimeError("No buy/short historical labels were available for the requested symbol history.")

    merged = search_df.merge(
        feature_frame,
        left_on=["symbol", "date"],
        right_on=["symbol", "date"],
        how="inner",
        suffixes=("", "_feature"),
    )
    if merged.empty:
        raise RuntimeError("Traditional ML labels were found, but none could be aligned to feature rows.")
    merged = merged.sort_values(["entry_date", "symbol", "ret_dec"], ascending=[True, True, False])
    merged = merged.drop_duplicates(subset=["symbol", "entry_date", "side", "freq", "k"], keep="first")
    return merged.reset_index(drop=True)


def clear_reference_trade_catalog_cache() -> None:
    _build_reference_trade_catalog_cached.cache_clear()


def save_trade_vector_artifacts(
    *,
    reference_catalog: pd.DataFrame,
    ae_model: Any,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str] = (),
    artifact_dir: str = str(DEFAULT_AE_ARTIFACT_DIR),
) -> dict[str, Any]:
    if reference_catalog is None or reference_catalog.empty:
        raise ValueError("reference_catalog is empty.")

    artifact_dir_path = _resolve_artifact_dir_path(artifact_dir)
    vector_dir = artifact_dir_path / TRADE_VECTOR_SUBDIR
    vector_dir.mkdir(parents=True, exist_ok=True)

    latent_input = _prepare_model_frame(
        reference_catalog,
        numeric_cols=list(numeric_cols or []),
        categorical_cols=list(categorical_cols or []),
    )
    latent_matrix = np.asarray(
        ae_model.latent(
            latent_input,
            numeric_cols=list(numeric_cols or []),
            categorical_cols=list(categorical_cols or []),
        ),
        dtype=np.float32,
    )
    familiarity = np.asarray(
        ae_model.familiarity(
            latent_input,
            numeric_cols=list(numeric_cols or []),
            categorical_cols=list(categorical_cols or []),
            quantile=99.9,
            mode="latent_reciprocal_soft",
        ),
        dtype=np.float32,
    ).reshape(-1)
    if latent_matrix.ndim != 2 or latent_matrix.shape[0] != len(reference_catalog):
        raise ValueError("Autoencoder latent matrix shape does not align to the reference catalog.")

    latent_cols = [f"z_{idx}" for idx in range(latent_matrix.shape[1])]
    frame = reference_catalog.reset_index(drop=True).copy()
    for idx, column in enumerate(latent_cols):
        frame[column] = latent_matrix[:, idx]
    frame["candidate_ae_familiarity"] = familiarity
    frame_path = vector_dir / TRADE_VECTOR_FRAME_FILENAME
    frame.to_pickle(frame_path)

    normalized_latent = _l2_normalize(latent_matrix.astype(np.float32, copy=False)).astype(np.float32, copy=False)
    backend = "numpy"
    faiss_backend_error = ""
    faiss_path = vector_dir / TRADE_VECTOR_FAISS_FILENAME
    matrix_path = vector_dir / TRADE_VECTOR_MATRIX_FILENAME
    np.save(matrix_path, normalized_latent.astype(np.float32, copy=False))

    try:
        import faiss  # type: ignore

        index = faiss.IndexFlatIP(int(normalized_latent.shape[1]))
        index.add(normalized_latent)
        faiss.write_index(index, str(faiss_path))
        backend = "faiss"
    except Exception as exc:
        faiss_backend_error = f"{type(exc).__name__}: {exc}"
        if faiss_path.exists():
            try:
                faiss_path.unlink()
            except Exception:
                pass

    metadata = {
        "backend": backend,
        "row_count": int(len(frame)),
        "latent_dim": int(len(latent_cols)),
        "latent_cols": latent_cols,
        "numeric_cols": [str(column) for column in list(numeric_cols or [])],
        "categorical_cols": [str(column) for column in list(categorical_cols or [])],
        "frame_path": str(frame_path),
        "matrix_path": str(matrix_path),
        "faiss_path": str(faiss_path),
        "faiss_backend_error": faiss_backend_error,
    }
    (vector_dir / TRADE_VECTOR_META_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _load_trade_vector_artifacts_cached.cache_clear()
    return metadata


def load_trade_vector_artifacts(
    *,
    artifact_dir: str = str(DEFAULT_AE_ARTIFACT_DIR),
) -> TradeVectorArtifacts | None:
    artifact_dir_path = _resolve_artifact_dir_path(artifact_dir)
    return _load_trade_vector_artifacts_cached(str(artifact_dir_path))


@lru_cache(maxsize=4)
def _load_trade_vector_artifacts_cached(artifact_dir: str) -> TradeVectorArtifacts | None:
    artifact_dir_path = Path(str(artifact_dir))
    vector_dir = artifact_dir_path / TRADE_VECTOR_SUBDIR
    meta = _load_json(vector_dir / TRADE_VECTOR_META_FILENAME)
    if not meta:
        return None

    frame_path = Path(str(meta.get("frame_path") or vector_dir / TRADE_VECTOR_FRAME_FILENAME))
    if not frame_path.exists():
        return None
    frame = pd.read_pickle(frame_path)
    latent_cols = [str(column) for column in list(meta.get("latent_cols") or [])]
    if not latent_cols:
        latent_cols = [str(column) for column in frame.columns if str(column).startswith("z_")]
    if not latent_cols:
        return None

    backend = str(meta.get("backend") or "numpy").strip().lower() or "numpy"
    normalized_latent = None
    faiss_index = None
    matrix_path = Path(str(meta.get("matrix_path") or vector_dir / TRADE_VECTOR_MATRIX_FILENAME))
    if matrix_path.exists():
        normalized_latent = np.asarray(np.load(matrix_path), dtype=np.float32)
    if backend == "faiss":
        try:
            import faiss  # type: ignore

            faiss_path = Path(str(meta.get("faiss_path") or vector_dir / TRADE_VECTOR_FAISS_FILENAME))
            if faiss_path.exists():
                faiss_index = faiss.read_index(str(faiss_path))
            else:
                backend = "numpy"
        except Exception:
            backend = "numpy"
    if normalized_latent is None:
        normalized_latent = _l2_normalize(frame[latent_cols].to_numpy(dtype=np.float32, copy=False)).astype(np.float32, copy=False)
    return TradeVectorArtifacts(
        frame=frame,
        latent_cols=latent_cols,
        backend=backend,
        faiss_index=faiss_index,
        normalized_latent=normalized_latent,
        metadata=meta,
    )


def _query_trade_vector_artifacts(
    artifacts: TradeVectorArtifacts,
    *,
    query_vector: np.ndarray,
    top_k: int,
    reference_symbols: Sequence[str] = (),
) -> pd.DataFrame:
    work = artifacts.frame.copy()
    query_norm = _l2_normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1)).astype(np.float32, copy=False)
    symbol_filter = {str(symbol).strip().upper() for symbol in list(reference_symbols or []) if str(symbol).strip()}
    if symbol_filter:
        work = work[work["symbol"].astype(str).str.upper().isin(symbol_filter)].copy()
        if work.empty:
            return work
        matrix = _l2_normalize(work[artifacts.latent_cols].to_numpy(dtype=np.float32, copy=False)).astype(np.float32, copy=False)
        similarity_scores = np.nan_to_num(matrix @ query_norm[0], nan=0.0, posinf=0.0, neginf=0.0).astype(float, copy=False)
        work["similarity_score"] = similarity_scores
        return work.sort_values(["similarity_score", "ret_dec"], ascending=[False, False]).head(max(int(top_k), 1))

    if artifacts.backend == "faiss" and artifacts.faiss_index is not None:
        distances, indices = artifacts.faiss_index.search(query_norm, int(max(1, min(int(top_k), len(work)))))
        picked = work.iloc[indices[0]].copy().reset_index(drop=True)
        picked["similarity_score"] = distances[0]
        return picked.sort_values(["similarity_score", "ret_dec"], ascending=[False, False]).reset_index(drop=True)

    matrix = artifacts.normalized_latent
    if matrix is None:
        matrix = _l2_normalize(work[artifacts.latent_cols].to_numpy(dtype=np.float32, copy=False)).astype(np.float32, copy=False)
    similarity_scores = np.nan_to_num(matrix @ query_norm[0], nan=0.0, posinf=0.0, neginf=0.0).astype(float, copy=False)
    ranked = work.copy()
    ranked["similarity_score"] = similarity_scores
    return ranked.sort_values(["similarity_score", "ret_dec"], ascending=[False, False]).head(max(int(top_k), 1)).reset_index(drop=True)


def embed_rows(frame: pd.DataFrame, bundle: AutoEncoderBundle) -> tuple[np.ndarray, np.ndarray]:
    work = _prepare_model_frame(frame, numeric_cols=bundle.numeric_cols, categorical_cols=bundle.categorical_cols)
    latent = np.asarray(
        bundle.model.latent(
            work,
            numeric_cols=bundle.numeric_cols,
            categorical_cols=bundle.categorical_cols,
        ),
        dtype=float,
    )
    familiarity = np.asarray(
        bundle.model.familiarity(
            work,
            numeric_cols=bundle.numeric_cols,
            categorical_cols=bundle.categorical_cols,
            quantile=99.9,
            mode="latent_reciprocal_soft",
        ),
        dtype=float,
    ).reshape(-1)
    return latent, familiarity


def cosine_similarity_matrix(matrix: np.ndarray, query_vector: Sequence[float]) -> np.ndarray:
    base = _l2_normalize(np.asarray(matrix, dtype=float))
    query = _l2_normalize(np.asarray(query_vector, dtype=float).reshape(1, -1))[0]
    return np.nan_to_num(base @ query, nan=0.0, posinf=0.0, neginf=0.0).astype(float, copy=False)


def resolve_numeric_feature_columns(
    *,
    query_row: pd.DataFrame,
    reference_catalog: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
) -> list[str]:
    if feature_columns is not None:
        return [
            str(column)
            for column in list(feature_columns or [])
            if str(column) in query_row.columns and str(column) in reference_catalog.columns
        ]

    columns: list[str] = []
    for column in reference_catalog.columns:
        name = str(column)
        if name in NON_FEATURE_COLUMNS:
            continue
        if name not in query_row.columns:
            continue
        ref_numeric = pd.to_numeric(reference_catalog[name], errors="coerce")
        query_numeric = pd.to_numeric(query_row[name], errors="coerce")
        if ref_numeric.notna().any() and query_numeric.notna().any():
            columns.append(name)
    return columns


def build_feature_similarity_inputs(
    *,
    query_row: pd.DataFrame,
    reference_catalog: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cols = [str(column) for column in list(feature_columns or [])]
    reference_numeric = reference_catalog[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    query_numeric = query_row[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    means = reference_numeric.mean(axis=0).to_numpy(dtype=float)
    stds = reference_numeric.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0).to_numpy(dtype=float)
    reference_matrix = (reference_numeric.to_numpy(dtype=float) - means) / stds
    query_vector = (query_numeric.iloc[0].to_numpy(dtype=float) - means) / stds
    reference_matrix = np.nan_to_num(reference_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    query_vector = np.nan_to_num(query_vector, nan=0.0, posinf=0.0, neginf=0.0)
    return query_vector, reference_matrix, means, stds


def build_indicator_summary(
    query_row: pd.Series | dict[str, Any],
    reference_frame: pd.DataFrame,
    *,
    feature_columns: Iterable[str],
    max_per_family: int = 6,
) -> pd.DataFrame:
    if isinstance(query_row, pd.Series):
        row = query_row.copy()
    else:
        row = pd.Series(dict(query_row or {}))
    cols = [str(column) for column in list(feature_columns or []) if str(column) in reference_frame.columns]
    if not cols:
        return pd.DataFrame(columns=["family", "feature", "value", "reference_mean", "zscore"])

    ref_numeric = reference_frame[cols].apply(pd.to_numeric, errors="coerce")
    means = ref_numeric.mean(axis=0)
    stds = ref_numeric.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    rows: list[dict[str, Any]] = []
    for column in cols:
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        mean_value = pd.to_numeric(pd.Series([means.get(column)]), errors="coerce").iloc[0]
        std_value = pd.to_numeric(pd.Series([stds.get(column)]), errors="coerce").iloc[0]
        zscore = 0.0 if pd.isna(std_value) or float(std_value) == 0.0 else (float(value) - float(mean_value)) / float(std_value)
        rows.append(
            {
                "family": _feature_family(column),
                "feature": column,
                "value": float(value),
                "reference_mean": float(0.0 if pd.isna(mean_value) else mean_value),
                "zscore": float(zscore),
                "abs_zscore": abs(float(zscore)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["family", "feature", "value", "reference_mean", "zscore"])
    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["family", "abs_zscore"], ascending=[True, False])
    summary = summary.groupby("family", group_keys=False).head(max(int(max_per_family), 1))
    return summary.drop(columns=["abs_zscore"]).reset_index(drop=True)


def build_feature_attribution_summary(
    *,
    query_row: pd.DataFrame,
    nearest_trades: pd.DataFrame,
    feature_columns: Iterable[str],
    top_n: int = 10,
) -> pd.DataFrame:
    cols = [str(column) for column in list(feature_columns or []) if str(column) in query_row.columns and str(column) in nearest_trades.columns]
    if not cols or nearest_trades.empty or query_row.empty:
        return pd.DataFrame(
            columns=[
                "family",
                "feature",
                "current_value",
                "top_k_mean",
                "direction_pct_change",
                "tree_importance",
                "tree_importance_pct",
                "sample_count",
            ]
        )

    baseline = query_row.iloc[0]
    nearest_numeric = (
        nearest_trades[cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    valid_columns = [column for column in cols if nearest_numeric[column].notna().any()]
    if not valid_columns:
        return pd.DataFrame(
            columns=[
                "family",
                "feature",
                "current_value",
                "top_k_mean",
                "direction_pct_change",
                "tree_importance",
                "tree_importance_pct",
                "sample_count",
            ]
        )

    tree_input = nearest_numeric[valid_columns].copy()
    tree_input = tree_input.fillna(tree_input.median(axis=0, numeric_only=True)).fillna(0.0)
    direction_labels = _resolve_trade_direction_labels(nearest_trades)
    sample_weights = _resolve_feature_attribution_sample_weights(nearest_trades)
    raw_importance = pd.Series(dtype=float)
    if len(direction_labels) == len(tree_input) and direction_labels.nunique(dropna=True) >= 2:
        encoded_target = direction_labels.map({"short": 0, "long": 1}).astype(int)
        forest = RandomForestClassifier(
            n_estimators=200,
            max_depth=min(5, max(2, int(len(tree_input) // 2) if len(tree_input) >= 6 else 3)),
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=1337,
        )
        fit_kwargs: dict[str, Any] = {}
        if sample_weights is not None and len(sample_weights) == len(tree_input):
            fit_kwargs["sample_weight"] = sample_weights
        forest.fit(tree_input, encoded_target, **fit_kwargs)
        raw_importance = pd.Series(forest.feature_importances_, index=valid_columns, dtype=float)
        raw_importance = raw_importance[raw_importance > 0.0].sort_values(ascending=False)
    if raw_importance.empty:
        variance_fallback = tree_input.var(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw_importance = variance_fallback[variance_fallback > 0.0].sort_values(ascending=False)
    if raw_importance.empty:
        return pd.DataFrame(
            columns=[
                "family",
                "feature",
                "current_value",
                "top_k_mean",
                "direction_pct_change",
                "tree_importance",
                "tree_importance_pct",
                "sample_count",
            ]
        )

    importance_total = float(raw_importance.sum()) if raw_importance.notna().any() else 0.0
    importance_pct = ((raw_importance / importance_total) * 100.0) if importance_total > 0.0 else raw_importance * 0.0
    rows: list[dict[str, Any]] = []
    for column in cols:
        if column not in raw_importance.index:
            continue
        current_value = pd.to_numeric(pd.Series([baseline.get(column)]), errors="coerce").iloc[0]
        trade_values = nearest_numeric[column].dropna()
        if pd.isna(current_value) or trade_values.empty:
            continue
        current_value = float(current_value)
        feature_direction = _infer_feature_direction_label(
            feature_name=column,
            current_value=current_value,
            nearest_trades=nearest_trades,
        )
        rows.append(
            {
                "family": _feature_family(column),
                "feature": column,
                "current_value": current_value,
                "top_k_mean": float(trade_values.mean()),
                "direction_label": feature_direction,
                "tree_importance": float(raw_importance.get(column, 0.0)),
                "tree_importance_pct": float(importance_pct.get(column, 0.0)),
                "sample_count": int(trade_values.shape[0]),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "family",
                "feature",
                "current_value",
                "top_k_mean",
                "direction_label",
                "tree_importance",
                "tree_importance_pct",
                "sample_count",
            ]
        )

    summary = pd.DataFrame(rows)
    summary = summary.replace([np.inf, -np.inf], np.nan)
    summary = summary.dropna(subset=["tree_importance_pct"]).copy()
    if summary.empty:
        return pd.DataFrame(
            columns=[
                "family",
                "feature",
                "current_value",
                "top_k_mean",
                "direction_label",
                "tree_importance",
                "tree_importance_pct",
                "sample_count",
            ]
        )
    summary = summary.sort_values(
        ["tree_importance_pct", "tree_importance"],
        ascending=[False, False],
    ).head(max(int(top_n), 1))
    numeric_columns = [
        "current_value",
        "top_k_mean",
        "tree_importance",
        "tree_importance_pct",
    ]
    summary[numeric_columns] = summary[numeric_columns].apply(pd.to_numeric, errors="coerce").round(3)
    return summary.reset_index(drop=True)


def _format_nearest_trades(frame: pd.DataFrame, *, include_familiarity: bool = True) -> pd.DataFrame:
    out = frame.copy()
    out["entry_date"] = pd.to_datetime(out["entry_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["exit_date"] = pd.to_datetime(out["exit_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["similarity_score"] = pd.to_numeric(out["similarity_score"], errors="coerce").round(6)
    if "candidate_ae_familiarity" in out.columns:
        out["candidate_ae_familiarity"] = pd.to_numeric(out["candidate_ae_familiarity"], errors="coerce").round(6)
    out["trade_return_pct"] = pd.to_numeric(out["trade_return_pct"], errors="coerce").round(2)
    side_sign = out.get("side", pd.Series([""] * len(out), index=out.index)).astype(str).str.strip().str.lower().map({"long": 1.0, "short": -1.0}).fillna(1.0)
    out["signed_trade_return_pct"] = (pd.to_numeric(out["trade_return_pct"], errors="coerce") * side_sign).round(2)
    out["hold_days"] = pd.to_numeric(out["hold_days"], errors="coerce").round(0)
    keep = [
        "symbol",
        "side",
        "entry_date",
        "exit_date",
        "signed_trade_return_pct",
        "hold_days",
    ]
    if include_familiarity and "candidate_ae_familiarity" in out.columns:
        keep.append("candidate_ae_familiarity")
    optional = [column for column in ["entry_px", "exit_px"] if column in out.columns]
    display = out[keep + optional].reset_index(drop=True)
    return display.rename(
        columns={
            "symbol": "Symbol",
            "side": "Side",
            "entry_date": "Entry Date",
            "exit_date": "Exit Date",
            "signed_trade_return_pct": "Signed Trade Return",
            "hold_days": "Hold Days",
            "candidate_ae_familiarity": "AE Familiarity",
            "entry_px": "Entry Price",
            "exit_px": "Exit Price",
        }
    )


def _prepare_model_frame(
    frame: pd.DataFrame,
    *,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> pd.DataFrame:
    work = frame.copy()
    missing_numeric = [column for column in list(numeric_cols or []) if column not in work.columns]
    if missing_numeric:
        filler = pd.DataFrame(0.0, index=work.index, columns=missing_numeric)
        work = pd.concat([work, filler], axis=1)
    for column in list(numeric_cols or []):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if numeric_cols:
        work[list(numeric_cols)] = work[list(numeric_cols)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    missing_categorical = [column for column in list(categorical_cols or []) if column not in work.columns]
    if missing_categorical:
        filler = pd.DataFrame("", index=work.index, columns=missing_categorical)
        work = pd.concat([work, filler], axis=1)
    for column in list(categorical_cols or []):
        work[column] = work[column].astype(str).fillna("")
    return work


def _prepare_prediction_input(frame: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    work = frame.copy()
    missing_cols = [column for column in list(feature_cols or []) if column not in work.columns]
    if missing_cols:
        filler = pd.DataFrame(0.0, index=work.index, columns=missing_cols)
        work = pd.concat([work, filler], axis=1)
    work[list(feature_cols)] = (
        work[list(feature_cols)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return work


def _resolve_runtime_feature_columns(model_obj: Any, fallback_features: Sequence[str] | None = None) -> list[str]:
    used_features = list(getattr(model_obj, "_used_features", []) or [])
    if used_features:
        return [str(column) for column in used_features]
    return [str(column) for column in list(fallback_features or [])]


def _predict_wrapper_value(model_obj: Any, prediction_df: pd.DataFrame, feature_cols: Sequence[str]) -> float:
    predict = getattr(model_obj, "predict", None)
    if callable(predict):
        try:
            values = predict(prediction_df, feature_cols=list(feature_cols))
        except TypeError:
            values = predict(prediction_df[list(feature_cols)])
        array = np.asarray(values, dtype=float).reshape(-1)
        return float(array[0]) if array.size else float("nan")
    return float("nan")


def _predict_wrapper_probability(model_obj: Any, prediction_df: pd.DataFrame, feature_cols: Sequence[str]) -> float:
    predict_proba = getattr(getattr(model_obj, "model", None), "predict_proba", None)
    if not callable(predict_proba):
        return float("nan")
    try:
        proba = np.asarray(predict_proba(prediction_df[list(feature_cols)]), dtype=float)
    except Exception:
        return float("nan")
    if proba.ndim != 2 or proba.shape[0] < 1 or proba.shape[1] < 1:
        return float("nan")
    if proba.shape[1] >= 2:
        return float(proba[0, 1])
    return float(proba[0, 0])


def _predict_wrapper_class_probabilities(
    model_obj: Any,
    prediction_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> dict[str, float]:
    predict_proba = getattr(getattr(model_obj, "model", None), "predict_proba", None)
    classes = getattr(getattr(model_obj, "model", None), "classes_", None)
    if not callable(predict_proba):
        return {}
    try:
        proba = np.asarray(predict_proba(prediction_df[list(feature_cols)]), dtype=float)
    except Exception:
        return {}
    if proba.ndim != 2 or proba.shape[0] < 1 or proba.shape[1] < 1:
        return {}
    if classes is None or len(classes) != proba.shape[1]:
        return {}
    class_mapping = dict(getattr(model_obj, "_class_mapping", {}) or {})
    rows: dict[str, float] = {}
    for idx, raw_class in enumerate(list(classes)):
        try:
            class_key: Any = int(raw_class)
        except Exception:
            class_key = raw_class
        mapped_label = _normalize_classifier_label(str(class_mapping.get(class_key, raw_class)))
        rows[mapped_label] = float(proba[0, idx])
    return rows


def _map_classifier_output_label(model_obj: Any, prediction_value: float) -> str:
    if pd.isna(prediction_value):
        return ""
    class_mapping = dict(getattr(model_obj, "_class_mapping", {}) or {})
    try:
        key = int(round(float(prediction_value)))
    except Exception:
        return _normalize_classifier_label(str(prediction_value))
    raw_label = str(class_mapping.get(key, key))
    return _normalize_classifier_label(raw_label)


def _normalize_classifier_label(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "long", "buy"}:
        return "Long"
    if normalized in {"0", "short", "sell", "cover"}:
        return "Short"
    if not normalized:
        return ""
    return str(value).strip().title()


def _resolve_trade_direction_labels(frame: pd.DataFrame) -> pd.Series:
    for column in ["side", "direction_label", "market_position"]:
        if column not in frame.columns:
            continue
        values = frame[column]
        if column == "direction_label":
            mapped = pd.to_numeric(values, errors="coerce").map({1.0: "long", 0.0: "short"})
        elif column == "market_position":
            numeric = pd.to_numeric(values, errors="coerce")
            mapped = numeric.apply(
                lambda value: "long" if pd.notna(value) and float(value) > 0.0 else "short" if pd.notna(value) and float(value) < 0.0 else ""
            )
        else:
            mapped = values.astype(str).str.strip().str.lower()
        normalized = mapped.replace({"buy": "long", "sell": "long", "cover": "short"})
        normalized = normalized.where(normalized.isin(["long", "short"]), "")
        if normalized.replace("", np.nan).notna().any():
            return normalized
    return pd.Series([""] * len(frame), index=frame.index, dtype=str)


def _resolve_feature_attribution_sample_weights(frame: pd.DataFrame) -> np.ndarray | None:
    if frame.empty:
        return None

    similarity = frame.get("similarity_score")
    if isinstance(similarity, pd.Series):
        similarity = pd.to_numeric(similarity, errors="coerce")
    else:
        similarity_value = pd.to_numeric(pd.Series([similarity]), errors="coerce").iloc[0]
        similarity = pd.Series([similarity_value] * len(frame), index=frame.index, dtype=float)
    similarity = similarity.replace([np.inf, -np.inf], np.nan)
    if similarity.notna().any():
        similarity = similarity.clip(lower=0.0)
        similarity = similarity.fillna(float(similarity.dropna().min() or 0.0))
        similarity_weights = similarity + 1e-6
    else:
        similarity_weights = None

    familiarity = frame.get("candidate_ae_familiarity")
    if isinstance(familiarity, pd.Series):
        familiarity = pd.to_numeric(familiarity, errors="coerce")
    else:
        familiarity_value = pd.to_numeric(pd.Series([familiarity]), errors="coerce").iloc[0]
        familiarity = pd.Series([familiarity_value] * len(frame), index=frame.index, dtype=float)
    familiarity = familiarity.replace([np.inf, -np.inf], np.nan)
    if familiarity.notna().any():
        familiarity = familiarity.clip(lower=0.0)
        familiarity = familiarity.fillna(float(familiarity.dropna().median() or 0.0))
        familiarity_weights = familiarity + 1e-6
    else:
        familiarity_weights = None

    if similarity_weights is not None and familiarity_weights is not None:
        weights = similarity_weights * familiarity_weights
    elif similarity_weights is not None:
        weights = similarity_weights
    elif familiarity_weights is not None:
        weights = familiarity_weights
    else:
        return None

    weight_array = np.asarray(weights, dtype=float).reshape(-1)
    if weight_array.size != len(frame) or not np.isfinite(weight_array).any():
        return None
    weight_array = np.nan_to_num(weight_array, nan=0.0, posinf=0.0, neginf=0.0)
    if float(weight_array.sum()) <= 0.0:
        return None
    return weight_array


def _infer_feature_direction_label(
    *,
    feature_name: str,
    current_value: float,
    nearest_trades: pd.DataFrame,
) -> str:
    directions = _resolve_trade_direction_labels(nearest_trades)
    if directions.replace("", np.nan).notna().sum() == 0:
        return ""
    feature_values = pd.to_numeric(nearest_trades.get(feature_name), errors="coerce")
    if feature_values.notna().sum() == 0:
        return str(directions.replace("", np.nan).mode().iloc[0]).title() if directions.replace("", np.nan).notna().any() else ""

    direction_means: dict[str, float] = {}
    for label in ["long", "short"]:
        label_values = feature_values[directions == label].dropna()
        if not label_values.empty:
            direction_means[label] = float(label_values.mean())
    if not direction_means:
        return ""
    if len(direction_means) == 1:
        return str(next(iter(direction_means.keys()))).title()

    long_distance = abs(float(current_value) - float(direction_means["long"]))
    short_distance = abs(float(current_value) - float(direction_means["short"]))
    return "Long" if long_distance <= short_distance else "Short"


def _resolve_numeric_cols(model: Any, metadata: dict[str, Any]) -> list[str]:
    artifact = getattr(model, "_artifact", None)
    numeric_cols = list(getattr(artifact, "numeric_cols", []) or [])
    if not numeric_cols:
        numeric_cols = [str(column) for column in list(metadata.get("ae_numeric_cols") or [])]
    if not numeric_cols:
        raise ValueError("Unable to resolve numeric columns for the autoencoder artifact.")
    return [str(column) for column in numeric_cols]


def _resolve_categorical_cols(model: Any) -> list[str]:
    artifact = getattr(model, "_artifact", None)
    return [str(column) for column in list(getattr(artifact, "cat_cols", []) or [])]


def _resolve_artifact_dir_path(artifact_dir: str) -> Path:
    artifact_dir_path = Path(str(artifact_dir or DEFAULT_AE_ARTIFACT_DIR)).expanduser()
    if not artifact_dir_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[1]
        artifact_dir_path = repo_root / artifact_dir_path
    return artifact_dir_path


def _resolve_query_start_timestamp(*, query: OptimalTradeQuery, as_of_ts: pd.Timestamp) -> pd.Timestamp:
    reference_start = str(query.reference_start_date or "").strip()
    if reference_start:
        try:
            return pd.Timestamp(reference_start)
        except Exception:
            pass
    return as_of_ts - pd.DateOffset(years=max(int(query.query_lookback_years), 1))


def _safe_pickle_load(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def _feature_family(column: str) -> str:
    name = str(column or "").lower()
    if name.startswith("px__"):
        return "technical"
    if name.startswith("macro__"):
        return "macro"
    return "fundamental"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")) or {})
    except Exception:
        return {}


def _normalize_symbol(value: str) -> str:
    text = str(value or "").strip().upper()
    if not text:
        raise ValueError("A non-empty symbol is required.")
    return text


def _normalize_symbols(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in list(values or []):
        symbol = _normalize_symbol(value)
        if symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.nan_to_num(arr / norms, nan=0.0, posinf=0.0, neginf=0.0)
