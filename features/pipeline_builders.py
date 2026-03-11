from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd

from data.historical_prices import load_adjusted_price_frames
from fmp.models import EconomicIndicatorSeries, Symbol, TreasuryRateSeries
from features.feature_builders import (
    build_event_features,
    build_fundamental_change_features,
    build_ownership_features,
    build_price_technical_features,
    build_statement_quality_features,
)
from features.macro import EconomicDataConfig, broadcast_series_to_daily, fetch_economic_data_series
from features.naming import feature_display_name
from features.section_utils import clear_section_record_cache, prime_section_record_cache
from settings import BASE_DIR


SECTION_ORDER = [
    "prices_div_adj",
    "key_metrics",
    "ratios",
    "income_statement",
    "income_statement_growth",
    "cash_flow",
    "cash_flow_growth",
    "balance_sheet",
    "balance_sheet_growth",
    "financial_growth",
    "earnings",
    "analyst_estimates",
    "ratings_historical",
    "grades_historical",
    "insider_trading",
    "economic_indicators",
    "treasury_rates",
    "representation_embedding",
]

SECTION_LABELS = {
    "prices_div_adj": "Prices Div Adj",
    "key_metrics": "Key Metrics",
    "ratios": "Ratios",
    "income_statement": "Income Statement",
    "income_statement_growth": "Income Statement Growth",
    "cash_flow": "Cash Flow",
    "cash_flow_growth": "Cash Flow Growth",
    "balance_sheet": "Balance Sheet",
    "balance_sheet_growth": "Balance Sheet Growth",
    "financial_growth": "Financial Growth",
    "earnings": "Earnings",
    "analyst_estimates": "Analyst Estimates",
    "ratings_historical": "Ratings Historical",
    "grades_historical": "Grades Historical",
    "insider_trading": "Insider Trading",
    "economic_indicators": "Economic Indicators",
    "treasury_rates": "Treasury Rates",
    "representation_embedding": "Representation Embedding",
}

REPRESENTATION_EMBEDDING_MODEL_VERSION = "semantic_grouped_v2"
REPRESENTATION_EMBEDDING_FAMILY_GROUPS: dict[str, tuple[str, ...]] = {
    "price_technical": ("prices_div_adj",),
    "valuation_quality": ("key_metrics", "ratios"),
    "income_statement": ("income_statement", "income_statement_growth"),
    "cash_flow": ("cash_flow", "cash_flow_growth"),
    "balance_sheet": ("balance_sheet", "balance_sheet_growth"),
    "broad_fundamental_growth": ("financial_growth",),
    "earnings_analyst_sentiment": ("earnings", "analyst_estimates", "ratings_historical", "grades_historical"),
    "insider_ownership": ("insider_trading",),
    "macro_rates": ("economic_indicators", "treasury_rates"),
}


def _default_feature_config() -> dict[str, Any]:
    return {
        "include_price_technicals": True,
        "include_fundamental_change": True,
        "include_statement_quality": True,
        "include_event_features": True,
        "include_ownership_features": True,
        "include_economic_indicators": True,
        "include_treasury_rates": True,
        "include_representation_embedding": False,
    }


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def feature_toggle_data(source: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = dict(source or {})

    defaults = _default_feature_config()
    return {
        "include_price_technicals": _as_bool(raw.get("include_price_technicals"), bool(defaults["include_price_technicals"])),
        "include_fundamental_change": _as_bool(raw.get("include_fundamental_change"), bool(defaults["include_fundamental_change"])),
        "include_statement_quality": _as_bool(raw.get("include_statement_quality"), bool(defaults["include_statement_quality"])),
        "include_event_features": _as_bool(raw.get("include_event_features"), bool(defaults["include_event_features"])),
        "include_ownership_features": _as_bool(raw.get("include_ownership_features"), bool(defaults["include_ownership_features"])),
        "include_economic_indicators": _as_bool(raw.get("include_economic_indicators"), bool(defaults["include_economic_indicators"])),
        "include_treasury_rates": _as_bool(raw.get("include_treasury_rates"), bool(defaults["include_treasury_rates"])),
        "include_representation_embedding": _as_bool(raw.get("include_representation_embedding"), bool(defaults["include_representation_embedding"])),
    }


def representation_embedding_config(source: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = dict(source or {})
    default_store_dir = Path(BASE_DIR) / "data" / "embedding_store"
    column_prefix = str(raw.get("representation_embedding_column_prefix") or "embedding_").strip() or "embedding_"
    return {
        "enabled": _as_bool(raw.get("include_representation_embedding"), False),
        "model_name": str(raw.get("representation_embedding_model_name") or "sentence-transformers/all-MiniLM-L6-v2").strip() or "sentence-transformers/all-MiniLM-L6-v2",
        "model_version": str(raw.get("representation_embedding_model_version") or REPRESENTATION_EMBEDDING_MODEL_VERSION).strip() or REPRESENTATION_EMBEDDING_MODEL_VERSION,
        "store_dir": str(raw.get("representation_embedding_store_dir") or default_store_dir),
        "column_prefix": column_prefix,
        "local_files_only": _as_bool(raw.get("representation_embedding_local_files_only"), False),
        "device": str(raw.get("representation_embedding_device") or "").strip() or None,
    }


def _stage(performance_tracer, name: str, *, category: str, workload_type: str, metadata: dict[str, Any] | None = None):
    if performance_tracer is None:
        return nullcontext()
    return performance_tracer.stage(
        name,
        category=category,
        workload_type=workload_type,
        metadata=metadata,
    )


def _resolve_feature_date_window(config: dict[str, Any] | None) -> tuple[str | None, str | None]:
    raw = dict(config or {})
    start_date = str(raw.get("feature_start_date") or raw.get("start_date") or "").strip() or None
    end_date = str(raw.get("feature_end_date") or raw.get("end_date") or "").strip() or None
    return start_date, end_date


def _needed_sparse_sections(feature_flags: dict[str, Any]) -> list[str]:
    sections: list[str] = []
    if feature_flags.get("include_fundamental_change"):
        sections.extend(["key_metrics", "ratios"])
    if feature_flags.get("include_statement_quality"):
        sections.extend(
            [
                "income_statement",
                "income_statement_growth",
                "cash_flow",
                "cash_flow_growth",
                "balance_sheet",
                "balance_sheet_growth",
                "financial_growth",
            ]
        )
    if feature_flags.get("include_event_features"):
        sections.extend(["earnings", "analyst_estimates", "ratings_historical", "grades_historical"])
    if feature_flags.get("include_ownership_features"):
        sections.append("insider_trading")
    return list(dict.fromkeys(sections))


def _build_feature_family_coverage_row(section_label: str, df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    if df.empty or not feature_cols:
        return {"section_label": section_label, "min_date": None, "max_date": None, "count": 0}
    usable_cols = [col for col in feature_cols if col in df.columns]
    if not usable_cols:
        return {"section_label": section_label, "min_date": None, "max_date": None, "count": 0}
    mask = df[usable_cols].notna().any(axis=1)
    if not mask.any():
        return {"section_label": section_label, "min_date": None, "max_date": None, "count": 0}
    valid_index = df.index[mask]
    if isinstance(valid_index, pd.MultiIndex):
        dates = pd.to_datetime(valid_index.get_level_values("date"))
    else:
        dates = pd.to_datetime(valid_index)
    return {
        "section_label": section_label,
        "min_date": dates.min().date().isoformat() if len(dates) else None,
        "max_date": dates.max().date().isoformat() if len(dates) else None,
        "count": int(mask.sum()),
    }


def _append_representation_embedding_columns(
    symbol_df: pd.DataFrame,
    grouped_feature_columns: dict[str, list[str]],
    *,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    if symbol_df.empty or not bool(config.get("enabled")):
        return symbol_df, [], {
            "enabled": False,
            "columns": [],
            "dimension": 0,
            "model_name": str(config.get("model_name") or ""),
            "model_version": str(config.get("model_version") or ""),
            "store_dir": str(config.get("store_dir") or ""),
        }

    build_dataset_embeddings, encoder = _resolve_representation_embedding_backend(config)
    dataset_rows = _representation_embedding_dataset_rows(symbol_df, grouped_feature_columns)
    if not dataset_rows:
        return symbol_df, [], {
            "enabled": False,
            "columns": [],
            "dimension": 0,
            "model_name": str(getattr(encoder, "model_name", config.get("model_name") or "")),
            "model_version": str(getattr(encoder, "model_version", config.get("model_version") or "")),
            "store_dir": str(config.get("store_dir") or ""),
        }

    embedding_rows = build_dataset_embeddings(
        dataset_rows,
        encoder=encoder,
        store_dir=str(config.get("store_dir") or ""),
    )
    first_vector_value = embedding_rows[0].get("embedding_vector")
    first_vector = list(first_vector_value) if first_vector_value is not None else []
    embedding_columns = [f"{str(config.get('column_prefix') or 'embedding_')}{idx}" for idx in range(len(first_vector))]
    embedding_df = pd.DataFrame(
        [
            {column: float(vector[idx]) for idx, column in enumerate(embedding_columns)}
            for vector in [
                list(item.get("embedding_vector")) if item.get("embedding_vector") is not None else []
                for item in embedding_rows
            ]
        ]
    )
    augmented = pd.concat([symbol_df.reset_index(drop=True), embedding_df], axis=1)
    return augmented, embedding_columns, {
        "enabled": bool(embedding_columns),
        "columns": list(embedding_columns),
        "dimension": len(embedding_columns),
        "model_name": str(getattr(encoder, "model_name", config.get("model_name") or "")),
        "model_version": str(getattr(encoder, "model_version", config.get("model_version") or "")),
        "store_dir": str(config.get("store_dir") or ""),
        "family_groups": {key: list(value) for key, value in REPRESENTATION_EMBEDDING_FAMILY_GROUPS.items()},
    }


def _representation_embedding_dataset_rows(
    symbol_df: pd.DataFrame,
    grouped_feature_columns: dict[str, list[str]],
) -> list[dict[str, Any]]:
    dataset_rows: list[dict[str, Any]] = []
    embedding_family_columns = _representation_embedding_grouped_feature_columns(grouped_feature_columns)
    usable_families = [
        str(family_name)
        for family_name, columns in embedding_family_columns.items()
        if list(columns or [])
    ]
    for row in symbol_df.to_dict(orient="records"):
        families: dict[str, dict[str, Any]] = {}
        for family_name in usable_families:
            values: dict[str, Any] = {}
            for column in list(embedding_family_columns.get(family_name) or []):
                if column not in row:
                    continue
                value = row.get(column)
                if _representation_embedding_missing_value(value):
                    continue
                display_name = _representation_embedding_feature_name(column, values)
                values[display_name] = value
            if values:
                families[family_name] = values
        if not families:
            raise ValueError(
                f"Representation embedding requested but no family features were available for "
                f"{row.get('symbol')} on {row.get('date')}."
            )
        dataset_rows.append(
            {
                "symbol": str(row.get("symbol") or ""),
                "date": str(row.get("date") or ""),
                "families": families,
            }
        )
    return dataset_rows


def _representation_embedding_grouped_feature_columns(
    grouped_feature_columns: dict[str, list[str]],
) -> dict[str, list[str]]:
    semantic_groups: dict[str, list[str]] = {}
    for family_name, source_families in REPRESENTATION_EMBEDDING_FAMILY_GROUPS.items():
        merged_columns: list[str] = []
        for source_family in source_families:
            merged_columns.extend(str(column) for column in list(grouped_feature_columns.get(source_family) or []))
        semantic_groups[family_name] = list(dict.fromkeys(merged_columns))
    return semantic_groups


def _representation_embedding_feature_name(column: str, existing_values: dict[str, Any]) -> str:
    display_name = str(feature_display_name(str(column)) or str(column)).strip() or str(column)
    if display_name not in existing_values:
        return display_name
    return f"{display_name} [{column}]"


def _representation_embedding_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return str(value).strip().lower() in {"", "nan", "none", "null", "<na>", "n/a", "na"}
    if isinstance(value, (list, tuple, set)):
        return not any(not _representation_embedding_missing_value(item) for item in value)
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _resolve_representation_embedding_backend(config: dict[str, Any]):
    from analysis.feature_embeddings.encoder import SentenceTransformerEncoder
    from analysis.feature_embeddings.pipeline import build_dataset_embeddings

    encoder = SentenceTransformerEncoder(
        model_name=str(config.get("model_name") or "sentence-transformers/all-MiniLM-L6-v2"),
        model_version=str(config.get("model_version") or "default"),
        local_files_only=bool(config.get("local_files_only")),
        device=config.get("device"),
    )
    return build_dataset_embeddings, encoder


def build_feature_panel_frame_for_symbols(
    *,
    symbols: list[str],
    config: dict[str, Any] | None = None,
    progress_callback=None,
    performance_tracer=None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    data = feature_toggle_data(config)
    embedding_config = representation_embedding_config(config)
    fieldnames: list[str] = ["date", "symbol"]
    global_grouped_feature_columns: dict[str, list[str]] = {key: [] for key in SECTION_ORDER}
    coverage_maps: dict[str, list[dict[str, Any]]] = {key: [] for key in SECTION_ORDER}
    symbols_processed = 0
    output_frames: list[pd.DataFrame] = []
    representation_meta: dict[str, Any] = {
        "enabled": False,
        "columns": [],
        "dimension": 0,
        "model_name": str(embedding_config.get("model_name") or ""),
        "model_version": str(embedding_config.get("model_version") or ""),
        "store_dir": str(embedding_config.get("store_dir") or ""),
        "family_groups": {key: list(value) for key, value in REPRESENTATION_EMBEDDING_FAMILY_GROUPS.items()},
    }

    input_symbols = list(symbols or [])
    total_symbols = len(input_symbols)
    if callable(progress_callback):
        progress_callback(completed=0, total=total_symbols, current_symbol="", force=True)

    start_date, end_date = _resolve_feature_date_window(config)
    normalized_symbols = [str(symbol or "").strip().upper() for symbol in input_symbols if str(symbol or "").strip()]

    with _stage(
        performance_tracer,
        "features.load_symbol_metadata",
        category="data_loading",
        workload_type="batched",
        metadata={"symbols_requested": len(normalized_symbols)},
    ):
        symbol_map = {
            str(symbol.symbol).strip().upper(): symbol
            for symbol in Symbol.objects.filter(symbol__in=normalized_symbols).only("id", "symbol")
        }

    with _stage(
        performance_tracer,
        "features.load_adjusted_prices",
        category="data_loading",
        workload_type="batched",
        metadata={"symbols_requested": len(normalized_symbols)},
    ):
        price_frames = load_adjusted_price_frames(normalized_symbols, start_date=start_date, end_date=end_date)

    needed_sections = _needed_sparse_sections(data)
    if needed_sections:
        with _stage(
            performance_tracer,
            "features.load_sparse_sections",
            category="data_loading",
            workload_type="batched",
            metadata={"section_count": len(needed_sections)},
        ):
            prime_section_record_cache(list(symbol_map.values()), needed_sections)

    non_empty_price_frames = [df for df in price_frames.values() if not df.empty]
    economic_df = pd.DataFrame()
    treasury_df = pd.DataFrame()
    if non_empty_price_frames:
        effective_start = min(df.index.min() for df in non_empty_price_frames).date().isoformat()
        effective_end = max(df.index.max() for df in non_empty_price_frames).date().isoformat()
        if data.get("include_economic_indicators"):
            with _stage(
                performance_tracer,
                "features.load_economic_series",
                category="data_loading",
                workload_type="batched",
                metadata={"start_date": effective_start, "end_date": effective_end},
            ):
                economic_series_codes = tuple(
                    str(code) for code in EconomicIndicatorSeries.objects.order_by("code").values_list("code", flat=True)
                )
                economic_df = fetch_economic_data_series(
                    api_key="",
                    start_date=effective_start,
                    end_date=effective_end,
                    config=EconomicDataConfig(economic_indicator_series=economic_series_codes, include_treasury_rates=False),
                )
        if data.get("include_treasury_rates"):
            with _stage(
                performance_tracer,
                "features.load_treasury_series",
                category="data_loading",
                workload_type="batched",
                metadata={"start_date": effective_start, "end_date": effective_end},
            ):
                treasury_series_codes = tuple(
                    str(code) for code in TreasuryRateSeries.objects.order_by("code").values_list("code", flat=True)
                )
                treasury_df = fetch_economic_data_series(
                    api_key="",
                    start_date=effective_start,
                    end_date=effective_end,
                    config=EconomicDataConfig(economic_indicator_series=treasury_series_codes, include_treasury_rates=False),
                )

    try:
        for idx, raw_symbol in enumerate(input_symbols, start=1):
            symbol = str(raw_symbol or "").strip().upper()
            if callable(progress_callback):
                progress_callback(completed=max(0, idx - 1), total=total_symbols, current_symbol=symbol)
            if not symbol:
                continue
            symbol_obj = symbol_map.get(symbol)
            df_prices = price_frames.get(symbol, pd.DataFrame())
            if symbol_obj is None or df_prices.empty:
                continue

            with _stage(
                performance_tracer,
                "features.compute_symbol_panel",
                category="feature_generation",
                workload_type="per_symbol",
                metadata={"symbol": symbol, "rows": int(len(df_prices))},
            ):
                target_index = pd.MultiIndex.from_arrays(
                    [df_prices.index, [symbol] * len(df_prices)],
                    names=["date", "symbol"],
                )
                merged = pd.DataFrame(index=target_index)
                grouped_feature_columns: dict[str, list[str]] = {key: [] for key in SECTION_ORDER}
                feature_columns: list[str] = []

                if data.get("include_price_technicals"):
                    with _stage(
                        performance_tracer,
                        "features.price_rolling",
                        category="rolling_calculations",
                        workload_type="vectorized",
                        metadata={"symbol": symbol},
                    ):
                        built = build_price_technical_features(symbol, df_prices)
                        if not built.df.empty:
                            merged = merged.join(built.df[built.feature_cols], how="left")
                            feature_columns.extend(built.feature_cols)
                            close_series = pd.to_numeric(df_prices["close"], errors="coerce")
                            legacy_price_df = pd.DataFrame(index=df_prices.index)
                            legacy_price_df["close"] = close_series
                            legacy_price_df["ret_1"] = close_series.pct_change()
                            legacy_price_df["sma_5"] = close_series.rolling(5, min_periods=1).mean()
                            legacy_price_df["sma_5_ratio"] = legacy_price_df["close"] / legacy_price_df["sma_5"]
                            legacy_price_df["vol_5"] = legacy_price_df["ret_1"].rolling(5, min_periods=2).std()
                            legacy_price_df.index.name = "date"
                            merged = merged.join(legacy_price_df[["close", "ret_1", "sma_5", "sma_5_ratio", "vol_5"]], how="left")
                            feature_columns.extend(["close", "ret_1", "sma_5", "sma_5_ratio", "vol_5"])
                            grouped_feature_columns["prices_div_adj"] = [
                                "close",
                                "ret_1",
                                "sma_5",
                                "sma_5_ratio",
                                "vol_5",
                            ] + list(built.feature_cols)

                if data.get("include_fundamental_change"):
                    built = build_fundamental_change_features(symbol_obj, target_index, df_prices=df_prices)
                    if not built.df.empty:
                        merged = merged.join(built.df[built.feature_cols], how="left")
                        feature_columns.extend(built.feature_cols)
                        grouped_feature_columns["key_metrics"] = [c for c in built.feature_cols if c.startswith("km__")]
                        grouped_feature_columns["ratios"] = [c for c in built.feature_cols if c.startswith("rt__")]

                if data.get("include_statement_quality"):
                    built = build_statement_quality_features(symbol_obj, target_index)
                    if not built.df.empty:
                        merged = merged.join(built.df[built.feature_cols], how="left")
                        feature_columns.extend(built.feature_cols)
                        grouped_feature_columns["income_statement"] = [c for c in built.feature_cols if c.startswith("is__")]
                        grouped_feature_columns["income_statement_growth"] = [c for c in built.feature_cols if c.startswith("isg__")]
                        grouped_feature_columns["cash_flow"] = [c for c in built.feature_cols if c.startswith("cf__")]
                        grouped_feature_columns["cash_flow_growth"] = [c for c in built.feature_cols if c.startswith("cfg__")]
                        grouped_feature_columns["balance_sheet"] = [c for c in built.feature_cols if c.startswith("bs__")]
                        grouped_feature_columns["balance_sheet_growth"] = [c for c in built.feature_cols if c.startswith("bsg__")]
                        grouped_feature_columns["financial_growth"] = [c for c in built.feature_cols if c.startswith("fg__")]

                if data.get("include_event_features"):
                    built = build_event_features(symbol_obj, target_index)
                    if not built.df.empty:
                        merged = merged.join(built.df[built.feature_cols], how="left")
                        feature_columns.extend(built.feature_cols)
                        grouped_feature_columns["earnings"] = [c for c in built.feature_cols if c.startswith("evt__earn_")]
                        grouped_feature_columns["analyst_estimates"] = [c for c in built.feature_cols if c.startswith("evt__ae_")]
                        grouped_feature_columns["ratings_historical"] = [c for c in built.feature_cols if c.startswith("evt__rating_")]
                        grouped_feature_columns["grades_historical"] = [c for c in built.feature_cols if c.startswith("evt__grade_")]

                if data.get("include_ownership_features"):
                    built = build_ownership_features(symbol_obj, target_index)
                    if not built.df.empty:
                        merged = merged.join(built.df[built.feature_cols], how="left")
                        feature_columns.extend(built.feature_cols)
                        grouped_feature_columns["insider_trading"] = [c for c in built.feature_cols if c.startswith("own__insider_")]

                if data.get("include_economic_indicators") and not economic_df.empty:
                    economic_daily = broadcast_series_to_daily(economic_df, target_index)
                    economic_cols = list(economic_daily.columns)
                    merged = merged.join(economic_daily[economic_cols], how="left")
                    feature_columns.extend(economic_cols)
                    grouped_feature_columns["economic_indicators"] = economic_cols

                if data.get("include_treasury_rates") and not treasury_df.empty:
                    treasury_daily = broadcast_series_to_daily(treasury_df, target_index)
                    treasury_cols = list(treasury_daily.columns)
                    merged = merged.join(treasury_daily[treasury_cols], how="left")
                    feature_columns.extend(treasury_cols)
                    grouped_feature_columns["treasury_rates"] = treasury_cols

                feature_columns = list(dict.fromkeys(feature_columns))
                symbol_df = merged.reset_index()
                symbol_df["date"] = pd.to_datetime(symbol_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                symbol_df = symbol_df.dropna(subset=["date"]).sort_values(["date", "symbol"]).reset_index(drop=True)
                if embedding_config.get("enabled"):
                    symbol_df, embedding_columns, embedding_meta = _append_representation_embedding_columns(
                        symbol_df,
                        grouped_feature_columns,
                        config=embedding_config,
                    )
                    if embedding_columns:
                        feature_columns.extend(embedding_columns)
                        grouped_feature_columns["representation_embedding"] = list(embedding_columns)
                        if representation_meta["columns"] and representation_meta["columns"] != list(embedding_columns):
                            raise ValueError("Representation embedding columns changed across symbols.")
                        representation_meta = dict(embedding_meta)

                output_frames.append(symbol_df)
                symbols_processed += 1

                for col in list(symbol_df.columns):
                    if col not in fieldnames:
                        fieldnames.append(col)
                coverage_symbol_df = symbol_df.copy()
                coverage_symbol_df["date"] = pd.to_datetime(coverage_symbol_df["date"], errors="coerce")
                coverage_symbol_df = coverage_symbol_df.dropna(subset=["date"]).set_index(["date", "symbol"]).sort_index()
                for key in SECTION_ORDER:
                    cols = [c for c in grouped_feature_columns[key] if c in symbol_df.columns]
                    for col in cols:
                        if col not in global_grouped_feature_columns[key]:
                            global_grouped_feature_columns[key].append(col)
                    coverage_source = coverage_symbol_df if key == "representation_embedding" else merged
                    coverage_maps[key].append(_build_feature_family_coverage_row(SECTION_LABELS[key], coverage_source, cols))

            if callable(progress_callback):
                progress_callback(completed=idx, total=total_symbols, current_symbol=symbol)
    finally:
        clear_section_record_cache()

    with _stage(
        performance_tracer,
        "features.concat_symbol_frames",
        category="joins_merges",
        workload_type="vectorized",
        metadata={"symbol_frames": len(output_frames)},
    ):
        if output_frames:
            output_frame = pd.concat(output_frames, ignore_index=True, sort=False)
            output_frame = output_frame.reindex(columns=fieldnames)
        else:
            output_frame = pd.DataFrame(columns=fieldnames)

    aggregated_coverage: list[dict[str, Any]] = []
    for key in SECTION_ORDER:
        counts = [int(row.get("count") or 0) for row in coverage_maps[key] if row]
        min_dates = [str(row.get("min_date") or "") for row in coverage_maps[key] if row.get("min_date")]
        max_dates = [str(row.get("max_date") or "") for row in coverage_maps[key] if row.get("max_date")]
        aggregated_coverage.append(
            {
                "section_key": key,
                "section_label": SECTION_LABELS[key],
                "feature_count": len(global_grouped_feature_columns[key]),
                "count": int(sum(counts)),
                "min_date": min(min_dates) if min_dates else None,
                "max_date": max(max_dates) if max_dates else None,
            }
        )

    metadata = {
        "feature_family_columns": global_grouped_feature_columns,
        "coverage_rows": aggregated_coverage,
        "feature_column_count": int(max(0, len(fieldnames) - 2)),
        "symbols_processed": int(symbols_processed),
        "config": dict(data),
        "representation_embedding_enabled": bool(representation_meta.get("enabled")),
        "representation_embedding_columns": list(representation_meta.get("columns") or []),
        "representation_embedding_dimension": int(representation_meta.get("dimension") or 0),
        "representation_embedding_model_name": str(representation_meta.get("model_name") or ""),
        "representation_embedding_model_version": str(representation_meta.get("model_version") or ""),
        "representation_embedding_store_dir": str(representation_meta.get("store_dir") or ""),
        "representation_embedding_family_groups": dict(representation_meta.get("family_groups") or {}),
        "feature_start_date": str(start_date or ""),
        "feature_end_date": str(end_date or ""),
    }
    if callable(progress_callback):
        progress_callback(completed=total_symbols, total=total_symbols, current_symbol="", force=True)
    return output_frame, fieldnames, metadata


def build_feature_panel_for_symbols(
    *,
    symbols: list[str],
    config: dict[str, Any] | None = None,
    progress_callback=None,
    performance_tracer=None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    frame, fieldnames, metadata = build_feature_panel_frame_for_symbols(
        symbols=symbols,
        config=config,
        progress_callback=progress_callback,
        performance_tracer=performance_tracer,
    )
    return frame.to_dict(orient="records"), fieldnames, metadata
