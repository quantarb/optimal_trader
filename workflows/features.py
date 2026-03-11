from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd

from data.historical_prices import load_adjusted_price_frames
from domain.features import (
    REPRESENTATION_EMBEDDING_FAMILY_GROUPS,
    REPRESENTATION_EMBEDDING_MODEL_VERSION,
    SECTION_LABELS,
    SECTION_ORDER,
    FeatureBuildSpec,
    append_representation_embedding_columns,
    build_feature_family_coverage_row,
    needed_sparse_sections,
)
from fmp.models import EconomicIndicatorSeries, Symbol, TreasuryRateSeries
from features.feature_builders import (
    build_event_features,
    build_fundamental_change_features,
    build_ownership_features,
    build_price_technical_features,
    build_statement_quality_features,
)
from features.macro import EconomicDataConfig, broadcast_series_to_daily, fetch_economic_data_series
from features.section_utils import clear_section_record_cache, prime_section_record_cache
from settings import BASE_DIR


def _stage(performance_tracer, name: str, *, category: str, workload_type: str, metadata: dict[str, Any] | None = None):
    if performance_tracer is None:
        return nullcontext()
    return performance_tracer.stage(name, category=category, workload_type=workload_type, metadata=metadata)


def build_feature_panel_frame_for_symbols(
    *,
    symbols: list[str],
    spec: FeatureBuildSpec | None = None,
    config: dict[str, Any] | None = None,
    progress_callback=None,
    performance_tracer=None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Build a multi-symbol feature panel using typed research specs."""

    build_spec = spec or FeatureBuildSpec.from_mapping(
        config,
        default_store_dir=str(Path(BASE_DIR) / "data" / "embedding_store"),
        default_model_version=REPRESENTATION_EMBEDDING_MODEL_VERSION,
    )
    fieldnames: list[str] = ["date", "symbol"]
    global_grouped_feature_columns: dict[str, list[str]] = {key: [] for key in SECTION_ORDER}
    coverage_maps: dict[str, list[dict[str, Any]]] = {key: [] for key in SECTION_ORDER}
    symbols_processed = 0
    output_frames: list[pd.DataFrame] = []
    representation_meta: dict[str, Any] = {
        "enabled": False,
        "columns": [],
        "dimension": 0,
        "model_name": str(build_spec.representation_embedding.model_name),
        "model_version": str(build_spec.representation_embedding.model_version),
        "store_dir": str(build_spec.representation_embedding.store_dir),
        "family_groups": {key: list(value) for key, value in REPRESENTATION_EMBEDDING_FAMILY_GROUPS.items()},
    }

    input_symbols = list(symbols or [])
    total_symbols = len(input_symbols)
    if callable(progress_callback):
        progress_callback(completed=0, total=total_symbols, current_symbol="", force=True)

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
        price_frames = load_adjusted_price_frames(
            normalized_symbols,
            start_date=build_spec.start_date,
            end_date=build_spec.end_date,
        )

    sparse_sections = needed_sparse_sections(build_spec.toggles)
    if sparse_sections:
        with _stage(
            performance_tracer,
            "features.load_sparse_sections",
            category="data_loading",
            workload_type="batched",
            metadata={"section_count": len(sparse_sections)},
        ):
            prime_section_record_cache(list(symbol_map.values()), sparse_sections)

    non_empty_price_frames = [df for df in price_frames.values() if not df.empty]
    economic_df = pd.DataFrame()
    treasury_df = pd.DataFrame()
    if non_empty_price_frames:
        effective_start = min(df.index.min() for df in non_empty_price_frames).date().isoformat()
        effective_end = max(df.index.max() for df in non_empty_price_frames).date().isoformat()
        if build_spec.toggles.include_economic_indicators:
            with _stage(
                performance_tracer,
                "features.load_economic_series",
                category="data_loading",
                workload_type="batched",
                metadata={"start_date": effective_start, "end_date": effective_end},
            ):
                economic_df = fetch_economic_data_series(
                    api_key="",
                    start_date=effective_start,
                    end_date=effective_end,
                    config=EconomicDataConfig(
                        economic_indicator_series=tuple(
                            str(code) for code in EconomicIndicatorSeries.objects.order_by("code").values_list("code", flat=True)
                        ),
                        include_treasury_rates=False,
                    ),
                )
        if build_spec.toggles.include_treasury_rates:
            with _stage(
                performance_tracer,
                "features.load_treasury_series",
                category="data_loading",
                workload_type="batched",
                metadata={"start_date": effective_start, "end_date": effective_end},
            ):
                treasury_df = fetch_economic_data_series(
                    api_key="",
                    start_date=effective_start,
                    end_date=effective_end,
                    config=EconomicDataConfig(
                        economic_indicator_series=tuple(
                            str(code) for code in TreasuryRateSeries.objects.order_by("code").values_list("code", flat=True)
                        ),
                        include_treasury_rates=False,
                    ),
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
                target_index = pd.MultiIndex.from_arrays([df_prices.index, [symbol] * len(df_prices)], names=["date", "symbol"])
                merged = pd.DataFrame(index=target_index)
                grouped_feature_columns: dict[str, list[str]] = {key: [] for key in SECTION_ORDER}
                feature_columns: list[str] = []

                if build_spec.toggles.include_price_technicals:
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

                if build_spec.toggles.include_fundamental_change:
                    built = build_fundamental_change_features(symbol_obj, target_index, df_prices=df_prices)
                    if not built.df.empty:
                        merged = merged.join(built.df[built.feature_cols], how="left")
                        feature_columns.extend(built.feature_cols)
                        grouped_feature_columns["key_metrics"] = [c for c in built.feature_cols if c.startswith("km__")]
                        grouped_feature_columns["ratios"] = [c for c in built.feature_cols if c.startswith(("rt__", "ratio__"))]

                if build_spec.toggles.include_statement_quality:
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

                if build_spec.toggles.include_event_features:
                    built = build_event_features(symbol_obj, target_index)
                    if not built.df.empty:
                        merged = merged.join(built.df[built.feature_cols], how="left")
                        feature_columns.extend(built.feature_cols)
                        grouped_feature_columns["earnings"] = [c for c in built.feature_cols if c.startswith("evt__earn_")]
                        grouped_feature_columns["analyst_estimates"] = [c for c in built.feature_cols if c.startswith("evt__ae_")]
                        grouped_feature_columns["ratings_historical"] = [c for c in built.feature_cols if c.startswith("evt__rating_")]
                        grouped_feature_columns["grades_historical"] = [c for c in built.feature_cols if c.startswith("evt__grade_")]

                if build_spec.toggles.include_ownership_features:
                    built = build_ownership_features(symbol_obj, target_index)
                    if not built.df.empty:
                        merged = merged.join(built.df[built.feature_cols], how="left")
                        feature_columns.extend(built.feature_cols)
                        grouped_feature_columns["insider_trading"] = [c for c in built.feature_cols if c.startswith("own__insider_")]

                if build_spec.toggles.include_economic_indicators and not economic_df.empty:
                    economic_daily = broadcast_series_to_daily(economic_df, target_index)
                    economic_cols = list(economic_daily.columns)
                    merged = merged.join(economic_daily[economic_cols], how="left")
                    feature_columns.extend(economic_cols)
                    grouped_feature_columns["economic_indicators"] = economic_cols

                if build_spec.toggles.include_treasury_rates and not treasury_df.empty:
                    treasury_daily = broadcast_series_to_daily(treasury_df, target_index)
                    treasury_cols = list(treasury_daily.columns)
                    merged = merged.join(treasury_daily[treasury_cols], how="left")
                    feature_columns.extend(treasury_cols)
                    grouped_feature_columns["treasury_rates"] = treasury_cols

                feature_columns = list(dict.fromkeys(feature_columns))
                symbol_df = merged.reset_index()
                symbol_df["date"] = pd.to_datetime(symbol_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                symbol_df = symbol_df.dropna(subset=["date"]).sort_values(["date", "symbol"]).reset_index(drop=True)
                if build_spec.representation_embedding.enabled:
                    symbol_df, embedding_columns, embedding_meta = append_representation_embedding_columns(
                        symbol_df,
                        grouped_feature_columns,
                        config=build_spec.representation_embedding,
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
                    coverage_maps[key].append(build_feature_family_coverage_row(SECTION_LABELS[key], coverage_source, cols))

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
        "config": build_spec.toggles.to_dict(),
        "representation_embedding_enabled": bool(representation_meta.get("enabled")),
        "representation_embedding_columns": list(representation_meta.get("columns") or []),
        "representation_embedding_dimension": int(representation_meta.get("dimension") or 0),
        "representation_embedding_model_name": str(representation_meta.get("model_name") or ""),
        "representation_embedding_model_version": str(representation_meta.get("model_version") or ""),
        "representation_embedding_store_dir": str(representation_meta.get("store_dir") or ""),
        "representation_embedding_family_groups": dict(representation_meta.get("family_groups") or {}),
        "feature_start_date": str(build_spec.start_date or ""),
        "feature_end_date": str(build_spec.end_date or ""),
    }
    if callable(progress_callback):
        progress_callback(completed=total_symbols, total=total_symbols, current_symbol="", force=True)
    return output_frame, fieldnames, metadata


def build_feature_panel_for_symbols(
    *,
    symbols: list[str],
    spec: FeatureBuildSpec | None = None,
    config: dict[str, Any] | None = None,
    progress_callback=None,
    performance_tracer=None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    frame, fieldnames, metadata = build_feature_panel_frame_for_symbols(
        symbols=symbols,
        spec=spec,
        config=config,
        progress_callback=progress_callback,
        performance_tracer=performance_tracer,
    )
    return frame.to_dict(orient="records"), fieldnames, metadata

