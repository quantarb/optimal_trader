from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import pandas as pd

from domain.features import (
    FeatureBuildSpec,
    REPRESENTATION_EMBEDDING_FAMILY_GROUPS,
    SECTION_LABELS,
    SECTION_ORDER,
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
    build_ta_classic_technical_features,
    build_time_calendar_feature_family,
)
from features.macro import EconomicDataConfig, broadcast_series_to_daily, fetch_economic_data_series
from features.section_utils import clear_section_record_cache, prime_section_record_cache


@dataclass(frozen=True)
class FeaturePanelDependencies:
    """Adapters required by the feature-panel workflow."""

    load_price_frames: Callable[..., dict[str, pd.DataFrame]]


@dataclass
class FeaturePanelAccumulator:
    """Mutable accumulation state for the multi-symbol feature panel."""

    fieldnames: list[str] = field(default_factory=lambda: ["date", "symbol"])
    global_grouped_feature_columns: dict[str, list[str]] = field(
        default_factory=lambda: {key: [] for key in SECTION_ORDER}
    )
    coverage_maps: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: {key: [] for key in SECTION_ORDER})
    output_frames: list[pd.DataFrame] = field(default_factory=list)
    symbols_processed: int = 0


@dataclass(frozen=True)
class FeaturePanelEnvironment:
    """Loaded inputs shared across per-symbol feature assembly."""

    input_symbols: tuple[str, ...]
    normalized_symbols: tuple[str, ...]
    build_spec: FeatureBuildSpec
    symbol_map: Mapping[str, Symbol]
    price_frames: Mapping[str, pd.DataFrame]
    economic_df: pd.DataFrame
    treasury_df: pd.DataFrame


@dataclass(frozen=True)
class SymbolFeatureResult:
    """Per-symbol feature assembly result."""

    symbol_frame: pd.DataFrame
    merged_frame: pd.DataFrame
    coverage_frame: pd.DataFrame
    grouped_feature_columns: dict[str, list[str]]
    representation_meta: dict[str, Any]


def _stage(
    performance_tracer,
    name: str,
    *,
    category: str,
    workload_type: str,
    metadata: dict[str, Any] | None = None,
):
    if performance_tracer is None:
        return nullcontext()
    return performance_tracer.stage(name, category=category, workload_type=workload_type, metadata=metadata)


def _initial_representation_meta(build_spec: FeatureBuildSpec) -> dict[str, Any]:
    return {
        "enabled": False,
        "columns": [],
        "dimension": 0,
        "model_name": str(build_spec.representation_embedding.model_name),
        "model_version": str(build_spec.representation_embedding.model_version),
        "store_dir": str(build_spec.representation_embedding.store_dir),
        "family_groups": {key: list(value) for key, value in REPRESENTATION_EMBEDDING_FAMILY_GROUPS.items()},
    }


def _normalize_symbols(symbols: list[str]) -> tuple[str, ...]:
    return tuple(str(symbol or "").strip().upper() for symbol in list(symbols or []) if str(symbol or "").strip())


def _load_symbol_map(normalized_symbols: tuple[str, ...], performance_tracer) -> dict[str, Symbol]:
    with _stage(
        performance_tracer,
        "features.load_symbol_metadata",
        category="data_loading",
        workload_type="batched",
        metadata={"symbols_requested": len(normalized_symbols)},
    ):
        return {
            str(symbol.symbol).strip().upper(): symbol
            for symbol in Symbol.objects.filter(symbol__in=normalized_symbols).only("id", "symbol", "payload")
        }


def _load_price_frames(
    *,
    normalized_symbols: tuple[str, ...],
    build_spec: FeatureBuildSpec,
    dependencies: FeaturePanelDependencies,
    performance_tracer,
) -> dict[str, pd.DataFrame]:
    with _stage(
        performance_tracer,
        "features.load_adjusted_prices",
        category="data_loading",
        workload_type="batched",
        metadata={"symbols_requested": len(normalized_symbols)},
    ):
        return dependencies.load_price_frames(
            normalized_symbols,
            start_date=build_spec.start_date,
            end_date=build_spec.end_date,
        )


def _prime_sparse_sections(
    *,
    build_spec: FeatureBuildSpec,
    symbol_map: Mapping[str, Symbol],
    performance_tracer,
) -> None:
    sparse_sections = needed_sparse_sections(build_spec.toggles)
    if not sparse_sections:
        return
    with _stage(
        performance_tracer,
        "features.load_sparse_sections",
        category="data_loading",
        workload_type="batched",
        metadata={"section_count": len(sparse_sections)},
    ):
        prime_section_record_cache(list(symbol_map.values()), sparse_sections)


def _effective_price_window(price_frames: Mapping[str, pd.DataFrame]) -> tuple[str, str] | None:
    non_empty_price_frames = [df for df in price_frames.values() if not df.empty]
    if not non_empty_price_frames:
        return None
    effective_start = min(df.index.min() for df in non_empty_price_frames).date().isoformat()
    effective_end = max(df.index.max() for df in non_empty_price_frames).date().isoformat()
    return effective_start, effective_end


def _load_macro_frame(
    *,
    codes: tuple[str, ...],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    return fetch_economic_data_series(
        api_key="",
        start_date=start_date,
        end_date=end_date,
        config=EconomicDataConfig(
            economic_indicator_series=codes,
            include_treasury_rates=False,
        ),
    )


def _load_macro_frames(
    *,
    build_spec: FeatureBuildSpec,
    price_frames: Mapping[str, pd.DataFrame],
    performance_tracer,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    window = _effective_price_window(price_frames)
    if window is None:
        return pd.DataFrame(), pd.DataFrame()
    effective_start, effective_end = window
    economic_df = pd.DataFrame()
    treasury_df = pd.DataFrame()
    if build_spec.toggles.include_economic_indicators:
        with _stage(
            performance_tracer,
            "features.load_economic_series",
            category="data_loading",
            workload_type="batched",
            metadata={"start_date": effective_start, "end_date": effective_end},
        ):
            economic_df = _load_macro_frame(
                codes=tuple(
                    str(code) for code in EconomicIndicatorSeries.objects.order_by("code").values_list("code", flat=True)
                ),
                start_date=effective_start,
                end_date=effective_end,
            )
    if build_spec.toggles.include_treasury_rates:
        with _stage(
            performance_tracer,
            "features.load_treasury_series",
            category="data_loading",
            workload_type="batched",
            metadata={"start_date": effective_start, "end_date": effective_end},
        ):
            treasury_df = _load_macro_frame(
                codes=tuple(
                    str(code) for code in TreasuryRateSeries.objects.order_by("code").values_list("code", flat=True)
                ),
                start_date=effective_start,
                end_date=effective_end,
            )
    return economic_df, treasury_df


def build_feature_panel_environment(
    *,
    symbols: list[str],
    build_spec: FeatureBuildSpec,
    dependencies: FeaturePanelDependencies,
    performance_tracer=None,
) -> FeaturePanelEnvironment:
    """Load shared data required for a multi-symbol feature panel."""

    input_symbols = tuple(list(symbols or []))
    normalized_symbols = _normalize_symbols(list(input_symbols))
    symbol_map = _load_symbol_map(normalized_symbols, performance_tracer)
    price_frames = _load_price_frames(
        normalized_symbols=normalized_symbols,
        build_spec=build_spec,
        dependencies=dependencies,
        performance_tracer=performance_tracer,
    )
    _prime_sparse_sections(build_spec=build_spec, symbol_map=symbol_map, performance_tracer=performance_tracer)
    economic_df, treasury_df = _load_macro_frames(
        build_spec=build_spec,
        price_frames=price_frames,
        performance_tracer=performance_tracer,
    )
    return FeaturePanelEnvironment(
        input_symbols=input_symbols,
        normalized_symbols=normalized_symbols,
        build_spec=build_spec,
        symbol_map=symbol_map,
        price_frames=price_frames,
        economic_df=economic_df,
        treasury_df=treasury_df,
    )


def _build_legacy_price_frame(df_prices: pd.DataFrame) -> pd.DataFrame:
    close_series = pd.to_numeric(df_prices["close"], errors="coerce")
    legacy_price_df = pd.DataFrame(index=df_prices.index)
    legacy_price_df["close"] = close_series
    legacy_price_df["ret_1"] = close_series.pct_change()
    legacy_price_df["sma_5"] = close_series.rolling(5, min_periods=1).mean()
    legacy_price_df["sma_5_ratio"] = legacy_price_df["close"] / legacy_price_df["sma_5"]
    legacy_price_df["vol_5"] = legacy_price_df["ret_1"].rolling(5, min_periods=2).std()
    legacy_price_df.index.name = "date"
    return legacy_price_df


def _add_price_features(
    *,
    symbol: str,
    df_prices: pd.DataFrame,
    merged: pd.DataFrame,
    feature_columns: list[str],
    grouped_feature_columns: dict[str, list[str]],
) -> pd.DataFrame:
    built = build_price_technical_features(symbol, df_prices)
    if built.df.empty:
        return merged
    merged = merged.join(built.df[built.feature_cols], how="left")
    feature_columns.extend(built.feature_cols)
    legacy_cols = ["close", "ret_1", "sma_5", "sma_5_ratio", "vol_5"]
    legacy_price_df = _build_legacy_price_frame(df_prices)
    merged = merged.join(legacy_price_df[legacy_cols], how="left")
    feature_columns.extend(legacy_cols)
    grouped_feature_columns["prices_div_adj"] = legacy_cols + list(built.feature_cols)
    return merged


def _add_ta_classic_technical_features(
    *,
    symbol: str,
    df_prices: pd.DataFrame,
    merged: pd.DataFrame,
    feature_columns: list[str],
    grouped_feature_columns: dict[str, list[str]],
) -> pd.DataFrame:
    built_by_family = build_ta_classic_technical_features(symbol, df_prices)
    for family_name, built in built_by_family.items():
        if built.df.empty:
            continue
        active_cols = [
            col
            for col in built.feature_cols
            if col in built.df.columns and pd.api.types.is_numeric_dtype(built.df[col])
        ]
        if not active_cols:
            continue
        merged = merged.join(built.df[active_cols], how="left")
        feature_columns.extend(active_cols)
        grouped_feature_columns[family_name] = list(active_cols)
    return merged


def _add_time_calendar_features(
    *,
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    merged: pd.DataFrame,
    feature_columns: list[str],
    grouped_feature_columns: dict[str, list[str]],
) -> pd.DataFrame:
    built = build_time_calendar_feature_family(symbol_obj, target_index)
    if built.df.empty:
        return merged
    merged = merged.join(built.df[built.feature_cols], how="left")
    feature_columns.extend(built.feature_cols)
    grouped_feature_columns["time_calendar"] = [col for col in built.feature_cols if col.startswith("time__")]
    return merged


def _add_fundamental_change_features(
    *,
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    df_prices: pd.DataFrame,
    merged: pd.DataFrame,
    feature_columns: list[str],
    grouped_feature_columns: dict[str, list[str]],
) -> pd.DataFrame:
    built = build_fundamental_change_features(symbol_obj, target_index, df_prices=df_prices)
    if built.df.empty:
        return merged
    merged = merged.join(built.df[built.feature_cols], how="left")
    feature_columns.extend(built.feature_cols)
    grouped_feature_columns["key_metrics"] = [col for col in built.feature_cols if col.startswith("km__")]
    grouped_feature_columns["ratios"] = [col for col in built.feature_cols if col.startswith(("rt__", "ratio__"))]
    return merged


def _add_statement_quality_features(
    *,
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    merged: pd.DataFrame,
    feature_columns: list[str],
    grouped_feature_columns: dict[str, list[str]],
) -> pd.DataFrame:
    built = build_statement_quality_features(symbol_obj, target_index)
    if built.df.empty:
        return merged
    merged = merged.join(built.df[built.feature_cols], how="left")
    feature_columns.extend(built.feature_cols)
    grouped_feature_columns["income_statement"] = [col for col in built.feature_cols if col.startswith("is__")]
    grouped_feature_columns["income_statement_growth"] = [col for col in built.feature_cols if col.startswith("isg__")]
    grouped_feature_columns["cash_flow"] = [col for col in built.feature_cols if col.startswith("cf__")]
    grouped_feature_columns["cash_flow_growth"] = [col for col in built.feature_cols if col.startswith("cfg__")]
    grouped_feature_columns["balance_sheet"] = [col for col in built.feature_cols if col.startswith("bs__")]
    grouped_feature_columns["balance_sheet_growth"] = [col for col in built.feature_cols if col.startswith("bsg__")]
    grouped_feature_columns["financial_growth"] = [col for col in built.feature_cols if col.startswith("fg__")]
    return merged


def _add_event_features(
    *,
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    merged: pd.DataFrame,
    feature_columns: list[str],
    grouped_feature_columns: dict[str, list[str]],
) -> pd.DataFrame:
    built = build_event_features(symbol_obj, target_index)
    if built.df.empty:
        return merged
    merged = merged.join(built.df[built.feature_cols], how="left")
    feature_columns.extend(built.feature_cols)
    grouped_feature_columns["earnings"] = [col for col in built.feature_cols if col.startswith("evt__earn_")]
    grouped_feature_columns["analyst_estimates"] = [col for col in built.feature_cols if col.startswith("evt__ae_")]
    grouped_feature_columns["ratings_historical"] = [col for col in built.feature_cols if col.startswith("evt__rating_")]
    grouped_feature_columns["grades_historical"] = [col for col in built.feature_cols if col.startswith("evt__grade_")]
    return merged


def _add_ownership_features(
    *,
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    merged: pd.DataFrame,
    feature_columns: list[str],
    grouped_feature_columns: dict[str, list[str]],
) -> pd.DataFrame:
    built = build_ownership_features(symbol_obj, target_index)
    if built.df.empty:
        return merged
    merged = merged.join(built.df[built.feature_cols], how="left")
    feature_columns.extend(built.feature_cols)
    grouped_feature_columns["insider_trading"] = [col for col in built.feature_cols if col.startswith("own__insider_")]
    return merged


def _add_macro_features(
    *,
    macro_df: pd.DataFrame,
    target_index: pd.MultiIndex,
    family_key: str,
    merged: pd.DataFrame,
    feature_columns: list[str],
    grouped_feature_columns: dict[str, list[str]],
) -> pd.DataFrame:
    if macro_df.empty:
        return merged
    daily_frame = broadcast_series_to_daily(macro_df, target_index)
    macro_cols = list(daily_frame.columns)
    merged = merged.join(daily_frame[macro_cols], how="left")
    feature_columns.extend(macro_cols)
    grouped_feature_columns[family_key] = macro_cols
    return merged


def _finalize_symbol_frame(merged: pd.DataFrame) -> pd.DataFrame:
    symbol_df = merged.reset_index()
    symbol_df["date"] = pd.to_datetime(symbol_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return symbol_df.dropna(subset=["date"]).sort_values(["date", "symbol"]).reset_index(drop=True)


def _apply_representation_embedding(
    *,
    symbol_df: pd.DataFrame,
    grouped_feature_columns: dict[str, list[str]],
    build_spec: FeatureBuildSpec,
    representation_meta: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not build_spec.representation_embedding.enabled:
        return symbol_df, representation_meta
    symbol_df, embedding_columns, embedding_meta = append_representation_embedding_columns(
        symbol_df,
        grouped_feature_columns,
        config=build_spec.representation_embedding,
    )
    if not embedding_columns:
        return symbol_df, representation_meta
    grouped_feature_columns["representation_embedding"] = list(embedding_columns)
    if representation_meta["columns"] and representation_meta["columns"] != list(embedding_columns):
        raise ValueError("Representation embedding columns changed across symbols.")
    return symbol_df, dict(embedding_meta)


def build_symbol_feature_result(
    *,
    symbol: str,
    symbol_obj: Symbol,
    df_prices: pd.DataFrame,
    build_spec: FeatureBuildSpec,
    economic_df: pd.DataFrame,
    treasury_df: pd.DataFrame,
    representation_meta: dict[str, Any],
) -> SymbolFeatureResult:
    """Build a single-symbol feature panel and associated coverage metadata."""

    target_index = pd.MultiIndex.from_arrays([df_prices.index, [symbol] * len(df_prices)], names=["date", "symbol"])
    merged = pd.DataFrame(index=target_index)
    grouped_feature_columns: dict[str, list[str]] = {key: [] for key in SECTION_ORDER}
    feature_columns: list[str] = []

    if build_spec.toggles.include_price_technicals:
        merged = _add_price_features(
            symbol=symbol,
            df_prices=df_prices,
            merged=merged,
            feature_columns=feature_columns,
            grouped_feature_columns=grouped_feature_columns,
        )
    if build_spec.toggles.include_ta_classic_technicals:
        merged = _add_ta_classic_technical_features(
            symbol=symbol,
            df_prices=df_prices,
            merged=merged,
            feature_columns=feature_columns,
            grouped_feature_columns=grouped_feature_columns,
        )
    if build_spec.toggles.include_time_calendar_features:
        merged = _add_time_calendar_features(
            symbol_obj=symbol_obj,
            target_index=target_index,
            merged=merged,
            feature_columns=feature_columns,
            grouped_feature_columns=grouped_feature_columns,
        )
    if build_spec.toggles.include_fundamental_change:
        merged = _add_fundamental_change_features(
            symbol_obj=symbol_obj,
            target_index=target_index,
            df_prices=df_prices,
            merged=merged,
            feature_columns=feature_columns,
            grouped_feature_columns=grouped_feature_columns,
        )
    if build_spec.toggles.include_statement_quality:
        merged = _add_statement_quality_features(
            symbol_obj=symbol_obj,
            target_index=target_index,
            merged=merged,
            feature_columns=feature_columns,
            grouped_feature_columns=grouped_feature_columns,
        )
    if build_spec.toggles.include_event_features:
        merged = _add_event_features(
            symbol_obj=symbol_obj,
            target_index=target_index,
            merged=merged,
            feature_columns=feature_columns,
            grouped_feature_columns=grouped_feature_columns,
        )
    if build_spec.toggles.include_ownership_features:
        merged = _add_ownership_features(
            symbol_obj=symbol_obj,
            target_index=target_index,
            merged=merged,
            feature_columns=feature_columns,
            grouped_feature_columns=grouped_feature_columns,
        )
    if build_spec.toggles.include_economic_indicators:
        merged = _add_macro_features(
            macro_df=economic_df,
            target_index=target_index,
            family_key="economic_indicators",
            merged=merged,
            feature_columns=feature_columns,
            grouped_feature_columns=grouped_feature_columns,
        )
    if build_spec.toggles.include_treasury_rates:
        merged = _add_macro_features(
            macro_df=treasury_df,
            target_index=target_index,
            family_key="treasury_rates",
            merged=merged,
            feature_columns=feature_columns,
            grouped_feature_columns=grouped_feature_columns,
        )

    feature_columns = list(dict.fromkeys(feature_columns))
    symbol_df = _finalize_symbol_frame(merged)
    symbol_df, next_representation_meta = _apply_representation_embedding(
        symbol_df=symbol_df,
        grouped_feature_columns=grouped_feature_columns,
        build_spec=build_spec,
        representation_meta=representation_meta,
    )
    coverage_frame = symbol_df.copy()
    coverage_frame["date"] = pd.to_datetime(coverage_frame["date"], errors="coerce")
    coverage_frame = coverage_frame.dropna(subset=["date"]).set_index(["date", "symbol"]).sort_index()
    return SymbolFeatureResult(
        symbol_frame=symbol_df,
        merged_frame=merged,
        coverage_frame=coverage_frame,
        grouped_feature_columns=grouped_feature_columns,
        representation_meta=next_representation_meta,
    )


def _append_symbol_result(accumulator: FeaturePanelAccumulator, result: SymbolFeatureResult) -> None:
    accumulator.output_frames.append(result.symbol_frame)
    accumulator.symbols_processed += 1
    for column in list(result.symbol_frame.columns):
        if column not in accumulator.fieldnames:
            accumulator.fieldnames.append(column)
    for key in SECTION_ORDER:
        cols = [col for col in result.grouped_feature_columns[key] if col in result.symbol_frame.columns]
        for column in cols:
            if column not in accumulator.global_grouped_feature_columns[key]:
                accumulator.global_grouped_feature_columns[key].append(column)
        coverage_source = result.coverage_frame if key == "representation_embedding" else result.merged_frame
        accumulator.coverage_maps[key].append(build_feature_family_coverage_row(SECTION_LABELS[key], coverage_source, cols))


def _concat_output_frames(accumulator: FeaturePanelAccumulator, performance_tracer) -> pd.DataFrame:
    with _stage(
        performance_tracer,
        "features.concat_symbol_frames",
        category="joins_merges",
        workload_type="vectorized",
        metadata={"symbol_frames": len(accumulator.output_frames)},
    ):
        if not accumulator.output_frames:
            return pd.DataFrame(columns=accumulator.fieldnames)
        output_frame = pd.concat(accumulator.output_frames, ignore_index=True, sort=False)
        return output_frame.reindex(columns=accumulator.fieldnames)


def _aggregate_coverage_rows(accumulator: FeaturePanelAccumulator) -> list[dict[str, Any]]:
    aggregated: list[dict[str, Any]] = []
    for key in SECTION_ORDER:
        counts = [int(row.get("count") or 0) for row in accumulator.coverage_maps[key] if row]
        min_dates = [str(row.get("min_date") or "") for row in accumulator.coverage_maps[key] if row.get("min_date")]
        max_dates = [str(row.get("max_date") or "") for row in accumulator.coverage_maps[key] if row.get("max_date")]
        aggregated.append(
            {
                "section_key": key,
                "section_label": SECTION_LABELS[key],
                "feature_count": len(accumulator.global_grouped_feature_columns[key]),
                "count": int(sum(counts)),
                "min_date": min(min_dates) if min_dates else None,
                "max_date": max(max_dates) if max_dates else None,
            }
        )
    return aggregated


def build_feature_panel_frame(
    *,
    environment: FeaturePanelEnvironment,
    progress_callback=None,
    performance_tracer=None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Build a feature panel from preloaded environment state."""

    accumulator = FeaturePanelAccumulator()
    representation_meta = _initial_representation_meta(environment.build_spec)
    total_symbols = len(environment.input_symbols)
    if callable(progress_callback):
        progress_callback(completed=0, total=total_symbols, current_symbol="", force=True)

    try:
        for idx, raw_symbol in enumerate(environment.input_symbols, start=1):
            symbol = str(raw_symbol or "").strip().upper()
            if callable(progress_callback):
                progress_callback(completed=max(0, idx - 1), total=total_symbols, current_symbol=symbol)
            if not symbol:
                continue
            symbol_obj = environment.symbol_map.get(symbol)
            df_prices = environment.price_frames.get(symbol, pd.DataFrame())
            if symbol_obj is None or df_prices.empty:
                continue
            with _stage(
                performance_tracer,
                "features.compute_symbol_panel",
                category="feature_generation",
                workload_type="per_symbol",
                metadata={"symbol": symbol, "rows": int(len(df_prices))},
            ):
                result = build_symbol_feature_result(
                    symbol=symbol,
                    symbol_obj=symbol_obj,
                    df_prices=df_prices,
                    build_spec=environment.build_spec,
                    economic_df=environment.economic_df,
                    treasury_df=environment.treasury_df,
                    representation_meta=representation_meta,
                )
            _append_symbol_result(accumulator, result)
            representation_meta = result.representation_meta
            if callable(progress_callback):
                progress_callback(completed=idx, total=total_symbols, current_symbol=symbol)
    finally:
        clear_section_record_cache()

    output_frame = _concat_output_frames(accumulator, performance_tracer)
    metadata = {
        "feature_family_columns": accumulator.global_grouped_feature_columns,
        "coverage_rows": _aggregate_coverage_rows(accumulator),
        "feature_column_count": int(max(0, len(accumulator.fieldnames) - 2)),
        "symbols_processed": int(accumulator.symbols_processed),
        "config": environment.build_spec.toggles.to_dict(),
        "representation_embedding_enabled": bool(representation_meta.get("enabled")),
        "representation_embedding_columns": list(representation_meta.get("columns") or []),
        "representation_embedding_dimension": int(representation_meta.get("dimension") or 0),
        "representation_embedding_model_name": str(representation_meta.get("model_name") or ""),
        "representation_embedding_model_version": str(representation_meta.get("model_version") or ""),
        "representation_embedding_store_dir": str(representation_meta.get("store_dir") or ""),
        "representation_embedding_family_groups": dict(representation_meta.get("family_groups") or {}),
        "feature_start_date": str(environment.build_spec.start_date or ""),
        "feature_end_date": str(environment.build_spec.end_date or ""),
    }
    if callable(progress_callback):
        progress_callback(completed=total_symbols, total=total_symbols, current_symbol="", force=True)
    return output_frame, accumulator.fieldnames, metadata
