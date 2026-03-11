from __future__ import annotations

import math
import uuid
from typing import Any

from django.utils import timezone

from domain.features import FeatureBuildSpec
from domain.labels import LabelBuildSpec
from fmp.models import Symbol
from settings import BASE_DIR
from workflows.features import build_feature_panel_frame_for_symbols
from workflows.labels import build_oracle_labels

from .progress import ProgressReporter
from .service_runtime import (
    BuiltOutput,
    as_bool,
    load_universe_symbols,
    normalize_symbol_list,
    stable_payload_hash,
    write_frame_artifact,
    write_payload_artifact,
)
from .universe_selection import parse_exchange_values, resolve_symbol_universe


def execute_universe(config: dict[str, Any], *, performance_tracer=None) -> BuiltOutput:
    stage_ctx = (
        performance_tracer.stage(
            "universe.resolve",
            category="data_loading",
            workload_type="batched",
            metadata={},
        )
        if performance_tracer is not None
        else None
    )
    if stage_ctx is None:
        symbols = normalize_symbol_list(list(config.get("symbols") or []))
        universe_filters: dict[str, Any] = {}
        if not symbols:
            limit_raw = config.get("limit")
            limit = int(limit_raw or 300)
            min_market_cap = config.get("min_market_cap")
            max_market_cap = config.get("max_market_cap")
            country = str(config.get("country") or "").strip()
            exchanges = parse_exchange_values(config.get("exchanges"))
            exclude_pooled_vehicles = as_bool(config.get("exclude_pooled_vehicles"), default=False)
            if min_market_cap not in (None, "") or max_market_cap not in (None, "") or country or exchanges:
                universe_filters = {
                    "min_market_cap": float(min_market_cap) if min_market_cap not in (None, "") else None,
                    "max_market_cap": float(max_market_cap) if max_market_cap not in (None, "") else None,
                    "country": country,
                    "exchanges": exchanges,
                    "exclude_pooled_vehicles": bool(exclude_pooled_vehicles),
                    "limit": max(1, limit),
                }
                symbols = normalize_symbol_list(
                    resolve_symbol_universe(
                        min_market_cap=universe_filters["min_market_cap"],
                        max_market_cap=universe_filters["max_market_cap"],
                        country=country or None,
                        exchanges=exchanges,
                        limit=max(1, limit),
                        exclude_pooled_vehicles=exclude_pooled_vehicles,
                    )
                )
            else:
                qs = Symbol.objects.order_by("symbol").values_list("symbol", flat=True)
                symbols = normalize_symbol_list(list(qs[: max(1, limit)]))
    else:
        with stage_ctx:
            symbols = normalize_symbol_list(list(config.get("symbols") or []))
            universe_filters = {}
            if not symbols:
                limit_raw = config.get("limit")
                limit = int(limit_raw or 300)
                min_market_cap = config.get("min_market_cap")
                max_market_cap = config.get("max_market_cap")
                country = str(config.get("country") or "").strip()
                exchanges = parse_exchange_values(config.get("exchanges"))
                exclude_pooled_vehicles = as_bool(config.get("exclude_pooled_vehicles"), default=False)
                if min_market_cap not in (None, "") or max_market_cap not in (None, "") or country or exchanges:
                    universe_filters = {
                        "min_market_cap": float(min_market_cap) if min_market_cap not in (None, "") else None,
                        "max_market_cap": float(max_market_cap) if max_market_cap not in (None, "") else None,
                        "country": country,
                        "exchanges": exchanges,
                        "exclude_pooled_vehicles": bool(exclude_pooled_vehicles),
                        "limit": max(1, limit),
                    }
                    symbols = normalize_symbol_list(
                        resolve_symbol_universe(
                            min_market_cap=universe_filters["min_market_cap"],
                            max_market_cap=universe_filters["max_market_cap"],
                            country=country or None,
                            exchanges=exchanges,
                            limit=max(1, limit),
                            exclude_pooled_vehicles=exclude_pooled_vehicles,
                        )
                    )
                else:
                    qs = Symbol.objects.order_by("symbol").values_list("symbol", flat=True)
                    symbols = normalize_symbol_list(list(qs[: max(1, limit)]))
    payload = {
        "symbols": symbols,
        "count": len(symbols),
        "created_at": timezone.now().isoformat(),
        "filters": universe_filters,
    }
    cache_key = stable_payload_hash({"symbols": symbols, "filters": universe_filters})
    key = f"universe_{uuid.uuid4().hex}"
    stored = write_payload_artifact(key, payload)
    return BuiltOutput(
        artifact_type="UNIVERSE",
        content={"count": len(symbols)},
        metadata={
            "symbols_preview": symbols[:20],
            "filters": universe_filters,
            "universe_cache_key": cache_key,
            **stored.storage_metadata(),
        },
        uri=stored.uri,
    )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def _std_sample(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    m = _mean(values)
    if m is None:
        return None
    var = sum((v - m) ** 2 for v in values) / float(len(values) - 1)
    return float(math.sqrt(var))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_values[mid])
    return float((sorted_values[mid - 1] + sorted_values[mid]) / 2.0)


def _build_label_statistics(label_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not label_rows:
        return {
            "trade_stats": {},
            "grouped_trade_stats": [],
            "symbol_grouped_trade_stats": [],
        }

    returns: list[float] = []
    long_returns: list[float] = []
    short_returns: list[float] = []
    symbol_bucket: dict[str, list[float]] = {}
    symbol_holds: dict[str, list[float]] = {}
    grouped_bucket: dict[tuple[str, str, int], list[float]] = {}
    grouped_holds: dict[tuple[str, str, int], list[float]] = {}

    for row in label_rows:
        try:
            ret = float(row.get("trade_return") or 0.0)
        except Exception:
            ret = 0.0
        side = str(row.get("side") or "").strip().lower()
        label = int(row.get("label") or 0)
        if side not in {"long", "short"}:
            side = "long" if label == 1 else "short"
        freq = str(row.get("freq") or "D1").strip() or "D1"
        try:
            k = int(row.get("k") or 1)
        except Exception:
            k = 1
        try:
            hold_days = float(row.get("hold_days") or 0.0)
        except Exception:
            hold_days = 0.0
        side_ret = ret
        returns.append(side_ret)
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol:
            symbol_bucket.setdefault(symbol, []).append(side_ret)
            symbol_holds.setdefault(symbol, []).append(hold_days)
        grouped_key = (side, freq, k)
        grouped_bucket.setdefault(grouped_key, []).append(side_ret)
        grouped_holds.setdefault(grouped_key, []).append(hold_days)
        if side == "long":
            long_returns.append(side_ret)
        else:
            short_returns.append(side_ret)

    wins = [v for v in returns if v > 0]
    losses = [v for v in returns if v < 0]
    breakeven = [v for v in returns if v == 0]

    trade_stats = {
        "total_trades": int(len(returns)),
        "symbols_count": int(len(symbol_bucket)),
        "long_trades": int(len(long_returns)),
        "short_trades": int(len(short_returns)),
        "winning_trades": int(len(wins)),
        "losing_trades": int(len(losses)),
        "breakeven_trades": int(len(breakeven)),
        "win_rate_pct": round((len(wins) / float(len(returns))) * 100.0, 4) if returns else 0.0,
        "loss_rate_pct": round((len(losses) / float(len(returns))) * 100.0, 4) if returns else 0.0,
        "avg_return_pct": round((_mean(returns) or 0.0) * 100.0, 4),
        "median_return_pct": round((_median(returns) or 0.0) * 100.0, 4),
    }

    grouped_trade_stats: list[dict[str, Any]] = []
    for (side, freq, k), vals in sorted(grouped_bucket.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        if not vals:
            continue
        mean_v = _mean(vals) or 0.0
        std_v = _std_sample(vals)
        hold_vals = grouped_holds.get((side, freq, k), [])
        hold_mean = _mean(hold_vals) or 0.0
        hold_std = _std_sample(hold_vals)
        sharpe = (mean_v / std_v) if std_v and abs(std_v) > 1e-12 else None
        grouped_trade_stats.append(
            {
                "side": side,
                "freq": freq,
                "k": int(k),
                "trades": int(len(vals)),
                "trade_return_mean_pct": round(mean_v * 100.0, 4),
                "trade_return_std_pct": round(std_v * 100.0, 4) if std_v is not None else None,
                "trade_duration_mean": round(float(hold_mean), 4),
                "trade_duration_std": round(float(hold_std), 4) if hold_std is not None else None,
                "sharpe": round(sharpe, 6) if sharpe is not None else None,
            }
        )

    symbol_grouped_trade_stats: list[dict[str, Any]] = []
    for symbol in sorted(symbol_bucket.keys()):
        vals = symbol_bucket[symbol]
        mean_v = _mean(vals) or 0.0
        std_v = _std_sample(vals)
        hold_vals = symbol_holds.get(symbol, [])
        hold_mean = _mean(hold_vals) or 0.0
        hold_std = _std_sample(hold_vals)
        sharpe = (mean_v / std_v) if std_v and abs(std_v) > 1e-12 else None
        symbol_grouped_trade_stats.append(
            {
                "symbol": symbol,
                "side": "mixed",
                "freq": "mixed",
                "k": 0,
                "trades": int(len(vals)),
                "trade_return_mean_pct": round(mean_v * 100.0, 4),
                "trade_return_std_pct": round(std_v * 100.0, 4) if std_v is not None else None,
                "trade_duration_mean": round(float(hold_mean), 4),
                "trade_duration_std": round(float(hold_std), 4) if hold_std is not None else None,
                "sharpe": round(sharpe, 6) if sharpe is not None else None,
            }
        )

    return {
        "trade_stats": trade_stats,
        "grouped_trade_stats": grouped_trade_stats,
        "symbol_grouped_trade_stats": symbol_grouped_trade_stats,
    }


def _min_profit_decimal_from_config(config: dict[str, Any]) -> float:
    raw = config.get("min_profit_pct")
    try:
        value = float(raw)
    except Exception:
        value = 1.0
    if value < 0:
        value = 0.0
    return value / 100.0


def execute_labels(
    config: dict[str, Any],
    universe_artifact,
    *,
    pipeline_run=None,
    job_run=None,
    performance_tracer=None,
) -> BuiltOutput:
    symbols = load_universe_symbols(universe_artifact)
    progress = ProgressReporter(pipeline_run=pipeline_run, job_run=job_run)
    spec = LabelBuildSpec.from_mapping(config)
    storage_format = str(config.get("artifact_storage_format") or "csv").strip().lower() or "csv"

    total_symbols = len(symbols)
    progress.update(
        phase="build_labels",
        phase_label="Generate oracle labels",
        phase_index=1,
        phase_total=1,
        unit_label="symbols",
        total_units=total_symbols,
        completed_units=0,
        current_item="",
        force=True,
    )
    stage_ctx = (
        performance_tracer.stage(
            "labels.generate",
            category="label_generation",
            workload_type="per_symbol",
            metadata={"symbols": total_symbols},
        )
        if performance_tracer is not None
        else None
    )
    if stage_ctx is None:
        built = build_oracle_labels(
            symbols,
            spec=spec,
            progress_callback=lambda *, completed, total, current_symbol="": progress.update(
                phase="build_labels",
                phase_label="Generate oracle labels",
                phase_index=1,
                phase_total=1,
                unit_label="symbols",
                total_units=total,
                completed_units=completed,
                current_item=current_symbol,
            ),
        )
    else:
        with stage_ctx:
            built = build_oracle_labels(
                symbols,
                spec=spec,
                progress_callback=lambda *, completed, total, current_symbol="": progress.update(
                    phase="build_labels",
                    phase_label="Generate oracle labels",
                    phase_index=1,
                    phase_total=1,
                    unit_label="symbols",
                    total_units=total,
                    completed_units=completed,
                    current_item=current_symbol,
                ),
            )
    progress.update(
        phase="finalize_labels",
        phase_label="Finalize labels output",
        phase_index=1,
        phase_total=1,
        unit_label="symbols",
        total_units=total_symbols,
        completed_units=total_symbols,
        current_item="",
        force=True,
    )
    label_rows = list(built.label_rows)
    key = f"labels_{uuid.uuid4().hex}"
    label_fieldnames = [
        "date",
        "symbol",
        "label",
        "market_position",
        "trade_return",
        "hold_days",
        "side",
        "freq",
        "k",
        "entry_date",
        "exit_date",
        "entry_px",
        "exit_px",
        "ret_pct",
    ]
    if performance_tracer is not None:
        with performance_tracer.stage(
            "labels.serialize_artifact",
            category="serialization",
            workload_type="batched",
            metadata={"rows": len(label_rows), "storage_format": storage_format},
        ):
            stored = write_frame_artifact(
                key,
                rows=label_rows,
                fieldnames=label_fieldnames,
                storage_format=storage_format,
            )
    else:
        stored = write_frame_artifact(
            key,
            rows=label_rows,
            fieldnames=label_fieldnames,
            storage_format=storage_format,
        )
    stats = dict(built.statistics or {})
    labels_cache_key = stable_payload_hash(
        {
            "source_universe_artifact_id": int(universe_artifact.id),
            "symbols": symbols,
            "k_params": spec.k_params,
            "min_profit_decimal": spec.min_profit_pct,
            "buy_col": spec.buy_execution,
            "sell_col": spec.sell_execution,
            "short_col": spec.short_execution,
            "cover_col": spec.cover_execution,
            "dedup_mode": spec.trade_dedup_mode,
        }
    )
    return BuiltOutput(
        artifact_type="LABELS",
        content={
            "rows": len(label_rows),
            "symbols": len(symbols),
            "min_profit_pct": round(spec.min_profit_pct * 100.0, 6),
            "statistics": stats,
        },
        metadata={
            "source_universe_artifact_id": universe_artifact.id,
            "min_profit_decimal": spec.min_profit_pct,
            "labels_cache_key": labels_cache_key,
            **stored.storage_metadata(),
        },
        uri=stored.uri,
    )


def execute_features(
    config: dict[str, Any],
    universe_artifact,
    *,
    pipeline_run=None,
    job_run=None,
    performance_tracer=None,
) -> BuiltOutput:
    symbols = load_universe_symbols(universe_artifact)
    progress = ProgressReporter(pipeline_run=pipeline_run, job_run=job_run)
    storage_format = str(config.get("artifact_storage_format") or "csv").strip().lower() or "csv"
    feature_spec = FeatureBuildSpec.from_mapping(config, default_store_dir=str(BASE_DIR / "data" / "embedding_store"))
    feature_frame, fieldnames, feature_meta = build_feature_panel_frame_for_symbols(
        symbols=symbols,
        spec=feature_spec,
        progress_callback=lambda *, completed, total, current_symbol="", force=False: progress.update(
            phase="build_features",
            phase_label="Build feature panel",
            phase_index=1,
            phase_total=1,
            unit_label="symbols",
            total_units=total,
            completed_units=completed,
            current_item=current_symbol,
            force=bool(force),
        ),
        performance_tracer=performance_tracer,
    )
    progress.complete(message="Feature panel completed")
    features_cache_key = stable_payload_hash(
        {
            "source_universe_artifact_id": int(universe_artifact.id),
            "symbols": symbols,
            "feature_config": dict(config or {}),
        }
    )
    key = f"features_{uuid.uuid4().hex}"
    if performance_tracer is not None:
        with performance_tracer.stage(
            "features.serialize_artifact",
            category="serialization",
            workload_type="batched",
            metadata={"rows": int(len(feature_frame)), "storage_format": storage_format},
        ):
            stored = write_frame_artifact(key, frame=feature_frame, fieldnames=fieldnames, storage_format=storage_format)
    else:
        stored = write_frame_artifact(key, frame=feature_frame, fieldnames=fieldnames, storage_format=storage_format)
    return BuiltOutput(
        artifact_type="FEATURES",
        content={
            "rows": int(len(feature_frame)),
            "symbols": len(symbols),
            "feature_column_count": int(feature_meta.get("feature_column_count") or 0),
        },
        metadata={
            "source_universe_artifact_id": universe_artifact.id,
            "feature_family_columns": dict(feature_meta.get("feature_family_columns") or {}),
            "coverage_rows": list(feature_meta.get("coverage_rows") or []),
            "feature_column_count": int(feature_meta.get("feature_column_count") or 0),
            "symbols_processed": int(feature_meta.get("symbols_processed") or 0),
            "feature_config": dict(feature_meta.get("config") or {}),
            "representation_embedding_enabled": bool(feature_meta.get("representation_embedding_enabled")),
            "representation_embedding_columns": list(feature_meta.get("representation_embedding_columns") or []),
            "representation_embedding_dimension": int(feature_meta.get("representation_embedding_dimension") or 0),
            "representation_embedding_model_name": str(feature_meta.get("representation_embedding_model_name") or ""),
            "representation_embedding_model_version": str(feature_meta.get("representation_embedding_model_version") or ""),
            "representation_embedding_store_dir": str(feature_meta.get("representation_embedding_store_dir") or ""),
            "representation_embedding_family_groups": dict(feature_meta.get("representation_embedding_family_groups") or {}),
            "feature_start_date": str(feature_meta.get("feature_start_date") or ""),
            "feature_end_date": str(feature_meta.get("feature_end_date") or ""),
            "features_cache_key": features_cache_key,
            **stored.storage_metadata(),
        },
        uri=stored.uri,
    )


__all__ = ["execute_features", "execute_labels", "execute_universe"]
