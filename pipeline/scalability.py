from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd

from django.utils import timezone

from pipeline.performance import PerformanceTracer

from .models import Artifact, PipelineRun, StrategyDefinition
from .services import ARTIFACT_DIR, execute_pipeline_run
from .strategy_definitions import ensure_default_strategy_definitions, upsert_strategy_definition
from .universe_selection import resolve_market_cap_tier_symbols


@dataclass(frozen=True)
class ScalabilityTier:
    key: str
    market_cap_key: str
    label: str
    target_symbol_count: int


SCALABILITY_TIERS: dict[str, ScalabilityTier] = {
    "tier1": ScalabilityTier("tier1", "1t", "1T+ market cap", 10),
    "tier2": ScalabilityTier("tier2", "100b", "100B+ market cap", 100),
    "tier3": ScalabilityTier("tier3", "10b", "10B+ market cap", 1000),
}

FEATURE_PROFILES: dict[str, dict[str, Any]] = {
    "baseline": {
        "include_price_technicals": True,
        "include_fundamental_change": False,
        "include_statement_quality": False,
        "include_event_features": False,
        "include_ownership_features": False,
        "include_economic_indicators": False,
        "include_treasury_rates": False,
        "include_representation_embedding": False,
    },
    "full": {
        "include_price_technicals": True,
        "include_fundamental_change": True,
        "include_statement_quality": True,
        "include_event_features": True,
        "include_ownership_features": True,
        "include_economic_indicators": True,
        "include_treasury_rates": True,
        "include_representation_embedding": False,
    },
}

DEFAULT_BENCHMARK_START_DATE = "2020-01-01"
DEFAULT_BENCHMARK_END_DATE = "2025-12-31"
DEFAULT_MAX_TIER2_RUNTIME_SECONDS = 180.0


def scalability_tier_names() -> list[str]:
    return list(SCALABILITY_TIERS.keys())


def resolve_scalability_tier(tier_name: str) -> ScalabilityTier:
    key = str(tier_name or "").strip().lower()
    if key not in SCALABILITY_TIERS:
        raise ValueError(f"Unknown scalability tier {tier_name!r}. Available: {', '.join(scalability_tier_names())}")
    return SCALABILITY_TIERS[key]


def run_scalability_benchmark_suite(
    *,
    tiers: list[str] | None = None,
    output_dir: str | Path | None = None,
    feature_profile: str = "baseline",
    start_date: str = DEFAULT_BENCHMARK_START_DATE,
    end_date: str = DEFAULT_BENCHMARK_END_DATE,
    artifact_storage_format: str = "parquet",
    max_tier2_runtime_seconds: float = DEFAULT_MAX_TIER2_RUNTIME_SECONDS,
    min_profit_pct: float = 2.0,
    label_k_params: dict[str, list[int]] | None = None,
    train_end_date: str | None = None,
    score_start_date: str | None = None,
    buy_execution: str | None = None,
    sell_execution: str | None = None,
    short_execution: str | None = None,
    cover_execution: str | None = None,
) -> dict[str, Any]:
    ordered_tiers = [resolve_scalability_tier(name) for name in (tiers or scalability_tier_names())]
    suite_started = time.perf_counter()
    tier_reports: list[dict[str, Any]] = []
    skipped_tiers: list[dict[str, Any]] = []

    for tier in ordered_tiers:
        if tier.key == "tier3":
            prior_tier2 = next((row for row in tier_reports if row.get("tier", {}).get("key") == "tier2"), None)
            if prior_tier2 is not None and float(prior_tier2.get("total_runtime_seconds") or 0.0) > float(max_tier2_runtime_seconds):
                skipped_tiers.append(
                    {
                        "tier": tier.key,
                        "reason": "tier2_runtime_exceeded_limit",
                        "tier2_runtime_seconds": round(float(prior_tier2.get("total_runtime_seconds") or 0.0), 6),
                        "max_tier2_runtime_seconds": float(max_tier2_runtime_seconds),
                    }
                )
                continue
        tier_reports.append(
            run_scalability_benchmark(
                tier=tier,
                output_dir=output_dir,
                feature_profile=feature_profile,
                start_date=start_date,
                end_date=end_date,
                artifact_storage_format=artifact_storage_format,
                min_profit_pct=min_profit_pct,
                label_k_params=label_k_params,
                train_end_date=train_end_date,
                score_start_date=score_start_date,
                buy_execution=buy_execution,
                sell_execution=sell_execution,
                short_execution=short_execution,
                cover_execution=cover_execution,
            )
        )

    suite_report = {
        "created_at": timezone.now().isoformat(),
        "feature_profile": str(feature_profile),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "artifact_storage_format": str(artifact_storage_format),
        "min_profit_pct": float(min_profit_pct),
        "label_k_params": dict(label_k_params or {"YE": [1]}),
        "train_end_date": str(train_end_date or ""),
        "score_start_date": str(score_start_date or ""),
        "buy_execution": str(buy_execution or ""),
        "sell_execution": str(sell_execution or ""),
        "short_execution": str(short_execution or ""),
        "cover_execution": str(cover_execution or ""),
        "total_runtime_seconds": round(float(time.perf_counter() - suite_started), 6),
        "tiers": tier_reports,
        "skipped_tiers": skipped_tiers,
    }
    if output_dir is not None:
        write_scalability_report_files(output_dir=output_dir, report=suite_report)
    return suite_report


def run_scalability_benchmark(
    *,
    tier: ScalabilityTier,
    output_dir: str | Path | None = None,
    feature_profile: str = "baseline",
    start_date: str = DEFAULT_BENCHMARK_START_DATE,
    end_date: str = DEFAULT_BENCHMARK_END_DATE,
    artifact_storage_format: str = "parquet",
    min_profit_pct: float = 2.0,
    label_k_params: dict[str, list[int]] | None = None,
    train_end_date: str | None = None,
    score_start_date: str | None = None,
    buy_execution: str | None = None,
    sell_execution: str | None = None,
    short_execution: str | None = None,
    cover_execution: str | None = None,
) -> dict[str, Any]:
    if feature_profile not in FEATURE_PROFILES:
        raise ValueError(f"Unknown feature profile {feature_profile!r}. Available: {', '.join(sorted(FEATURE_PROFILES))}")

    symbols = resolve_market_cap_tier_symbols(tier_key=tier.market_cap_key, limit=tier.target_symbol_count)
    if not symbols:
        raise ValueError(f"No symbols were resolved for {tier.key}.")

    ensure_default_strategy_definitions()
    benchmark_strategy = upsert_strategy_definition(
        slug="scalability-benchmark-topk-v1",
        name="Scalability Benchmark Top-K",
        strategy_type=StrategyDefinition.StrategyType.NOTEBOOK_TOPK_V1,
        config={
            "gate_quantile": 0.0,
            "top_k": min(25, max(5, len(symbols))),
            "rebalance_freq": "W",
            "gross_exposure": 1.0,
            "selection_side": "long_only",
            "signal_combination": "multiply",
        },
        description="Low-gate benchmark strategy used by the scalability suite.",
    )

    output_root = Path(output_dir) if output_dir is not None else (Path("docs") / "performance")
    artifact_root = output_root / "artifacts" / tier.key
    tracer = PerformanceTracer(enabled=True)
    benchmark_started = time.perf_counter()

    resolved_train_end_date, resolved_score_start_date = (
        str(train_end_date or "").strip() or None,
        str(score_start_date or "").strip() or None,
    )
    if not resolved_train_end_date or not resolved_score_start_date:
        resolved_train_end_date, resolved_score_start_date = _train_score_split_dates(start_date=start_date, end_date=end_date)

    resolved_label_k_params = dict(label_k_params or {"YE": [1]})
    resolved_label_execution = {
        "buy_execution": str(buy_execution or "").strip() or None,
        "sell_execution": str(sell_execution or "").strip() or None,
        "short_execution": str(short_execution or "").strip() or None,
        "cover_execution": str(cover_execution or "").strip() or None,
    }
    with _artifact_dir_override(artifact_root):
        universe_artifact = _run_benchmark_job(
            name=f"{tier.key}-universe",
            target_job="universe",
            config={"symbols": symbols},
            performance_tracer=tracer,
        )
        label_artifact = _run_benchmark_job(
            name=f"{tier.key}-labels",
            target_job="labels",
            input_artifact_ids=[int(universe_artifact.id)],
            config={
                "k_params": resolved_label_k_params,
                "min_profit_pct": float(min_profit_pct),
                "label_start_date": start_date,
                "label_end_date": end_date,
                **{key: value for key, value in resolved_label_execution.items() if value},
                "artifact_storage_format": artifact_storage_format,
            },
            performance_tracer=tracer,
        )
        feature_artifact = _run_benchmark_job(
            name=f"{tier.key}-features",
            target_job="features",
            input_artifact_ids=[int(universe_artifact.id)],
            config={
                **FEATURE_PROFILES[feature_profile],
                "feature_start_date": start_date,
                "feature_end_date": end_date,
                "artifact_storage_format": artifact_storage_format,
            },
            performance_tracer=tracer,
        )
        model_artifact = _run_benchmark_job(
            name=f"{tier.key}-fit-regressor",
            target_job="fit_regressor",
            input_artifact_ids=[int(label_artifact.id), int(feature_artifact.id)],
            config={
                "target_col": "trade_return",
                "framework": "sklearn",
                "split_ratio": 1.0,
                "train_start_date": start_date,
                "train_end_date": resolved_train_end_date,
                "params": {
                    "n_estimators": 64,
                    "max_depth": 8,
                    "n_jobs": -1,
                },
            },
            performance_tracer=tracer,
        )
        prediction_artifact = _run_benchmark_job(
            name=f"{tier.key}-score-regressor",
            target_job="score_regressor",
            input_artifact_ids=[int(model_artifact.id), int(feature_artifact.id)],
            config={
                "score_start_date": resolved_score_start_date,
                "score_end_date": end_date,
                "label_artifact_id": int(label_artifact.id),
                "artifact_storage_format": artifact_storage_format,
            },
            performance_tracer=tracer,
        )
        strategy_artifact = _run_benchmark_job(
            name=f"{tier.key}-strategy-dataset",
            target_job="build_strategy_dataset",
            input_artifact_ids=[int(feature_artifact.id)],
            config={
                "prediction_artifact_ids": [int(prediction_artifact.id)],
                "strategy_definition_id": int(benchmark_strategy.id),
                "strategy_start_date": resolved_score_start_date,
                "strategy_end_date": end_date,
                "artifact_storage_format": artifact_storage_format,
            },
            performance_tracer=tracer,
        )
        backtest_artifact = _run_benchmark_job(
            name=f"{tier.key}-backtest",
            target_job="backtest_strategy",
            input_artifact_ids=[int(strategy_artifact.id)],
            config={
                "backtest_start_date": resolved_score_start_date,
                "backtest_end_date": end_date,
                "fee_bps": 2.0,
                "slippage_bps": 4.0,
                "execution_delay_days": 1,
                "turnover_half_l1": True,
                "artifact_storage_format": artifact_storage_format,
            },
            performance_tracer=tracer,
        )

    tracer_summary = tracer.summary()
    report = {
        "created_at": timezone.now().isoformat(),
        "tier": {
            "key": str(tier.key),
            "label": str(tier.label),
            "market_cap_key": str(tier.market_cap_key),
            "target_symbol_count": int(tier.target_symbol_count),
            "actual_symbol_count": int(len(symbols)),
            "symbols_preview": list(symbols[:20]),
        },
        "feature_profile": str(feature_profile),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "train_end_date": str(resolved_train_end_date),
        "score_start_date": str(resolved_score_start_date),
        "artifact_storage_format": str(artifact_storage_format),
        "min_profit_pct": float(min_profit_pct),
        "label_k_params": resolved_label_k_params,
        "buy_execution": str(resolved_label_execution["buy_execution"] or ""),
        "sell_execution": str(resolved_label_execution["sell_execution"] or ""),
        "short_execution": str(resolved_label_execution["short_execution"] or ""),
        "cover_execution": str(resolved_label_execution["cover_execution"] or ""),
        "artifacts": {
            "universe": _artifact_snapshot(universe_artifact),
            "labels": _artifact_snapshot(label_artifact),
            "features": _artifact_snapshot(feature_artifact),
            "model": _artifact_snapshot(model_artifact),
            "predictions": _artifact_snapshot(prediction_artifact),
            "strategy_dataset": _artifact_snapshot(strategy_artifact),
            "backtest": _artifact_snapshot(backtest_artifact),
        },
        "performance": tracer_summary,
        "total_runtime_seconds": round(float(time.perf_counter() - benchmark_started), 6),
    }
    if output_dir is not None:
        write_scalability_report_files(output_dir=output_dir, report=report, filename_prefix=tier.key)
    return report


def write_scalability_report_files(
    *,
    output_dir: str | Path,
    report: dict[str, Any],
    filename_prefix: str | None = None,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    prefix = str(filename_prefix or f"scalability_{timezone.now().strftime('%Y%m%d_%H%M%S')}")
    json_path = root / f"{prefix}.json"
    md_path = root / f"{prefix}.md"
    json_path.write_text(json.dumps(report, sort_keys=True, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_scalability_report_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def render_scalability_report_markdown(report: dict[str, Any]) -> str:
    if isinstance(report.get("tiers"), list):
        lines = ["# Scalability Benchmark Suite", ""]
        lines.append("| Tier | Symbols | Runtime (s) | Top Stage |")
        lines.append("| --- | ---: | ---: | --- |")
        for tier_report in list(report.get("tiers") or []):
            tier_meta = dict(tier_report.get("tier") or {})
            stages = sorted(
                list((tier_report.get("performance") or {}).get("stages") or []),
                key=lambda row: float(row.get("wall_seconds") or 0.0),
                reverse=True,
            )
            top_stage = stages[0]["name"] if stages else ""
            lines.append(
                f"| {tier_meta.get('key','')} | {tier_meta.get('actual_symbol_count',0)} | "
                f"{float(tier_report.get('total_runtime_seconds') or 0.0):.3f} | {top_stage} |"
            )
        skipped = list(report.get("skipped_tiers") or [])
        if skipped:
            lines.extend(["", "## Skipped Tiers", ""])
            for row in skipped:
                lines.append(f"- `{row.get('tier')}` skipped: `{row.get('reason')}`")
        return "\n".join(lines) + "\n"

    tier_meta = dict(report.get("tier") or {})
    lines = [
        f"# Scalability Benchmark {tier_meta.get('key', '')}",
        "",
        f"- Tier: {tier_meta.get('label', '')}",
        f"- Symbols: {tier_meta.get('actual_symbol_count', 0)}",
        f"- Runtime: {float(report.get('total_runtime_seconds') or 0.0):.3f}s",
        f"- Feature profile: {report.get('feature_profile', '')}",
        f"- Date window: {report.get('start_date', '')} -> {report.get('end_date', '')}",
        "",
        "## Hottest Stages",
        "",
        "| Stage | Category | Workload | Runtime (s) | Read MB | Write MB |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    stages = sorted(
        list((report.get("performance") or {}).get("stages") or []),
        key=lambda row: float(row.get("wall_seconds") or 0.0),
        reverse=True,
    )[:12]
    for stage in stages:
        lines.append(
            f"| {stage.get('name','')} | {stage.get('category','')} | {stage.get('workload_type','')} | "
            f"{float(stage.get('wall_seconds') or 0.0):.3f} | "
            f"{float(stage.get('read_bytes') or 0) / (1024.0 * 1024.0):.2f} | "
            f"{float(stage.get('write_bytes') or 0) / (1024.0 * 1024.0):.2f} |"
        )
    return "\n".join(lines) + "\n"


def _artifact_snapshot(artifact: Artifact) -> dict[str, Any]:
    return {
        "id": int(artifact.id),
        "artifact_type": str(artifact.artifact_type),
        "uri": str(artifact.uri),
        "content": dict(artifact.content or {}),
        "metadata": dict(artifact.metadata or {}),
    }


def _train_score_split_dates(*, start_date: str, end_date: str) -> tuple[str, str]:
    start_ts = pd.Timestamp(str(start_date))
    end_ts = pd.Timestamp(str(end_date))
    if end_ts <= start_ts:
        raise ValueError("end_date must be after start_date for scalability benchmarks.")
    split_ts = (start_ts + ((end_ts - start_ts) * 0.8)).normalize()
    score_start = (split_ts + pd.Timedelta(days=1)).normalize()
    if score_start > end_ts:
        score_start = end_ts
    return split_ts.date().isoformat(), score_start.date().isoformat()


def _run_benchmark_job(
    *,
    name: str,
    target_job: str,
    config: dict[str, Any],
    input_artifact_ids: list[int] | None = None,
    performance_tracer: PerformanceTracer | None = None,
) -> Artifact:
    pipeline_run = PipelineRun.objects.create(
        name=str(name),
        requested_job=str(target_job),
        mode=PipelineRun.Mode.STRICT,
        status=PipelineRun.Status.PENDING,
    )
    return execute_pipeline_run(
        pipeline_run=pipeline_run,
        target_job=target_job,
        mode=PipelineRun.Mode.STRICT,
        config=dict(config or {}),
        input_artifact_ids=list(input_artifact_ids or []),
        performance_tracer=performance_tracer,
    )


@contextmanager
def _artifact_dir_override(path: Path):
    from . import services

    previous = ARTIFACT_DIR
    services.ARTIFACT_DIR = Path(path)
    try:
        yield
    finally:
        services.ARTIFACT_DIR = previous


__all__ = [
    "DEFAULT_BENCHMARK_END_DATE",
    "DEFAULT_BENCHMARK_START_DATE",
    "DEFAULT_MAX_TIER2_RUNTIME_SECONDS",
    "FEATURE_PROFILES",
    "SCALABILITY_TIERS",
    "ScalabilityTier",
    "render_scalability_report_markdown",
    "resolve_scalability_tier",
    "run_scalability_benchmark",
    "run_scalability_benchmark_suite",
    "scalability_tier_names",
    "write_scalability_report_files",
]
