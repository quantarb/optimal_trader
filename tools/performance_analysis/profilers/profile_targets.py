from __future__ import annotations

import os

from ..config import BenchmarkDefaults


def _bootstrap() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    import django

    django.setup()


def _ensure_fixture() -> None:
    _bootstrap()
    from fmp.models import Symbol
    from pipeline.test_support import ScalabilityFixtureMixin

    enough = (
        Symbol.objects.filter(symbol__startswith="TIER1").count() >= 10
        and Symbol.objects.filter(symbol__startswith="TIER2").count() >= 100
        and Symbol.objects.filter(symbol__startswith="TIER3").count() >= 1000
    )
    if not enough:
        ScalabilityFixtureMixin.seed_scalability_universe(start_date="2024-01-02", business_days=90)


def prepare_profile_environment() -> None:
    _ensure_fixture()


def run_scalability_target(tier: str, *, benchmark: BenchmarkDefaults | None = None) -> dict:
    defaults = benchmark or BenchmarkDefaults()
    _ensure_fixture()
    from pipeline.scalability import run_scalability_benchmark_suite

    report = run_scalability_benchmark_suite(
        tiers=[tier],
        output_dir=None,
        feature_profile=defaults.feature_profile,
        start_date=defaults.start_date,
        end_date=defaults.end_date,
        train_end_date=defaults.train_end_date,
        score_start_date=defaults.score_start_date,
        artifact_storage_format=defaults.artifact_storage_format,
        min_profit_pct=defaults.min_profit_pct,
        label_k_params=defaults.label_k_params,
        buy_execution=defaults.buy_execution,
        sell_execution=defaults.sell_execution,
        short_execution=defaults.short_execution,
        cover_execution=defaults.cover_execution,
        max_tier2_runtime_seconds=999999.0,
    )
    return dict(report["tiers"][0])


def available_profile_targets() -> dict[str, str]:
    return {
        "scalability_tier1": "10-symbol end-to-end scalability workflow",
        "scalability_tier2": "100-symbol end-to-end scalability workflow",
        "scalability_tier3": "1000-symbol end-to-end scalability workflow",
    }


def resolve_profile_target(target: str, *, benchmark: BenchmarkDefaults | None = None):
    name = str(target or "scalability_tier2").strip().lower()
    if name == "scalability_tier1":
        return lambda: run_scalability_target("tier1", benchmark=benchmark)
    if name == "scalability_tier2":
        return lambda: run_scalability_target("tier2", benchmark=benchmark)
    if name == "scalability_tier3":
        return lambda: run_scalability_target("tier3", benchmark=benchmark)
    raise ValueError(f"Unknown profile target {target!r}. Available: {', '.join(sorted(available_profile_targets()))}")
