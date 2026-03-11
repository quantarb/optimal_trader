from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand

from pipeline.scalability import (
    DEFAULT_BENCHMARK_END_DATE,
    DEFAULT_BENCHMARK_START_DATE,
    DEFAULT_MAX_TIER2_RUNTIME_SECONDS,
    FEATURE_PROFILES,
    render_scalability_report_markdown,
    run_scalability_benchmark_suite,
    scalability_tier_names,
    write_scalability_report_files,
)


class Command(BaseCommand):
    help = "Run the gated scalability/performance benchmark suite and write JSON/Markdown reports."

    def add_arguments(self, parser):
        parser.add_argument(
            "--tiers",
            default=",".join(scalability_tier_names()),
            help="Comma-separated tier names. Available: " + ", ".join(scalability_tier_names()),
        )
        parser.add_argument(
            "--feature-profile",
            default="baseline",
            choices=sorted(FEATURE_PROFILES.keys()),
            help="Feature profile used for the benchmark suite.",
        )
        parser.add_argument("--start-date", default=DEFAULT_BENCHMARK_START_DATE)
        parser.add_argument("--end-date", default=DEFAULT_BENCHMARK_END_DATE)
        parser.add_argument("--artifact-storage-format", default="parquet", choices=["csv", "parquet"])
        parser.add_argument("--max-tier2-runtime", type=float, default=DEFAULT_MAX_TIER2_RUNTIME_SECONDS)
        parser.add_argument("--min-profit-pct", type=float, default=2.0)
        parser.add_argument("--output-dir", default=str(Path("docs") / "performance"))

    def handle(self, *args, **options):
        tiers = [token.strip() for token in str(options["tiers"]).split(",") if token.strip()]
        output_dir = Path(str(options["output_dir"]))
        report = run_scalability_benchmark_suite(
            tiers=tiers,
            output_dir=None,
            feature_profile=str(options["feature_profile"]),
            start_date=str(options["start_date"]),
            end_date=str(options["end_date"]),
            artifact_storage_format=str(options["artifact_storage_format"]),
            max_tier2_runtime_seconds=float(options["max_tier2_runtime"]),
            min_profit_pct=float(options["min_profit_pct"]),
        )
        paths = write_scalability_report_files(output_dir=output_dir, report=report)
        self.stdout.write(render_scalability_report_markdown(report))
        self.stdout.write(f"JSON report: {paths['json']}")
        self.stdout.write(f"Markdown report: {paths['markdown']}")
