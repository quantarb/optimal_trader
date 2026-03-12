from __future__ import annotations

from pathlib import Path

from .models import AnalysisConfig, RouteTarget


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "product_quality"
DEFAULT_SNAPSHOT_DIR = DEFAULT_OUTPUT_DIR / "snapshots"
DEFAULT_CRAWL_DIR = DEFAULT_OUTPUT_DIR / "crawl_artifacts"
DEFAULT_BASELINE_LABEL = "baseline"
DEFAULT_CURRENT_LABEL = "after"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_BROWSER_TIMEOUT_MS = 18_000
DEFAULT_LARGE_TABLE_WARNING = 100
DEFAULT_LARGE_TABLE_CRITICAL = 300
DEFAULT_DOM_WARNING = 1_500
DEFAULT_DOM_CRITICAL = 3_000
DEFAULT_MAG7_SYMBOLS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA")
DEFAULT_SYMBOL_FALLBACKS = ("AMD", "AVGO", "CRM")
DEFAULT_TIERS = ("tier1", "tier2", "tier3")

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "current_price": ("close", "adj_close", "px__close", "price"),
    "return_20d": ("return_20d", "ret_20d", "px__ret_20d"),
    "return_60d": ("return_60d", "ret_60d", "px__ret_60d", "px__ret_63d"),
    "return_120d": ("return_120d", "ret_120d", "px__ret_120d", "px__ret_126d"),
    "signal_score": ("signal_score", "prediction_score", "ranking", "combined_score", "strategy_score", "prob_buy"),
    "feature_summary": ("feature_summary", "cluster_description", "stock_insight", "market_situation_explanation"),
}

DISPLAY_LABELS: dict[str, tuple[str, ...]] = {
    "current_price": ("Current Price", "Price"),
    "return_20d": ("20d", "20D", "Return (20D)"),
    "return_60d": ("60d", "60D", "Return (60D)"),
    "return_120d": ("120d", "120D", "Return (120D)"),
    "signal_score": ("Signal", "Signal Score", "Prediction Score", "Opportunity Score"),
}


def default_routes() -> list[RouteTarget]:
    return [
        RouteTarget(
            name="pipeline_ui",
            path="/pipeline/ui/",
            group="pipeline",
            description="Shared pipeline shell and navigation surface.",
            tags=["dashboard", "shell"],
        ),
        RouteTarget(
            name="pipeline_stock_root",
            path="/pipeline/stock-intelligence/",
            group="pipeline",
            description="Default stock-intelligence landing page.",
            tags=["stock", "intelligence"],
        ),
        RouteTarget(
            name="pipeline_stock_amd",
            path="/pipeline/stock/AMD/",
            group="pipeline",
            description="Known-working stock detail route backed by the latest tier artifacts.",
            symbol="AMD",
            tags=["stock", "intelligence", "detail"],
        ),
        RouteTarget(
            name="pipeline_opportunities",
            path="/pipeline/opportunities/?limit=20",
            group="pipeline",
            description="Opportunity ranking page.",
            tags=["opportunities", "core-flow"],
        ),
        RouteTarget(
            name="pipeline_portfolio_default",
            path="/pipeline/portfolio-analysis/?symbols=AMD,AVGO,CRM",
            group="pipeline",
            description="Portfolio analysis with three representative holdings.",
            tags=["portfolio", "core-flow"],
        ),
        RouteTarget(
            name="fmp_universe_screener",
            path="/fmp/universe-screener/form/",
            group="fmp",
            description="Universe screener form.",
            tags=["screener", "form"],
        ),
        RouteTarget(
            name="fmp_symbol_aapl",
            path="/fmp/symbol/AAPL/",
            group="fmp",
            description="FMP symbol detail page.",
            symbol="AAPL",
            tags=["symbol-detail", "fmp"],
        ),
    ]


def default_config(
    *,
    base_url: str = DEFAULT_BASE_URL,
    output_dir: str | Path | None = None,
    label: str = DEFAULT_CURRENT_LABEL,
    tiers: tuple[str, ...] = DEFAULT_TIERS,
) -> AnalysisConfig:
    resolved_output = Path(output_dir).resolve() if output_dir is not None else DEFAULT_OUTPUT_DIR
    return AnalysisConfig(
        base_url=str(base_url).rstrip("/"),
        output_dir=resolved_output,
        snapshot_dir=resolved_output / "snapshots",
        crawl_dir=resolved_output / "crawl_artifacts",
        label=str(label or DEFAULT_CURRENT_LABEL),
        browser_timeout_ms=DEFAULT_BROWSER_TIMEOUT_MS,
        dom_warning_threshold=DEFAULT_DOM_WARNING,
        dom_critical_threshold=DEFAULT_DOM_CRITICAL,
        table_warning_threshold=DEFAULT_LARGE_TABLE_WARNING,
        table_critical_threshold=DEFAULT_LARGE_TABLE_CRITICAL,
        critical_symbols=list(DEFAULT_MAG7_SYMBOLS),
        default_symbol_fallbacks=list(DEFAULT_SYMBOL_FALLBACKS),
        symbol_tiers=list(tiers),
        field_aliases={key: list(value) for key, value in FIELD_ALIASES.items()},
        display_labels={key: list(value) for key, value in DISPLAY_LABELS.items()},
        routes=default_routes(),
    )
