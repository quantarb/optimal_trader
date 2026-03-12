from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

from ..config import default_routes
from ..models import AnalysisConfig, ArtifactInventory, RouteTarget
from ..integrations.data_quality_runner import latest_rows, load_artifact_frame, resolve_candidate_symbols


def configured_routes(config: AnalysisConfig) -> list[RouteTarget]:
    return list(config.routes or default_routes())


def artifact_backed_routes(config: AnalysisConfig, inventories: list[ArtifactInventory]) -> list[RouteTarget]:
    routes: list[RouteTarget] = []
    symbols = resolve_candidate_symbols(inventories, preferred=config.default_symbol_fallbacks)
    for inventory in inventories:
        strategy = inventory.artifacts.get("STRATEGY_DATASET")
        feature = inventory.artifacts.get("FEATURES")
        label = inventory.artifacts.get("LABELS")
        prediction = inventory.artifacts.get("REGRESSOR_PREDICTIONS")
        if strategy is None:
            continue
        params = [("strategy_artifact_id", strategy.artifact_id)]
        if feature is not None:
            params.append(("feature_artifact_id", feature.artifact_id))
        if label is not None:
            params.append(("label_artifact_id", label.artifact_id))
        if prediction is not None:
            params.append(("prediction_artifact_id", prediction.artifact_id))
        routes.append(
            RouteTarget(
                name=f"pipeline_opportunities_{inventory.tier}",
                path=f"/pipeline/opportunities/?{urlencode([*params, ('limit', 20)], doseq=True)}",
                group="pipeline",
                tier=inventory.tier,
                description=f"Opportunities using the latest {inventory.tier} artifact stack.",
                tags=["opportunities", "scalability", str(inventory.tier)],
            )
        )
        if symbols and inventory.tier in {"tier1", "tier2"}:
            route_symbols = symbols[: min(10 if inventory.tier == "tier1" else 25, len(symbols))]
            routes.append(
                RouteTarget(
                    name=f"pipeline_portfolio_{inventory.tier}",
                    path=f"/pipeline/portfolio-analysis/?{urlencode([*params, ('symbols', ','.join(route_symbols))], doseq=True)}",
                    group="pipeline",
                    tier=inventory.tier,
                    description=f"Portfolio analysis using the latest {inventory.tier} artifact stack.",
                    tags=["portfolio", "scalability", str(inventory.tier)],
                    metadata={"symbol_count": len(route_symbols)},
                )
            )
        frame = latest_rows(load_artifact_frame(strategy.uri))
        if inventory.tier == "tier1" and not frame.empty and "symbol" in frame.columns:
            detail_symbol = str(frame["symbol"].astype(str).iloc[0]).strip().upper()
            if detail_symbol:
                routes.append(
                    RouteTarget(
                        name=f"pipeline_stock_{inventory.tier}_{detail_symbol.lower()}",
                        path=f"/pipeline/stock/{detail_symbol}/?{urlencode(params, doseq=True)}",
                        group="pipeline",
                        tier=inventory.tier,
                        symbol=detail_symbol,
                        description=f"Stock intelligence detail backed by the latest {inventory.tier} artifact stack.",
                        tags=["stock", "detail", "scalability", str(inventory.tier)],
                    )
                )
    return routes


def discover_routes(config: AnalysisConfig, inventories: list[ArtifactInventory]) -> list[RouteTarget]:
    routes = configured_routes(config)
    route_index = {route.name: route for route in routes}
    for route in artifact_backed_routes(config, inventories):
        route_index.setdefault(route.name, route)
    return list(route_index.values())
