from __future__ import annotations

from ..integrations.playwright_runner import crawl_routes
from ..models import AnalysisConfig, PageSnapshot, RouteTarget


def crawl_pages(config: AnalysisConfig, routes: list[RouteTarget]) -> list[PageSnapshot]:
    return crawl_routes(config, routes)
