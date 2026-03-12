from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from ..models import AnalysisConfig, PageSnapshot, RouteTarget
from ..utils.dom_utils import coerce_table_metrics
from ..utils.path_utils import ensure_directory, safe_slug


DOM_METRICS_SCRIPT = """
() => {
  const sampleNodes = Array.from(document.querySelectorAll('body *')).slice(0, 900);
  const styleSet = (mapper) => {
    const values = new Set();
    for (const node of sampleNodes) {
      const style = window.getComputedStyle(node);
      const value = mapper(style);
      if (value) values.add(String(value).trim().toLowerCase());
    }
    return Array.from(values);
  };
  const spacingValues = styleSet((style) => {
    const tokens = [style.marginTop, style.marginBottom, style.paddingTop, style.paddingBottom, style.gap]
      .map((value) => String(value || '').trim())
      .filter(Boolean);
    return tokens.join('|');
  }).flatMap((value) => value.split('|').filter(Boolean));
  const emptyNeedles = /(n\\/a|no data|not available|no rows|no .*available|provide a .*|missing .*|error)/i;
  const textNodes = sampleNodes.map((node) => String(node.textContent || '').trim()).filter(Boolean);
  const emptyMarkers = textNodes.filter((text) => emptyNeedles.test(text)).slice(0, 30);
  const errorMarkers = textNodes.filter((text) => /(traceback|internal server error|could not|failed|timeout)/i.test(text)).slice(0, 20);
  const tables = Array.from(document.querySelectorAll('table')).map((table, index) => {
    const rows = Array.from(table.querySelectorAll('tbody tr'));
    const visibleRows = rows.filter((row) => window.getComputedStyle(row).display !== 'none');
    const headerRow = table.querySelector('thead tr') || table.querySelector('tr');
    const pageSizeAttr = table.getAttribute('data-page-size') || table.dataset.pageSize || '';
    const shell = table.closest('.data-table-shell') || table.closest('.pager') || table.parentElement;
    const shellText = shell ? String(shell.textContent || '') : '';
    const hasPagination = /(page|rows per page|prev|next|first|last)/i.test(shellText);
    return {
      index,
      identifier: table.id || table.getAttribute('aria-label') || table.className || `table-${index}`,
      row_count: rows.length,
      column_count: headerRow ? headerRow.children.length : 0,
      visible_row_count: visibleRows.length,
      visible_column_count: headerRow ? headerRow.children.length : 0,
      has_pagination: hasPagination,
      page_size: pageSizeAttr ? Number(pageSizeAttr) : null,
      has_sort_controls: table.querySelectorAll('button.sort-btn,[data-sortable],th button').length > 0,
      text_density: Number(((rows.length || 1) * ((headerRow ? headerRow.children.length : 0) || 1)).toFixed(2)),
    };
  });
  const layoutSignature = [
    document.querySelector('.app-shell') ? 'app-shell' : '',
    document.querySelector('.app-sidebar') ? 'sidebar' : '',
    document.querySelector('.app-topbar') ? 'topbar' : '',
    document.querySelector('.page-grid') ? 'page-grid' : '',
    document.querySelector('.filter-panel') ? 'filter-panel' : '',
    document.querySelector('.section-card') ? 'section-card' : '',
    document.querySelector('.panel') ? 'panel' : '',
    document.querySelector('.pager') ? 'pager' : '',
  ].filter(Boolean);
  const componentSignatures = Array.from(new Set(sampleNodes
    .map((node) => {
      const classes = Array.from(node.classList || []).filter(Boolean).slice(0, 3).join('.');
      return classes ? `${node.tagName.toLowerCase()}.${classes}` : node.tagName.toLowerCase();
    })
    .filter(Boolean)
  )).slice(0, 60);
  return {
    dom_node_count: document.querySelectorAll('*').length,
    interactive_count: document.querySelectorAll('a,button,input,select,textarea,[role="button"],[tabindex]').length,
    heading_count: document.querySelectorAll('h1,h2,h3').length,
    card_count: document.querySelectorAll('.section-card,.row-card,.panel,.card,.metric-tile,[class*="card"]').length,
    chart_like_count: document.querySelectorAll('svg,canvas,[class*="chart"],[data-chart]').length,
    headings: Array.from(document.querySelectorAll('h1,h2,h3')).map((el) => String(el.textContent || '').trim()).filter(Boolean).slice(0, 20),
    text_sample: textNodes.slice(0, 40).join(' ').slice(0, 1200),
    empty_markers: emptyMarkers,
    error_markers: errorMarkers,
    unique_colors: styleSet((style) => style.color),
    unique_font_sizes: styleSet((style) => style.fontSize),
    unique_spacing_values: Array.from(new Set(spacingValues.map((value) => String(value).trim().toLowerCase()).filter(Boolean))),
    layout_signature: layoutSignature,
    component_signatures: componentSignatures,
    tables,
  };
}
"""


def _should_capture_screenshot(route: RouteTarget) -> bool:
    if "scalability" in route.tags:
        return False
    if route.tier in {"tier2", "tier3"}:
        return False
    return True


def crawl_routes(config: AnalysisConfig, routes: list[RouteTarget]) -> list[PageSnapshot]:
    ensure_directory(config.crawl_dir)
    snapshots: list[PageSnapshot] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        for route in routes:
            context = browser.new_context(viewport={"width": 1440, "height": 2200})
            page = context.new_page()
            page.set_default_navigation_timeout(config.browser_timeout_ms)
            page.set_default_timeout(config.browser_timeout_ms)
            console_errors: list[str] = []
            page.on(
                "console",
                lambda message: console_errors.append(str(message.text))
                if message.type == "error"
                else None,
            )
            page.on("pageerror", lambda exc: console_errors.append(str(exc)))
            target_url = route.full_url(config.base_url)
            html_path = config.crawl_dir / f"{safe_slug(config.label)}__{safe_slug(route.name)}.html"
            screenshot_path = config.crawl_dir / f"{safe_slug(config.label)}__{safe_slug(route.name)}.png"
            started = time.perf_counter()
            status_code: int | None = None
            ok = False
            response_error = ""
            metrics: dict[str, Any] = {}
            try:
                response = page.goto(target_url, wait_until="domcontentloaded", timeout=config.browser_timeout_ms)
                status_code = response.status if response is not None else None
                try:
                    page.wait_for_load_state("networkidle", timeout=min(config.browser_timeout_ms, 5_000))
                except PlaywrightTimeoutError:
                    console_errors.append("networkidle wait timed out")
                page.wait_for_timeout(250)
                metrics = dict(page.evaluate(DOM_METRICS_SCRIPT) or {})
                if _should_capture_screenshot(route):
                    page.screenshot(path=str(screenshot_path), full_page=False)
                ok = status_code is not None and int(status_code) < 500
            except PlaywrightTimeoutError as exc:
                response_error = f"timeout: {exc}"
                try:
                    metrics = dict(page.evaluate(DOM_METRICS_SCRIPT) or {})
                    if _should_capture_screenshot(route):
                        page.screenshot(path=str(screenshot_path), full_page=False)
                except PlaywrightError:
                    pass
            except PlaywrightError as exc:
                response_error = str(exc)
            finally:
                try:
                    html_path.write_text(page.content(), encoding="utf-8")
                except Exception:
                    html_path.write_text("", encoding="utf-8")
                context.close()
            snapshot = PageSnapshot(
                name=route.name,
                url=target_url,
                group=route.group,
                tier=route.tier,
                ok=ok,
                status_code=status_code,
                load_time_ms=round((time.perf_counter() - started) * 1000.0, 2),
                dom_node_count=int(metrics.get("dom_node_count") or 0),
                interactive_count=int(metrics.get("interactive_count") or 0),
                table_metrics=coerce_table_metrics(list(metrics.get("tables") or [])),
                card_count=int(metrics.get("card_count") or 0),
                chart_like_count=int(metrics.get("chart_like_count") or 0),
                heading_count=int(metrics.get("heading_count") or 0),
                unique_colors_used=len(list(metrics.get("unique_colors") or [])),
                unique_font_sizes=len(list(metrics.get("unique_font_sizes") or [])),
                unique_spacing_values=len(list(metrics.get("unique_spacing_values") or [])),
                layout_signature=[str(item) for item in list(metrics.get("layout_signature") or [])],
                component_signatures=[str(item) for item in list(metrics.get("component_signatures") or [])],
                text_sample=str(metrics.get("text_sample") or ""),
                headings=[str(item) for item in list(metrics.get("headings") or [])],
                empty_markers=[str(item) for item in list(metrics.get("empty_markers") or [])],
                error_markers=[str(item) for item in list(metrics.get("error_markers") or [])],
                console_errors=console_errors[:20],
                html_path=str(html_path),
                screenshot_path=str(screenshot_path),
                response_error=response_error,
                metadata=dict(route.metadata),
            )
            snapshots.append(snapshot)
        browser.close()
    return snapshots
