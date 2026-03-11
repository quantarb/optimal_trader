# Pipeline UI Design System

## Audit

The pipeline UI had several repeated inconsistencies:

- Multiple page shells with duplicated `topbar`, `hero`, and `card` CSS blocks.
- Different color systems between research pages, insight pages, and internal tooling.
- Inconsistent spacing and typography across dashboard, reports, and symbol research.
- Large report pages implemented as one-off table pages with no shared report scaffold.
- Raw internal feature names such as `ev_dividedby_ebitda` and `ret_5d` leaking into tables and previews.
- Wide tables without a shared search, sort, or pagination behavior.
- Navigation structure differing page to page, which made the app feel stitched together instead of intentional.

## Layout System

The shared app shell now follows:

- `AppLayout`
  - `SidebarNavigation`
  - `TopNavigationBar`
  - `PageContent`

Pages should follow:

- `PageHeader`
- `Toolbar`
- `SummaryMetrics`
- `MainContent`
- `SecondaryPanels`

Report pages follow:

- `ReportHeader`
- `SummaryMetrics`
- `InsightCards`
- `DataTables`
- `Charts`

## Shared Components

Implemented shared components and primitives:

- `templates/pipeline/base_app.html`
- `templates/pipeline/base_report.html`
- `templates/pipeline/components/_design_system_css.html`
- `templates/pipeline/components/_design_system_js.html`
- `templates/pipeline/components/_page_header.html`
- `templates/pipeline/components/_empty_state.html`

Core CSS classes are:

- `page-header`
- `section-card`
- `metric-tile`
- `filter-panel`
- `data-table-shell`
- `status-badge`
- `action-button`
- `empty-state`
- `error-state`
- `row-card`
- `summary-list`
- `summary-item`

## Feature Presentation

Canonical feature presentation is centralized in:

- `pipeline/feature_presentation.py`
- `pipeline/templatetags/pipeline_ui.py`

This registry standardizes:

- display name
- family name
- numeric formatting
- canonical line rendering
- embedding serialization

All new pages should use:

- `{% feature_label feature_name %}`
- `{% feature_value feature_name value %}`
- `{{ family_name|feature_family_label }}`
- `{{ signature|feature_family_signature_label }}`

## Refactored Pages

These pages now use the shared shell:

- `templates/pipeline/dashboard.html`
- `templates/pipeline/opportunities.html`
- `templates/pipeline/stock_intelligence.html`
- `templates/pipeline/portfolio_analysis.html`
- `templates/pipeline/market_situations.html`
- `templates/pipeline/research_reports.html`
- `templates/pipeline/oracle_reports.html`
- `templates/pipeline/feature_attribution_reports.html`

Feature previews were also standardized in:

- `templates/pipeline/symbol_research.html`
- `templates/pipeline/artifact_detail.html`

## Table Standardization

The shared table system is class-based:

- apply `app-table js-data-table`
- optional `data-page-size`

The shared JS now provides:

- client-side filtering
- sortable columns
- pagination

## Migration Plan

Remaining high-priority pages to migrate next:

1. `templates/pipeline/diagnostic_reports.html`
2. `templates/pipeline/rl_policy_reports.html`
3. `templates/pipeline/backtest_detail.html`
4. `templates/pipeline/strategy_detail.html`
5. `templates/pipeline/lab.html`
6. `templates/pipeline/cohorts.html`

Migration rules:

1. Extend `base_app.html` or `base_report.html`.
2. Remove page-specific root CSS variables unless the page needs a deliberate variation.
3. Replace one-off stats with `metric-tile`.
4. Replace one-off tables with `data-table-shell` and `app-table js-data-table`.
5. Replace internal feature names with canonical feature presentation helpers.
6. Preserve route names and underlying view logic.
