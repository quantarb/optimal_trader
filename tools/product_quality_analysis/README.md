# Product Quality Analysis

Thin orchestration around mature tooling for UI and data-quality checks in this repo. The toolkit uses:

- `playwright` for real browser crawling, screenshots, DOM metrics, and route timing
- `rich` + `typer` for CLI UX
- `pydantic` for structured snapshots
- `pandas` for artifact coverage checks
- `Pillow` for screenshot diffing
- `lighthouse`, `axe-core`, and `stylelint` when they are installed locally; otherwise those sections are skipped cleanly

## Purpose

The goal is to catch product failures that hurt trust and usability:

- missing key fields like `60d` and `120d` returns
- broken or empty insight panels
- slow routes that stop being usable as symbol counts grow
- tables that overload the DOM or readability budget
- inconsistent shells, spacing, typography, and page structure

## Install

Required baseline:

```bash
pip install typer rich pydantic pandas pillow playwright beautifulsoup4
python -m playwright install chromium
```

Optional integrations:

```bash
pip install great_expectations
pip install soda-core
npm install -D lighthouse
npm install -D @axe-core/playwright
npm install -D stylelint
```

Repo-local interpreter example for this workspace:

```bash
/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m pip install typer rich pydantic pandas pillow playwright beautifulsoup4
/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m playwright install chromium
```

## CLI

```bash
python -m tools.product_quality_analysis.cli crawl
python -m tools.product_quality_analysis.cli analyze --label baseline
python -m tools.product_quality_analysis.cli data-quality
python -m tools.product_quality_analysis.cli ui-consistency
python -m tools.product_quality_analysis.cli scalability
python -m tools.product_quality_analysis.cli rank-issues
python -m tools.product_quality_analysis.cli verify-fixes --baseline-label baseline --current-label after
python -m tools.product_quality_analysis.cli report --label after
```

Useful options:

- `--routes pipeline_ui,pipeline_stock_amd`
- `--output docs/product_quality`
- `--tiers tier1,tier2,tier3`
- `--visual-baseline-dir docs/product_quality/visual_baseline`
- `--base-url http://127.0.0.1:8000`

## Route Discovery

The tool combines:

- configured high-value routes in `config.py`
- artifact-backed tier routes discovered from `db.sqlite3`

That means the crawler can hit both general product pages and tier-specific scalability routes without hardcoding artifact ids into the CLI.

## Baseline and Verification Workflow

1. Start the local app.
2. Capture a baseline:

```bash
python -m tools.product_quality_analysis.cli analyze --label baseline
```

3. Implement product fixes.
4. Rerun after the changes:

```bash
python -m tools.product_quality_analysis.cli analyze --label after
python -m tools.product_quality_analysis.cli verify-fixes --baseline-label baseline --current-label after
```

## Outputs

Reports are written under `docs/product_quality/`:

- `product_quality_summary.md`
- `prioritized_issues.md`
- `data_quality_report.md`
- `ui_consistency_report.md`
- `pagination_report.md`
- `readability_report.md`
- `scalability_report.md`
- `fix_verification.md`

Machine-readable snapshots live under `docs/product_quality/snapshots/`.

## Issue Ranking

Issues are ranked with a weighted score over:

- severity
- trust impact
- frequency
- scalability risk
- usability impact
- implementation feasibility

The formula intentionally biases toward fixes that meaningfully improve trust and usability instead of cosmetic churn.

## How To Interpret The Reports

- `product_quality_summary.md`: short executive summary and top issues
- `prioritized_issues.md`: ranked fix queue with severity and evidence
- `data_quality_report.md`: field coverage and missing-display problems
- `ui_consistency_report.md`: cross-page shell and token drift
- `pagination_report.md`: large tables and missing paging
- `readability_report.md`: row/column overload
- `scalability_report.md`: tier growth, timeouts, and usability degradation
- `fix_verification.md`: before/after issue comparison

## How LLM Agents Should Use It

1. Run `analyze --label baseline`.
2. Start with `prioritized_issues.md` and verify the evidence in the JSON snapshot.
3. Prefer fixes that improve route usability, trust, and tier scalability at the same time.
4. After changes, rerun `analyze --label after` and `verify-fixes`.
5. Only declare a fix complete when the metric improves in `fix_verification.md`.
