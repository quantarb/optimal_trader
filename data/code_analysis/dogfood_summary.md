# Dogfood Summary

## Commands Executed

- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m compileall tools/code_analysis tests`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis generate_architecture_report --root . --output data/code_analysis`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis capture_quality_snapshot --root . --output data/code_analysis --label baseline`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m unittest discover -s tests/tools/code_analysis -p 'test_*.py' -q`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis capture_quality_snapshot --root . --output data/code_analysis --label after`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis compare_quality_snapshots baseline after --output data/code_analysis`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis capture_quality_snapshot --root . --output data/code_analysis --label after` (rerun after adding blast-radius/refactor-priority reports)
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m unittest tests.test_tool_driven_refactors -v`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis capture_quality_snapshot --label refactor_pass_2`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis compare_quality_snapshots after refactor_pass_2`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis capture_quality_snapshot --label self_improve_loop1`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis compare_quality_snapshots refactor_pass_2 self_improve_loop1`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis capture_quality_snapshot --label self_improve_loop2`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis compare_quality_snapshots self_improve_loop1 self_improve_loop2`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis capture_quality_snapshot --label self_improve_loop3`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis compare_quality_snapshots self_improve_loop2 self_improve_loop3`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis capture_quality_snapshot --label self_improve_loop4`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis compare_quality_snapshots self_improve_loop3 self_improve_loop4`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis capture_quality_snapshot --label self_improve_loop5`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis compare_quality_snapshots self_improve_loop4 self_improve_loop5`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis compare_quality_snapshots refactor_pass_2 self_improve_loop5`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis capture_quality_snapshot --label self_improve_loop6`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis compare_quality_snapshots self_improve_loop5 self_improve_loop6`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis capture_quality_snapshot --label self_improve_loop7`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis compare_quality_snapshots self_improve_loop6 self_improve_loop7`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis compare_quality_snapshots self_improve_loop5 self_improve_loop7`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis capture_quality_snapshot --label self_improve_loop8`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis compare_quality_snapshots self_improve_loop7 self_improve_loop8`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis capture_quality_snapshot --label self_improve_loop9`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis compare_quality_snapshots self_improve_loop8 self_improve_loop9`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis capture_quality_snapshot --label self_improve_loop10`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis compare_quality_snapshots self_improve_loop9 self_improve_loop10`
- `/Users/johnnylee/miniconda3/envs/optimal_trader/bin/python -m tools.code_analysis compare_quality_snapshots self_improve_loop5 self_improve_loop10`

## What The First Run Revealed

- The end-to-end integration worked and produced all required reports plus a usable baseline snapshot.
- The first `baseline` snapshot scored the repo at `65.72`.
- Anti-pattern noise was too high at `1502` findings.
- The noisiest heuristics were:
  - `possible N+1 expensive calls inside loops`: `309`
  - `mixed concerns modules`: `162`
  - `hidden side effects`: `96`
- Good-pattern recall was strong, but the scorecard proxies were too punitive on large modules:
  - `good_patterns`: `1554`
  - `llm_editability`: `48.51`
  - `change_safety`: `28.20`
- Architecture validation already surfaced real repo issues instead of pure noise:
  - `30` rule violations
  - notable boundary leaks from `domain.*` into `analysis.*`, `features.*`, and `data.*`

## Changes Made To Improve The Analyzer

- Tightened the expensive-call heuristic so cheap collection helpers like `.get()` and `.append()` no longer trigger N+1 findings.
- Narrowed hidden-side-effect detection so local container mutation is not treated like external I/O or global state mutation.
- Raised the mixed-concern threshold to require both broader concern spread and a stronger mixing score.
- Switched pattern aggregation in the metrics layer from symbol guessing to path-based grouping so module counts are accurate for methods and classes.
- Softened the `llm_editability`, `change_safety`, and `complexity_health` proxy formulas so large modules do not collapse to zero by default.
- Improved snapshot comparison so raw inventory counts are ignored and newly added modules do not masquerade as score improvements.

## What The Final Run Revealed

- The `after` snapshot scored the repo at `71.80`, up `+6.08`.
- Anti-pattern findings dropped from `1502` to `1107`.
- The biggest noise reduction came from the targeted heuristics:
  - `possible N+1 expensive calls inside loops`: `309 -> 61`
  - `mixed concerns modules`: `162 -> 54`
  - `hidden side effects`: `96 -> 56`
- Architecture findings stayed stable at `30`, which is a good sign that the improvements removed noise without hiding actual boundary problems.
- The new change-impact layer identified the most interconnected modules, the highest-risk modules to touch, and the safest high-value refactor targets.
- Repo-level scorecard deltas:
  - `anti_pattern_burden`: `38.63 -> 46.87`
  - `llm_editability`: `48.51 -> 64.43`
  - `change_safety`: `28.20 -> 50.84`
  - `complexity_health`: `27.02 -> 44.53`
- Blast-radius highlights from the final repo run:
  - top blast-radius module: `fmp.models`
  - top highest-risk module to change: `pipeline.cohort_runner`
  - top safest high-value refactor: `pipeline.models`
- Some comparison regressions reflect the repository growing during implementation, especially the new analyzer test files.

## What The Latest Self-Improvement Passes Revealed

- The latest snapshot is `self_improve_loop10`, with repo score `72.59`.
- The latest cumulative comparison against the start of the second self-improvement batch is `quality_comparison_self_improve_loop5_vs_self_improve_loop10.*`, for an overall score delta of `+0.10`.
- Measured improvements over that batch:
  - `artifact_boundary_usage`: `119 -> 121`
  - `config_object_usage`: `109 -> 110`
  - `dependency_health`: `90.63 -> 90.85`
  - `complexity_health`: `45.94 -> 46.12`
  - `llm_editability`: `65.72 -> 65.90`
  - `repo_score`: `72.49 -> 72.59`
- The last single pass (`self_improve_loop9 -> self_improve_loop10`) still improved the score:
  - `repo_score`: `72.57 -> 72.59`
  - `anti_pattern_burden`: `1084 -> 1083`
- High-value cleanup that landed during the self-improvement loops:
  - `tools.product_quality_analysis.cli` was split into `snapshot_support.py` and `reporting_support.py` and no longer shows an anti-pattern finding.
  - `workflows.strategy` shed its `possible N+1 expensive calls inside loops` finding after signal-building helpers moved into `strategy_signal_support.py`.
  - `analysis.diagnostics` was split so RL and reporting responsibilities moved into `diagnostic_rl.py` and `diagnostic_reporting.py`.
  - `pipeline.artifact_support` now delegates backtest/equity summarization through `artifact_backtest_support.py`.
  - `analysis.market_insight_schema` dropped to three findings and now uses smaller input-assembly helpers.

## Top 10 Findings In This Repo

1. `analysis.alpha_flavors.cluster_alpha_flavors` is a `428` line function with estimated complexity `98`, deep nesting, and nested loops.
2. `analysis.oracle_reports.build_oracle_trade_report` is a `300` line reporting/orchestration function with nested loops and repeated workflow logic.
3. `analysis.insights.build_stock_intelligence` is a `202` line mixed-concern function in a mixed-concern module.
4. `analysis.diagnostics.build_diagnostic_report` is a `166` line mixed-concern function with nested loops.
5. `analysis.feature_attribution._enrich_rows_with_oracle_coverage` still shows a credible N+1 pattern around `Artifact.objects.filter.order_by`.
6. Duplicate workflow cluster `5` spans `5` members across `5` modules in the feature-attribution/reporting area.
7. `domain.features.panel` violates the architecture rules by importing `analysis.feature_embeddings.*` and `features.naming`.
8. `domain.labels.directional` violates the architecture rules by importing `data.schema`.
9. `utils.workflow` violates shared-layer boundaries by importing `fmp.models` and `pipeline.models`.
10. `pipeline.time_series_momentum_market_cap_policy_comparison` remains the lowest scoring module at `40.89`, driven by large size, duplicate clusters, and weak editability.

## Top 5 High-Value Refactors Suggested

1. Split `analysis.alpha_flavors.cluster_alpha_flavors` into precomputation, clustering, scoring, and reporting helpers.
2. Introduce a domain-facing boundary or adapter so `domain.features.panel` and `domain.labels.directional` stop importing application/infrastructure modules directly.
3. Extract the duplicated attribution/report workflow cluster around `analysis.feature_attribution` into a reusable pipeline stage or helper module.
4. Break `analysis.diagnostics`, `analysis.insights`, and `analysis.oracle_reports` into separate orchestration, data-shaping, and rendering/report modules.
5. Decompose `pipeline.time_series_momentum_market_cap_policy_comparison` into smaller build/fit/report entrypoints with explicit boundary objects.

## Top 10 Highest-Blast-Radius Modules

1. `fmp.models`
2. `pipeline.models`
3. `ml.base`
4. `domain.models.specs`
5. `pipeline.service_runtime`
6. `domain.models.interfaces`
7. `ml.execution`
8. `data`
9. `pipeline.cohort_runner`
10. `domain.models.datasets`

## Top 10 Safest High-Value Refactors

1. `pipeline.models`
2. `fmp.models`
3. `fmp.endpoints.base`
4. `tools.product_quality_analysis.cli`
5. `pipeline.artifact_support`
6. `ml.model_runtime`
7. `analysis.diagnostics`
8. `analysis.alpha_flavors`
9. `workflows.strategy`
10. `analysis.market_insight_schema`

## Top 10 Highest-Risk Modules To Change

1. `pipeline.cohort_runner`
2. `pipeline.characteristics_factor_model`
3. `pipeline.research_suite`
4. `pipeline.services`
5. `pipeline.strategy_definitions`
6. `pipeline.time_series_momentum_market_cap_policy_comparison`
7. `domain.features.panel`
8. `pipeline.experiments`
9. `pipeline.service_jobs_data`
10. `analysis.oracle_reports`

## Which Metrics Are Objective Vs Heuristic

- Mostly objective:
  - cyclomatic complexity summary
  - maintainability summary
  - function LOC distribution
  - class size distribution
  - nesting depth distribution
  - type hint coverage
  - dependency fan-in / fan-out
  - import cycle count
  - architecture rule violations
  - artifact boundary usage
- Deterministic but heuristic:
  - magic number count
  - duplicate workflow shapes
  - mixed concerns modules
  - broad exception swallowing
  - hidden side effects
  - good-pattern strengths
  - interface reuse count
  - config object usage
- Composite proxy scores:
  - quality scorecard
  - blast radius score
  - architectural badness score
  - estimated refactor leverage
  - LLM editability proxy score
  - change safety proxy score
- Advisory / model-assisted heuristics:
  - duplicate code clusters
  - dead code count
