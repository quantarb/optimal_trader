# Complexity Hotspots

- Engine: `radon`
- Files analyzed: `332`

## Modules

| Path | Complexity | Rank | MI | LOC | Score |
| --- | --- | --- | --- | --- | --- |
| analysis/alpha_flavors.py | 111.00 | F | 0.00 | 1065 | 470.76 |
| pipeline/artifact_support.py | 90.00 | F | 0.00 | 401 | 396.33 |
| analysis/oracle_reports.py | 87.00 | F | 2.49 | 518 | 387.64 |
| pipeline/experiments.py | 70.00 | F | 23.75 | 133 | 342.82 |
| pipeline/research_suite.py | 70.00 | F | 1.82 | 674 | 325.24 |
| pipeline/cohort_runner.py | 67.00 | F | 0.00 | 928 | 303.82 |
| features/views.py | 68.00 | F | 0.00 | 796 | 296.55 |
| analysis/insights.py | 44.00 | F | 14.70 | 443 | 261.63 |
| fmp/views.py | 58.00 | F | 0.00 | 2269 | 258.82 |
| pipeline/service_jobs_modeling.py | 51.00 | F | 25.25 | 331 | 255.30 |
| analysis/diagnostics.py | 51.00 | F | 8.84 | 322 | 244.18 |
| pipeline/views_artifacts.py | 46.00 | F | 10.73 | 357 | 243.15 |
| pipeline/views_reports.py | 47.00 | F | 18.35 | 247 | 236.82 |
| pipeline/service_jobs_data.py | 45.00 | F | 11.49 | 512 | 217.90 |
| domain/trades/operations.py | 45.00 | F | 15.84 | 250 | 217.42 |
| pipeline/views_workbench.py | 43.00 | F | 18.64 | 372 | 216.82 |
| pipeline/services.py | 45.00 | F | 15.71 | 605 | 216.18 |
| pipeline/management/commands/run_market_insight_reasoning.py | 41.00 | F | 38.23 | 105 | 213.26 |
| workflows/modeling.py | 43.00 | F | 29.00 | 167 | 211.77 |
| pipeline/progress.py | 45.00 | F | 18.24 | 220 | 208.15 |
| analysis/feature_attribution.py | 40.00 | E | 18.93 | 267 | 200.51 |
| analysis/insight_composer.py | 38.00 | E | 26.38 | 215 | 194.22 |
| pipeline/management/commands/run_market_situation_clustering.py | 35.00 | E | 37.97 | 100 | 183.27 |
| analysis/situation_clustering.py | 38.00 | E | 20.57 | 329 | 180.26 |
| backtest/strategies/stateful.py | 37.00 | E | 27.95 | 289 | 171.63 |

## Functions

| Path | Function | Complexity | Rank | Score |
| --- | --- | --- | --- | --- |
| analysis/alpha_flavors.py | cluster_alpha_flavors | 111.00 | F | 555.00 |
| pipeline/artifact_support.py | _build_equity_curve_context | 90.00 | F | 450.00 |
| analysis/oracle_reports.py | build_oracle_trade_report | 87.00 | F | 447.19 |
| pipeline/research_suite.py | _build_report_summary | 70.00 | F | 362.27 |
| pipeline/experiments.py | expand_model_cohort_configs | 70.00 | F | 359.53 |
| features/views.py | _build_feature_preview_result | 68.00 | F | 340.00 |
| pipeline/cohort_runner.py | run_model_cohort_backtests | 67.00 | F | 335.00 |
| fmp/views.py | symbol_detail | 58.00 | F | 290.00 |
| analysis/diagnostics.py | build_diagnostic_report | 51.00 | F | 266.40 |
| pipeline/service_jobs_modeling.py | execute_fit_model | 51.00 | F | 264.34 |
| analysis/diagnostics.py | _build_recommendations | 49.00 | F | 256.40 |
| pipeline/cohort_runner.py | _aggregate_walk_forward_rows | 50.00 | F | 250.00 |
| pipeline/views_reports.py | pipeline_cohorts_view | 47.00 | F | 245.21 |
| pipeline/research_suite.py | run_optimal_trade_research_suite | 46.00 | F | 242.27 |
| pipeline/views_artifacts.py | backtest_detail | 46.00 | F | 241.16 |
| pipeline/service_jobs_data.py | _build_label_statistics | 45.00 | F | 236.06 |
| pipeline/services.py | _execute_rl_train | 45.00 | F | 235.54 |
| domain/trades/operations.py | build_label_statistics | 45.00 | F | 235.52 |
| pipeline/progress.py | ProgressReporter.update | 45.00 | F | 235.22 |
| pipeline/cohort_runner.py | run_walk_forward_model_cohort_backtests | 47.00 | F | 235.00 |
| analysis/insights.py | build_stock_intelligence | 44.00 | F | 230.66 |
| pipeline/views_workbench.py | pipeline_lab_view | 43.00 | F | 225.17 |
| workflows/modeling.py | build_model_training_spec | 43.00 | F | 223.88 |
| pipeline/management/commands/run_market_insight_reasoning.py | Command.handle | 41.00 | F | 212.72 |
| analysis/feature_attribution.py | run_feature_family_attribution_suite | 40.00 | E | 210.13 |
