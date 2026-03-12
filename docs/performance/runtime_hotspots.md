# Runtime Hotspots

- Engine: `cProfile`
- Target: `scalability_tier2`
- Total runtime: `5.835s`

## Stage Hotspots

| Stage | Wall (s) | CPU (s) | RSS Delta MB |
| --- | --- | --- | --- |
| labels.generate | 0.602 | 0.544 | 5.969 |
| model.fit | 0.503 | 0.938 | 79.078 |
| strategy.serialize_dataset | 0.447 | 0.446 | 0.000 |
| features.load_adjusted_prices | 0.316 | 0.316 | 1.891 |
| model.serialize_predictions | 0.230 | 0.230 | 0.438 |
| features.serialize_artifact | 0.215 | 0.215 | 35.141 |
| model.score | 0.107 | 0.111 | 36.297 |
| strategy.build_dataset | 0.095 | 0.095 | 23.328 |
| backtest.load_inputs | 0.070 | 0.070 | 7.078 |
| features.compute_symbol_panel | 0.031 | 0.031 | 0.000 |
| features.compute_symbol_panel | 0.031 | 0.031 | 0.000 |
| features.compute_symbol_panel | 0.030 | 0.030 | 0.188 |

## Function Hotspots

| Path | Line | Function | Calls | Cum (s) | Total (s) | % |
| --- | --- | --- | --- | --- | --- | --- |
| pipeline/scalability.py | 75 | run_scalability_benchmark_suite | 1/1 | 5.707 | 0.000 | 97.80 |
| pipeline/scalability.py | 153 | run_scalability_benchmark | 1/1 | 5.707 | 0.000 | 97.80 |
| pipeline/scalability.py | 436 | _run_benchmark_job | 7/7 | 5.589 | 0.000 | 95.79 |
| pipeline/services.py | 426 | execute_pipeline_run | 7/7 | 5.586 | 0.000 | 95.72 |
| pipeline/services.py | 459 | ensure_job | 7/7 | 5.575 | 0.000 | 95.54 |
| pipeline/services.py | 263 | _run_job_executor | 7/7 | 5.551 | 0.001 | 95.13 |
| pipeline/service_jobs_data.py | 435 | execute_features | 1/1 | 3.058 | 0.001 | 52.40 |
| workflows/features.py | 18 | build_feature_panel_frame_for_symbols | 1/1 | 2.841 | 0.000 | 48.69 |
| workflows/feature_runtime.py | 567 | build_feature_panel_frame | 1/1 | 2.523 | 0.002 | 43.23 |
| workflows/feature_runtime.py | 422 | build_symbol_feature_result | 94/94 | 2.407 | 0.003 | 41.26 |
| workflows/feature_runtime.py | 275 | _add_price_features | 94/94 | 2.035 | 0.001 | 34.87 |
| features/feature_builders.py | 25 | build_price_technical_features | 94/94 | 1.785 | 0.002 | 30.59 |
| domain/features/technical.py | 213 | build_price_technical_features | 94/94 | 1.783 | 0.009 | 30.56 |
| domain/features/technical.py | 71 | compute_features_worldclass | 94/94 | 1.488 | 0.014 | 25.50 |
| pipeline/service_runtime.py | 138 | write_frame_artifact | 5/5 | 0.902 | 0.000 | 15.46 |
| pipeline/service_jobs_strategy.py | 62 | execute_build_strategy_dataset | 1/1 | 0.793 | 0.000 | 13.59 |
| data/historical_prices.py | 8 | load_adjusted_price_frames | 2/2 | 0.683 | 0.022 | 11.70 |
| pipeline/service_jobs_data.py | 288 | execute_labels | 1/1 | 0.607 | 0.000 | 10.41 |
| workflows/labels.py | 133 | build_oracle_labels | 1/1 | 0.602 | 0.000 | 10.31 |
| workflows/labels.py | 36 | build_trade_results | 1/1 | 0.597 | 0.050 | 10.23 |

## Notes

- Fell back to cProfile because pyinstrument is not installed.
