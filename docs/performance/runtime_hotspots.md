# Runtime Hotspots

- Engine: `cProfile`
- Target: `scalability_tier2`
- Total runtime: `9.279s`

## Stage Hotspots

| Stage | Wall (s) | CPU (s) | RSS Delta MB |
| --- | --- | --- | --- |
| labels.generate | 0.791 | 0.745 | 9.219 |
| model.fit | 0.563 | 0.863 | 63.203 |
| strategy.serialize_dataset | 0.520 | 0.516 | 0.422 |
| features.load_adjusted_prices | 0.406 | 0.404 | 1.297 |
| model.serialize_predictions | 0.264 | 0.261 | 0.000 |
| features.serialize_artifact | 0.260 | 0.254 | 27.625 |
| model.score | 0.115 | 0.122 | 39.859 |
| strategy.build_dataset | 0.109 | 0.109 | 7.062 |
| backtest.load_inputs | 0.085 | 0.084 | 13.359 |
| features.compute_symbol_panel | 0.050 | 0.039 | 0.000 |
| features.compute_symbol_panel | 0.033 | 0.033 | 0.891 |
| features.compute_symbol_panel | 0.032 | 0.032 | 0.047 |

## Function Hotspots

| Path | Line | Function | Calls | Cum (s) | Total (s) | % |
| --- | --- | --- | --- | --- | --- | --- |
| pipeline/scalability.py | 75 | run_scalability_benchmark_suite | 1/1 | 7.619 | 0.000 | 82.12 |
| pipeline/scalability.py | 153 | run_scalability_benchmark | 1/1 | 7.619 | 0.000 | 82.12 |
| pipeline/scalability.py | 436 | _run_benchmark_job | 7/7 | 7.498 | 0.000 | 80.81 |
| pipeline/services.py | 426 | execute_pipeline_run | 7/7 | 7.494 | 0.000 | 80.77 |
| pipeline/services.py | 459 | ensure_job | 7/7 | 7.481 | 0.000 | 80.63 |
| pipeline/services.py | 263 | _run_job_executor | 7/7 | 7.454 | 0.001 | 80.33 |
| pipeline/service_jobs_data.py | 435 | execute_features | 1/1 | 3.862 | 0.002 | 41.63 |
| workflows/features.py | 18 | build_feature_panel_frame_for_symbols | 1/1 | 3.599 | 0.002 | 38.79 |
| workflows/feature_runtime.py | 567 | build_feature_panel_frame | 1/1 | 3.188 | 0.004 | 34.36 |
| workflows/feature_runtime.py | 422 | build_symbol_feature_result | 99/99 | 3.017 | 0.004 | 32.52 |
| workflows/feature_runtime.py | 275 | _add_price_features | 99/99 | 2.499 | 0.001 | 26.93 |
| features/feature_builders.py | 25 | build_price_technical_features | 99/99 | 2.160 | 0.001 | 23.28 |
| domain/features/technical.py | 213 | build_price_technical_features | 99/99 | 2.159 | 0.014 | 23.27 |
| domain/features/technical.py | 71 | compute_features_worldclass | 99/99 | 1.684 | 0.016 | 18.15 |
| pipeline/service_jobs_strategy.py | 62 | execute_build_strategy_dataset | 1/1 | 1.667 | 0.001 | 17.97 |
| workflows/strategy.py | 226 | build_strategy_dataset_frame | 1/1 | 1.141 | 0.001 | 12.30 |
| pipeline/service_runtime.py | 138 | write_frame_artifact | 5/5 | 1.056 | 0.000 | 11.38 |
| pipeline/strategy_definitions.py | 181 | apply_strategy_definition | 1/1 | 0.999 | 0.006 | 10.77 |
| data/historical_prices.py | 8 | load_adjusted_price_frames | 2/2 | 0.865 | 0.025 | 9.32 |
| ml/frameworks/sklearn/__init__.py | 14 | __getattr__ | 2/2 | 0.832 | 0.000 | 8.97 |

## Notes

- Fell back to cProfile because pyinstrument is not installed.
