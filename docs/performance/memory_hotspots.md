# Memory Hotspots

- Engine: `tracemalloc`
- Target: `scalability_tier2`
- Peak RSS: `672.16 MB`
- Traced peak: `58.52 MB`

## Stage Hotspots

| Stage | RSS Delta MB | Wall (s) |
| --- | --- | --- |
| model.score | 12.656 | 0.183 |
| strategy.build_dataset | 8.125 | 0.104 |
| backtest.load_inputs | 8.016 | 0.081 |
| features.serialize_artifact | 7.906 | 1.883 |
| labels.generate | 3.953 | 1.082 |
| model.fit | 1.891 | 2.906 |
| strategy.serialize_dataset | 1.312 | 4.289 |
| model.serialize_predictions | 0.547 | 2.320 |
| features.compute_symbol_panel | 0.125 | 0.037 |
| features.load_adjusted_prices | 0.109 | 0.532 |
| features.compute_symbol_panel | 0.078 | 0.037 |
| features.compute_symbol_panel | 0.047 | 0.069 |

## Allocation Hotspots

| Path | Line | Size MB | Count |
| --- | --- | --- | --- |
| /Users/johnnylee/miniconda3/envs/optimal_trader/lib/python3.13/site-packages/django/db/backends/sqlite3/operations.py | 181 | 1.8482 | 143 |
| /Users/johnnylee/miniconda3/envs/optimal_trader/lib/python3.13/json/decoder.py | 361 | 0.2160 | 3496 |
| data/historical_prices.py | 34 | 0.1595 | 1900 |
| /Users/johnnylee/miniconda3/envs/optimal_trader/lib/python3.13/site-packages/django/db/utils.py | 101 | 0.1063 | 1754 |
| pipeline/service_runtime.py | 107 | 0.0779 | 614 |
| pipeline/performance.py | 26 | 0.0513 | 232 |
| ml/models.py | 41 | 0.0262 | 475 |
| pipeline/performance.py | 91 | 0.0201 | 230 |
| pipeline/performance.py | 38 | 0.0201 | 230 |
| pipeline/performance.py | 80 | 0.0186 | 116 |
| pipeline/service_runtime.py | 109 | 0.0185 | 162 |
| /Users/johnnylee/miniconda3/envs/optimal_trader/lib/python3.13/site-packages/pandas/core/indexes/range.py | 1178 | 0.0177 | 2 |
| /Users/johnnylee/miniconda3/envs/optimal_trader/lib/python3.13/site-packages/pandas/core/array_algos/take.py | 155 | 0.0177 | 2 |
| /Users/johnnylee/miniconda3/envs/optimal_trader/lib/python3.13/site-packages/pandas/io/parsers/c_parser_wrapper.py | 93 | 0.0130 | 255 |
| /Users/johnnylee/miniconda3/envs/optimal_trader/lib/python3.13/site-packages/pandas/core/indexes/base.py | 2764 | 0.0109 | 228 |
| /Users/johnnylee/miniconda3/envs/optimal_trader/lib/python3.13/site-packages/django/db/backends/utils.py | 146 | 0.0098 | 161 |
| /Users/johnnylee/miniconda3/envs/optimal_trader/lib/python3.13/contextlib.py | 109 | 0.0092 | 80 |
| /Users/johnnylee/miniconda3/envs/optimal_trader/lib/python3.13/site-packages/django/db/models/sql/compiler.py | 574 | 0.0089 | 186 |
| /Users/johnnylee/miniconda3/envs/optimal_trader/lib/python3.13/site-packages/pandas/core/dtypes/cast.py | 598 | 0.0079 | 180 |
| /Users/johnnylee/miniconda3/envs/optimal_trader/lib/python3.13/site-packages/django/db/models/sql/compiler.py | 2166 | 0.0079 | 69 |

## Notes

- Fell back to tracemalloc because scalene/memory_profiler are not installed.
