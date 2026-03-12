# Memory Hotspots

- Engine: `tracemalloc`
- Target: `scalability_tier2`
- Peak RSS: `615.31 MB`
- Traced peak: `50.80 MB`

## Stage Hotspots

| Stage | RSS Delta MB | Wall (s) |
| --- | --- | --- |
| backtest.load_inputs | 5.969 | 0.108 |
| strategy.build_dataset | 5.781 | 0.163 |
| model.fit | 5.406 | 3.247 |
| labels.generate | 2.422 | 1.676 |
| model.serialize_predictions | 1.672 | 2.711 |
| features.serialize_artifact | 1.234 | 2.207 |
| features.load_adjusted_prices | 0.812 | 0.756 |
| model.score | 0.438 | 0.206 |
| features.compute_symbol_panel | 0.156 | 0.090 |
| features.compute_symbol_panel | 0.156 | 0.148 |
| features.compute_symbol_panel | 0.141 | 0.055 |
| features.concat_symbol_frames | 0.141 | 0.012 |

## Allocation Hotspots

| Path | Line | Size MB | Count |
| --- | --- | --- | --- |
| /Users/johnnylee/.local/lib/python3.13/site-packages/django/db/backends/sqlite3/operations.py | 181 | 1.8443 | 149 |
| data/historical_prices.py | 34 | 0.1668 | 1987 |
| /Users/johnnylee/miniconda3/envs/optimal_trader/lib/python3.13/json/decoder.py | 361 | 0.1386 | 2101 |
| /Users/johnnylee/.local/lib/python3.13/site-packages/django/db/utils.py | 101 | 0.1055 | 1740 |
| pipeline/service_runtime.py | 107 | 0.0792 | 624 |
| /Users/johnnylee/.local/lib/python3.13/site-packages/pandas/core/arrays/arrow/array.py | 829 | 0.0556 | 995 |
| pipeline/performance.py | 26 | 0.0535 | 242 |
| ml/models.py | 41 | 0.0211 | 363 |
| pipeline/performance.py | 91 | 0.0210 | 240 |
| pipeline/performance.py | 38 | 0.0210 | 240 |
| pipeline/performance.py | 80 | 0.0194 | 121 |
| pipeline/service_runtime.py | 109 | 0.0187 | 162 |
| /Users/johnnylee/.local/lib/python3.13/site-packages/django/db/backends/utils.py | 146 | 0.0102 | 167 |
| /Users/johnnylee/miniconda3/envs/optimal_trader/lib/python3.13/contextlib.py | 109 | 0.0090 | 79 |
| /Users/johnnylee/.local/lib/python3.13/site-packages/django/db/models/sql/compiler.py | 2166 | 0.0086 | 75 |
| /Users/johnnylee/.local/lib/python3.13/site-packages/django/db/backends/utils.py | 148 | 0.0075 | 170 |
| /Users/johnnylee/.local/lib/python3.13/site-packages/django/db/models/sql/compiler.py | 574 | 0.0073 | 153 |
| ml/artifact_datasets.py | 56 | 0.0071 | 91 |
| /Users/johnnylee/.local/lib/python3.13/site-packages/pandas/core/internals/blocks.py | 182 | 0.0058 | 114 |
| pipeline/performance.py | 141 | 0.0055 | 242 |

## Notes

- Fell back to tracemalloc because scalene/memory_profiler are not installed.
