# Optimization Changes

## Implemented Changes

1. `data/historical_prices.py`
   Reworked batched adjusted-price loading to use `values_list()` instead of materializing Django model objects, deduplicated rows during load, and switched the intermediate container from `list[dict]` to compact record tuples.

2. `workflows/labels.py`
   Stopped copying cached price frames for every symbol during label generation and used `values_list()` on the database fallback path.

3. `ml/execution.py`
   Removed the redundant full feature-artifact read that happened before model training. Training now builds the joined dataset once instead of scanning the feature artifact to recover symbol names first.

4. `tools/performance_analysis/`
   Fixed two measurement issues in the new tool: runtime/memory profiling now seeds the scalability fixture before measurement, and macOS RSS reporting no longer inflates peak memory by treating bytes as kilobytes.

## Benchmark Deltas

Measured with the same `baseline` and `after` benchmark targets from `tools/performance_analysis`:

| Workload | Tier | Baseline (s) | After (s) | Change |
| --- | --- | ---: | ---: | ---: |
| End-to-end | `tier2` | 4.516 | 4.124 | -8.67% |
| End-to-end | `tier3` | 37.148 | 34.115 | -8.16% |
| Feature generation | `tier3` | 16.378 | 14.977 | -8.56% |
| Label generation | `tier3` | 4.221 | 3.827 | -9.32% |
| Model training | `tier2` | 0.608 | 0.537 | -11.61% |
| Model training | `tier3` | 4.398 | 3.990 | -9.29% |

Small `tier1` regressions remain in the report, but those workloads are sub-second and were measured with a single sample, so they are dominated by fixed overhead and normal run-to-run variance much more than the higher-scale tiers.

## Remaining Bottlenecks

- `labels.generate` is still the hottest traced stage on the `tier2` profile target.
- `features.load_adjusted_prices` and the downstream technical-feature path in `workflows/feature_runtime.py` still dominate feature-generation time.
- `pipeline/services.py` remains central in the call graph and is still prominent in function-level runtime output because the pipeline job orchestration wraps every stage.
