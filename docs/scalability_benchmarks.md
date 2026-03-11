# Scalability Benchmarks

## Test Layers

Smoke tests:

```bash
conda activate optimal_trader
MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp python manage.py test pipeline.tests_mag7 --verbosity 1
```

Scalability tests:

```bash
conda activate optimal_trader
RUN_SCALABILITY_TESTS=1 SCALABILITY_TEST_TIERS=tier1,tier2,tier3 \
MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp python manage.py test pipeline.tests_scalability --verbosity 1
```

The scalability suite is skipped by default and only runs when `RUN_SCALABILITY_TESTS=1` is set.

## Benchmark Command

Real-data benchmark run:

```bash
conda activate optimal_trader
MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp python manage.py run_scalability_benchmarks \
  --tiers tier1,tier2,tier3 \
  --feature-profile baseline \
  --start-date 2020-01-01 \
  --end-date 2025-12-31 \
  --artifact-storage-format parquet \
  --max-tier2-runtime 180 \
  --output-dir docs/performance
```

Feature profiles:

- `baseline`: price technicals only, intended to scale like the Mag 7 smoke path.
- `full`: broad feature families enabled for deeper profiling.

Tier gating:

- `tier1`: `1T+` market cap, target `10` symbols.
- `tier2`: `100B+` market cap, target `100` symbols.
- `tier3`: `10B+` market cap, target `1,000` symbols.
- If `tier2` runtime exceeds `--max-tier2-runtime`, `tier3` is skipped automatically.

## Performance Improvements

### Implemented

- Batched adjusted-price loading across the full symbol set instead of per-symbol ORM reads.
- Raw section payload caching so feature builders reuse a batched section query.
- Fast payload timestamp normalization in `features.section_utils.payload_to_row`, removing the per-row `pd.to_datetime` hotspot.
- DataFrame-first feature assembly so the pipeline stops round-tripping large feature panels through `list[dict]`.
- Batched label generation with shared price frames.
- `O(k*n)` joint-trade dynamic program replacing the previous `O(n^2)` interval enumeration in `labels.strategy_solver`.
- Optional parquet artifact storage for large benchmark runs.
- Stage-level performance tracing with runtime, CPU, memory delta, and I/O byte counters.

### Measured Before / After

| Workload | Before | After | Improvement |
| --- | ---: | ---: | ---: |
| Labels, 10-symbol `1T+`, `YE:k=1` | 20.74s | 0.57s | 36.4x |
| Features, 2-symbol full profile | 17.07s | 1.78s | 9.6x |

## Real Benchmark Results

Reference run:

- Date window: `2020-01-01` to `2025-12-31`
- Feature profile: `baseline`
- Artifact storage: `parquet`
- Report: [docs/performance/scalability_20260311_202309.md](/Users/johnnylee/PycharmProjects/optimal_trader/docs/performance/scalability_20260311_202309.md)

| Tier | Actual Symbols | Runtime |
| --- | ---: | ---: |
| `tier1` | 10 | 1.606s |
| `tier2` | 100 | 10.188s |
| `tier3` | 863 | 78.327s |

Top bottlenecks from that run:

- `model.fit` dominated every tier.
- `labels.generate` remained the largest non-model stage at scale.
- `features.load_adjusted_prices` is still a material cost for large universes.
- Serialization is now visible but no longer dominant.

## Remaining Bottlenecks

- `model.fit` is the primary large-universe bottleneck and should be the next optimization target.
- `labels.generate` is much faster after the solver rewrite, but still scales linearly with symbol count.
- `features.load_adjusted_prices` still reads the entire requested price panel into memory; chunked reads or symbol partitions would help the `tier3` path.
- Strategy and prediction artifact serialization still create full materialized outputs; row-grouped parquet writes would reduce peak memory.

## Recommended Next Optimizations

- Add date-partitioned / symbol-partitioned feature and label execution for `tier3+` runs.
- Evaluate `HistGradientBoostingRegressor` or lighter RF settings for the benchmark profile so model fit stops dominating the suite.
- Push more feature families onto shared batched sparse loaders instead of per-builder reconstruction.
- Add optional `n_jobs` / worker-level partitioning for symbol-local feature families once correctness coverage is expanded.
