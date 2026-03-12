# Scaling Benchmarks

- Engine: `internal_perf_counter`
- Label: `after`
- Warmup iterations: `0`
- Measured iterations: `1`

| Workload | Tier | Median (s) | Mean (s) | StdDev (s) | Samples |
| --- | --- | --- | --- | --- | --- |
| backtest | tier1 | 0.020 | 0.020 | 0.000 | 0.020 |
| backtest | tier2 | 0.092 | 0.092 | 0.000 | 0.092 |
| backtest | tier3 | 0.724 | 0.724 | 0.000 | 0.724 |
| end_to_end | tier1 | 0.890 | 0.890 | 0.000 | 0.890 |
| end_to_end | tier2 | 4.124 | 4.124 | 0.000 | 4.124 |
| end_to_end | tier3 | 34.115 | 34.115 | 0.000 | 34.115 |
| feature_generation | tier1 | 0.202 | 0.202 | 0.000 | 0.202 |
| feature_generation | tier2 | 1.632 | 1.632 | 0.000 | 1.632 |
| feature_generation | tier3 | 14.977 | 14.977 | 0.000 | 14.977 |
| label_generation | tier1 | 0.059 | 0.059 | 0.000 | 0.059 |
| label_generation | tier2 | 0.437 | 0.437 | 0.000 | 0.437 |
| label_generation | tier3 | 3.827 | 3.827 | 0.000 | 3.827 |
| model_scoring | tier1 | 0.056 | 0.056 | 0.000 | 0.056 |
| model_scoring | tier2 | 0.120 | 0.120 | 0.000 | 0.120 |
| model_scoring | tier3 | 0.587 | 0.587 | 0.000 | 0.587 |
| model_training | tier1 | 0.169 | 0.169 | 0.000 | 0.169 |
| model_training | tier2 | 0.537 | 0.537 | 0.000 | 0.537 |
| model_training | tier3 | 3.990 | 3.990 | 0.000 | 3.990 |
| strategy_dataset | tier1 | 0.099 | 0.099 | 0.000 | 0.099 |
| strategy_dataset | tier2 | 0.605 | 0.605 | 0.000 | 0.605 |
| strategy_dataset | tier3 | 5.696 | 5.696 | 0.000 | 5.696 |
