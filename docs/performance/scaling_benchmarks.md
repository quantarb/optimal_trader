# Scaling Benchmarks

- Engine: `internal_perf_counter`
- Label: `tsmom_after_refactor2`
- Warmup iterations: `0`
- Measured iterations: `1`

| Workload | Tier | Median (s) | Mean (s) | StdDev (s) | Samples |
| --- | --- | --- | --- | --- | --- |
| backtest | tier1 | 0.014 | 0.014 | 0.000 | 0.014 |
| backtest | tier2 | 0.111 | 0.111 | 0.000 | 0.111 |
| backtest | tier3 | 0.802 | 0.802 | 0.000 | 0.802 |
| end_to_end | tier1 | 0.720 | 0.720 | 0.000 | 0.720 |
| end_to_end | tier2 | 4.569 | 4.569 | 0.000 | 4.569 |
| end_to_end | tier3 | 40.813 | 40.813 | 0.000 | 40.813 |
| feature_generation | tier1 | 0.177 | 0.177 | 0.000 | 0.177 |
| feature_generation | tier2 | 1.753 | 1.753 | 0.000 | 1.753 |
| feature_generation | tier3 | 16.808 | 16.808 | 0.000 | 16.808 |
| label_generation | tier1 | 0.044 | 0.044 | 0.000 | 0.044 |
| label_generation | tier2 | 0.464 | 0.464 | 0.000 | 0.464 |
| label_generation | tier3 | 4.058 | 4.058 | 0.000 | 4.058 |
| model_scoring | tier1 | 0.039 | 0.039 | 0.000 | 0.039 |
| model_scoring | tier2 | 0.093 | 0.093 | 0.000 | 0.093 |
| model_scoring | tier3 | 0.516 | 0.516 | 0.000 | 0.516 |
| model_training | tier1 | 0.134 | 0.134 | 0.000 | 0.134 |
| model_training | tier2 | 0.490 | 0.490 | 0.000 | 0.490 |
| model_training | tier3 | 3.816 | 3.816 | 0.000 | 3.816 |
| strategy_dataset | tier1 | 0.075 | 0.075 | 0.000 | 0.075 |
| strategy_dataset | tier2 | 0.662 | 0.662 | 0.000 | 0.662 |
| strategy_dataset | tier3 | 6.814 | 6.814 | 0.000 | 6.814 |
