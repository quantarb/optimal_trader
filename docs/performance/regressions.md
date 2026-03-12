# Regressions

| Workload | Tier | Baseline (s) | Current (s) | Delta (s) | Delta % | Class |
| --- | --- | --- | --- | --- | --- | --- |
| backtest | tier1 | 0.015 | 0.020 | 0.005 | 35.28 | regressed |
| model_scoring | tier1 | 0.041 | 0.056 | 0.014 | 35.06 | regressed |
| label_generation | tier1 | 0.049 | 0.059 | 0.011 | 21.99 | regressed |
| end_to_end | tier1 | 0.743 | 0.890 | 0.148 | 19.88 | regressed |
| strategy_dataset | tier1 | 0.083 | 0.099 | 0.016 | 19.85 | regressed |
| model_scoring | tier2 | 0.108 | 0.120 | 0.012 | 10.70 | regressed |
| feature_generation | tier1 | 0.186 | 0.202 | 0.016 | 8.55 | regressed |
| strategy_dataset | tier2 | 0.881 | 0.605 | -0.276 | -31.35 | improved |
| model_training | tier2 | 0.608 | 0.537 | -0.071 | -11.61 | improved |
| model_scoring | tier3 | 0.662 | 0.587 | -0.075 | -11.30 | improved |
| label_generation | tier3 | 4.221 | 3.827 | -0.393 | -9.32 | improved |
| model_training | tier3 | 4.398 | 3.990 | -0.409 | -9.29 | improved |
| end_to_end | tier2 | 4.516 | 4.124 | -0.391 | -8.67 | improved |
| feature_generation | tier3 | 16.378 | 14.977 | -1.401 | -8.56 | improved |
| end_to_end | tier3 | 37.148 | 34.115 | -3.033 | -8.16 | improved |
| backtest | tier2 | 0.097 | 0.092 | -0.005 | -4.88 | neutral |
| strategy_dataset | tier3 | 5.953 | 5.696 | -0.257 | -4.32 | neutral |
| model_training | tier1 | 0.162 | 0.169 | 0.006 | 3.88 | neutral |
| feature_generation | tier2 | 1.692 | 1.632 | -0.060 | -3.55 | neutral |
| backtest | tier3 | 0.745 | 0.724 | -0.021 | -2.80 | neutral |
| label_generation | tier2 | 0.428 | 0.437 | 0.009 | 2.13 | neutral |
