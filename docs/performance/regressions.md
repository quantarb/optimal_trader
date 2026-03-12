# Regressions

| Workload | Tier | Baseline (s) | Current (s) | Delta (s) | Delta % | Class |
| --- | --- | --- | --- | --- | --- | --- |
| strategy_dataset | tier3 | 5.953 | 6.814 | 0.860 | 14.45 | regressed |
| backtest | tier2 | 0.097 | 0.111 | 0.014 | 14.11 | regressed |
| end_to_end | tier3 | 37.148 | 40.813 | 3.665 | 9.87 | regressed |
| label_generation | tier2 | 0.428 | 0.464 | 0.036 | 8.36 | regressed |
| backtest | tier3 | 0.745 | 0.802 | 0.057 | 7.62 | regressed |
| strategy_dataset | tier2 | 0.881 | 0.662 | -0.219 | -24.83 | improved |
| model_scoring | tier3 | 0.662 | 0.516 | -0.146 | -22.02 | improved |
| model_training | tier2 | 0.608 | 0.490 | -0.118 | -19.40 | improved |
| model_training | tier1 | 0.162 | 0.134 | -0.028 | -17.25 | improved |
| model_scoring | tier2 | 0.108 | 0.093 | -0.016 | -14.52 | improved |
| model_training | tier3 | 4.398 | 3.816 | -0.582 | -13.23 | improved |
| strategy_dataset | tier1 | 0.083 | 0.075 | -0.008 | -9.53 | improved |
| label_generation | tier1 | 0.049 | 0.044 | -0.004 | -9.17 | improved |
| model_scoring | tier1 | 0.041 | 0.039 | -0.003 | -6.44 | improved |
| backtest | tier1 | 0.015 | 0.014 | -0.001 | -5.23 | improved |
| feature_generation | tier1 | 0.186 | 0.177 | -0.009 | -4.60 | neutral |
| label_generation | tier3 | 4.221 | 4.058 | -0.163 | -3.85 | neutral |
| feature_generation | tier2 | 1.692 | 1.753 | 0.061 | 3.58 | neutral |
| end_to_end | tier1 | 0.743 | 0.720 | -0.023 | -3.05 | neutral |
| feature_generation | tier3 | 16.378 | 16.808 | 0.429 | 2.62 | neutral |
| end_to_end | tier2 | 4.516 | 4.569 | 0.053 | 1.17 | neutral |
