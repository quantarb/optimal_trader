# Runtime Hotspots

- Engine: `performance_tracer`
- Target: `tsmom_market_cap_policy_comparison__1t`
- Total runtime: `512.801s`

## Stage Hotspots

| Stage | Wall (s) | CPU (s) | RSS Delta MB |
| --- | --- | --- | --- |
| model.train.1t.wf_2025 | 39.915 | 39.635 | -1367.094 |
| model.train.1t.wf_2024 | 38.436 | 38.104 | -1023.703 |
| model.train.1t.wf_2023 | 36.578 | 36.340 | -1160.188 |
| model.train.1t.wf_2022 | 35.022 | 34.848 | -546.016 |
| model.train.1t.wf_2021 | 33.972 | 33.770 | -916.516 |
| model.train.1t.wf_2020 | 33.937 | 33.551 | -91.047 |
| model.train.1t.wf_2019 | 33.251 | 31.759 | -1138.484 |
| model.train.1t.wf_2018 | 31.860 | 31.354 | 237.297 |
| baseline.train.1t.wf_2025 | 19.312 | 19.194 | -55.578 |
| baseline.train.1t.wf_2024 | 18.452 | 18.401 | -34.375 |
| baseline.train.1t.wf_2023 | 17.693 | 17.608 | 89.625 |
| baseline.train.1t.wf_2022 | 17.100 | 17.044 | -289.828 |

## Notes

- Generated from the experiment's existing PerformanceTracer stages and rendered with tools/performance_analysis.
