# Runtime Hotspots

- Engine: `performance_tracer`
- Target: `tsmom_market_cap_policy_comparison__1t`
- Total runtime: `525.582s`

## Stage Hotspots

| Stage | Wall (s) | CPU (s) | RSS Delta MB |
| --- | --- | --- | --- |
| model.train.1t.wf_2025 | 44.750 | 44.084 | -923.031 |
| model.train.1t.wf_2024 | 40.007 | 39.294 | -1320.484 |
| model.train.1t.wf_2023 | 39.729 | 39.212 | -744.219 |
| model.train.1t.wf_2022 | 37.857 | 37.433 | -624.078 |
| model.train.1t.wf_2021 | 36.633 | 36.144 | -799.438 |
| model.train.1t.wf_2020 | 35.438 | 35.067 | -958.531 |
| model.train.1t.wf_2019 | 34.320 | 33.903 | -438.062 |
| model.train.1t.wf_2018 | 33.019 | 32.496 | 108.375 |
| baseline.train.1t.wf_2025 | 20.501 | 20.344 | -784.078 |
| baseline.train.1t.wf_2024 | 18.762 | 18.700 | -594.438 |
| baseline.train.1t.wf_2023 | 18.466 | 18.371 | -56.641 |
| baseline.train.1t.wf_2022 | 18.091 | 17.955 | -461.500 |

## Notes

- Generated from the experiment's existing PerformanceTracer stages and rendered with tools/performance_analysis.
