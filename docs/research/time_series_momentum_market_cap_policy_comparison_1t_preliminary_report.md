# Preliminary 1T TSMOM Market-Cap Policy Comparison

## Status

- Generated on March 12, 2026 while the full `1t,100b,10b` experiment was still running.
- Scope: completed `1t` no-filter folds `wf_2018` through `wf_2025`.
- The full filter summary plus the `100b` and `10b` universes are still pending from the ongoing command.

## Critical Caveat

- The current `1t` universe is contaminated by synthetic scalability fixtures.
- Resolved `1t` symbols: `GOOG`, `AVGO`, `TIER10000`-`TIER10009`, `BRK-B`, `BRK-A`.
- That is 14 total symbols, 10 of which are synthetic `TIER*` records.
- This makes the current output useful as a platform diagnostic, but not yet a trustworthy real-US-stock research conclusion.

## Preliminary Performance Comparison

| Policy | Sharpe | Total Return | Max DD | Turnover | Trades | Positive Fold Rate | Valid Fold Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline TSMOM | 0.051 | -7.25% | -47.42% | 47.57 | 8707 | 37.5% | 100.0% |
| Oracle-label RF | 1.038 | 415.26% | -31.81% | 5.57 | 8966 | 87.5% | 0.0% |

Notes:

- The model policy materially outperformed the baseline on raw `1t` no-filter backtest outcomes.
- Every model fold failed the configured `min_trained_rows` validity gate because training rows only ranged from 83 to 111 versus the configured minimum of 200.
- The baseline folds all passed the configured validity gates.

## Fold Detail

| Fold | Baseline Return | Baseline Sharpe | Model Return | Model Sharpe | Model Trained Rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2018 | -2.33% | -0.032 | 2.33% | 0.216 | 83 |
| 2019 | -8.65% | -0.577 | 21.02% | 1.240 | 87 |
| 2020 | -23.66% | -0.657 | 19.16% | 0.676 | 91 |
| 2021 | 16.87% | 1.212 | 48.31% | 2.680 | 95 |
| 2022 | -4.76% | -0.149 | -12.19% | -0.363 | 99 |
| 2023 | 8.38% | 0.722 | 46.24% | 2.386 | 103 |
| 2024 | -4.78% | -0.208 | 33.76% | 1.828 | 107 |
| 2025 | 18.53% | 0.899 | 37.07% | 1.589 | 111 |

## Runtime Comparison

| Policy | Total Runtime | Model Train | Model Score | Strategy Build | Backtest |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline TSMOM | 10.030s | 0.000s | 0.000s | 9.410s | 0.620s |
| Oracle-label RF | 26.854s | 0.480s | 6.600s | 18.850s | 0.923s |

Notes:

- The model run took about 2.68x the baseline runtime on the current `1t` scope.
- Most of the extra cost came from prediction scoring and larger strategy-dataset construction, not from the RandomForest fit itself.

## Symbol-Level Readout

- Baseline top symbols were synthetic fixtures `TIER10000` through `TIER10009`, which is another direct sign that the current universe is polluted by test data.
- Model top real symbols on the completed `1t` run were:
  - `AVGO` with aggregate symbol Sharpe 1.10
  - `GOOG` with aggregate symbol Sharpe 0.93
  - `BRK-B` with aggregate symbol Sharpe 0.78
  - `BRK-A` with aggregate symbol Sharpe 0.76

## Interim Conclusions

- On the current contaminated `1t` universe, the oracle-label RF policy beat the baseline TSMOM signal decisively on raw backtest performance.
- That result is not yet trustworthy enough to answer the research question because:
  - the universe is contaminated by synthetic `TIER*` records
  - every model fold failed the configured minimum training-row gate
- The correct next step is to exclude synthetic scalability symbols from the market-cap test universes, rerun the study, and then generate the final all-tier report.
