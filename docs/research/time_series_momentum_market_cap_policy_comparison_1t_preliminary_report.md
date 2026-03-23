# In-Progress TSMOM Market-Cap Research Report

## Status

- Updated on March 12, 2026 while the clean real-data `1t,100b,10b` command is still running.
- Fully available today: clean `1t` no-filter folds `wf_2018` through `wf_2025`.
- Partially available today: clean `100b` universe/feature/label build plus the first `baseline_train` fold.
- Not available yet: final all-tier aggregate summary, combined filter diagnostics, `10b` results, and the final report at `docs/research/time_series_momentum_market_cap_policy_comparison_report.md`.

## Experiment Setup

- Universe family: US-listed stocks on the platform's predefined market-cap tier helpers.
- Tiers currently resolved from the live DB:
  - `1t`: `4` symbols
  - `100b`: `114` symbols
  - `10b`: `762` symbols
- Policies:
  - Baseline TSMOM signal using the existing paper formula and monthly long/short sign policy
  - Oracle-label RandomForestRegressor policy using the existing full feature artifact and `trade_return` labels
- Walk-forward design:
  - Train through December 31 of year `N`
  - Test out of sample during year `N+1`
- Backtest cost settings:
  - `fee_bps=2`
  - `slippage_bps=8`
  - `short_borrow_bps_annual=25`
  - `execution_delay_days=1`

## Universe Health Check

- Synthetic `TIER*` symbols remaining in the live FMP DB: `0`.
- Clean real `1t` membership in the current run: `GOOG`, `AVGO`, `BRK-B`, `BRK-A`.
- This means the partial results below are no longer contaminated by the old scalability fixtures.

## Performance So Far

Current reliable aggregate is the clean `1t` no-filter comparison.

| Universe | Policy | Variant | Mean Fold Sharpe | Total Return | Max DD | Turnover | Trades | Positive Fold Rate | Valid Fold Rate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1t | Baseline TSMOM | no_filter | 0.167 | -5.71% | -31.89% | 46.25 | 8005 | 37.5% | 100.0% |
| 1t | Oracle-label RF | no_filter | 0.000 | 481.97% | -12.19% | 0.00 | 8044 | 87.5% | 0.0% |

Notes:

- On raw compounded fold returns, the oracle-label model still dominates the baseline on the clean `1t` universe.
- Every clean `1t` model fold still failed the configured `min_trained_rows=200` gate. The model only trained on `83` to `111` rows per fold.
- The clean `1t` baseline passed the current validity gates in all `8` folds.
- The model fold summaries currently report `0.0` Sharpe in every clean `1t` fold JSON even while cumulative returns are strongly positive. Based on the emitted artifacts, that metric should be treated as provisional until we inspect why the model-level Sharpe output is flat.

## 1T Fold Detail

| Fold | Baseline Return | Baseline Sharpe | Model Return | Model Trained Rows | Model Passed Gates |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2018 | -2.33% | -0.032 | 2.33% | 83 | no |
| 2019 | -8.65% | -0.577 | 21.02% | 87 | no |
| 2020 | -23.66% | -0.657 | 19.16% | 91 | no |
| 2021 | 16.87% | 1.212 | 48.31% | 95 | no |
| 2022 | -4.76% | -0.149 | -12.19% | 99 | no |
| 2023 | 8.38% | 0.722 | 46.24% | 103 | no |
| 2024 | -3.20% | -0.083 | 51.07% | 107 | no |
| 2025 | 18.53% | 0.899 | 37.07% | 111 | no |

## Runtime So Far

Clean `1t` no-filter runtime:

| Universe | Policy | Total Runtime | Model Train | Model Score | Strategy Build | Backtest |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1t | Baseline TSMOM | 12.57s | 0.00s | 0.00s | 11.89s | 0.68s |
| 1t | Oracle-label RF | 27.65s | 0.47s | 7.05s | 13.57s | 0.89s |

Early clean `100b` runtime checkpoints:

| Universe | Stage | Time |
| --- | --- | ---: |
| 100b | universe build | 0.00s |
| 100b | feature artifact build | 225.81s |
| 100b | label artifact build | 6.62s |
| 100b | first baseline train strategy build (`wf_2018`) | 3175.52s |
| 100b | first baseline train backtest (`wf_2018`) | 27.72s |

Runtime observations:

- The clean `100b` feature artifact is already about `6.2 GB`.
- The first clean `100b` baseline strategy build took about `52.9` minutes by itself.
- That makes runtime scaling a first-class research finding already: moving from `1t` to broader universes increases compute cost much faster than the `1t` slice alone would suggest.

## Symbol Insights So Far

Clean `1t` no-filter aggregate symbol readout:

- Baseline strongest symbols by average symbol Sharpe:
  - `BRK-B` `0.3075`
  - `AVGO` `0.0759`
  - `BRK-A` `0.0679`
  - `GOOG` `0.0157`
- Model strongest symbols by average symbol Sharpe:
  - `AVGO` `1.1190`
  - `GOOG` `0.9876`
  - `BRK-B` `0.8436`
  - `BRK-A` `0.8320`

Interpretation:

- On the tiny clean `1t` universe, the model strength is broad rather than concentrated in just one name.
- Because `1t` only has four symbols, sector and industry filter interpretation is not very meaningful yet.

## Filter Status

- The full experiment is designed to evaluate `no_filter`, `profitable_filter`, and `beats_buy_hold_filter`.
- The final combined filter diagnostics have not been emitted yet because the all-tier command has not completed.
- For now, the most defensible partial comparison is the clean `1t` `no_filter` baseline versus model readout above.

## Interim Conclusions

- On the clean real-data `1t` universe, the oracle-label model still beats baseline TSMOM on raw compounded return, drawdown, and positive fold rate.
- That does not yet clear the research bar because the model fails the current training-row validity gate in every `1t` fold.
- The added model complexity is already more expensive even on `1t`, and the first clean `100b` fold shows that runtime cost escalates sharply on broader universes.
- The most important unanswered questions now are:
  - whether the model still wins once `100b` and `10b` complete
  - whether simple metadata filters add signal without breaking coverage
  - whether the performance gain, if it persists, is worth the very large runtime jump on broader universes

## Next Output To Expect

- Once the ongoing command finishes, the final combined report should be written to:
  - `docs/research/time_series_momentum_market_cap_policy_comparison_report.md`
- That final report should add:
  - all three universes
  - all filter variants
  - aggregate symbol diagnostics
  - filter diagnostics and interpretable tree rules
  - full runtime comparison across universes and variants
