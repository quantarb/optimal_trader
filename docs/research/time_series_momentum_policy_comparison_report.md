# Time Series Momentum Policy Comparison Report

## 1. Experiment setup

- Baseline policy: direct TSMOM sign signal using `(1 + px__ret_252_d) / (1 + px__ret_21_d) - 1`, monthly rebalanced long/short.
- Model policy: random-forest regressor trained on oracle `trade_return` labels with the platform's full feature artifact, converted into a monthly sign long/short policy.
- Universe: GOOG, AVGO, BRK-B, BRK-A, WMT, LLY, JPM, XOM, V, JNJ, MA, COST, MU, ORCL, NFLX, ABBV, CVX, PG, PLTR, HD, BAC, GE, KO, AMD
- Evaluation: yearly walk-forward folds with training through the prior December 31 and out-of-sample testing in the next calendar year.
- Filters tested per strategy: `no_filter`, `simple_filter` (top training-window historical performers), and `learned_filter` (shallow symbol-profitability model trained on training-window symbol summaries).

## 2. Walk-forward comparison

| Variant | Sharpe | Total Return | Max DD | Turnover | Trades | Mean Selected Symbols | Positive Fold Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline__no_filter | 0.234 | 20.60% | -40.25% | 46.91 | 47372 | 24.0 | 0.75 |
| baseline__simple_filter | -0.038 | -5.91% | -26.14% | 24.03 | 24034 | 12.0 | 0.62 |
| baseline__learned_filter | 0.117 | 5.29% | -23.93% | 24.08 | 24052 | 12.0 | 0.75 |
| model__no_filter | 1.188 | 451.70% | -31.65% | 4.04 | 47572 | 24.0 | 0.88 |
| model__simple_filter | 0.951 | 97.87% | -17.67% | 2.04 | 24132 | 12.0 | 0.88 |
| model__learned_filter | 1.003 | 106.54% | -19.08% | 2.04 | 24132 | 12.0 | 0.88 |

## 3. Baseline strategy results

- No filter: Sharpe 0.234, total return 20.60%, positive fold rate 0.75.
- Simple filter vs no filter: no.
- Learned filter vs no filter: no.

## 4. Model strategy results

- No filter: Sharpe 1.188, total return 451.70%, positive fold rate 0.88.
- Model vs baseline without filter: yes.
- Simple filter vs model no filter: no.
- Learned filter vs model no filter: no.

## 5. Symbol-level performance analysis

- Strongest baseline symbols (aggregate no-filter test diagnostics): ORCL (Sharpe 0.59), GE (Sharpe 0.50), BAC (Sharpe 0.34), ABBV (Sharpe 0.30), MA (Sharpe 0.30)
- Strongest model symbols (aggregate no-filter test diagnostics): LLY (Sharpe 1.29), COST (Sharpe 1.23), PLTR (Sharpe 1.13), AVGO (Sharpe 1.12), GOOG (Sharpe 0.99)
- Most frequently selected by the baseline simple filter: ABBV (8), AMD (8), JNJ (8), MA (8), MU (8)
- Most frequently selected by the model learned filter: BRK-A (8), CVX (8), GOOG (8), JNJ (8), LLY (8)

## 6. Interpretation

- Does the model beat the simple signal? yes.
- Does filtering help the baseline? simple=no, learned=no.
- Does filtering help the model? simple=no, learned=no.
- Are gains consistent across folds? Use positive-fold rate plus fold Sharpe dispersion in the table above; improvements that raise Sharpe but lower positive-fold rate are treated as mixed rather than robust.
- Does added complexity pay for itself? The answer should follow the out-of-sample Sharpe and stability deltas, not in-sample training diagnostics.

## 7. Platform capabilities added

- Reusable symbol-subset backtests via `allowed_symbols` in `StrategyBacktestSpec` and the strategy backtest workflow.
- Reusable symbol-level strategy diagnostics in `pipeline/symbol_diagnostics.py`.
- Reusable symbol filtering helpers in `pipeline/symbol_filters.py`.
- A reusable research comparison runner for baseline-vs-model TSMOM policy studies in `pipeline/time_series_momentum_policy_comparison.py`.

## 8. Output artifacts

- Summary JSON: `data/pipeline_artifacts/time_series_momentum_policy_comparison_run1.json`
- Summary CSV: `data/pipeline_artifacts/time_series_momentum_policy_comparison_run1.csv`
- Symbol diagnostics (test folds): `data/pipeline_artifacts/time_series_momentum_policy_comparison_run1__symbol_diagnostics_test.csv`
- Symbol diagnostics (aggregate): `data/pipeline_artifacts/time_series_momentum_policy_comparison_run1__symbol_diagnostics_aggregate.csv`
