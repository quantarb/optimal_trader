# Oracle Residual Label Experiment

## 1. Design

- Objective: compare the same oracle ranking pipeline trained on raw forward-return ranks versus factor-residualized forward-return ranks.
- Label horizon: 21 business days with monthly rebalance dates.
- Raw target: `future_rank_pct`, the cross-sectional percentile rank of `trade_return`.
- Residual target: `residual_rank_pct`, where `residual_return = trade_return - factor_expected_return` from a per-date cross-sectional regression.
- Residualization proxies: existing size, momentum, volatility features plus sector dummies and stock/ETF type from symbol metadata.
- Universe: 20 symbols.
- Symbols: GOOG, AVGO, BRK-B, BRK-A, WMT, LLY, JPM, XOM, V, JNJ, MA, COST, MU, ORCL, NFLX, ABBV, CVX, PG, PLTR, HD

## 2. Results

| Variant | Sharpe | Total Return | Max DD | Turnover | Trades | IC vs Raw Return | IC vs Residual Return | Positive Fold Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_momentum | 0.767 | 87.08% | -33.01% | 15.25 | 4065 | 0.089 | -0.012 | 0.75 |
| raw_label_model | -0.619 | -43.14% | -48.57% | 25.00 | 4104 | -0.012 | -0.013 | 0.25 |
| residual_label_model | -0.674 | -45.69% | -58.63% | 25.25 | 4104 | 0.025 | 0.015 | 0.25 |

## 3. Interpretation

- Momentum baseline: Sharpe 0.767, total return 87.08%.
- Raw-label model: Sharpe -0.619, total return -43.14%, raw-return IC -0.012.
- Residual-label model: Sharpe -0.674, total return -45.69%, residual-return IC 0.015.
- Conclusion: Raw labels helped: the profitable edge appears to rely on systematic factor exposures. Neither ML variant beat the simple momentum baseline in this pilot.

## 4. Artifacts

- Summary JSON: `data/pipeline_artifacts/oracle_residual_experiment/summary.json`
- Comparison summary CSV: `data/pipeline_artifacts/oracle_residual_experiment/comparison_summary.csv`
- Fold results CSV: `data/pipeline_artifacts/oracle_residual_experiment/fold_results.csv`
- Prediction diagnostics CSV: `data/pipeline_artifacts/oracle_residual_experiment/prediction_diagnostics.csv`
- Aggregate prediction diagnostics CSV: `data/pipeline_artifacts/oracle_residual_experiment/prediction_diagnostics_aggregate.csv`
