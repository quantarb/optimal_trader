# TSMOM Market-Cap Policy Comparison Report

## Status

- Generated on March 12, 2026 from all clean real-data outputs available after the experiment process stopped before final aggregation.
- This report is the best available final write-up from completed artifacts and pipeline-run records.
- The canonical all-tier summary JSON, CSV, filter diagnostics CSV, and runtime-analysis CSV were not written because the command exited before the final serialization step.

## Data Coverage

- Clean run coverage recovered for this report:
  - `1t`: full out-of-sample folds `wf_2018` through `wf_2025` for baseline, model, and both filter variants
  - `100b`: completed `wf_2018` baseline train, baseline test, `profitable_filter`, `beats_buy_hold_filter`, and model train
  - `10b`: no completed folds available
- Universe sizes resolved from the live DB at report time:
  - `1t`: `4` symbols
  - `100b`: `114` symbols
  - `10b`: `762` symbols
- Synthetic `TIER*` symbols remaining in the live FMP DB: `0`
- Data source note:
  - `1t` no-filter metrics were taken from fold JSONs and corrected with backtest-artifact runtime metrics where needed
  - completed filter variants were reconstructed from the latest successful backtest artifacts per exact run name
  - stale duplicate rows from earlier aborted attempts were ignored

## Experiment Setup

- Universes: existing predefined US market-cap tiers from the platform
- Policies:
  - Baseline TSMOM paper signal, monthly sign-based long/short policy
  - Oracle-label RandomForestRegressor policy using the full feature artifact and `trade_return` labels
- Filters:
  - `no_filter`
  - `profitable_filter`
  - `beats_buy_hold_filter`
- Walk-forward design:
  - train through December 31 of year `N`
  - test during year `N+1`
- Backtest costs:
  - `fee_bps=2`
  - `slippage_bps=8`
  - `short_borrow_bps_annual=25`
  - `execution_delay_days=1`

## Performance Comparison

### Complete Clean `1t` Out-of-Sample Results

| Universe | Policy | Variant | Sharpe | Return | Max DD | Turnover | Trades | Positive Fold Rate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1t | Baseline TSMOM | no_filter | 0.167 | -5.71% | -31.89% | 46.25 | 8005 | 37.5% |
| 1t | Baseline TSMOM | profitable_filter | 0.167 | -5.71% | -31.89% | 46.25 | 8005 | 37.5% |
| 1t | Baseline TSMOM | beats_buy_hold_filter | 0.167 | -5.71% | -31.89% | 46.25 | 8005 | 37.5% |
| 1t | Oracle-label RF | no_filter | 1.338 | 481.97% | -12.19% | 4.00 | 8044 | 87.5% |
| 1t | Oracle-label RF | profitable_filter | 1.338 | 481.97% | -12.19% | 4.00 | 8044 | 87.5% |
| 1t | Oracle-label RF | beats_buy_hold_filter | 1.338 | 481.97% | -12.19% | 4.00 | 8044 | 87.5% |

Notes:

- On clean raw out-of-sample `1t` returns, the model policy decisively beat baseline.
- The two `1t` filter variants were complete no-ops: each selected all four symbols in every fold, so they produced the same results as `no_filter`.
- Model fold JSONs left Sharpe and turnover blank, so those were recomputed from the completed backtest artifacts. Return, drawdown, and trade counts matched the saved fold summaries.
- Validity caveat:
  - baseline `1t` passed its fold gates
  - model `1t` failed the configured `min_trained_rows=200` gate in every fold, with only `83` to `111` trained rows

### Completed Clean `100b` Out-of-Sample Results So Far

Only `wf_2018` baseline-side out-of-sample variants completed before the process stopped.

| Universe | Policy | Variant | Scope | Sharpe | Return | Max DD | Turnover | Trades | Symbols Selected |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100b | Baseline TSMOM | no_filter | OOS `wf_2018` | -0.372 | -4.05% | -12.73% | 5.34 | 24957 | 114 |
| 100b | Baseline TSMOM | profitable_filter | OOS `wf_2018` | 0.458 | 1.15% | -1.97% | 0.84 | 4191 | 17 |
| 100b | Baseline TSMOM | beats_buy_hold_filter | OOS `wf_2018` | -0.587 | -0.21% | -0.53% | 0.12 | 504 | 5 |

Takeaways from completed `100b` OOS data:

- On the one completed `100b` out-of-sample fold, `profitable_filter` improved baseline materially.
- `beats_buy_hold_filter` reduced risk and activity sharply, but it did not improve return or Sharpe.
- There is not yet enough completed `100b` OOS data to say whether this holds across years.

### Completed Clean `100b` Model Result So Far

Only the `wf_2018` training-window model run completed before the process stopped.

| Universe | Policy | Variant | Scope | Sharpe | Return | Max DD | Turnover | Trades | Trained Rows |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100b | Oracle-label RF | no_filter | Train `wf_2018` | 0.565 | 2224.81% | -46.72% | 72.89 | 489497 | 3639 |

Important caveat:

- This is an in-sample training-window result, not an out-of-sample test fold.
- It is useful for runtime and capacity context, but it cannot answer whether the model beats baseline on `100b` out of sample.

## Runtime Comparison

### Complete Clean `1t` Runtime

| Universe | Policy | Variant | Total Runtime | Model Train | Model Score | Strategy Build | Backtest |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1t | Baseline TSMOM | no_filter | 12.57s | 0.00s | 0.00s | 11.89s | 0.68s |
| 1t | Oracle-label RF | no_filter | 27.65s | 0.47s | 7.05s | 13.57s | 0.89s |

Interpretation:

- On the tiny `1t` universe, the model is already about `2.2x` slower than baseline.
- Most of that extra cost comes from scoring and strategy-dataset construction, not from the RandomForest fit itself.

### Completed Clean `100b` Runtime Checkpoints

| Universe | Stage | Runtime |
| --- | --- | ---: |
| 100b | universe build | 0.00s |
| 100b | feature artifact build | 225.81s |
| 100b | label artifact build | 6.62s |
| 100b | baseline train strategy build `wf_2018` | 3175.52s |
| 100b | baseline train backtest `wf_2018` | 27.72s |
| 100b | baseline test strategy build `wf_2018` | 41.15s |
| 100b | baseline test backtest `wf_2018` | 0.82s |
| 100b | profitable_filter backtest `wf_2018` | 0.72s |
| 100b | beats_buy_hold_filter backtest `wf_2018` | 0.69s |
| 100b | model fit `wf_2018` | 166.96s |
| 100b | model score `wf_2018` | 172.37s |
| 100b | model train strategy build `wf_2018` | 3653.79s |
| 100b | model train backtest `wf_2018` | 60.49s |

Runtime conclusions so far:

- Runtime scales far faster than the small `1t` slice would suggest.
- The clean `100b` feature artifact was about `6.2 GB`.
- Strategy-dataset construction is the dominant cost center on broader universes.
- Even before `10b` started, runtime cost was already a serious part of the research answer.

## Symbol Insights

### Clean `1t` Aggregate Symbol Readout

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

- The model edge on `1t` was broad across all four names, not dependent on a single winner.
- Because the `1t` universe only has four symbols, metadata filtering had no room to specialize.

### Completed `100b` Baseline Test Fold (`wf_2018`) Top Symbols

- Strongest symbols by symbol-level Sharpe on the completed baseline `100b` test fold:
  - `PG` `1.8459`
  - `NEM` `1.7141`
  - `PANW` `1.6228`
  - `AMD` `1.6151`
  - `MO` `1.0869`
  - `HCA` `1.0124`

Interpretation:

- The strongest completed `100b` baseline fold exposures were not concentrated only in mega-cap tech.
- Defensive, healthcare, and materials-related names also showed up among the strongest completed symbol diagnostics.

## Filter Interpretation

### `1t`

- Both filter variants selected all four `1t` names in every completed fold.
- Result:
  - no incremental filtering effect
  - no interpretable sector or industry segmentation
  - no performance change relative to `no_filter`

### `100b` `wf_2018` Baseline Filters

`profitable_filter` selected `17` symbols:

- Symbols:
  - `KLAC`, `AMD`, `LRCX`, `ABBV`, `MRK`, `WFC`, `VZ`, `T`, `ORCL`, `PEP`, `INTC`, `QCOM`, `TXN`, `AVGO`, `AMAT`, `MU`, `ADI`
- Sector mix:
  - `Technology`: `11`
  - `Healthcare`: `2`
  - `Communication Services`: `2`
  - `Financial Services`: `1`
  - `Consumer Defensive`: `1`
- Industry mix:
  - `Semiconductors`: `10`
  - `Drug Manufacturers - General`: `2`
  - `Telecommunications Services`: `2`

`beats_buy_hold_filter` selected `5` symbols:

- Symbols:
  - `PLTR`, `GEGGL`, `VZA`, `GOOG`, `XOM`
- Sector mix:
  - `Communication Services`: `2`
  - `Technology`: `1`
  - `Financial Services`: `1`
  - `Energy`: `1`

Interpretation:

- The completed `100b` `profitable_filter` leaned heavily into semiconductors and broader technology, and that was the strongest completed filter result so far.
- The completed `beats_buy_hold_filter` was much narrower and did not outperform the unfiltered baseline on the single completed OOS fold.
- Tree depth, feature importance, and exact split rules were not recoverable because the final filter diagnostics file was never written before the process stopped.

## Conclusions

### Does the ML policy beat the baseline?

- Clean `1t`: yes on raw out-of-sample performance.
- Broader universes: still unproven because the clean `100b` out-of-sample model test never completed and `10b` never started.
- Important caveat:
  - the `1t` model still fails the configured minimum-training-row gate in every fold, so the apparent edge does not yet meet the experiment's own validity threshold.

### Does filtering help?

- `1t`: no. Both filters were complete no-ops because they selected the full universe.
- `100b` so far: maybe.
  - `profitable_filter` improved the completed `wf_2018` baseline OOS result
  - `beats_buy_hold_filter` did not
- We do not yet have enough broader-universe completed folds to call the filter result stable.

### Which universe works best?

- Based on completed out-of-sample data alone, only `1t` is fully available.
- `100b` has one promising completed baseline filter fold, but that is not enough to crown a winner.
- `10b` has no completed data available.

### Does the added complexity justify the runtime cost?

- Not yet.
- The model complexity looks promising on `1t`, but it fails the experiment's training-row gate there.
- Runtime cost explodes on broader universes:
  - `1t` model was only moderately slower than baseline
  - `100b` strategy building took tens of minutes per major step
  - `10b` would likely be even more expensive
- On the available evidence, the performance upside is not yet strong enough to justify the broader-universe runtime burden with confidence.

## Recommended Next Step

- Resume or rerun the experiment starting from clean `100b` model test `wf_2018` onward and carry it through all `100b` and `10b` folds.
- Persist filter diagnostics and runtime summaries even if the outer command exits early.
- Investigate why the model fold JSONs omitted Sharpe and turnover while the underlying backtest artifacts clearly contained enough information to compute them.
