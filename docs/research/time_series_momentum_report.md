# Time Series Momentum Research Report

## 1. Strategy implementation

- Paper reference: Moskowitz, Ooi, and Pedersen (2012), "Time Series Momentum."
- Paper universe: 58 liquid futures across equity indexes, government bonds, currencies, and commodities.
- Platform implementation: monthly-rebalanced direct-signal strategy on liquid multi-asset ETF proxies available in the local FMP-backed database.
- Trading universe used here: SPY, QQQ, IWM, EFA, EEM, VNQ, TLT, IEF, SHY, LQD, HYG, GLD, SLV, DBC, USO, UNG, FXE, FXY
- Missing requested proxy symbols not present locally: none
- Signal: 12-month return excluding the most recent month, implemented as `(1 + px__ret_252_d) / (1 + px__ret_21_d) - 1`.
- Position rule: sign transform applied to the direct signal so positive signals are equally weighted longs and negative signals are equally weighted shorts.
- Rebalance frequency: M
- Gross exposure target: 1.0
- Train/test split: yearly walk-forward validation; each fold trains on data through the prior December 31 and tests the next calendar year.
- Evaluation metrics: Sharpe ratio, total return, max drawdown, turnover, trade count, plus fold-level stability statistics.

## 2. Code changes made

- Added `Ret21d` to the reusable price technical feature set in `domain/features/technical.py`.
- Extended direct strategy definitions to support `action_transform` in `pipeline/strategy_definitions.py`.
- Extended direct strategy score computation to use `combined_score_expr` in `workflows/strategy.py`.
- Added a reusable walk-forward runner for direct feature-driven strategies in `pipeline/direct_strategy_runner.py`.
- Persisted exact walk-forward rollup metrics from fold backtest daily rows so research reports can use true total-return, Sharpe, drawdown, turnover, and trade-count outputs.
- Added the paper replication command in `pipeline/management/commands/run_time_series_momentum_research.py`.
- Hardened universe filtering in `pipeline/universe_selection.py` so pooled vehicles flagged as funds are excluded consistently when requested.
- Added regression coverage in `tests/test_research_core_unit.py`, `pipeline/tests.py`, and `pipeline/tests_mag7.py`.

## 3. Experiment results

- Fold count: 15
- Walk-forward test window: 2011-01-03 to 2025-12-31
- Positive folds: 7
- Negative folds: 8
- Walk-forward Sharpe ratio: -0.164
- Walk-forward total return: -24.99%
- Walk-forward final equity: 0.7501
- Walk-forward max drawdown: -44.72%
- Walk-forward excess cumulative return vs equal-weight benchmark: -187.17%
- Mean fold Sharpe: -0.077
- Median fold Sharpe: -0.144
- Mean fold excess return: -8.60%
- Avg daily turnover: 0.0228
- Total turnover: 85.8681
- Trade count: 52004
- Best fold: wf_2015 (11.13%, Sharpe 1.329)
- Worst fold: wf_2020 (-24.31%, Sharpe -1.317)
- Success criteria assessment: trend-following behavior appears only episodically in this proxy universe, and the overall Sharpe remained negative.
- Secondary clean US 50 sanity check: 8 positive folds / 7 negative folds, mean fold Sharpe 0.112, overall Sharpe -0.052, total return -19.11%, and max drawdown -54.27%; this showed somewhat better trend-following character but still did not clear the positive-Sharpe bar after costs.

## 4. Differences from the paper

- The paper studies 58 liquid futures across equity indexes, bonds, currencies, and commodities; this implementation uses liquid ETF proxies because the platform does not yet have a native futures dataset.
- The paper works with excess returns and volatility-scaled positions; this implementation uses simple total returns from adjusted prices and equal-weight long/short sign positions.
- The paper sample spans 1965 to 2009 for futures; this backtest uses the locally available proxy sample and a 2011-2025 yearly walk-forward evaluation window.
- The paper reports pooled t-statistics, factor regressions, and decomposition across horizons; this run focuses on platform-native backtest metrics and walk-forward stability.

## 5. Improvements made to the platform

- Direct strategies can now express reusable feature formulas without requiring a model-scoring stage.
- Direct strategies can now normalize signed signals into equal-weight long/short portfolios.
- The platform now has a generic walk-forward runner for deterministic feature-driven strategies, which broadens the research surface beyond supervised models.
- Walk-forward summaries now preserve exact backtest-level research metrics for direct strategies instead of relying only on fold aggregates.
- Universe selection is more reliable for equity research because payload-based pooled-vehicle flags are filtered consistently.
- Post-implementation `code_analysis` reduced `pipeline.direct_strategy_runner` max function complexity from 64 to 34 and lowered its mixed-concern score materially.
- Post-implementation `performance_analysis` no longer flags `pipeline/direct_strategy_runner.py` as a scaling-risk hotspot; the remaining prominent scaling risks are elsewhere in the platform.
- `product_quality_analysis` was intentionally skipped in this session because you asked not to run the long product-quality pass.

## 6. Suggestions for future improvements

- Add first-class futures and excess-return data support so paper universes can be replicated literally instead of through ETF proxies.
- Add volatility-targeting and inverse-vol position scaling to the strategy definition schema.
- Add richer direct-signal transforms such as winsorization, z-scoring, and top/bottom quantile selection.
- Add factor-regression evaluation and Newey-West t-stat reporting for closer academic comparison.

## Backtest config

- Fee bps: 2.0
- Slippage bps: 8.0
- Short borrow bps annual: 25.0
- Execution delay days: 1
