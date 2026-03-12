# Jegadeesh & Titman (1993) Research Report

## 1. Strategy implementation

- Paper reference: Jegadeesh and Titman (1993), "Returns to Buying Winners and Selling Losers."
- Core paper signal: rank stocks cross-sectionally on cumulative past returns over the prior `J` months, then buy the winner decile and short the loser decile.
- Canonical paper construction: equal-weight deciles, monthly portfolio formation, overlapping `K`-month holding sleeves, with a one-week skip between ranking measurement and portfolio entry.
- Platform implementation: reusable cross-sectional quantile portfolio construction on price-momentum features, monthly rebalanced, decile winner/loser sleeves, `5` trading-day lag, and overlapping `K`-month holdings.
- Universe used here: 40 locally available US large-cap equities with sufficient price history and platform liquidity filters.
- Selected symbols: GOOG, AVGO, BRK-B, BRK-A, WMT, LLY, JPM, XOM, V, JNJ, MA, COST, MU, ORCL, NFLX, ABBV, CVX, PG, HD, BAC, GE, KO, AMD, CAT, CSCO, MRK, RTX, PM, AMAT, LRCX, UNH, MS, GS, WFC, TMUS, IBM, MCD, INTC, PEP, VZ
- Coverage snapshot: GOOG (2012-12-24 to 2025-12-31), AVGO (2012-12-24 to 2025-12-31), BRK-B (2012-12-24 to 2025-12-31), BRK-A (2012-12-24 to 2025-12-31), WMT (2012-12-24 to 2025-12-31), LLY (2012-12-24 to 2025-12-31), JPM (2012-12-24 to 2025-12-31), XOM (2012-12-24 to 2025-12-31), V (2012-12-24 to 2025-12-31), JNJ (2012-12-24 to 2025-12-31)
- Direct signal fields used: `px__ret_63_d`, `px__ret_126_d`, `px__ret_189_d`, and `px__ret_252_d` depending on the formation window.

## 2. Experiment results

- Variant count: 16
- Positive-Sharpe variants: 4
- Variants passing stability gates: 16
- Best variant: jt1993_j3_k3_lag5 | Sharpe 0.043 | total return -0.07% | max drawdown -20.05%
- Best variant fold stability: 5/12 positive folds
- Best variant turnover/trades: 41.85 total turnover | 42823 trades
- Worst variant: jt1993_j3_k9_lag5 | Sharpe -0.104 | total return -12.34% | max drawdown -18.63%
- Interpretation: in this large-cap survivor universe, changing the formation window from 3 to 12 months barely moved the ranked portfolios; the holding horizon drove most of the performance difference.
- `K=3` variants averaged Sharpe 0.043, versus -0.079 for longer holds.

Top variants:
- jt1993_j3_k3_lag5: Sharpe 0.043, total return -0.07%, drawdown -20.05%, positive folds 5/12
- jt1993_j6_k3_lag5: Sharpe 0.043, total return -0.07%, drawdown -20.05%, positive folds 5/12
- jt1993_j9_k3_lag5: Sharpe 0.043, total return -0.07%, drawdown -20.05%, positive folds 5/12
- jt1993_j12_k3_lag5: Sharpe 0.043, total return -0.07%, drawdown -20.05%, positive folds 5/12
- jt1993_j3_k6_lag5: Sharpe -0.035, total return -6.99%, drawdown -19.43%, positive folds 5/12

Best-variant symbol diagnostics:
- Top symbol COST: Sharpe 0.862, avg trade return 0.27%, hit rate 58.33%, trades 20
- Top symbol BAC: Sharpe 0.538, avg trade return 0.30%, hit rate 75.00%, trades 10
- Top symbol TMUS: Sharpe 0.491, avg trade return 0.28%, hit rate 72.67%, trades 19
- Top symbol XOM: Sharpe 0.392, avg trade return 0.19%, hit rate 75.00%, trades 23
- Top symbol GE: Sharpe 0.387, avg trade return 0.15%, hit rate 62.50%, trades 20
- Weak symbol BRK-B: Sharpe -0.685, avg trade return -0.33%, hit rate 50.00%, trades 4
- Weak symbol INTC: Sharpe -0.682, avg trade return -0.33%, hit rate 34.17%, trades 25
- Weak symbol BRK-A: Sharpe -0.609, avg trade return -0.23%, hit rate 33.33%, trades 8
- Weak symbol MA: Sharpe -0.404, avg trade return -0.21%, hit rate 29.17%, trades 13
- Weak symbol IBM: Sharpe -0.391, avg trade return -0.11%, hit rate 22.22%, trades 13

## 3. Platform capabilities added

- Added reusable cross-sectional quantile portfolio construction to `pipeline/strategy_definitions.py`, so any score field can now drive winner/loser decile portfolios.
- Added overlapping multi-rebalance holding support, which closes a major gap between the original academic construction and the platform's previous single-sleeve carry-forward logic.
- Added `Ret189d` to the shared price-technical feature set so 9-month momentum windows are first-class platform features.
- Added coverage-aware symbol history filtering in `pipeline/universe_selection.py`, which makes both direct-strategy and oracle-based studies less brittle on sparse data.

## 4. Research workflow inefficiencies discovered

- The platform still rehydrates the full feature artifact for each fold/variant pair, which is simple but wastes time for strategy-family sweeps.
- Academic equity replications remain limited by the local FMP-style symbol store; there is no native CRSP-like universe, delisting return support, or historical membership handling.
- Strategy family comparison is still assembled in research code instead of through a first-class family runner with shared caching and reporting.

## 5. Lessons for the oracle-based workflow

- Oracle and model outputs can now be translated into cross-sectional long/short decile portfolios by setting `portfolio_construction=cross_sectional_quantiles`, rather than only thresholding scores independently by symbol.
- Coverage-aware universe screening should happen before oracle-label generation as well, so models are not trained on symbols with fragmented price history.
- A natural next extension is a reusable cross-sectional label builder that emits winner/loser portfolio membership or relative-rank labels, which would let models learn portfolio-level momentum structure directly.

## 6. Differences from the paper

- The paper uses the broad CRSP equity universe with historical listings and delisting returns; this run uses a locally available, large-cap, survivor-biased US equity subset.
- The paper's one-week skip is implemented here as a `5` trading-day lag on the ranking signal, which is close but not identical to calendar-week portfolio timing.
- The platform backtest focuses on practical metrics such as Sharpe, total return, drawdown, turnover, and fold stability, rather than the original paper's full significance tables.
