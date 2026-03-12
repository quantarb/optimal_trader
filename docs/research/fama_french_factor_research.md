# Fama-French Style Factor Research

Reference framing:
- Fama and French (1993): [Common Risk Factors in the Returns on Stocks and Bonds](https://www.jstor.org/stable/2329112)
- Ken French Data Library overview: [Description of Fama/French factors](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library/f-f_factors.html)

## 1. Implemented factors

- `MKT`: equal-weight market proxy built from the full selected universe and run through the shared direct-strategy backtest path.
- `SMB`: long the smaller-cap sleeve and short the larger-cap sleeve using the existing `km__marketcap` feature.
- `HML`: long the value sleeve and short the growth sleeve using the platform's existing valuation feature set.
- Multi-factor ranking strategy: equal-weight blend of cross-sectional momentum, value, and size component ranks, then fed into `portfolio_construction=cross_sectional_quantiles`.
- Universe used here: 20 US large-cap equities with sufficient history and platform liquidity filters.
- Selected symbols: GOOG, AVGO, BRK-B, BRK-A, WMT, LLY, JPM, XOM, V, JNJ, MA, COST, MU, ORCL, NFLX, ABBV, CVX, PG, PLTR, HD
- Coverage snapshot: GOOG (2020-12-31 to 2025-12-31), AVGO (2020-12-31 to 2025-12-31), BRK-B (2020-12-31 to 2025-12-31), BRK-A (2020-12-31 to 2025-12-31), WMT (2020-12-31 to 2025-12-31), LLY (2020-12-31 to 2025-12-31), JPM (2020-12-31 to 2025-12-31), XOM (2020-12-31 to 2025-12-31), V (2020-12-31 to 2025-12-31), JNJ (2020-12-31 to 2025-12-31)

## 2. Factor results

- MKT: Sharpe 1.199, total return 190.19%, drawdown -22.31%, positive folds 3/4
- SMB: Sharpe 0.923, total return 26.22%, drawdown -5.72%, positive folds 3/4
- HML: Sharpe 0.120, total return 2.71%, drawdown -24.53%, positive folds 1/4

Factor correlations:
- HML vs MKT: correlation -0.131
- HML vs SMB: correlation -0.259
- MKT vs SMB: correlation 0.313

## 3. Strategy comparison

- Momentum baseline: Sharpe 0.952, total return 81.83%, drawdown -17.35%.
- Best strategy: multi_factor_rank | Sharpe 1.719, total return 135.46%, drawdown -9.00%.
- Interpretation: this experiment asks whether reusable factor primitives improve cross-sectional ranking versus a simple momentum sort, not whether the local universe reproduces the exact original paper coefficients.

## 4. Factor exposure diagnostics

- momentum_baseline: alpha 0.00070, R^2 0.068, residual return -5.44%, betas [HML -0.495, MKT 0.053, SMB -0.338]
- multi_factor_rank: alpha 0.00060, R^2 0.203, residual return -2.65%, betas [HML 0.414, MKT 0.079, SMB 0.716]

## 5. Platform capabilities added

- Added a reusable `portfolio_construction=long_short_factor` mode so any existing score or feature expression can become a factor portfolio.
- Added reusable cross-sectional factor-component scoring so weighted factor blends can drive the existing ranking portfolio constructor.
- Added reusable factor analytics for return-series metrics, correlation analysis, and strategy exposure regression.

## 6. Differences from the original paper

- The original Fama-French construction uses value-weighted portfolios, NYSE breakpoints, and a market-minus-risk-free series; this implementation uses a local equal-weight proxy universe and practical backtest costs.
- The research stack uses existing platform features such as `km__marketcap` and valuation ratios, rather than rebuilding CRSP/Compustat style accounting pipelines.
- The main goal here is reusable platform capability for future multi-factor and ML ranking work, so the report emphasizes factor returns, cross-strategy exposures, and ranking portability.

## 7. Implications for future ML ranking models

- The new factor-component path means future ML ranking models can blend model scores with reusable value/size/momentum context inside the same ranking workflow.
- Strategy exposure regression now makes it easier to tell whether a future ML model is producing genuine alpha or just repackaging factor loads.
