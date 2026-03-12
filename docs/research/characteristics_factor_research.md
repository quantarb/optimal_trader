# Characteristics Factor Research Report

## 1. Experiment

- Objective: test whether existing platform features can learn latent factor exposures from returns and convert them into a stronger cross-sectional ranking signal than simple momentum.
- Research framing: inspired by Kelly, Pruitt, and Su's idea that characteristics can map to factor exposures, but implemented as a reusable cross-sectional research capability rather than a paper-specific reproduction.
- Latent factor basis: 3 PCA-style factors estimated from training-window daily returns only across up to 20 symbols per fold.
- Exposure target: rolling OLS betas over 63 business days, then cross-sectional factor premia estimated from the oracle rank-percentile label.
- Trading path: predicted score -> existing `cross_sectional_quantiles` portfolio construction -> long top bucket / short bottom bucket.
- Universe: 20 symbols.
- Symbols: GOOG, AVGO, BRK-B, BRK-A, WMT, LLY, JPM, XOM, V, JNJ, MA, COST, MU, ORCL, NFLX, ABBV, CVX, PG, PLTR, HD

## 2. Results

- Baseline momentum: Sharpe 0.767, total return 87.08%, max drawdown -33.01%.
- Best ML variant: characteristics_factor_rf_all_features | Sharpe 0.896 | total return 111.06% | max drawdown -42.63%.

| Variant | Sharpe | Total Return | Max DD | Turnover | Trades | Mean IC | Long-Short Spread | Positive Fold Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| characteristics_factor_rf_all_features | 0.896 | 111.06% | -42.63% | 7.25 | 4033 | 0.028 | 3.62% | 0.75 |
| baseline_momentum | 0.767 | 87.08% | -33.01% | 15.25 | 4065 | 0.089 | 3.17% | 0.75 |
| oracle_rank_rf_all_features | -0.695 | -46.22% | -55.88% | 25.75 | 4107 | -0.054 | -2.15% | 0.25 |

## 3. Diagnostics

- Baseline mean Spearman IC: 0.089.
- Best ML mean Spearman IC: 0.028.
- Baseline top bucket forward return: 4.53%.
- Best ML top bucket forward return: 4.44%.
- Baseline top-bucket sector mix: Technology (0.65), Energy (0.48), Communication Services (0.15)
- Best-ML top-bucket sector mix: Technology (0.65), Energy (0.47), Communication Services (0.17)
- Best-ML stock/ETF mix: stock (1.00)
- characteristics_factor_rf_all_features: overlap with momentum winners 0.375, fold stability 0.259.
- oracle_rank_rf_all_features: overlap with momentum winners 0.125, fold stability 0.315.

## 4. Interpretation

- Does a feature-based factor model beat simple momentum? yes on portfolio returns, but only mixed on ranking diagnostics.
- The direct ML baseline tests whether broader features help without factor structure. The characteristics-factor variant tests whether routing those same features through learned exposures adds value beyond direct score regression.
- In this pilot the characteristics-factor model improved Sharpe and total return versus momentum, but it did so with lower rank IC and a deeper drawdown, so the edge looks portfolio-level rather than a cleaner cross-sectional oracle.
- Winner overlap and fold stability help separate genuine ranking improvements from strategies that simply rotate into a different but noisier subset of names.

## 5. Platform Capabilities Added

- Reusable characteristic-panel builder that joins existing features with cross-sectional rank-percentile labels.
- Reusable latent factor estimation from returns plus rolling realized exposure targets on rebalance dates.
- Reusable feature-to-exposure regressor path that emits standard prediction artifacts for the existing strategy engine.

## 6. Output Artifacts

- Summary JSON: `data/pipeline_artifacts/characteristics_factor_research_final.json`
- Aggregate results CSV: `data/pipeline_artifacts/characteristics_factor_research_final.csv`
- Fold results CSV: `data/pipeline_artifacts/characteristics_factor_research_final__fold_rows.csv`
- Prediction rows CSV: `data/pipeline_artifacts/characteristics_factor_research_final__predictions.csv`
- Factor premia CSV: `data/pipeline_artifacts/characteristics_factor_research_final__factor_premia.csv`
- Exposure targets CSV: `data/pipeline_artifacts/characteristics_factor_research_final__factor_exposures.csv`
- Model diagnostics CSV: `data/pipeline_artifacts/characteristics_factor_research_final__model_diagnostics.csv`
- Ranking diagnostics CSV: `data/pipeline_artifacts/characteristics_factor_research_final__ranking_summary_aggregate.csv`
