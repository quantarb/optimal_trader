# Oracle Ranking Signal Research Report

## 1. Strategy implementation

- Objective: test whether an oracle-trained regression model that predicts future cross-sectional rank percentiles can produce better cross-sectional portfolios than a simple momentum ranking baseline.
- Oracle target: `future_rank_pct`, computed on monthly rebalance dates from 65 labeled cross-sections using a 21-day forward return horizon and a 1-day execution offset.
- Baseline ranking signal: `(1.0 + px__ret_252d) / (1.0 + px__ret_21d) - 1.0`.
- Portfolio construction: existing `portfolio_construction="cross_sectional_quantiles"` with equal-weight long top bucket / short bottom bucket and monthly rebalancing.
- Universe: 20 US large-cap symbols with sufficient price history.
- Symbols: GOOG, AVGO, BRK-B, BRK-A, WMT, LLY, JPM, XOM, V, JNJ, MA, COST, MU, ORCL, NFLX, ABBV, CVX, PG, PLTR, HD

## 2. Experiment results

- Baseline Sharpe 0.767, total return 87.08%, max drawdown -33.01%, positive fold rate 0.75.
- Best model variant: oracle_rank_rf_context_only | Sharpe 0.025 | total return -6.25% | max drawdown -46.45%.

| Variant | Sharpe | Total Return | Max DD | Turnover | Trades | Mean IC | Mean Long-Short Spread | Positive Fold Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_momentum | 0.767 | 87.08% | -33.01% | 15.25 | 4065 | 0.089 | 3.17% | 0.75 |
| oracle_rank_rf_context_only | 0.025 | -6.25% | -46.45% | 21.25 | 4089 | -0.002 | 0.02% | 0.50 |
| oracle_rank_rf_all_features | -0.528 | -39.03% | -53.59% | 26.50 | 4110 | -0.039 | -1.32% | 0.50 |
| oracle_rank_rf_prices_only | -0.626 | -40.66% | -41.15% | 35.50 | 4136 | -0.006 | -0.91% | 0.00 |

## 3. Ranking diagnostics

- Baseline mean Spearman IC: 0.089.
- Best model mean Spearman IC: -0.002.
- Baseline top-bottom bucket spread: 3.17%.
- Best model top-bottom bucket spread: 0.02%.
- Baseline top bucket avg forward return: 4.53%.
- Best model top bucket avg forward return: 2.89%.

## 4. Cohort diagnostics

- Baseline top-bucket sector mix: Technology (0.65), Energy (0.48), Communication Services (0.15)
- Best-model top-bucket sector mix: Technology (0.46), Healthcare (0.26), Consumer Defensive (0.15)
- Baseline stock/ETF mix: stock (1.00)
- Best-model stock/ETF mix: stock (1.00)
- oracle_rank_rf_all_features: overlap with baseline winners 0.118 average Jaccard, fold stability 0.336.
- oracle_rank_rf_prices_only: overlap with baseline winners 0.056 average Jaccard, fold stability 0.419.
- oracle_rank_rf_context_only: overlap with baseline winners 0.174 average Jaccard, fold stability 0.387.

## 5. Interpretation

- Does ML beat simple momentum? mixed/no.
- Does broader context add value beyond price-only features? Compare `oracle_rank_rf_all_features` against `oracle_rank_rf_prices_only` in the table above; the gap is the cleanest test because both use the same target, model family, and portfolio construction.
- Are improvements consistent across folds? Use positive fold rate plus the fold-stability rows; higher Sharpe with weak IC or low fold consistency should be treated cautiously.
- Is extra complexity justified? The answer depends on whether the best ML variant improves both portfolio outcomes and ranking IC over the baseline, not just one of them.

## 6. Platform capabilities added

- Reusable cross-sectional rank-percentile label artifacts backed by the standard `LABELS` artifact type.
- Reusable ranking diagnostics for rank IC, bucket spreads, cohort composition, winner overlap, and fold stability.
- A reusable oracle-ranking research runner that composes the existing feature, model, strategy, and backtest infrastructure instead of building a parallel notebook path.

## 7. Output artifacts

- Summary JSON: `data/pipeline_artifacts/oracle_ranking_signal_research_final.json`
- Aggregate results CSV: `data/pipeline_artifacts/oracle_ranking_signal_research_final.csv`
- Fold results CSV: `data/pipeline_artifacts/oracle_ranking_signal_research_final__fold_rows.csv`
- Ranking summary CSV: `data/pipeline_artifacts/oracle_ranking_signal_research_final__ranking_summary_aggregate.csv`
- Bucket returns CSV: `data/pipeline_artifacts/oracle_ranking_signal_research_final__bucket_aggregate.csv`
- Cohort diagnostics CSV: `data/pipeline_artifacts/oracle_ranking_signal_research_final__cohort_aggregate.csv`
- Winner overlap CSV: `data/pipeline_artifacts/oracle_ranking_signal_research_final__overlap_aggregate.csv`
- Fold stability CSV: `data/pipeline_artifacts/oracle_ranking_signal_research_final__stability_summary.csv`
