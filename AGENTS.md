# Repository Rules

## Dependency Source Of Truth

- Use `quant-warehouse` from `https://github.com/quantarb/quant-warehouse.git@main`.
- Do not use local editable `quant-warehouse` paths in committed dependency files.
- Do not call OpenBB or vendor market-data APIs directly from `optimal_trader` for data that belongs in the warehouse.

## Data Boundary

- `optimal_trader` should consume warehouse data, feature families, target engineering labels, and orchestrator workflows through `quant-warehouse` and `quant-orchestrator`.
- If data is missing or incomplete, fix the source route in `quantarb/OpenBB`, then refresh `quant-warehouse`.

## Application Responsibility

- `optimal_trader` is the custom frontend/runtime that may submit orders to brokers.
- Research workflows, model training, and backtests should be delegated to `quant-orchestrator`.
- Market data, features, labels, and refresh behavior should come through `quant-warehouse`; do not add direct OpenBB or vendor API calls for research data here.

## Migration Policy

- This repo is a live-money monolith. Preserve current behavior unless the user explicitly asks for a migration change.
- New research or data code should move toward `quant-warehouse` and `quant-orchestrator`, but do not break existing live trading paths as part of cleanup work.

## Build Vs Buy Policy

- Prefer widely used, actively maintained third-party packages or small forks of proven projects over custom implementations.
- For trading UI, broker adapters, scheduling, serialization, analytics, model loading, and strategy/reporting utilities, use battle-tested libraries when they fit the live-app boundary.
- Build from scratch only when no reliable package fits the requirement or the live app needs a thin custom wrapper around a proven dependency; document that reason in the change.

## Scale-Tier Script Policy

- Never create separate scripts, notebooks, or implementations for fixed market-cap tiers such as `$1T`, `$100B`, or `$10B`.
- Market-cap thresholds, artifact tags, estimator counts, smoke/full behavior, and other scale controls must be command-line or configuration parameters on one canonical workflow.
- Use the exact same code path at every scale. Run the highest threshold first as a smoke test, then lower the threshold only after the prior run succeeds.
- Artifact directories may include a caller-supplied tag, but tier names must not be embedded in implementation names or control flow.
- Remove superseded one-off scale scripts instead of retaining wrappers that can drift from the canonical runner.
- Curated technical feature families are part of the default equity model contract at every scale. Do not create a technical/non-technical script split or an opt-out that silently removes them.
- Scripts must call the current public optimized APIs in `quant-warehouse` and `quant-orchestrator`; do not copy notebook cells, private feature builders, model training loops, or full-panel filtering implementations into scripts.

## Temporal Semantics For Research And Backtests

- Never introduce a reporting, disclosure, or filing lag into model feature construction or feature-family alignment unless the user explicitly requests that lag. Preserve the source observation date exactly; forward-fill may carry a value only from that recorded date into later rows. This rule applies to employee counts, institutional position summaries, fundamentals, and all other issuer features.
- Event labels must also use exact `(symbol, event_date)` matching. Never forward-align, smear, or move an event label to a later panel/trading date through a tolerance window unless the user explicitly requests it.
- Historical backfills must request the absolute warehouse floor `1900-01-01`; a provider may return a later documented minimum, but ingestion must preserve every returned historical observation and must not collapse it onto the ingestion date.

- Do not call a model non-deployable merely because its historical training labels were generated from future outcomes within the historical training window. This applies to Oracle trades, congressional trades, HITS-derived trades, event graphs, and any other future-outcome-derived target. Such labels are valid supervised-learning targets when every input and label used for training is available by the training cutoff and the model is applied only after that cutoff.
- For example, generating Oracle, congressional-event, or HITS trade labels from 2024 data, training on 2024 features and labels, and predicting 2025 with 2025-available features is a valid point-in-time 2025 deployment test. This is conceptually the same as computing a 200-day moving average from historical prices and applying it later.
- Distinguish historical label generation from leakage: using future 2024 outcomes to label 2024 training rows is acceptable for a 2025 test; using future 2025 prices or undisclosed 2025 event outcomes to create 2025 predictions, or directly executing known 2025 future-derived trades, is leakage.
- Before making a temporal-validity claim, identify the train cutoff, test/deployment period, feature availability time, label availability time, and any disclosure or reporting lag. Do not repeat a blanket claim that Oracle-derived supervised models are non-deployable without checking those boundaries.

## Feature-Family Coverage

- Feature families have independent historical coverage. Never inner-join feature-family panels when constructing the model input; preserve the union of `(symbol, date)` observations and use fold-safe imputation plus family-presence indicators for missing coverage. Inner joins are allowed only for joining a prepared feature panel to its required price/label rows, not for intersecting independent feature families.

## Experiment Baseline Policy

- The current baseline is always the strongest completed, apples-to-apples experiment established so far—not an older hard-coded configuration.
- When a new variant is stronger under the agreed primary evaluation criteria, promote that exact configuration and its completed WFO results to the current baseline for all subsequent comparisons.
- Preserve the prior baseline as a named historical benchmark; never silently replace it or compare new work against a stale “best” result.
- Before promoting a baseline, record the full comparison context: task layout, model architecture, data coverage and cutoff, WFO years, universe, trading costs, top-k/portfolio rules, and runtime-relevant settings.
- Partial runs, interrupted runs, and runs with mismatched tasks, data, leakage boundaries, or backtest settings cannot establish or replace a baseline.
- If a result is reported as stronger, verify that the comparison is configuration-matched and reproducible before using it to choose the next experiment.

## Current Transformer Baseline

- The official current baseline is the completed 100B anchored WFO with coverage-aware feature-family adapters and trunk-specific feature-family routing, plus masked-token reconstruction and causal next-token prediction auxiliary objectives.
- Architecture: two shared transformer trunks; each trunk learns a soft mixture over feature-family adapter states before temporal/cross-sectional attention; task heads retain learned task-to-trunk routing; shared low-rank feature mixer, linear cross-sectional context, hybrid asset adapters, and one lightweight bottleneck adapter plus state-dependent gate per feature family. Every existing MTL head consumes the shared fused state.
- Coverage behavior: feature-family panels are outer-joined on `(symbol, date)`; fold-safe imputation is accompanied by explicit family-presence masks. No family is silently removed because another family has narrower historical coverage.
- Data: 116 screened 100B equity symbols with the current backfilled feature-family cache; technical feature families remain disabled in accordance with the raw-feature/auto-feature-engineering experiment.
- Tasks: six equity HITS return targets, six speed-HITS targets, 45 event targets, sector/industry/year temporal auxiliary tasks, cross-sectional return and speed token tasks, and cross-sectional year supervision. Auxiliary objectives reconstruct 15% randomly masked complete tokens and predict the next temporal token. Macro tasks are disabled.
- WFO: 2021–2025, anchored historical training through each test year, CUDA, 12 epochs, 5.5 bps transaction cost, top-k=20, long-only and short-only evaluation, with temporal and cross-sectional trading heads reported. Runtime was 1,460.6 seconds.
- Full-WFO long-only results: temporal return mean return 30.76% / Sharpe 1.55; temporal speed 29.31% / 1.72; cross-sectional return 26.25% / 1.66; cross-sectional speed 25.80% / 1.67. Mean holding periods were 75.8, 105.8, 112.4, and 113.0 days respectively. The model was profitable long-only in 2022 through the temporal-speed strategy (+15.59%, Sharpe 1.02).
- Reproduction defaults in `scripts/run_symbol_year_transformer_mtl.py`: `TRUNKS=2`, `CROSS_SECTIONAL_ENABLED=1`, `CROSS_SECTIONAL_TOKEN_TASKS=1`, `CROSS_SECTIONAL_COMPARE_HEADS=1`, `SHARED_FEATURE_MIXER=1`, `CROSS_SECTIONAL_SET_CONTEXT=1`, `COVERAGE_AWARE_FAMILY_ADAPTERS=1`, `MASKED_TOKEN_ENABLED=1`, `NEXT_TOKEN_ENABLED=1`, `MASKED_TOKEN_RATE=0.15`, `MASKED_TOKEN_WEIGHT=0.05`, `NEXT_TOKEN_WEIGHT=0.05`, `MACRO_ENABLED=0`, `SPEED_STRATEGY_ENABLED=1`, `EPOCHS=12`, and both `long_only,short_only` variants. The tier remains an explicit run choice; use `TRANSFORMER_TIERS=100B` and the corresponding feature cache for the official baseline universe.
- The prior coverage-aware baseline without trunk-specific feature routing remains a named historical benchmark. The prior dual-tower issuer/instrument model is also historical; future comparisons should use this new routed baseline unless the user explicitly promotes another completed apples-to-apples result.
