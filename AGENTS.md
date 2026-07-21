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

- Do not call a model non-deployable merely because its historical training labels were generated from future outcomes within the historical training window. This applies to Oracle trades, congressional trades, HITS-derived trades, event graphs, and any other future-outcome-derived target. Such labels are valid supervised-learning targets when every input and label used for training is available by the training cutoff and the model is applied only after that cutoff.
- For example, generating Oracle, congressional-event, or HITS trade labels from 2024 data, training on 2024 features and labels, and predicting 2025 with 2025-available features is a valid point-in-time 2025 deployment test. This is conceptually the same as computing a 200-day moving average from historical prices and applying it later.
- Distinguish historical label generation from leakage: using future 2024 outcomes to label 2024 training rows is acceptable for a 2025 test; using future 2025 prices or undisclosed 2025 event outcomes to create 2025 predictions, or directly executing known 2025 future-derived trades, is leakage.
- Before making a temporal-validity claim, identify the train cutoff, test/deployment period, feature availability time, label availability time, and any disclosure or reporting lag. Do not repeat a blanket claim that Oracle-derived supervised models are non-deployable without checking those boundaries.
