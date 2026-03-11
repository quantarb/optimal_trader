# Repository Target Architecture

This document is the concrete migration map for turning the current repository
from "three overlapping monorepos" into one coherent application.

The codebase historically had three competing centers of gravity:

1. `modules/`
   Notebook-era quant engine and workflows. This root has now been removed.
2. Django apps like `features/`, `labels/`, `ml/`, `fmp/`
   App-specific models, views, forms, and some business logic.
3. `pipeline/`
   A newer orchestration/reporting/intelligence layer that also contains
   analysis logic and presentation logic.

The goal is not to split into multiple repos. The goal is to make ownership
clear inside one repo.

## Migration Status

Completed so far:

- downstream evaluation and market-intelligence modules were extracted from
  `pipeline/` into the new top-level `analysis/` package
- `pipeline/views.py` was split into a thinner route layer plus focused modules:
  `pipeline/view_support.py`, `pipeline/views_insights.py`,
  `pipeline/views_artifacts.py`, `pipeline/views_reports.py`, and
  `pipeline/views_workbench.py`
- canonical namespaces now exist and are used directly for the active app
  surface: `data/`, `backtest/`, `ml/`, `analysis/`, and `pipeline/`
- the concrete implementations from `modules/data/` and `modules/engine/`
  were moved into `data/` and `backtest/`
- the raw-stack helpers from `modules/workflows/` were moved into canonical
  owners: `ml/raw_stack.py`, `backtest/raw_stack.py`, and
  `analysis/raw_stack_inference.py`
- canonical implementations now exist for the RL and autoencoder stacks in
  `ml/rl/` and `ml/autoencoder/`
- framework-owned model implementations now live under `ml/frameworks/`
  for sklearn, transformers, and AutoGluon
- shared model contracts and metrics now live in `ml/`
- the old `modules/features/`, `modules/labels/`, `modules/models/`,
  `modules/analysis/`, `modules/data/`, `modules/engine/`,
  `modules/workflows/`, `modules/qcore/`, `modules/signals/`,
  `modules/strategies/`, and `modules/utils/` roots were deleted after their
  remaining code was moved
- the old `modules/api.py` façade was replaced by `pipeline/api.py`
- `pipeline/api.py` was split into focused service modules:
  `pipeline/api_common.py`, `pipeline/api_features.py`,
  `pipeline/api_datasets.py`, `pipeline/api_labels.py`, and
  `pipeline/api_models.py`, leaving `pipeline/api.py` as a thin curated façade
- the compatibility bridges `ml.legacy`, `data.legacy`, and
  `backtest.legacy` were deleted
- the wrapper-only model files under `modules/models/base.py`,
  `modules/models/metrics.py`, `modules/models/sklearn/*`,
  `modules/models/stable_baselines3/*`, and
  `modules/models/torch/autoencoder/*` were deleted
- active app/orchestration code no longer imports those deleted compatibility
  paths

Still pending:

- break up the remaining orchestration-heavy parts of `pipeline/`
- split `pipeline/services.py` and `pipeline/view_support.py`, which are now the
  main mixed-responsibility hotspots
- resolve the remaining circular dependencies in `ml/` and `fmp/`
- keep regenerating the code-analysis reports as the canonical architecture
  changes

## Core Decisions

1. `pipeline/` owns application orchestration, artifacts, runs, and reports.
2. `features/` owns feature construction and feature metadata only.
3. `labels/` owns label recipes and oracle-trade generation only.
4. `ml/` owns model training, scoring, storage, and model-family abstractions.
5. `fmp/` owns external provider integration and provider persistence only.
6. A new `analysis/` package should own downstream evaluation and intelligence:
   feature attribution, oracle coverage, clustering, diagnostics, analog search,
   reasoning, and insight composition.
7. The former `modules/` architecture should not be reintroduced. New code
   should be added only to canonical owners.

## Target Logical Package Map

This is the target logical architecture, even if the physical moves happen in phases.

```text
fmp/           external data provider integration and provider persistence
data/          pure dataset loading, PIT joins, preparation, storage helpers
features/      feature builders, feature naming, feature family metadata
labels/        label builders, oracle recipes, solver logic
ml/            model contracts, fit/score execution, storage, multitask
ml/frameworks/ framework-specific model implementations
backtest/      execution engine, rebalance logic, portfolio mechanics
analysis/      evaluation, diagnostics, oracle coverage, clustering, insights
pipeline/      orchestration, artifacts, experiment runs, report registry
web/           optional future thin UI layer; for now existing Django apps stay
```

## Ownership Rules

Use these rules when deciding where code belongs.

- If code answers "how do we build this dataset?" it belongs with the producer:
  `features/`, `labels/`, `ml/`, `data/`, or `fmp/`.
- If code answers "what happened when we used this dataset/model?" it belongs in
  `analysis/`.
- If code answers "run this workflow end to end and persist artifacts" it belongs
  in `pipeline/`.
- If code answers "render this in HTML or JSON" it belongs in a UI layer and
  should call services, not contain business logic.

## Immediate Package Strategy

Do not rename the whole repo at once.

Phase 1 should be:

1. Create `analysis/`.
2. Create `data/`.
3. Create `backtest/`.
4. Move logic out of `modules/` into canonical owners.
5. Split `pipeline.views` and other large view modules into thin view/controller
   code plus service calls.

## File Move Map

The tables below use these actions:

- `KEEP` means keep the file in its current package, but slim it if needed.
- `MOVE` means the file should move to the target package.
- `SPLIT` means one file currently mixes multiple concerns and should be broken up.
- `RETIRE` means delete after callers are migrated.

## 1. Keep Existing Domain Producers

These packages already align with ownership and should remain the canonical producers.

### `features/`

KEEP:

- `features/analyst_estimates_features.py`
- `features/balance_sheet_features.py`
- `features/balance_sheet_growth_features.py`
- `features/cash_flow_features.py`
- `features/cash_flow_growth_features.py`
- `features/earnings_features.py`
- `features/feature_builders.py`
- `features/financial_growth_features.py`
- `features/fundamentals.py`
- `features/grades_historical_features.py`
- `features/income_statement_features.py`
- `features/income_statement_growth_features.py`
- `features/insider_trading_features.py`
- `features/key_metrics_features.py`
- `features/macro.py`
- `features/naming.py`
- `features/prices_div_adj_features.py`
- `features/ratings_historical_features.py`
- `features/ratios_features.py`
- `features/section_utils.py`
- `features/technical.py`
- `features/time_features.py`

SPLIT:

- `features/pipeline_builders.py`
  - keep pure feature-build logic in `features/`
  - move orchestration/reporting helpers into `pipeline/` or `analysis/`
- `features/views.py`
  - keep UI only
  - move data prep and duplicate helpers into `features/` service modules

### `labels/`

KEEP:

- `labels/directional.py`
- `labels/events.py`
- `labels/ranking.py`
- `labels/strategy_solver.py`
- `labels/trades.py`

SPLIT:

- `labels/views.py`
  - keep UI only
  - move report and evaluation logic out

### `ml/`

KEEP:

- `ml/execution.py`
- `ml/feature_families.py`
- `ml/multitask.py`
- `ml/store.py`

SPLIT:

- `ml/execution.py`
  - training frame build and score execution stay in `ml/`
  - reporting/orchestration helpers move to `pipeline/`
- `ml/views.py`
  - keep UI only

### `fmp/`

KEEP:

- `fmp/endpoints/*`
- `fmp/models.py`
- `fmp/tasks.py`

SPLIT:

- `fmp/views.py`
  - keep UI only
  - move provider freshness / stale-state logic into `fmp/services/`

## 2. Create `analysis/` and Move Downstream Evaluation There

These files are not producers. They are downstream evaluation, explanation,
clustering, or market-intelligence logic. They should not live under `features/`
or `labels/`.

### Move to `analysis/oracle/`

MOVE:

- `pipeline/oracle_reports.py` -> `analysis/oracle/reports.py`
- `pipeline/oracle_state_dataset.py` -> `analysis/oracle/state_dataset.py`

Likely extracted from existing files:

- oracle coverage logic from `pipeline/research.py` -> `analysis/oracle/coverage.py`
- oracle-cluster evaluation logic from `pipeline/research_suite.py` -> `analysis/oracle/evaluation.py`

Reason:

- oracle labeling belongs in `labels/`
- oracle recovery, oracle coverage, oracle reports, and oracle clustering do not

### Move to `analysis/feature_attribution/`

MOVE:

- `pipeline/feature_attribution.py` -> `analysis/feature_attribution.py`

Reason:

- feature family construction belongs in `features/`
- attribution of downstream model/backtest/oracle effects belongs in `analysis/`

### Move to `analysis/market_situations/`

MOVE:

- `pipeline/cluster_explanations.py` -> `analysis/market_situations/cluster_explanations.py`
- `pipeline/cluster_outcomes.py` -> `analysis/market_situations/cluster_outcomes.py`
- `pipeline/situation_clustering.py` -> `analysis/market_situations/clustering.py`
- `pipeline/situation_similarity.py` -> `analysis/market_situations/similarity.py`
- `pipeline/historical_outcomes.py` -> `analysis/market_situations/historical_outcomes.py`
- `pipeline/historical_situation_search.py` -> `analysis/market_situations/search.py`
- `pipeline/similarity_engine.py` -> `analysis/market_situations/neighbor_search.py`

### Move to `analysis/insights/`

MOVE:

- `pipeline/analog_reasoning.py` -> `analysis/insights/analog_reasoning.py`
- `pipeline/familiarity_reasoning.py` -> `analysis/insights/familiarity_reasoning.py`
- `pipeline/feature_reasoning.py` -> `analysis/insights/feature_reasoning.py`
- `pipeline/model_reasoning.py` -> `analysis/insights/model_reasoning.py`
- `pipeline/insight_composer.py` -> `analysis/insights/composer.py`
- `pipeline/llm_prompt_builder.py` -> `analysis/insights/llm_prompt_builder.py`
- `pipeline/market_insight_schema.py` -> `analysis/insights/schema.py`
- `pipeline/opportunity_scoring.py` -> `analysis/insights/opportunity_scoring.py`
- `pipeline/insights.py` -> `analysis/insights/service.py`

### Move to `analysis/representations/`

MOVE:

- `pipeline/market_state.py` -> `analysis/representations/market_state.py`
- `pipeline/state_embedding.py` -> `analysis/representations/state_embedding.py`
- `pipeline/state_representations.py` -> `analysis/representations/state_representations.py`

Reason:

- these are shared representation/search assets
- they are not orchestration concerns

### Move to `analysis/diagnostics/`

MOVE:

- `pipeline/diagnostics.py` -> `analysis/diagnostics/pipeline_diagnostics.py`
- `pipeline/research.py` -> `analysis/diagnostics/symbol_research.py`

KEEP in `pipeline/`:

- the commands that orchestrate these analyses

## 3. Keep `pipeline/` for Orchestration Only

These files should stay in `pipeline/`, but many need to be split internally.

KEEP:

- `pipeline/models.py`
- `pipeline/contracts.py`
- `pipeline/report_catalog.py`
- `pipeline/strategy_definitions.py`
- `pipeline/universe_selection.py`
- `pipeline/tasks.py`

KEEP but SPLIT into subpackages:

- `pipeline/services.py`
  - split into:
    - `pipeline/services/jobs.py`
    - `pipeline/services/artifacts.py`
    - `pipeline/services/strategy_backtest.py`
    - `pipeline/services/model_jobs.py`
- `pipeline/cohort_runner.py`
  - keep orchestration
  - extract shared cache helpers to `pipeline/common/cache_io.py`
- `pipeline/research_suite.py`
  - keep orchestration
  - extract shared summary writers/loaders to `pipeline/common/report_io.py`
- `pipeline/experiments.py`
  - keep only run assembly and config expansion

KEEP UI only:

- `pipeline/views.py`
- `pipeline/forms.py`
- `pipeline/urls.py`
- `pipeline/templatetags/pipeline_ui.py`

But `pipeline/views.py` must be split immediately into:

- `pipeline/views/research.py`
- `pipeline/views/reports.py`
- `pipeline/views/insights.py`
- `pipeline/views/artifacts.py`
- `pipeline/views/backtests.py`

and the current file retired.

## 4. Migrate `modules/` Into Canonical Owners

`modules/` should not survive as a parallel product architecture.

### `modules/features/*`

MOVE:

- `modules/features/fundamentals_fmp.py` -> merge into `features/fundamentals.py`
- `modules/features/macro_fmp.py` -> merge into `features/macro.py`
- `modules/features/technical.py` -> merge into `features/technical.py`
- `modules/features/time_features.py` -> merge into `features/time_features.py`

RETIRE:

- `modules/features/__init__.py`

### `modules/labels/*`

MOVE:

- `modules/labels/directional.py` -> merge into `labels/directional.py`
- `modules/labels/events.py` -> merge into `labels/events.py`
- `modules/labels/ranking.py` -> merge into `labels/ranking.py`
- `modules/labels/strategy_solver.py` -> merge into `labels/strategy_solver.py`
- `modules/labels/trades.py` -> merge into `labels/trades.py`

RETIRE:

- `modules/labels/__init__.py`

### `modules/models/*`

MOVE:

- `modules/models/base.py` -> `ml/adapters/base.py`
- `modules/models/metrics.py` -> `ml/metrics.py`
- `modules/models/autogluon/classifier.py` -> `ml/adapters/autogluon_classifier.py`
- `modules/models/autogluon/regressor.py` -> `ml/adapters/autogluon_regressor.py`
- `modules/models/sklearn/classifier.py` -> `ml/adapters/sklearn_classifier.py`
- `modules/models/sklearn/regressor.py` -> `ml/adapters/sklearn_regressor.py`
- `modules/models/stable_baselines3/a2c.py` -> `ml/rl/a2c.py`
- `modules/models/stable_baselines3/common.py` -> `ml/rl/common.py`
- `modules/models/stable_baselines3/ppo.py` -> `ml/rl/ppo.py`
- `modules/models/torch/autoencoder/adapter.py` -> `ml/autoencoder/adapter.py`
- `modules/models/torch/autoencoder/config.py` -> `ml/autoencoder/config.py`
- `modules/models/torch/autoencoder/diagnostics.py` -> `ml/autoencoder/diagnostics.py`
- `modules/models/torch/autoencoder/model.py` -> `ml/autoencoder/model.py`
- `modules/models/torch/autoencoder/trainer.py` -> `ml/autoencoder/trainer.py`
- `modules/models/torch/autoencoder/vector_db.py` -> `ml/autoencoder/vector_db.py`
- `modules/models/transformers/seq2seq.py` -> `ml/adapters/seq2seq.py`

RETIRE after callers migrate:

- `modules/models/__init__.py`
- `modules/models/autogluon/__init__.py`
- `modules/models/sklearn/__init__.py`
- `modules/models/stable_baselines3/__init__.py`
- `modules/models/torch/__init__.py`
- `modules/models/torch/autoencoder/__init__.py`
- `modules/models/transformers/__init__.py`

### `modules/data/*`

Create a new `data/` package and move these there.

MOVE:

- `modules/data/build.py` -> `data/build.py`
- `modules/data/context.py` -> `data/context.py`
- `modules/data/dataset_rows.py` -> `data/dataset_rows.py`
- `modules/data/feature_name_map.py` -> `data/feature_name_map.py`
- `modules/data/fmp_client.py` -> `data/fmp_client.py`
- `modules/data/pit.py` -> `data/pit.py`
- `modules/data/preparation.py` -> `data/preparation.py`
- `modules/data/prices_sqlite.py` -> `data/prices_sqlite.py`
- `modules/data/quality.py` -> `data/quality.py`
- `modules/data/storage.py` -> `data/storage.py`
- `modules/data/universe_fmp.py` -> `data/universe.py`

### `modules/engine/*` and `modules/strategies/*`

Create a new `backtest/` package and move these there.

MOVE:

- `modules/engine/backtest.py` -> `backtest/engine.py`
- `modules/engine/latest.py` -> split:
  - live scoring helpers to `analysis/latest.py`
  - stale unused helpers deleted
- `modules/strategies/benchmark.py` -> `backtest/benchmark.py`
- `modules/strategies/panel_utils.py` -> `backtest/panel_utils.py`
- `modules/strategies/rebalance.py` -> `backtest/rebalance.py`
- `modules/strategies/stateful.py` -> `backtest/stateful.py`

### `modules/workflows/*`

Most of this is orchestration and should not remain in `modules/`.

MOVE:

- `modules/workflows/evaluation.py` -> `pipeline/workflows/evaluation.py`
- `modules/workflows/inference.py` -> `pipeline/workflows/inference.py`
- `modules/workflows/training.py` -> `pipeline/workflows/training.py`

Then RETIRE:

- `modules/workflows/__init__.py`

### `modules/utils/*`

Split by concern.

MOVE:

- `modules/utils/normalize.py` -> `data/normalize.py`
- `modules/utils/panel.py` -> `data/panel.py`
- `modules/utils/workflow.py` -> `pipeline/common/workflow.py`
- `modules/utils/llm_prompts.py` -> `analysis/insights/prompts_legacy.py` or delete if superseded
- `modules/utils/cfg.py` -> `pipeline/common/config.py`

### `modules/analysis/*`

MOVE:

- `modules/analysis/alpha_flavors.py` -> `analysis/regimes/alpha_flavors.py`

Reason:

- this is pure downstream research/analysis

### `modules/api.py`

SPLIT and RETIRE.

Do not move this file whole. It is a mixed helper bag.

Move functions from `modules/api.py` into these owners:

- feature summarization helpers -> `analysis/diagnostics/`
- dataset and artifact builders -> `pipeline/` or `data/`
- model training / prediction wrappers -> `ml/`
- backtest helpers -> `backtest/`
- formatting helpers -> `features/feature_presentation` or `analysis/insights`

Then delete `modules/api.py`.

### `modules/qcore/*`

MOVE selectively:

- `modules/qcore/contracts.py` -> `ml/contracts.py` or `pipeline/contracts.py`
- `modules/qcore/window.py` -> `data/window.py`

### `modules/config.py`, `modules/schema.py`, `modules/signals/*`

Review individually:

- `modules/config.py` -> likely `pipeline/common/config.py` or delete
- `modules/schema.py` -> likely `data/schema.py` or delete
- `modules/signals/predictors.py` -> `analysis/signals/predictors.py` if still used, otherwise delete

## 5. UI Modules Should Become Thin

These files should remain as route/form/template entrypoints, but stop owning
business logic.

SPLIT:

- `pipeline/views.py`
- `fmp/views.py`
- `features/views.py`
- `labels/views.py`
- `ml/views.py`

The target rule is:

- view reads request
- view calls service
- view renders response

No heavy dataframe building, report assembly, or scoring logic in the view file.

## 6. Canonical Owner Decisions for the Files You Asked About

### Should feature attribution live inside `features/`?

No.

Canonical answer:

- `features/` owns feature production
- `analysis/feature_attribution.py` owns downstream attribution

### Should oracle-related code live inside `labels/`?

Only oracle label generation.

Canonical answer:

- `labels/` owns oracle trade recipes and label generation
- `analysis/oracle/*` owns oracle coverage, oracle clustering, oracle reports,
  and oracle-based evaluation

## 7. First Refactor Wave

These are the highest-value moves to do first.

1. Create `analysis/`, `data/`, and `backtest/`.
2. Split `pipeline/views.py`.
3. Move:
   - `pipeline/feature_attribution.py`
   - `pipeline/oracle_reports.py`
   - `pipeline/oracle_state_dataset.py`
   - `pipeline/diagnostics.py`
   - `pipeline/insights.py`
   into `analysis/`.
4. Consolidate duplicate helpers shared by:
   - `pipeline/cohort_runner.py`
   - `pipeline/feature_attribution.py`
   - `pipeline/research_suite.py`
5. Merge `modules/features/*` into `features/*`.
6. Merge `modules/labels/*` into `labels/*`.
7. Start moving `modules/models/*` into `ml/`.
8. Break the remaining cycles:
   - `fmp.tasks <-> fmp.views`
   - `modules.models <-> modules.models.stable_baselines3`

## 8. Success Criteria

The architecture is improving when:

- new feature logic is never added to `modules/`
- every workflow has one canonical implementation
- `pipeline/` becomes orchestration only
- `analysis/` becomes the single home for evaluation and market intelligence
- views stop containing dataframe/report/business logic
- `modules/` shrinks every week until it can be deleted
