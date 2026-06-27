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
