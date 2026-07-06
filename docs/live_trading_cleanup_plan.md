# optimal_trader Live-Trading Boundary

`optimal_trader` is now the live execution app. Historical data refresh,
feature engineering, target engineering, backtesting, model training, and
research reporting are owned by the GitHub-installed downstream packages:

- `quant-warehouse`
- `quant-orchestrator`
- `tradingagents`

## Kept Here

- `platforms/brokers/`: real broker adapters for Alpaca, Robinhood, and
  Interactive Brokers.
- `platforms/agents/`: LLM/agent adapters such as TradingAgents.
- `trading/`: the minimal Django app that reads v2 live artifacts.
- `app/trading_app_v2_runtime.py`: thin notebook/runtime glue that delegates
  data and model work downstream, builds live order plans, and writes the
  Streamlit app.
- `notebooks/trading_app_v2.ipynb`: the live trading control notebook.
- `templates/trading/`: the minimal Django leaderboard reader for v2 artifacts.
- Focused broker/order tests.

## Removed Here

The old all-in-one research stack has been deleted from this repo, including
local FMP ingestion, local feature and label builders, local backtesting,
local model training, research reports, stale generated artifacts, Celery
workers, and legacy notebooks.

## Import Boundary

New live code may import:

- `quant_warehouse`
- `quant_orchestrator`
- `platforms.brokers.*`
- `platforms.agents.*`
- `app.trading_app_v2_runtime`

New live code should not recreate local warehouse, feature, target, backtest,
or ML-training modules in this repo.
