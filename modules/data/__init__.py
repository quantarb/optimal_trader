
"""Data access layer.

Key modules:
  - prices_sqlite: SQLite-first daily prices cache + incremental refresh
  - fmp_client: lightweight FinancialModelingPrep client (prices, quotes, fundamentals)
"""

from modules.data.fmp_client import FMPClient, fundamentals_to_daily_panel
from modules.data.pit import asof_join_pit, broadcast_asof_to_target_index
from modules.data.preparation import (
    Entry2ExitTextConfig,
    MLDatasetConfig,
    prepare_entry2exit_dataset,
    prepare_ml_dataset,
)
