# ============================================================
# modules/data/context.py
# Shared context for data access (API + SQLite + runtime knobs)
# ============================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from modules.data.storage import SQLiteStore


@dataclass(frozen=True)
class DataContext:
    """
    Central object you pass around instead of:
      - api_key
      - db_path / store
      - sleep_s
      - verbose flags
      - data_dir

    history_years:
      - If set (e.g., 30), the prices loader can backfill older history
        when your DB only contains a shorter window (e.g., 5y).
      - Leave None to disable earliest-history backfill.
    """
    api_key: str
    store: SQLiteStore
    sleep_s: float = 0.0
    verbose: bool = True
    history_years: Optional[int] = None  # ✅ NEW

    @staticmethod
    def from_data_dir(
        *,
        api_key: str,
        data_dir: str,
        db_name: str = "quant.db",
        sleep_s: float = 0.0,
        verbose: bool = True,
        history_years: Optional[int] = None,  # ✅ NEW
    ) -> "DataContext":
        store = SQLiteStore(db_path=f"{data_dir.rstrip('/')}/{db_name}")
        store.init_schema()
        return DataContext(
            api_key=api_key,
            store=store,
            sleep_s=sleep_s,
            verbose=verbose,
            history_years=history_years,  # ✅ NEW
        )
