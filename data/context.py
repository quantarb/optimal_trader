# ============================================================
# data/context.py
# Shared context for data access (API + runtime knobs)
# ============================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DataContext:
    """
    Central object you pass around instead of:
      - api_key
      - data_dir / db_name metadata
      - sleep_s
      - verbose flags

    history_years:
      - Retained for compatibility with older callers.
    """
    api_key: str
    data_dir: str = "."
    db_name: str = "quant.db"
    sleep_s: float = 0.0
    verbose: bool = True
    history_years: Optional[int] = None

    @staticmethod
    def from_data_dir(
        *,
        api_key: str,
        data_dir: str,
        db_name: str = "quant.db",
        sleep_s: float = 0.0,
        verbose: bool = True,
        history_years: Optional[int] = None,
    ) -> "DataContext":
        return DataContext(
            api_key=api_key,
            data_dir=data_dir,
            db_name=db_name,
            sleep_s=sleep_s,
            verbose=verbose,
            history_years=history_years,
        )
