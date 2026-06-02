from __future__ import annotations


def build_yearly_folds(start_year: int, end_year: int) -> list[dict[str, str]]:
    """Return walk-forward yearly fold specs for backtest research."""
    return [
        {
            "name": f"wf_{year}",
            "train_end_date": f"{year - 1}-12-31",
            "backtest_start_date": f"{year}-01-01",
            "backtest_end_date": f"{year}-12-31",
        }
        for year in range(int(start_year), int(end_year) + 1)
    ]
