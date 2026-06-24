from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from quant_warehouse.target_engineering import (
    add_rank_regression_labels,
    build_label_panel,
    deduplicate_labels,
)


def _summarize_labels_for_llm(df: pd.DataFrame, dedup_stats: Optional[Dict] = None) -> None:
    """Prints a structured table for Oracle performance and deduplication stats."""
    print("\n" + "=" * 80)
    print("  ORACLE LABEL PERFORMANCE & DEDUPLICATION SUMMARY")
    print("=" * 80)

    if dedup_stats:
        print("DEDUPLICATION METRICS:")
        print(f"  - Raw Signal Count:    {dedup_stats['raw_count']:,}")
        print(f"  - Unique Signal Count: {dedup_stats['unique_count']:,}")
        print(f"  - Redundancy Removed:  {dedup_stats['pct_removed']:.1f}%")
        print("-" * 80)

    if "trade_return" not in df.columns or "side" not in df.columns:
        print("Missing required columns for performance statistics.")
        return

    report_rows = []
    horizons = sorted(df["horizon"].unique())

    for horizon in horizons:
        h_df = df[df["horizon"] == horizon]
        for side in ["long", "short"]:
            s_df = h_df[h_df["side"] == side]
            if s_df.empty:
                continue

            n_trades = len(s_df) // 2
            avg_ret = s_df["trade_return"].mean()
            win_rate = (s_df["trade_return"] > 0).mean()

            avg_dur = np.nan
            if "trade_duration_days" in s_df.columns:
                avg_dur = s_df["trade_duration_days"].mean()

            report_rows.append(
                {
                    "Horizon": horizon,
                    "Side": "BUY" if side == "long" else "SHORT",
                    "Trades": n_trades,
                    "Mean Return %": round(avg_ret * 100, 2),
                    "Win Rate %": round(win_rate * 100, 1),
                    "Avg Duration": round(avg_dur, 1),
                }
            )

    summary_table = pd.DataFrame(report_rows)
    if not summary_table.empty:
        def _extract_k(h):
            import re
            match = re.search(r"k(\d+)", str(h))
            return int(match.group(1)) if match else 0

        summary_table["_k_val"] = summary_table["Horizon"].apply(_extract_k)
        summary_table = summary_table.sort_values(["_k_val", "Side"], ascending=[True, False]).drop(
            columns=["_k_val"]
        )
        print(summary_table.to_string(index=False))

    print("=" * 80 + "\n")


def build_label_dataframe(
    *,
    daily_by_symbol: Dict[str, pd.DataFrame],
    k_params: Dict[str, Union[int, List[int]]],
    execution_params: Dict[str, Any],
    weighting: Dict[str, Any],
    add_rank_labels: bool = True,
    deduplicate: bool = True,
    verbose: bool = True,
    max_workers: int = 1,
) -> pd.DataFrame:
    """Standard API entry point with deduplication tracking.

    Set max_workers > 1 to process symbols in parallel via ProcessPoolExecutor.
    """

    df_raw = build_label_panel(
        daily_by_symbol=daily_by_symbol,
        k_params=k_params,
        execution_params=execution_params,
        weighting=weighting,
        add_rank_labels=False,
        deduplicate=False,
        max_workers=max_workers,
    )

    raw_count = len(df_raw)

    if deduplicate:
        df_final = deduplicate_labels(df_raw)
    else:
        df_final = df_raw

    if add_rank_labels:
        df_final = add_rank_regression_labels(df_final)

    unique_count = len(df_final)
    pct_removed = ((raw_count - unique_count) / raw_count * 100) if raw_count > 0 else 0

    stats = {
        "raw_count": raw_count,
        "unique_count": unique_count,
        "pct_removed": pct_removed,
    }

    if verbose and not df_final.empty:
        _summarize_labels_for_llm(df_final, dedup_stats=stats if deduplicate else None)

    return df_final


__all__ = ["build_label_dataframe"]
