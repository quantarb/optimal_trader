from __future__ import annotations

import pandas as pd


def compute_cluster_outcome_stats(assignments: pd.DataFrame) -> pd.DataFrame:
    if assignments.empty:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "cluster_code",
                "side",
                "sample_size",
                "median_return",
                "mean_return",
                "win_rate",
                "worst_case",
                "best_case",
                "avg_hold_days",
                "return_std",
                "yearly_median_return_std",
                "yearly_win_rate_std",
            ]
        )
    work = assignments.copy()
    work["trade_return"] = pd.to_numeric(work["trade_return"], errors="coerce")
    work["hold_days"] = pd.to_numeric(work["hold_days"], errors="coerce")
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["cluster_id", "trade_return", "hold_days"]).copy()
    if work.empty:
        return pd.DataFrame()
    work["year"] = work["date"].dt.year.fillna(0).astype(int)

    grouped = (
        work.groupby(["cluster_id", "cluster_code", "side"], observed=True)
        .agg(
            sample_size=("cluster_id", "size"),
            median_return=("trade_return", "median"),
            mean_return=("trade_return", "mean"),
            win_rate=("trade_return", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            worst_case=("trade_return", "min"),
            best_case=("trade_return", "max"),
            avg_hold_days=("hold_days", "mean"),
            return_std=("trade_return", lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=0) if len(s) > 1 else 0.0)),
        )
        .reset_index()
    )

    yearly_rows = (
        work.groupby(["cluster_id", "year"], observed=True)
        .agg(
            yearly_median_return=("trade_return", "median"),
            yearly_win_rate=("trade_return", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
        )
        .reset_index()
    )
    yearly_stats = (
        yearly_rows.groupby("cluster_id", observed=True)
        .agg(
            yearly_median_return_std=("yearly_median_return", lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=0) if len(s) > 1 else 0.0)),
            yearly_win_rate_std=("yearly_win_rate", lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=0) if len(s) > 1 else 0.0)),
        )
        .reset_index()
    )
    merged = grouped.merge(yearly_stats, on="cluster_id", how="left")
    numeric_cols = [
        "median_return",
        "mean_return",
        "win_rate",
        "worst_case",
        "best_case",
        "avg_hold_days",
        "return_std",
        "yearly_median_return_std",
        "yearly_win_rate_std",
    ]
    for column in numeric_cols:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
        merged[column] = merged[column].round(6)
    merged["sample_size"] = pd.to_numeric(merged["sample_size"], errors="coerce").fillna(0).astype(int)
    merged["cluster_code"] = pd.to_numeric(merged["cluster_code"], errors="coerce").fillna(0).astype(int)
    return merged.sort_values(["side", "sample_size", "median_return"], ascending=[True, False, False]).reset_index(drop=True)
