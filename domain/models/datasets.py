from __future__ import annotations

import pandas as pd


def filter_frame_by_date(
    df: pd.DataFrame,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Apply an inclusive date window to a frame with a `date` column."""

    if df.empty or "date" not in df.columns:
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    if start_date:
        out = out[out["date"] >= pd.Timestamp(str(start_date))]
    if end_date:
        out = out[out["date"] <= pd.Timestamp(str(end_date))]
    return out.reset_index(drop=True)


def dedupe_label_frame(label_df: pd.DataFrame) -> pd.DataFrame:
    """Keep the highest-absolute-return label per (date, symbol)."""

    if label_df.empty or "date" not in label_df.columns or "symbol" not in label_df.columns:
        return label_df
    out = label_df.copy()
    if "trade_return" in out.columns:
        out["__trade_return_abs"] = pd.to_numeric(out["trade_return"], errors="coerce").abs().fillna(-1.0)
    else:
        out["__trade_return_abs"] = -1.0
    if "hold_days" in out.columns:
        out["__hold_days_num"] = pd.to_numeric(out["hold_days"], errors="coerce").fillna(10**9)
    else:
        out["__hold_days_num"] = 10**9
    out = out.sort_values(["date", "symbol", "__trade_return_abs", "__hold_days_num"], ascending=[True, True, False, True])
    out = out.drop_duplicates(subset=["date", "symbol"], keep="first")
    out = out.drop(columns=["__trade_return_abs", "__hold_days_num"], errors="ignore")
    return out.reset_index(drop=True)


def feature_columns_from_frame(feature_df: pd.DataFrame) -> list[str]:
    """Return all non-key columns from a feature frame."""

    return [str(col) for col in feature_df.columns if col not in {"date", "symbol"}]

