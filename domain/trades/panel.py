from __future__ import annotations

import numpy as np
import pandas as pd


def _assign_trade_id(out: pd.DataFrame) -> pd.DataFrame:
    if "trade_id" in out.columns and out["trade_id"].notna().all():
        return out

    base_cols = {
        "symbol": out.get("symbol", pd.Series([""] * len(out))).astype(str),
        "entry_date": pd.to_datetime(out.get("entry_date"), errors="coerce").dt.strftime("%Y%m%d"),
        "exit_date": pd.to_datetime(out.get("exit_date"), errors="coerce").dt.strftime("%Y%m%d"),
        "side": out.get("side", pd.Series([""] * len(out))).astype(str),
        "horizon": out.get("horizon", pd.Series([""] * len(out))).astype(str),
    }
    base = (
        "T|"
        + base_cols["symbol"].fillna("")
        + "|E" + base_cols["entry_date"].fillna("NA")
        + "|X" + base_cols["exit_date"].fillna("NA")
        + "|S" + base_cols["side"].fillna("")
        + "|H" + base_cols["horizon"].fillna("")
    )
    dup_n = base.groupby(base).cumcount()
    out["trade_id"] = np.where(dup_n > 0, base + "|N" + dup_n.astype(str), base)
    return out


def _finalize_trades_df(out: pd.DataFrame, *, trade_id_as_index: bool) -> pd.DataFrame:
    out = out.dropna(subset=["entry_date", "exit_date"])
    out = out[out["exit_date"] > out["entry_date"]].sort_values(["symbol", "entry_date", "exit_date"]).reset_index(drop=True)
    out = _assign_trade_id(out)
    if trade_id_as_index:
        out = out.set_index("trade_id", drop=False).sort_index()
    return out


def labels_panel_to_trades_df(label_df: pd.DataFrame, *, trade_id_as_index: bool = False) -> pd.DataFrame:
    df = label_df.copy()
    if "date" not in df.columns:
        if isinstance(df.index, pd.MultiIndex) and "date" in df.index.names:
            df = df.reset_index()
        elif df.index.name == "date":
            df = df.reset_index()
        else:
            raise ValueError("label_df must have a 'date' column or an index level named 'date'.")
    df["date"] = pd.to_datetime(df["date"])
    if "symbol" not in df.columns:
        if isinstance(label_df.index, pd.MultiIndex) and "symbol" in label_df.index.names:
            df["symbol"] = label_df.index.get_level_values("symbol")
        else:
            raise ValueError("label_df must include 'symbol' column or index level 'symbol'.")

    if {"entry_date", "exit_date"}.issubset(df.columns):
        out = df.copy()
        out["entry_date"] = pd.to_datetime(out["entry_date"])
        out["exit_date"] = pd.to_datetime(out["exit_date"])
        cols = ["symbol", "entry_date", "exit_date"]
        if "trade_id" in out.columns:
            cols.append("trade_id")
        for column in ["side", "horizon", "entry_px", "exit_px", "trade_return", "trade_duration_days", "sample_weight"]:
            if column in out.columns:
                cols.append(column)
        return _finalize_trades_df(out[cols], trade_id_as_index=trade_id_as_index)

    id_candidates = [column for column in ["trade_id", "pair_id", "signal_id", "event_id"] if column in df.columns]
    if id_candidates:
        trade_id_column = id_candidates[0]
        event_col = next((column for column in ["event", "action", "event_type", "kind"] if column in df.columns), None)
        rows = []
        for (symbol, trade_id), group in df.groupby(["symbol", trade_id_column], sort=False):
            group = group.sort_values("date")
            entry_row = group.iloc[0]
            exit_row = group.iloc[-1]
            if event_col:
                lc = group[event_col].astype(str).str.lower()
                entry_idx = lc[lc.str.contains("entry|buy|open")].index
                exit_idx = lc[lc.str.contains("exit|sell|close|cover")].index
                if len(entry_idx) > 0:
                    entry_row = group.loc[entry_idx[0]]
                if len(exit_idx) > 0:
                    exit_row = group.loc[exit_idx[-1]]
            row = {
                "symbol": symbol,
                "entry_date": pd.to_datetime(entry_row["date"]),
                "exit_date": pd.to_datetime(exit_row["date"]),
                "trade_id": str(trade_id),
            }
            for column in ["side", "horizon", "entry_px", "exit_px", "trade_return", "trade_duration_days", "sample_weight"]:
                if column in group.columns:
                    row[column] = entry_row.get(column, np.nan)
                    if column in ["exit_px", "trade_return", "trade_duration_days"]:
                        row[column] = exit_row.get(column, row[column])
            rows.append(row)
        return _finalize_trades_df(pd.DataFrame(rows), trade_id_as_index=trade_id_as_index)

    group_cols = ["symbol"]
    if "horizon" in df.columns:
        group_cols.append("horizon")
    if "side" in df.columns:
        group_cols.append("side")

    rows = []
    for _, group in df.groupby(group_cols, sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        count = len(group)
        if count < 2:
            continue
        if count % 2 == 1:
            group = group.iloc[:-1, :]
            count -= 1
        for index in range(0, count, 2):
            entry_row = group.iloc[index]
            exit_row = group.iloc[index + 1]
            row = {
                "symbol": entry_row["symbol"],
                "entry_date": entry_row["date"],
                "exit_date": exit_row["date"],
                "trade_id": (
                    f"PAIR|{entry_row['symbol']}|"
                    f"{pd.Timestamp(entry_row['date']).strftime('%Y%m%d')}|"
                    f"{pd.Timestamp(exit_row['date']).strftime('%Y%m%d')}|{index // 2}"
                ),
            }
            for column in ["side", "horizon", "entry_px", "exit_px", "trade_return", "trade_duration_days", "sample_weight"]:
                if column in group.columns:
                    row[column] = entry_row.get(column, np.nan)
                    if column in ["exit_px", "trade_return", "trade_duration_days"]:
                        row[column] = exit_row.get(column, row[column])
            rows.append(row)
    return _finalize_trades_df(pd.DataFrame(rows), trade_id_as_index=trade_id_as_index)

