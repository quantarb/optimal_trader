import pandas as pd
import numpy as np

def labels_panel_to_trades_df(label_df: pd.DataFrame) -> pd.DataFrame:
    df = label_df.copy()

    # --- ensure we have a date column
    if "date" not in df.columns:
        if isinstance(df.index, pd.MultiIndex) and "date" in df.index.names:
            df = df.reset_index()
        elif df.index.name == "date":
            df = df.reset_index()
        else:
            raise ValueError("label_df must have a 'date' column or an index level named 'date'.")

    df["date"] = pd.to_datetime(df["date"])

    # --- ensure symbol exists
    if "symbol" not in df.columns:
        if isinstance(label_df.index, pd.MultiIndex) and "symbol" in label_df.index.names:
            df["symbol"] = label_df.index.get_level_values("symbol")
        else:
            raise ValueError("label_df must include 'symbol' column or index level 'symbol'.")

    # Case A: already has entry/exit columns
    if {"entry_date", "exit_date"}.issubset(df.columns):
        out = df.copy()
        out["entry_date"] = pd.to_datetime(out["entry_date"])
        out["exit_date"] = pd.to_datetime(out["exit_date"])
        cols = ["symbol", "entry_date", "exit_date"]
        for c in ["side", "horizon", "entry_px", "exit_px", "trade_return", "trade_duration_days", "sample_weight"]:
            if c in out.columns:
                cols.append(c)
        out = out[cols].dropna(subset=["entry_date", "exit_date"])
        out = out[out["exit_date"] > out["entry_date"]].sort_values(["symbol", "entry_date"]).reset_index(drop=True)
        return out

    # Case B: there is an explicit trade/pair id
    id_candidates = [c for c in ["trade_id", "pair_id", "signal_id", "event_id"] if c in df.columns]
    if id_candidates:
        tid = id_candidates[0]
        # expect one "entry" row and one "exit" row per tid
        # If you have an explicit event/action column, use it; otherwise we use earliest/ latest.
        event_col = None
        for c in ["event", "action", "event_type", "kind"]:
            if c in df.columns:
                event_col = c
                break

        rows = []
        for (sym, trade_id), g in df.groupby(["symbol", tid], sort=False):
            g = g.sort_values("date")
            entry_row = g.iloc[0]
            exit_row = g.iloc[-1]

            if event_col:
                # try to pick entry/exit by text if possible
                lc = g[event_col].astype(str).str.lower()
                entry_idx = lc[lc.str.contains("entry|buy|open")].index
                exit_idx  = lc[lc.str.contains("exit|sell|close|cover")].index
                if len(entry_idx) > 0:
                    entry_row = g.loc[entry_idx[0]]
                if len(exit_idx) > 0:
                    exit_row = g.loc[exit_idx[-1]]

            row = {
                "symbol": sym,
                "entry_date": pd.to_datetime(entry_row["date"]),
                "exit_date": pd.to_datetime(exit_row["date"]),
            }
            for c in ["side", "horizon", "entry_px", "exit_px", "trade_return", "trade_duration_days", "sample_weight"]:
                if c in g.columns:
                    row[c] = entry_row.get(c, np.nan)
                    # some fields make more sense from the exit row
                    if c in ["exit_px", "trade_return", "trade_duration_days"]:
                        row[c] = exit_row.get(c, row[c])
            rows.append(row)

        out = pd.DataFrame(rows).dropna(subset=["entry_date", "exit_date"])
        out = out[out["exit_date"] > out["entry_date"]].sort_values(["symbol", "entry_date"]).reset_index(drop=True)
        return out

    # Case C: No id — pair consecutive rows (common when there are exactly 2 events per trade)
    # We pair (0,1), (2,3), ... within each (symbol,horizon,side) bucket.
    group_cols = ["symbol"]
    if "horizon" in df.columns:
        group_cols.append("horizon")
    if "side" in df.columns:
        group_cols.append("side")

    rows = []
    for keys, g in df.groupby(group_cols, sort=False):
        g = g.sort_values("date").reset_index(drop=True)

        # Pair consecutive rows
        n = len(g)
        if n < 2:
            continue
        # if odd, drop the last dangling row
        if n % 2 == 1:
            g = g.iloc[:-1, :]
            n -= 1

        for i in range(0, n, 2):
            entry_row = g.iloc[i]
            exit_row  = g.iloc[i + 1]

            row = {
                "symbol": entry_row["symbol"],
                "entry_date": entry_row["date"],
                "exit_date": exit_row["date"],
            }
            for c in ["side", "horizon", "entry_px", "exit_px", "trade_return", "trade_duration_days", "sample_weight"]:
                if c in g.columns:
                    row[c] = entry_row.get(c, np.nan)
                    if c in ["exit_px", "trade_return", "trade_duration_days"]:
                        row[c] = exit_row.get(c, row[c])
            rows.append(row)

    out = pd.DataFrame(rows).dropna(subset=["entry_date", "exit_date"])
    out = out[out["exit_date"] > out["entry_date"]].sort_values(["symbol", "entry_date"]).reset_index(drop=True)
    return out
