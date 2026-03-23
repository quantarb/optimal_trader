from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


STATE_NUMERIC_COLS = ["adj_open", "adj_high", "adj_low", "adj_close", "volume"]
ENTRY_NUMERIC_COLS = [f"entry_{col}" for col in STATE_NUMERIC_COLS]
EXIT_NUMERIC_COLS = [f"exit_{col}" for col in STATE_NUMERIC_COLS]


def build_state_text(date_text: str, symbol: str, company_name: str, sector: str, industry: str) -> str:
    return (
        f"date={date_text} ; symbol={symbol} ; company_name={company_name} ; "
        f"sector={sector} ; industry={industry}"
    )


def map_direction_label(side: str) -> str:
    normalized = str(side or "").strip().lower()
    if normalized == "long":
        return "long"
    if normalized == "short":
        return "short"
    raise ValueError(f"Unsupported side for direction label: {side}")


def build_symbol_metadata_lookup(universe_df: pd.DataFrame) -> pd.DataFrame:
    metadata_lookup_df = universe_df[["symbol", "company_name", "sector", "industry"]].drop_duplicates("symbol").copy()
    metadata_lookup_df["symbol"] = metadata_lookup_df["symbol"].astype(str).str.strip().str.upper()
    for col in ["company_name", "sector", "industry"]:
        metadata_lookup_df[col] = metadata_lookup_df[col].fillna("").astype(str).str.strip().replace("", "Unknown")
    return metadata_lookup_df


def build_trade_rows_from_oracle_sources(
    *,
    label_df: pd.DataFrame,
    completed_trades_df: pd.DataFrame,
) -> list[dict[str, object]]:
    trade_rows: list[dict[str, object]] = []
    label_pair_columns = {"trade_id", "event", "action_label", "symbol", "entry_date", "exit_date", "trade_return", "hold_days", "side"}
    if not label_df.empty and label_pair_columns.issubset(set(label_df.columns)):
        label_pairs_df = label_df.copy()
        label_pairs_df["trade_id"] = label_pairs_df["trade_id"].astype(str)
        label_pairs_df["event"] = label_pairs_df["event"].fillna("").astype(str).str.strip().str.lower()
        label_pairs_df["action_label"] = (
            label_pairs_df["action_label"].fillna(label_pairs_df.get("label", "")).astype(str).str.strip().str.lower()
        )
        for trade_id, trade_group in label_pairs_df.groupby("trade_id", sort=False):
            if not trade_id or str(trade_id).strip().lower() in {"", "nan"}:
                continue
            entry_rows = trade_group.loc[trade_group["event"] == "entry"]
            exit_rows = trade_group.loc[trade_group["event"] == "exit"]
            if entry_rows.empty or exit_rows.empty:
                continue

            entry_row = entry_rows.iloc[0]
            exit_row = exit_rows.iloc[0]
            side = str(entry_row.get("side") or exit_row.get("side") or "").strip().lower()
            symbol = str(entry_row.get("symbol") or exit_row.get("symbol") or "").strip().upper()
            entry_date = pd.to_datetime(entry_row.get("date", entry_row.get("entry_date")), errors="coerce")
            exit_date = pd.to_datetime(exit_row.get("date", exit_row.get("exit_date")), errors="coerce")
            hold_days = pd.to_numeric(entry_row.get("hold_days", exit_row.get("hold_days")), errors="coerce")
            ret_dec = pd.to_numeric(entry_row.get("trade_return", exit_row.get("trade_return")), errors="coerce")
            entry_action = str(entry_row.get("action_label") or entry_row.get("label") or "").strip().lower()
            exit_action = str(exit_row.get("action_label") or exit_row.get("label") or "").strip().lower()

            if not symbol or side not in {"long", "short"}:
                continue
            if entry_action not in {"buy", "short"} or exit_action not in {"sell", "cover"}:
                continue
            if pd.isna(entry_date) or pd.isna(exit_date) or pd.isna(hold_days) or pd.isna(ret_dec):
                continue

            trade_rows.append(
                {
                    "trade_id": str(trade_id),
                    "symbol": symbol,
                    "side": side,
                    "entry_action": entry_action,
                    "exit_action": exit_action,
                    "entry_date": pd.Timestamp(entry_date),
                    "exit_date": pd.Timestamp(exit_date),
                    "hold_days": float(hold_days),
                    "ret_dec": float(ret_dec),
                    "freq": entry_row.get("freq", exit_row.get("freq")),
                    "k": entry_row.get("k", exit_row.get("k")),
                }
            )
    else:
        for trade_id, row in enumerate(completed_trades_df.to_dict(orient="records")):
            side = str(row.get("side") or "").strip().lower()
            symbol = str(row.get("symbol") or "").strip().upper()
            entry_date = pd.to_datetime(row.get("entry_date"), errors="coerce")
            exit_date = pd.to_datetime(row.get("exit_date"), errors="coerce")
            hold_days = pd.to_numeric(row.get("hold_days"), errors="coerce")
            ret_dec = pd.to_numeric(row.get("ret_dec"), errors="coerce")

            if not symbol or side not in {"long", "short"}:
                continue
            if pd.isna(entry_date) or pd.isna(exit_date) or pd.isna(hold_days) or pd.isna(ret_dec):
                continue

            entry_action, exit_action = ("buy", "sell") if side == "long" else ("short", "cover")
            trade_rows.append(
                {
                    "trade_id": str(trade_id),
                    "symbol": symbol,
                    "side": side,
                    "entry_action": entry_action,
                    "exit_action": exit_action,
                    "entry_date": pd.Timestamp(entry_date),
                    "exit_date": pd.Timestamp(exit_date),
                    "hold_days": float(hold_days),
                    "ret_dec": float(ret_dec),
                    "freq": row.get("freq"),
                    "k": row.get("k"),
                }
            )
    return trade_rows


def attach_trade_percentile_targets(trade_pair_df: pd.DataFrame) -> pd.DataFrame:
    out = trade_pair_df.copy()
    out["trade_return_pct"] = out["ret_dec"].rank(method="average", pct=True)
    out["signed_ret_dec"] = np.where(out["side"].astype(str).str.lower() == "long", out["ret_dec"], -out["ret_dec"])
    out["signed_trade_return_pct"] = out["signed_ret_dec"].rank(method="average", pct=True)
    out["duration_pct"] = out["hold_days"].rank(method="average", pct=True)
    out["inverse_duration_pct"] = 1.0 - out["duration_pct"]
    return out


def build_trade_pair_frame(
    *,
    universe_df: pd.DataFrame,
    label_df: pd.DataFrame,
    completed_trades_df: pd.DataFrame,
    price_lookup_df: pd.DataFrame,
) -> pd.DataFrame:
    trade_rows = build_trade_rows_from_oracle_sources(label_df=label_df, completed_trades_df=completed_trades_df)
    trade_pair_df = pd.DataFrame(trade_rows)
    if trade_pair_df.empty:
        raise ValueError("No trade pairs could be built from oracle labels or completed trades.")

    metadata_lookup_df = build_symbol_metadata_lookup(universe_df)
    trade_pair_df = trade_pair_df.merge(metadata_lookup_df, on="symbol", how="left")
    for col in ["company_name", "sector", "industry"]:
        trade_pair_df[col] = trade_pair_df[col].fillna("").astype(str).str.strip().replace("", "Unknown")

    trade_pair_df["entry_date_text"] = pd.to_datetime(trade_pair_df["entry_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    trade_pair_df["exit_date_text"] = pd.to_datetime(trade_pair_df["exit_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    entry_price_lookup_df = price_lookup_df.rename(
        columns={
            "date_text": "entry_date_text",
            "adj_open": "entry_adj_open",
            "adj_high": "entry_adj_high",
            "adj_low": "entry_adj_low",
            "adj_close": "entry_adj_close",
            "volume": "entry_volume",
        }
    )
    exit_price_lookup_df = price_lookup_df.rename(
        columns={
            "date_text": "exit_date_text",
            "adj_open": "exit_adj_open",
            "adj_high": "exit_adj_high",
            "adj_low": "exit_adj_low",
            "adj_close": "exit_adj_close",
            "volume": "exit_volume",
        }
    )
    trade_pair_df = trade_pair_df.merge(entry_price_lookup_df, on=["symbol", "entry_date_text"], how="left")
    trade_pair_df = trade_pair_df.merge(exit_price_lookup_df, on=["symbol", "exit_date_text"], how="left")
    for col in ENTRY_NUMERIC_COLS + EXIT_NUMERIC_COLS:
        trade_pair_df[col] = pd.to_numeric(trade_pair_df[col], errors="coerce")

    trade_pair_df = attach_trade_percentile_targets(trade_pair_df)
    trade_pair_df["entry_direction_label"] = trade_pair_df["side"].map(map_direction_label)
    trade_pair_df["exit_direction_label"] = trade_pair_df["side"].map(map_direction_label)
    trade_pair_df["entry_text"] = trade_pair_df.apply(
        lambda row: build_state_text(
            str(row["entry_date_text"]),
            str(row["symbol"]),
            str(row["company_name"]),
            str(row["sector"]),
            str(row["industry"]),
        ),
        axis=1,
    )
    trade_pair_df["exit_text"] = trade_pair_df.apply(
        lambda row: build_state_text(
            str(row["exit_date_text"]),
            str(row["symbol"]),
            str(row["company_name"]),
            str(row["sector"]),
            str(row["industry"]),
        ),
        axis=1,
    )
    return trade_pair_df


def build_state_frame_from_trade_pairs(trade_pair_df: pd.DataFrame) -> pd.DataFrame:
    state_rows: list[dict[str, object]] = []
    for row in trade_pair_df.to_dict(orient="records"):
        state_rows.append(
            {
                "trade_id": row["trade_id"],
                "symbol": row["symbol"],
                "side": row["side"],
                "event_role": "entry",
                "action": row["entry_action"],
                "direction_label": row["entry_direction_label"],
                "date_text": row["entry_date_text"],
                "company_name": row["company_name"],
                "sector": row["sector"],
                "industry": row["industry"],
                "trade_return_pct": row["trade_return_pct"],
                "signed_trade_return_pct": row["signed_trade_return_pct"],
                "hold_days": row["hold_days"],
                "adj_open": row["entry_adj_open"],
                "adj_high": row["entry_adj_high"],
                "adj_low": row["entry_adj_low"],
                "adj_close": row["entry_adj_close"],
                "volume": row["entry_volume"],
                "text": row["entry_text"],
            }
        )
        state_rows.append(
            {
                "trade_id": row["trade_id"],
                "symbol": row["symbol"],
                "side": row["side"],
                "event_role": "exit",
                "action": row["exit_action"],
                "direction_label": row["exit_direction_label"],
                "date_text": row["exit_date_text"],
                "company_name": row["company_name"],
                "sector": row["sector"],
                "industry": row["industry"],
                "trade_return_pct": row["trade_return_pct"],
                "signed_trade_return_pct": row["signed_trade_return_pct"],
                "hold_days": row["hold_days"],
                "adj_open": row["exit_adj_open"],
                "adj_high": row["exit_adj_high"],
                "adj_low": row["exit_adj_low"],
                "adj_close": row["exit_adj_close"],
                "volume": row["exit_volume"],
                "text": row["exit_text"],
            }
        )

    state_df = pd.DataFrame(state_rows)
    if state_df.empty:
        raise ValueError("No state rows could be built from completed trades.")
    for col in STATE_NUMERIC_COLS:
        state_df[col] = pd.to_numeric(state_df[col], errors="coerce")
    return state_df


def build_oracle_entry_exit_frames(
    *,
    universe_df: pd.DataFrame,
    label_df: pd.DataFrame,
    completed_trades_df: pd.DataFrame,
    price_lookup_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trade_pair_df = build_trade_pair_frame(
        universe_df=universe_df,
        label_df=label_df,
        completed_trades_df=completed_trades_df,
        price_lookup_df=price_lookup_df,
    )
    state_df = build_state_frame_from_trade_pairs(trade_pair_df)
    return trade_pair_df, state_df


__all__ = [
    "STATE_NUMERIC_COLS",
    "ENTRY_NUMERIC_COLS",
    "EXIT_NUMERIC_COLS",
    "attach_trade_percentile_targets",
    "build_oracle_entry_exit_frames",
    "build_state_frame_from_trade_pairs",
    "build_state_text",
    "build_symbol_metadata_lookup",
    "build_trade_pair_frame",
    "build_trade_rows_from_oracle_sources",
    "map_direction_label",
]
