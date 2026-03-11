# ============================================================
# modules/data/storage.py
# SQLite storage layer (prices + meta + symbol_status)
#
# NOTE:
# - Removed dependency on sqlite-utils (not installed by default).
# - Uses stdlib sqlite3 + pandas for a small, explicit implementation.
#
# Public API preserved (so modules/data/prices.py can stay unchanged):
#   - init_schema()
#   - get_meta(key), set_meta(key, value)
#   - load_prices_daily(symbol) -> DataFrame indexed by date
#   - upsert_prices_daily(df, symbol) -> int rows
#   - get_last_price_date(symbol) -> Timestamp|None
#   - get_last_price_dates_from_prices() -> DataFrame(symbol,last_price_date)
#   - get_last_price_dates_from_prices_for_symbols(symbols) -> DataFrame(...)
#   - get_symbol_status(symbol) -> dict
# ============================================================
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


# ============================================================
# Helpers
# ============================================================
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


def _connect(db_path: str) -> sqlite3.Connection:
    _ensure_dir_for_file(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Pragmas (similar intent to your previous sqlite3 setup)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# ============================================================
# Store
# ============================================================
@dataclass(frozen=True)
class SQLiteStore:
    db_path: str

    # ============================================================
    # Schema
    # ============================================================
    def init_schema(self) -> None:
        """
        Safe to call repeatedly.
        Creates only the tables required for price caching.
        """
        with _connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS prices_daily (
                    symbol TEXT NOT NULL,
                    date   TEXT NOT NULL,   -- YYYY-MM-DD
                    open   REAL,
                    high   REAL,
                    low    REAL,
                    close  REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, date)
                );

                CREATE TABLE IF NOT EXISTS meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE TABLE IF NOT EXISTS symbol_status (
                    symbol          TEXT PRIMARY KEY,
                    last_price_date TEXT,  -- YYYY-MM-DD
                    last_update_utc TEXT,  -- ISO8601
                    row_count       INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_prices_daily_symbol_date
                ON prices_daily(symbol, date);
                """
            )

            # Schema marker
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?);",
                ("schema_version", "prices_only_v1"),
            )

    # ============================================================
    # Meta
    # ============================================================
    def get_meta(self, key: str) -> Optional[str]:
        self.init_schema()
        with _connect(self.db_path) as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?;", (key,)).fetchone()
            return str(row["value"]) if row and row["value"] is not None else None

    def set_meta(self, key: str, value: str) -> None:
        self.init_schema()
        with _connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?);", (key, value))

    # ============================================================
    # Prices (read helpers)
    # ============================================================
    def get_last_price_date(self, *, symbol: str) -> Optional[pd.Timestamp]:
        self.init_schema()
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(date) AS m FROM prices_daily WHERE symbol = ?;",
                (symbol,),
            ).fetchone()
            if not row or row["m"] is None:
                return None
            return pd.to_datetime(row["m"])

    def get_last_price_dates_from_prices(self) -> pd.DataFrame:
        """
        Compute latest date per symbol directly from prices_daily in ONE query.

        Returns a DataFrame with:
          - symbol
          - last_price_date (Timestamp)
        """
        self.init_schema()
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT symbol, MAX(date) AS last_price_date
                FROM prices_daily
                GROUP BY symbol;
                """
            ).fetchall()

        if not rows:
            return pd.DataFrame(columns=["symbol", "last_price_date"])

        df = pd.DataFrame([dict(r) for r in rows], columns=["symbol", "last_price_date"])
        df["last_price_date"] = pd.to_datetime(df["last_price_date"])
        return df

    def get_last_price_dates_from_prices_for_symbols(self, symbols: List[str]) -> pd.DataFrame:
        """
        Same as get_last_price_dates_from_prices(), but only for a provided symbol list.
        Useful if your DB has more symbols than your current universe.
        """
        if not symbols:
            return pd.DataFrame(columns=["symbol", "last_price_date"])

        self.init_schema()
        placeholders = ", ".join(["?"] * len(symbols))
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT symbol, MAX(date) AS last_price_date
                FROM prices_daily
                WHERE symbol IN ({placeholders})
                GROUP BY symbol;
                """,
                symbols,
            ).fetchall()

        if not rows:
            return pd.DataFrame(columns=["symbol", "last_price_date"])

        df = pd.DataFrame([dict(r) for r in rows], columns=["symbol", "last_price_date"])
        df["last_price_date"] = pd.to_datetime(df["last_price_date"])
        return df

    def load_prices_daily(self, *, symbol: str) -> pd.DataFrame:
        self.init_schema()
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT date, open, high, low, close, volume
                FROM prices_daily
                WHERE symbol = ?
                ORDER BY date ASC;
                """,
                (symbol,),
            ).fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(r) for r in rows], columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()

    # ============================================================
    # Prices (write helpers)
    # ============================================================
    def upsert_prices_daily(self, df: pd.DataFrame, *, symbol: str) -> int:
        """
        Upserts daily OHLCV rows into prices_daily.
        df must contain a 'date' column (case-insensitive).
        """
        if df is None or df.empty:
            return 0

        self.init_schema()

        cols = {c.lower().strip(): c for c in df.columns}
        if "date" not in cols:
            raise ValueError("upsert_prices_daily requires a 'date' column")

        dfi = df.copy()

        # Normalize date into YYYY-MM-DD strings
        dfi["date"] = pd.to_datetime(dfi[cols["date"]]).dt.strftime("%Y-%m-%d")

        def _maybe_float(colname: str) -> pd.Series:
            if colname in cols:
                return pd.to_numeric(dfi[cols[colname]], errors="coerce")
            return pd.Series([None] * len(dfi))

        open_s = _maybe_float("open")
        high_s = _maybe_float("high")
        low_s = _maybe_float("low")
        close_s = _maybe_float("close")
        volume_s = _maybe_float("volume")

        records = []
        for i in range(len(dfi)):
            records.append(
                (
                    symbol,
                    dfi["date"].iat[i],
                    None if pd.isna(open_s.iat[i]) else float(open_s.iat[i]),
                    None if pd.isna(high_s.iat[i]) else float(high_s.iat[i]),
                    None if pd.isna(low_s.iat[i]) else float(low_s.iat[i]),
                    None if pd.isna(close_s.iat[i]) else float(close_s.iat[i]),
                    None if pd.isna(volume_s.iat[i]) else float(volume_s.iat[i]),
                )
            )

        with _connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO prices_daily(symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                records,
            )

        self._refresh_symbol_status(symbol=symbol)
        return len(records)

    def _refresh_symbol_status(self, *, symbol: str) -> None:
        self.init_schema()
        with _connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT MAX(date) AS last_date, COUNT(*) AS n
                FROM prices_daily
                WHERE symbol = ?;
                """,
                (symbol,),
            ).fetchone()

            last_date = row["last_date"] if row else None
            n = int(row["n"]) if row and row["n"] is not None else 0

            conn.execute(
                """
                INSERT OR REPLACE INTO symbol_status(symbol, last_price_date, last_update_utc, row_count)
                VALUES (?, ?, ?, ?);
                """,
                (symbol, last_date, _utc_now_iso(), n),
            )

    # ============================================================
    # Symbol status
    # ============================================================
    def get_symbol_status(self, *, symbol: str) -> Dict[str, Any]:
        self.init_schema()
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT symbol, last_price_date, last_update_utc, row_count FROM symbol_status WHERE symbol = ?;",
                (symbol,),
            ).fetchone()

        if not row:
            return {"symbol": symbol, "last_price_date": None, "last_update_utc": None, "row_count": 0}

        return {
            "symbol": row["symbol"],
            "last_price_date": pd.to_datetime(row["last_price_date"]) if row["last_price_date"] else None,
            "last_update_utc": row["last_update_utc"],
            "row_count": int(row["row_count"]) if row["row_count"] is not None else 0,
        }
