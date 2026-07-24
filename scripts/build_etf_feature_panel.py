"""Build an adjusted ETF document corpus for transformer representation training."""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE = ROOT.parent / "quant-warehouse"
sys.path.insert(0, str(WAREHOUSE))
from quant_warehouse.platforms.data_providers.fmp.target_engineering.hits import (  # noqa: E402
    HitsLabelSpec,
    build_hits_labels,
    build_inverse_holding_time_hits_labels,
)

OUT = ROOT / "artifacts" / "graph_oracle_feature_family_gnn_improved_hits" / "cache"
RAW_COLS = ("open", "high", "low", "close", "volume")


def key() -> str:
    if os.getenv("FMP_API_KEY"):
        return os.environ["FMP_API_KEY"]
    for line in (WAREHOUSE / ".env").read_text().splitlines():
        if line.startswith("FMP_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("FMP_API_KEY is required")


def etf_symbols(api_key: str, limit: int) -> list[str]:
    response = requests.get(
        "https://financialmodelingprep.com/stable/company-screener",
        params={"isEtf": "true", "isFund": "false", "isActivelyTrading": "true", "limit": 10000, "apikey": api_key},
        timeout=60,
    )
    data = response.json()
    frame = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()
    if frame.empty or "symbol" not in frame.columns:
        return []
    frame["marketCap"] = pd.to_numeric(frame.get("marketCap"), errors="coerce").fillna(0.0)
    frame = frame.sort_values(["marketCap", "symbol"], ascending=[False, True]).drop_duplicates("symbol")
    return frame.symbol.astype(str).str.upper().head(limit).tolist()


def fetch(symbol: str, api_key: str) -> pd.DataFrame:
    response = requests.get(
        "https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted",
        params={"symbol": symbol, "from": "1900-01-01", "to": "2025-12-31", "apikey": api_key},
        timeout=60,
    )
    data = response.json()
    frame = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()
    if frame.empty:
        return frame
    frame = frame.rename(columns={"adjOpen": "open", "adjHigh": "high", "adjLow": "low", "adjClose": "close"})
    return frame[[column for column in ("date", "open", "high", "low", "close", "volume") if column in frame.columns]]


def build(limit: int = 250) -> Path:
    api_key = key()
    symbols = etf_symbols(api_key, limit)
    prices: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch, symbol, api_key): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                frame = future.result()
            except Exception as exc:
                print({"symbol": symbol, "error": str(exc)}, flush=True)
                continue
            if frame.empty or not set(RAW_COLS).issubset(frame.columns):
                continue
            frame["date"] = pd.to_datetime(frame.date, errors="coerce").dt.normalize()
            for column in RAW_COLS:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame = frame.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
            if len(frame) >= 30:
                prices[symbol] = frame
    if not prices:
        raise RuntimeError("No ETF historical prices were returned")

    spec = HitsLabelSpec(max_hold=120, iterations=50, tail_quantile=0.20, start_date="1900-01-01", end_date="2025-12-31")
    graph = build_hits_labels(prices, spec=spec)
    speed = build_inverse_holding_time_hits_labels(prices, spec=spec)
    for column in ("long_hub", "long_authority", "short_hub", "short_authority", "long_pagerank", "short_pagerank"):
        if column not in graph.columns:
            graph[column] = 0.0
        if column not in speed.columns:
            speed[column] = 0.0
    speed = speed.rename(columns={column: f"speed_{column}" for column in ("long_hub", "long_authority", "short_hub", "short_authority", "long_pagerank", "short_pagerank")})
    labels = graph.merge(speed, on=["symbol", "date"], how="left", validate="one_to_one")
    rows: list[pd.DataFrame] = []
    for symbol, frame in prices.items():
        out = frame[["date", *RAW_COLS]].copy()
        out.insert(0, "symbol", symbol)
        out = out.merge(labels, on=["symbol", "date"], how="left", validate="one_to_one")
        for column in ["long_hub", "long_authority", "short_hub", "short_authority", "long_pagerank", "short_pagerank", *[f"speed_{x}" for x in ("long_hub", "long_authority", "short_hub", "short_authority", "long_pagerank", "short_pagerank")]]:
            if column in out:
                out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).astype("float32")
        rows.append(out)
    panel = pd.concat(rows, ignore_index=True)
    rename = {column: f"etf__{column}" for column in RAW_COLS}
    panel = panel.rename(columns=rename).sort_values(["symbol", "date"]).reset_index(drop=True)
    output = OUT / "etf_document_corpus_100b_adjusted.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output, index=False)
    print({"symbols": len(prices), "rows": len(panel), "output": str(output)}, flush=True)
    return output


if __name__ == "__main__":
    build(int(os.getenv("ETF_CORPUS_LIMIT", "250")))
