"""Materialize FMP preferred-security raw features for an equity universe."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE = ROOT.parent / "quant-warehouse"
import sys
sys.path.insert(0, str(WAREHOUSE))

from quant_warehouse.platforms.data_providers.fmp.feature_engineering import (  # noqa: E402
    build_preferred_stock_features,
)


def _api_key() -> str:
    value = str(os.getenv("FMP_API_KEY") or "").strip()
    if value:
        return value
    env_path = WAREHOUSE / ".env"
    for line in env_path.read_text().splitlines():
        if line.startswith("FMP_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("FMP_API_KEY is required")


def _symbols(tier: str) -> list[str]:
    cap = {"1T": 1_000_000_000_000, "100B": 100_000_000_000, "10B": 10_000_000_000}[tier]
    index = pd.read_csv(
        ROOT / "artifacts" / "trading_app_v2" / f"equity_meta_model_{tier.lower()}"
        / f"mcap_{cap}_train_2020-12-31_seed_20260707" / "feature_family_panels" / "index.csv"
    )
    return sorted(set().union(*[
        set(pd.read_parquet(path, columns=["symbol"]).symbol.astype(str).str.upper())
        for path in index.panel_path
    ]))


def _one(symbol: str, key: str) -> tuple[str, str, list[str], str | None]:
    session = requests.Session()
    try:
        profile = session.get(
            "https://financialmodelingprep.com/stable/profile",
            params={"symbol": symbol, "apikey": key}, timeout=30,
        ).json()
        name = profile[0].get("companyName", symbol) if isinstance(profile, list) and profile else symbol
        results = session.get(
            "https://financialmodelingprep.com/stable/search-name",
            params={"query": name, "apikey": key}, timeout=30,
        ).json()
        candidates: list[str] = []
        for item in results if isinstance(results, list) else []:
            candidate = str(item.get("symbol") or "").upper()
            candidate_name = str(item.get("name") or "").lower()
            if candidate == symbol or not candidate:
                continue
            is_preferred = (
                "preferred" in candidate_name or "pfd" in candidate_name or "pref" in candidate_name
                or "-P" in candidate or candidate.endswith("P")
            )
            if is_preferred:
                candidates.append(candidate)
        return symbol, name, sorted(set(candidates)), None
    except Exception as exc:  # pragma: no cover - provider failure is reported
        return symbol, symbol, [], str(exc)


def _prices(symbol: str, key: str) -> list[dict]:
    response = requests.get(
        "https://financialmodelingprep.com/stable/historical-price-eod/full",
        params={
            "symbol": symbol, "from": "1980-01-01", "to": "2025-12-31", "apikey": key,
        }, timeout=60,
    )
    response.raise_for_status()
    return response.json() if isinstance(response.json(), list) else []


def build(tier: str = "100B") -> Path:
    key = _api_key()
    symbols = _symbols(tier)
    discovered: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(_one, symbol, key) for symbol in symbols]
        for future in as_completed(futures):
            symbol, _, candidates, error = future.result()
            if error:
                print({"symbol": symbol, "error": error}, flush=True)
            discovered[symbol] = candidates
    requests_by_symbol = sorted(set(candidate for values in discovered.values() for candidate in values))
    print({"tier": tier, "issuer_symbols": len(symbols), "preferred_candidates": len(requests_by_symbol)}, flush=True)
    raw: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_prices, symbol, key): symbol for symbol in requests_by_symbol}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                rows = future.result()
                for row in rows:
                    row["symbol"] = symbol
                raw.extend(rows)
            except Exception as exc:
                print({"preferred_symbol": symbol, "error": str(exc)}, flush=True)
    rows_by_issuer: list[pd.DataFrame] = []
    for issuer, preferred_symbols in discovered.items():
        frame = pd.DataFrame([row for row in raw if str(row.get("symbol", "")).upper() in preferred_symbols])
        if frame.empty:
            continue
        built = build_preferred_stock_features(issuer, frame)
        if not built.df.empty:
            rows_by_issuer.append(built.df.reset_index())
    output = ROOT / "artifacts" / "graph_oracle_feature_family_gnn_improved_hits" / "cache" / f"preferred_stock_features_{tier.lower()}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    panel = pd.concat(rows_by_issuer, ignore_index=True) if rows_by_issuer else pd.DataFrame()
    panel.to_parquet(output, index=False)
    print({"output": str(output), "rows": len(panel), "issuers": panel.symbol.nunique() if not panel.empty else 0}, flush=True)
    return output


if __name__ == "__main__":
    build(os.getenv("PREFERRED_TIER", "100B").strip().upper())
