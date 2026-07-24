"""Build isolated raw FMP feature families for related issuer securities."""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE = ROOT.parent / "quant-warehouse"
sys.path.insert(0, str(WAREHOUSE))
from quant_warehouse.platforms.data_providers.fmp.feature_engineering import build_price_technical_features  # noqa: E402

CLASSES = ("preferred", "warrant", "unit", "note_bond", "adr", "ordinary", "etf")


def key() -> str:
    if os.getenv("FMP_API_KEY"):
        return os.environ["FMP_API_KEY"]
    for line in (WAREHOUSE / ".env").read_text().splitlines():
        if line.startswith("FMP_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("FMP_API_KEY is required")


def symbols(tier: str) -> list[str]:
    cap = {"1T": 1_000_000_000_000, "100B": 100_000_000_000, "10B": 10_000_000_000}[tier]
    index = pd.read_csv(ROOT / "artifacts" / "trading_app_v2" / f"equity_meta_model_{tier.lower()}" / f"mcap_{cap}_train_2020-12-31_seed_20260707" / "feature_family_panels" / "index.csv")
    return sorted(set().union(*[set(pd.read_parquet(path, columns=["symbol"]).symbol.astype(str).str.upper()) for path in index.panel_path]))


def classify(name: str, symbol: str, issuer: str) -> str | None:
    text, sym = str(name).lower(), str(symbol).upper()
    if sym == issuer:
        return None
    if "warrant" in text or sym.endswith("W") or "-W" in sym:
        return "warrant"
    if "unit" in text or sym.endswith("U") or "-UN" in sym:
        return "unit"
    if any(x in text for x in ("note", "bond", "debenture", "senior debt")):
        return "note_bond"
    if any(x in text for x in ("preferred", "pfd", "pref")) or "-P" in sym or (sym.endswith("P") and len(sym) > len(issuer)):
        return "preferred"
    if "depositary" in text or "adr" in text:
        return "adr"
    if "ordinary" in text:
        return "ordinary"
    if " etf" in f" {text}" or text.endswith(" etf"):
        return "etf"
    return None


def maturity_date(name: str) -> str | None:
    match = re.search(
        r"(?:expir(?:es|ing)?|due|matur(?:es|ity))\s*(?:on\s*)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        str(name),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    parsed = pd.to_datetime(match.group(1), errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed).strftime("%Y-%m-%d")


def discover(issuer: str, api_key: str) -> tuple[str, list[tuple[str, str, str | None]]]:
    session = requests.Session()
    profile = session.get("https://financialmodelingprep.com/stable/profile", params={"symbol": issuer, "apikey": api_key}, timeout=30).json()
    name = profile[0].get("companyName", issuer) if isinstance(profile, list) and profile else issuer
    results = session.get("https://financialmodelingprep.com/stable/search-name", params={"query": name, "apikey": api_key}, timeout=30).json()
    candidates = []
    for item in results if isinstance(results, list) else []:
        symbol = str(item.get("symbol") or "").upper()
        security_class = classify(item.get("name", ""), symbol, issuer)
        if security_class:
            candidates.append((symbol, security_class, maturity_date(item.get("name", ""))))
    return issuer, sorted(set(candidates))


def fetch(symbol: str, api_key: str) -> pd.DataFrame:
    # Match the equity warehouse basis: FMP's dividend-adjusted daily route
    # returns adjusted OHLC fields as adjOpen/adjHigh/adjLow/adjClose.
    response = requests.get(
        "https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted",
        params={"symbol": symbol, "from": "1900-01-01", "to": "2025-12-31", "apikey": api_key},
        timeout=60,
    )
    data = response.json()
    frame = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()
    if frame.empty:
        return frame
    renamed = {"adjOpen": "open", "adjHigh": "high", "adjLow": "low", "adjClose": "close"}
    frame = frame.rename(columns=renamed)
    return frame[[column for column in ("date", "symbol", "open", "high", "low", "close", "volume") if column in frame.columns]]


def build(tier: str = "100B") -> Path:
    api_key = key(); issuers = symbols(tier); discovered = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(discover, issuer, api_key) for issuer in issuers]):
            issuer, candidates = future.result(); discovered[issuer] = candidates
    requests_by_symbol = sorted({symbol for values in discovered.values() for symbol, _, _ in values})
    raw = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, symbol, api_key): symbol for symbol in requests_by_symbol}
        for future in as_completed(futures):
            try: raw[futures[future]] = future.result()
            except Exception as exc: print({"symbol": futures[future], "error": str(exc)}, flush=True)
    class_outputs = {security_class: [] for security_class in CLASSES}
    for issuer, candidates in discovered.items():
        for security_class in CLASSES:
            for instrument_symbol, cls, maturity in candidates:
                if cls != security_class or instrument_symbol not in raw or raw[instrument_symbol].empty:
                    continue
                prices = raw[instrument_symbol].copy()
                prices["symbol"] = instrument_symbol
                try:
                    built = build_price_technical_features(instrument_symbol, prices)
                except Exception as exc:
                    print({"issuer": issuer, "instrument": instrument_symbol, "error": str(exc)}, flush=True)
                    continue
                if built.df.empty:
                    continue
                technical = built.df.reset_index().rename(columns={"symbol": "instrument_symbol"})
                technical["date"] = pd.to_datetime(technical["date"], errors="coerce").dt.normalize()
                technical = technical.drop_duplicates(["date", "instrument_symbol"], keep="last")
                raw_frame = prices[["date", "open", "high", "low", "close", "volume"]].copy()
                raw_frame["date"] = pd.to_datetime(raw_frame["date"], errors="coerce").dt.normalize()
                raw_frame.insert(1, "instrument_symbol", instrument_symbol)
                frame = raw_frame.merge(technical, on=["date", "instrument_symbol"], how="left", validate="one_to_one")
                feature_cols = [column for column in frame.columns if column.startswith("px__") or column in {"open", "high", "low", "close", "volume"}]
                frame = frame[["date", "instrument_symbol", *feature_cols]]
                frame = frame.rename(columns={column: f"{security_class}__{column}" for column in feature_cols})
                frame.insert(0, "symbol", issuer)
                frame.insert(2, "asset_class", security_class)
                frame.insert(3, "maturity_date", maturity)
                class_outputs[security_class].append(frame)
    outputs = [pd.concat(frames, ignore_index=True) for frames in class_outputs.values() if frames]
    output = ROOT / "artifacts" / "graph_oracle_feature_family_gnn_improved_hits" / "cache" / f"related_asset_features_{tier.lower()}_adjusted.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    if outputs:
        panel = pd.concat(outputs, ignore_index=True)
        panel = panel.sort_values(["symbol", "asset_class", "instrument_symbol", "date"]).reset_index(drop=True)
    else:
        panel = pd.DataFrame()
    panel.to_parquet(output, index=False)
    print({"tier": tier, "issuers": len(issuers), "candidate_symbols": len(requests_by_symbol), "rows": len(panel), "covered_issuers": panel.symbol.nunique() if not panel.empty else 0, "unique_instruments": panel.instrument_symbol.nunique() if not panel.empty else 0, "classes": sorted(panel.asset_class.unique().tolist()) if not panel.empty else []}, flush=True)
    return output


if __name__ == "__main__":
    build(os.getenv("RELATED_ASSET_TIER", "100B").strip().upper())
