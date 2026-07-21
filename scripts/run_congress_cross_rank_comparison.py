"""Cross-rank Oracle/HITS comparison with one congressional-event HITS variant.

Oracle and regular HITS are delegated to the canonical cross-sectional
workflow.  The only additional model is ``congress_hits``: its HITS labels are
created from congressional buy/sell event-date nodes and all valid later
event-date combinations weighted by realized return.  All models are scored
and backtested on every daily date in each forward test year.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_cross_rank_oracle_hits_anchored_wfo as cross  # noqa: E402
import run_oracle_hits_anchored_wfo as wfo  # noqa: E402
from quant_warehouse.platforms.data_providers.fmp.target_engineering.event_pairs.store import (  # noqa: E402
    build_event_pairs_from_historical_data,
)

OUT = Path(os.getenv("CONGRESS_COMPARE_OUT", str(ROOT / "artifacts" / "congress_cross_rank_comparison")))
OUT.mkdir(parents=True, exist_ok=True)
FIRST_YEAR = int(os.getenv("CROSS_RANK_FIRST_YEAR", "2021"))
LAST_YEAR = int(os.getenv("CROSS_RANK_LAST_YEAR", "2025"))
TOP_K = int(os.getenv("CROSS_RANK_TOP_K", "10"))
HITS_THRESHOLD = float(os.getenv("CROSS_RANK_HITS_THRESHOLD", str(wfo.base.HITS_THRESHOLD)))
MAX_EVENT_HOLD = int(os.getenv("CONGRESS_EVENT_MAX_HOLD", "0"))
FAMILY_FILTER = {x.strip() for x in os.getenv("CROSS_RANK_FAMILIES", "").split(",") if x.strip()}


def stable_seed(*parts: str) -> int:
    return wfo.base.SEED + int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:4], "little") % 10000


def congress_events(symbols: list[str], prices: pd.DataFrame) -> pd.DataFrame:
    warehouse = wfo.base.Warehouse()
    rows: list[pd.DataFrame] = []
    for symbol in symbols:
        pairs = build_event_pairs_from_historical_data(
            symbol,
            fundamentals=warehouse.fundamentals,
            event_families=("congress",),
            provider="fmp",
            start_date=str(wfo.DATA_START.date()),
            end_date=str(wfo.DATA_END.date()),
        )
        if pairs is None or pairs.empty:
            continue
        pair = pairs.copy()
        chamber = pair.get("actor_chamber", pd.Series("unknown", index=pair.index)).astype(str).str.lower()
        pair = pair.loc[chamber.isin({"house", "senate"})].copy()
        pair["date"] = pd.to_datetime(pair["event_date"], errors="coerce").dt.normalize()
        pair["side"] = pair["event_type"].map({"congress_buy": "buy", "congress_sell": "sell"})
        pair = pair.dropna(subset=["date", "side"])
        if pair.empty:
            continue
        counts = pair.groupby(["date", "side"], as_index=False).size().pivot(index="date", columns="side", values="size").fillna(0.0)
        counts = counts.rename_axis(None, axis=1).reset_index()
        for side in ("buy", "sell"):
            if side not in counts:
                counts[side] = 0.0
        counts["symbol"] = symbol.upper()
        counts["buy_count"] = counts.buy.astype(float)
        counts["sell_count"] = counts.sell.astype(float)
        counts["buy"] = counts.buy.gt(0).astype(float)
        counts["sell"] = counts.sell.gt(0).astype(float)
        rows.append(counts[["symbol", "date", "buy", "sell", "buy_count", "sell_count"]])
    if not rows:
        return pd.DataFrame(columns=["symbol", "date", "buy", "sell", "buy_count", "sell_count"])
    return pd.concat(rows, ignore_index=True).drop_duplicates(["symbol", "date"])


def node_hits(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(weights) or not np.any(weights > 0):
        return np.zeros(len(weights)), np.zeros(len(weights))
    hub = np.ones(len(weights), dtype=float)
    authority = np.ones(len(weights), dtype=float)
    for _ in range(50):
        authority = weights.T @ hub
        authority /= np.linalg.norm(authority) or 1.0
        hub = weights @ authority
        hub /= np.linalg.norm(hub) or 1.0
    return hub / (hub.max() or 1.0), authority / (authority.max() or 1.0)


def congress_graph_labels(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Create HITS targets from congressional event nodes and all later pairs."""
    price_map = {symbol: frame.sort_values("date").reset_index(drop=True) for symbol, frame in prices.groupby("symbol", sort=False)}
    outputs: list[pd.DataFrame] = []
    for (symbol, year), group in events.assign(year=events.date.dt.year).groupby(["symbol", "year"], sort=False):
        nodes = group.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        frame = price_map.get(symbol)
        if frame is None or len(nodes) < 2:
            continue
        close = frame.set_index("date").close.reindex(nodes.date).to_numpy(float)
        if not np.isfinite(close).all():
            continue
        positions = np.searchsorted(frame.date.to_numpy(), nodes.date.to_numpy())
        long_edges = np.zeros((len(nodes), len(nodes)), dtype=float)
        short_edges = np.zeros_like(long_edges)
        for i in range(len(nodes) - 1):
            end = len(nodes) if MAX_EVENT_HOLD <= 0 else min(len(nodes), i + MAX_EVENT_HOLD + 1)
            for j in range(i + 1, end):
                if MAX_EVENT_HOLD > 0 and positions[j] - positions[i] > MAX_EVENT_HOLD:
                    break
                if nodes.buy.iloc[i] and nodes.sell.iloc[j]:
                    long_edges[i, j] = max(float(close[j] / close[i] - 1.0), 0.0)
                if nodes.sell.iloc[i] and nodes.buy.iloc[j]:
                    short_edges[i, j] = max(float(close[i] / close[j] - 1.0), 0.0)
        long_hub, long_authority = node_hits(long_edges)
        short_hub, short_authority = node_hits(short_edges)
        out = nodes[["symbol", "date", "buy", "sell", "buy_count", "sell_count"]].copy()
        out["congress_long_hub"] = long_hub
        out["congress_long_authority"] = long_authority
        out["congress_short_hub"] = short_hub
        out["congress_short_authority"] = short_authority
        outputs.append(out)
    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True).drop_duplicates(["symbol", "date"])


def congress_hits_score(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str, family: str, year: int) -> np.ndarray | None:
    sparse = train.loc[train[target].notna()].copy()
    if sparse.empty:
        return None
    rank = sparse.groupby(["symbol", sparse.date.dt.year], sort=False)[target].rank(pct=True, method="first")
    sparse = sparse.loc[rank.le(0.2) | rank.ge(0.8)]
    if len(sparse) < max(20, len(features) * 2) or sparse[target].nunique() < 2:
        return None
    values = wfo.base.reg_predict(sparse, test, features, target, stable_seed(family, target, str(year)))
    return pd.Series(values, index=test.index).groupby(test.date).rank(pct=True, method="average").to_numpy()


def run_congress_family(meta: pd.Series, labels: pd.DataFrame, close: pd.DataFrame, year: int, tier: str) -> list[dict]:
    if FAMILY_FILTER and str(meta.family) not in FAMILY_FILTER:
        return []
    panel = pd.read_parquet(meta.panel_path)
    metadata = pd.read_parquet(meta.metadata_path)
    family, source = str(meta.family), str(meta.source)
    features = [c for c in metadata.feature.astype(str) if c in panel.columns]
    frame = panel[["symbol", "date", *features]].copy()
    frame.symbol = frame.symbol.astype(str).str.upper()
    frame.date = pd.to_datetime(frame.date).dt.normalize()
    frame = frame.merge(labels, on=["symbol", "date"], how="inner")
    train = frame.loc[frame.date.dt.year < year].copy()
    test = frame.loc[frame.date.dt.year.eq(year)].copy()
    if train.empty or test.empty:
        return []
    med = train[features].apply(pd.to_numeric, errors="coerce").median().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train = wfo.base.clean_features(train, features, med)
    test = wfo.base.clean_features(test, features, med)
    dates = pd.DatetimeIndex(close.index[(close.index >= f"{year}-01-01") & (close.index <= f"{year}-12-31")])
    result: list[dict] = []
    for side in ("long", "short"):
        components = {}
        for role in ("hub", "authority"):
            target = f"congress_{side}_{role}"
            values = congress_hits_score(train, test, features, target, family, year)
            if values is not None:
                components[role] = values
        if "hub" not in components or "authority" not in components:
            continue
        pred = test[["symbol", "date"]].copy()
        hub, authority = components["hub"], components["authority"]
        if side == "long":
            pred["long_score"], pred["short_score"] = hub, 0.0
            pred["long_exit_score"], pred["short_exit_score"] = 0.0, authority
        else:
            pred["long_score"], pred["short_score"] = 0.0, hub
            pred["long_exit_score"], pred["short_exit_score"] = authority, 0.0
        eligible = sorted(pred.symbol.astype(str).str.upper().unique())
        metrics = wfo.base.backtest_scores(pred, close.loc[:, close.columns.intersection(eligible)], side, dates, HITS_THRESHOLD, TOP_K)
        metrics.update({"tier": tier, "year": year, "model": "congress_hits", "variant": side, "ranker": "off", "family": family, "source": source, "top_k": TOP_K, "threshold": HITS_THRESHOLD})
        result.append(metrics)
    return result


def run_tier(tier: str) -> pd.DataFrame:
    index, prices, labels = wfo.load_data(tier)
    labels = cross.add_cross_sectional_ranks(prices, labels)
    symbols = sorted(prices.symbol.astype(str).str.upper().unique())
    events = congress_events(symbols, prices)
    graph = congress_graph_labels(events, prices)
    if not graph.empty:
        graph = graph.drop(columns=["buy", "sell"], errors="ignore")
        labels = labels.merge(graph, on=["symbol", "date"], how="left")
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    rows: list[dict] = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        year_rows: list[dict] = []
        for _, meta in index.iterrows():
            year_rows.extend(cross.run_family(meta, prices, labels, close, year, tier))
            year_rows.extend(run_congress_family(meta, labels, close, year, tier))
        rows.extend(year_rows)
        pd.DataFrame(rows).to_parquet(OUT / f"{tier.lower()}_through_{year}.parquet", index=False)
        print({"tier": tier, "year": year, "rows": len(year_rows)}, flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default=os.getenv("CROSS_RANK_TIERS", "1T,100B,10B"))
    args = parser.parse_args()
    tiers = tuple(x.strip().upper() for x in args.universe.split(",") if x.strip())
    frames = [run_tier(tier) for tier in tiers]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(OUT / "all_results.csv", index=False)
    if not out.empty:
        out.groupby(["tier", "year", "model", "variant", "ranker"], as_index=False).agg(
            families=("family", "nunique"), mean_return=("total_return", "mean"), median_return=("total_return", "median"), mean_sharpe=("sharpe", "mean"), total_trades=("trades", "sum")
        ).to_csv(OUT / "summary.csv", index=False)


if __name__ == "__main__":
    main()
