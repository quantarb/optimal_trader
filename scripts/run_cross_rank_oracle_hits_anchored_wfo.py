"""Cross-sectional conditional-rank augmentation for Oracle and HITS.

The event/entry model remains unchanged.  A rank model is trained only on
valid entry candidates and its predicted conditional rank multiplies the
entry score.  Exit scores remain the original Oracle sell/cover classifier or
HITS authority model.  The script runs long-only and short-only books
separately using the shared-book backtest engine.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import run_oracle_hits_anchored_wfo as wfo

from quant_warehouse.platforms.data_providers.fmp.target_engineering import LabelBuildSpec, build_trade_results

OUT = Path(os.getenv("CROSS_RANK_OUT", str(ROOT / "artifacts" / "cross_rank_oracle_hits_anchored_wfo")))
OUT.mkdir(parents=True, exist_ok=True)
FIRST_YEAR = int(os.getenv("CROSS_RANK_FIRST_YEAR", "2021"))
LAST_YEAR = int(os.getenv("CROSS_RANK_LAST_YEAR", "2025"))
RF_ESTIMATORS = int(os.getenv("CROSS_RANK_RF_ESTIMATORS", "40"))
TOP_K = int(os.getenv("CROSS_RANK_TOP_K", "10"))
ORACLE_THRESHOLD = float(os.getenv("CROSS_RANK_ORACLE_THRESHOLD", str(wfo.base.ORACLE_THRESHOLD)))
HITS_THRESHOLD = float(os.getenv("CROSS_RANK_HITS_THRESHOLD", str(wfo.base.HITS_THRESHOLD)))
# The production HITS graph uses a longer edge horizon, but materializing all
# cross-sectional edge groups for rank labels is quadratic.  A 20-session
# candidate horizon keeps this ranker test memory-safe and matches the short
# holding-period candidates used by the portfolio entry model.
MAX_HOLD = int(os.getenv("CROSS_RANK_MAX_HOLD", "20"))


def seed(*parts: str) -> int:
    return wfo.base.SEED + int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:4], "little") % 10000


def _edge_rows(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    frame = frame.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(frame.date).to_numpy()
    high = pd.to_numeric(frame.high, errors="coerce").to_numpy(float)
    low = pd.to_numeric(frame.low, errors="coerce").to_numpy(float)
    rows = []
    for i in range(len(frame) - 1):
        end = min(len(frame), i + MAX_HOLD + 1)
        if side == "long":
            values = low[i + 1:end] / high[i] - 1.0
        else:
            values = low[i] / high[i + 1:end] - 1.0
        for j, value in enumerate(values, i + 1):
            if np.isfinite(value):
                rows.append((dates[i], dates[j], float(value)))
    return pd.DataFrame(rows, columns=["entry_date", "exit_date", "ret"])


def add_cross_sectional_ranks(prices: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Attach Oracle trade and HITS-edge entry ranks to date-level labels."""
    oracle_rows = []
    hits_rows = []
    spec = LabelBuildSpec(k_params={"YE": [3]}, min_profit_pct=wfo.base.MIN_RETURN,
                          buy_execution="high", sell_execution="low",
                          short_execution="low", cover_execution="high")
    for (symbol, year), group in prices.groupby(["symbol", prices.date.dt.year], sort=False):
        bare = group.sort_values("date")[['date', 'open', 'high', 'low', 'close']].reset_index(drop=True)
        result = build_trade_results(["S"], spec=spec, price_frames={"S": bare})
        trades = pd.DataFrame(result.completed_trades)
        if not trades.empty:
            trades["symbol"] = str(symbol).upper()
            trades["entry_date"] = pd.to_datetime(trades.entry_date)
            trades["exit_date"] = pd.to_datetime(trades.exit_date)
            trades["ret"] = pd.to_numeric(trades.ret_dec, errors="coerce")
            for side, entry, exit_ in (("long", "buy", "sell"), ("short", "short", "cover")):
                part = trades.loc[trades.side.astype(str).str.lower().eq(side)].copy()
                if part.empty:
                    continue
                # Rank after all symbols have been collected; ranking inside
                # this per-symbol loop would make every observation rank 1.
                oracle_rows.append(part[["symbol", "entry_date", "exit_date", "ret"]].assign(side=side))
        for side in ("long", "short"):
            edges = _edge_rows(bare, side)
            if edges.empty:
                continue
            edges["symbol"] = str(symbol).upper()
            edges["side"] = side
            hold_days = (pd.to_datetime(edges.exit_date) - pd.to_datetime(edges.entry_date)).dt.days
            edges["hold_bucket"] = pd.cut(hold_days, bins=[0, 5, 10, MAX_HOLD], labels=False, include_lowest=True)
            # Exact entry/exit singleton edge groups are removed after all
            # symbols have been collected, for a true cross-section.
            hits_rows.append(edges[["symbol", "entry_date", "exit_date", "side", "ret"]])

    out = labels.copy()
    if oracle_rows:
        oracle_all = pd.concat(oracle_rows, ignore_index=True)
        group_size = oracle_all.groupby(["side", "entry_date"], dropna=False).ret.transform("size")
        oracle_all = oracle_all.loc[group_size.ge(2)].copy()
        oracle_all["rank"] = oracle_all.groupby(["side", "entry_date"], dropna=False).ret.rank(pct=True, method="average")
    if hits_rows:
        hits_all = pd.concat(hits_rows, ignore_index=True)
        group_size = hits_all.groupby(["side", "entry_date", "exit_date"], dropna=False).ret.transform("size")
        hits_all = hits_all.loc[group_size.ge(2)].copy()
        hits_all["rank"] = hits_all.groupby(["side", "entry_date", "exit_date"], dropna=False).ret.rank(pct=True, method="average")
    for side, event, target in (("long", "buy", "oracle_long_entry_rank"), ("short", "short", "oracle_short_entry_rank")):
        if oracle_rows and not oracle_all.empty:
            r = oracle_all.loc[oracle_all.side.eq(side)]
            # One entry row can have multiple generated trades; retain the best
            # cross-sectional opportunity available from that entry date.
            r = r.groupby(["symbol", "entry_date"], as_index=False)["rank"].mean().rename(columns={"entry_date": "date", "rank": target})
            out = out.merge(r[["symbol", "date", target]], on=["symbol", "date"], how="left")
        else:
            out[target] = np.nan
    if hits_rows:
        h = hits_all
        for side, target in (("long", "hits_long_entry_rank"), ("short", "hits_short_entry_rank")):
            r = h.loc[h.side.eq(side)].groupby(["symbol", "entry_date"], as_index=False)["rank"].mean().rename(columns={"entry_date": "date", "rank": target})
            out = out.merge(r[["symbol", "date", target]], on=["symbol", "date"], how="left")
    else:
        out["hits_long_entry_rank"] = np.nan
        out["hits_short_entry_rank"] = np.nan
    return out


def rank_prediction(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str, candidate: pd.Series, family: str, year: int) -> np.ndarray | None:
    rows = train.loc[candidate & train[target].notna()].copy()
    if len(rows) < max(10, len(features) // 2) or train.loc[rows.index, target].nunique() < 2:
        return None
    values = wfo.base.reg_predict(rows, test, features, target, seed(family, target, str(year)))
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def run_family(meta: pd.Series, prices: pd.DataFrame, labels: pd.DataFrame, close: pd.DataFrame, year: int, tier: str) -> list[dict]:
    panel = pd.read_parquet(meta.panel_path)
    metadata = pd.read_parquet(meta.metadata_path)
    family, source = str(meta.family), str(meta.source)
    features = [c for c in metadata.feature.astype(str) if c in panel.columns]
    if not features:
        return []
    frame = panel[["symbol", "date", *features]].copy()
    frame.symbol = frame.symbol.astype(str).str.upper()
    frame.date = pd.to_datetime(frame.date).dt.normalize()
    frame = frame.merge(labels, on=["symbol", "date"], how="inner")
    train_end = pd.Timestamp(f"{year - 1}-12-31")
    test_start, test_end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
    frame = frame.loc[frame.date.between(wfo.DATA_START, test_end)].reset_index(drop=True)
    train = frame.loc[frame.date <= train_end].copy()
    test = frame.loc[frame.date.between(test_start, test_end)].copy()
    if train.empty or test.empty:
        return []
    med = train[features].apply(pd.to_numeric, errors="coerce").median().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train = wfo.base.clean_features(train, features, med)
    test = wfo.base.clean_features(test, features, med)
    dates = pd.DatetimeIndex(close.index[(close.index >= test_start) & (close.index <= test_end)])
    result = []
    for side in ("long", "short"):
        oracle = wfo.oracle_scores(train, test, features, side)
        if oracle is not None:
            entry, exit_ = oracle
            rank_target = f"oracle_{side}_entry_rank"
            rank = rank_prediction(train, test, features, rank_target, train["buy" if side == "long" else "short"].eq(1), family, year)
            ranked_entry = np.where(entry >= ORACLE_THRESHOLD, entry * rank, -1.0) if rank is not None else entry
            for ranker, entry_score, entry_threshold in (("off", entry, ORACLE_THRESHOLD), ("on", ranked_entry, 0.0)):
                pred = test[["symbol", "date"]].copy()
                if side == "long":
                    pred["long_score"], pred["short_score"] = entry_score, 0.0
                    pred["long_exit_score"], pred["short_exit_score"] = 0.0, exit_
                else:
                    pred["long_score"], pred["short_score"] = 0.0, entry_score
                    pred["long_exit_score"], pred["short_exit_score"] = exit_, 0.0
                metrics = wfo.base.backtest_scores(pred, close, side, dates, entry_threshold, TOP_K)
                metrics.update({"tier": tier, "year": year, "model": "oracle", "variant": side, "ranker": ranker, "family": family, "source": source, "top_k": TOP_K, "threshold": ORACLE_THRESHOLD})
                result.append(metrics)
        components = {}
        for role in ("hub", "authority"):
            target = f"{side}_{role}"
            values = wfo.hits_score(train, test, features, target, family, year)
            if values is not None:
                components[role] = values
        if "hub" in components and "authority" in components:
            rank_target = f"hits_{side}_entry_rank"
            rank = rank_prediction(train, test, features, rank_target, wfo.base.select_top_bottom_rows(train, f"{side}_hub").index.to_series().reindex(train.index).notna(), family, year)
            ranked_entry = np.where(components["hub"] >= HITS_THRESHOLD, components["hub"] * rank, -1.0) if rank is not None else components["hub"]
            for ranker, entry_score, entry_threshold in (("off", components["hub"], HITS_THRESHOLD), ("on", ranked_entry, 0.0)):
                pred = test[["symbol", "date"]].copy()
                if side == "long":
                    pred["long_score"], pred["short_score"] = entry_score, 0.0
                    pred["long_exit_score"], pred["short_exit_score"] = 0.0, components["authority"]
                else:
                    pred["long_score"], pred["short_score"] = 0.0, entry_score
                    pred["long_exit_score"], pred["short_exit_score"] = components["authority"], 0.0
                metrics = wfo.base.backtest_scores(pred, close, side, dates, entry_threshold, TOP_K)
                metrics.update({"tier": tier, "year": year, "model": "hits", "variant": side, "ranker": ranker, "family": family, "source": source, "top_k": TOP_K, "threshold": HITS_THRESHOLD})
                result.append(metrics)
    return result


def run_tier(tier: str) -> pd.DataFrame:
    index, prices, labels = wfo.load_data(tier)
    labels = add_cross_sectional_ranks(prices, labels)
    index["tier"] = tier
    requested = tuple(x.strip() for x in os.getenv("CROSS_RANK_FAMILIES", "").split(",") if x.strip())
    if requested:
        index = index.loc[index.family.astype(str).isin(requested)]
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    rows = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        t0 = perf_counter()
        for _, meta in index.iterrows():
            rows.extend(run_family(meta, prices, labels, close, year, tier))
        print({"tier": tier, "year": year, "rows": len(rows), "seconds": round(perf_counter() - t0, 1)}, flush=True)
        # Preserve completed anchored-WFO years so long tier runs can be
        # inspected or resumed without waiting for the remaining years.
        pd.DataFrame(rows).to_parquet(OUT / f"{tier.lower()}_results_through_{year}.parquet", index=False)
    out = pd.DataFrame(rows)
    out.to_parquet(OUT / f"{tier.lower()}_results.parquet", index=False)
    return out


def main() -> None:
    tiers = tuple(x.strip().upper() for x in os.getenv("CROSS_RANK_TIERS", "1T").split(",") if x.strip())
    frames = [run_tier(tier) for tier in tiers]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(OUT / "all_results.csv", index=False)
    if not out.empty:
        out.groupby(["tier", "year", "model", "variant", "ranker"], as_index=False).agg(
            families=("family", "nunique"), mean_return=("total_return", "mean"), median_return=("total_return", "median"),
            mean_sharpe=("sharpe", "mean"), min_return=("total_return", "min"), max_return=("total_return", "max"),
        ).to_csv(OUT / "summary.csv", index=False)


if __name__ == "__main__":
    main()
