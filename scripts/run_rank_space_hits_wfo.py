from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
for repo in (REPO_ROOT, WORKSPACE_ROOT / "quant-warehouse", WORKSPACE_ROOT / "quant-orchestrator"):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

from quant_warehouse.platforms.data_providers.fmp.target_engineering import LabelBuildSpec, build_trade_results
from quant_warehouse.warehouse.api import Warehouse
from quant_orchestrator.platforms.backtesting_frameworks.shared_book import SharedBookCostModel, run_shared_book_framework_comparison
from quant_orchestrator.platforms.ml_frameworks.rapids.random_forest import RapidsRandomForestClassifier
from quant_orchestrator.research_tools import build_family_prediction_frame, build_strategy_score_frame, prepare_family_dataset

FIRST_TEST_YEAR, LAST_TEST_YEAR = 2021, 2025
DP_K, MIN_RETURN = 3, 0.01
RF_ESTIMATORS = int(os.getenv("GRAPH_ORACLE_FAMILY_RF_ESTIMATORS", "40"))
REQUESTED = tuple(x.strip().upper() for x in os.getenv("GRAPH_ORACLE_TIERS", "").split(",") if x.strip())
TIERS = {"1T": 1_000_000_000_000, "100B": 100_000_000_000, "10B": 10_000_000_000}
if REQUESTED:
    TIERS = {k: v for k, v in TIERS.items() if k in REQUESTED}
BASE_CACHE = {"1T": "equity_meta_model_1t", "100B": "equity_meta_model_100b", "10B": "equity_meta_model_10b"}
OUT = REPO_ROOT / "artifacts" / "graph_oracle_feature_family_wfo"
OUT.mkdir(parents=True, exist_ok=True)


def normalize_prices(raw: pd.DataFrame) -> pd.DataFrame:
    f = raw.copy().reset_index() if "date" not in raw.columns else raw.copy()
    f.columns = [str(c).lower() for c in f.columns]
    f["date"] = pd.to_datetime(f["date"], errors="coerce").dt.normalize()
    return f[["date", "open", "high", "low", "close"]].dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)


def hits_scores(frame: pd.DataFrame) -> pd.DataFrame:
    n = len(frame)
    dates = frame.date.to_numpy()
    high = pd.to_numeric(frame.high, errors="coerce").to_numpy(float)
    low = pd.to_numeric(frame.low, errors="coerce").to_numpy(float)
    valid = np.triu(np.ones((n, n), dtype=bool), 1)
    horizon = np.arange(n)[None, :] - np.arange(n)[:, None]
    valid &= horizon <= 120
    out = {"date": dates}
    for side, returns in [("long", low[None, :] / high[:, None] - 1), ("short", low[:, None] / high[None, :] - 1)]:
        w = np.where(valid, np.maximum(returns - MIN_RETURN, 0), 0)
        hub = np.ones(n); authority = np.ones(n)
        for _ in range(40):
            authority = w.T @ hub; authority /= np.linalg.norm(authority) or 1
            hub = w @ authority; hub /= np.linalg.norm(hub) or 1
        out[f"{side}_hub"] = hub
        out[f"{side}_authority"] = authority
    return pd.DataFrame(out)


def dp_entries(frame: pd.DataFrame, side: str) -> set[pd.Timestamp]:
    spec = LabelBuildSpec(k_params={"YE": [DP_K]}, min_profit_pct=MIN_RETURN, buy_execution="high", sell_execution="low", short_execution="low", cover_execution="high")
    trades = pd.DataFrame(build_trade_results(["S"], spec=spec, price_frames={"S": frame}).completed_trades)
    if trades.empty:
        return set()
    return set(pd.to_datetime(trades.loc[trades.side.str.lower().eq(side), "entry_date"], errors="coerce"))


def build_targets(price_frames: dict[str, pd.DataFrame], rank_mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    hits_rank_rows, price_rank_rows = [], []
    spec = LabelBuildSpec(k_params={"YE": [DP_K]}, min_profit_pct=MIN_RETURN, buy_execution="high", sell_execution="low", short_execution="low", cover_execution="high")
    for year in range(1970, LAST_TEST_YEAR + 1):
        scores = {}
        for symbol, frame in price_frames.items():
            part = frame.loc[frame.date.dt.year.eq(year)]
            if len(part) < 20:
                continue
            scores[symbol] = hits_scores(part).set_index("date")
        if not scores:
            continue
        close_wide = pd.concat({s: price_frames[s].loc[price_frames[s].date.dt.year.eq(year)].set_index("date").close for s in price_frames if year in set(price_frames[s].date.dt.year)}, axis=1)
        price_goodness = close_wide.rank(axis=1, pct=True, ascending=True)
        for side in ("long", "short"):
            cols = [f"{side}_hub", f"{side}_authority"]
            wide = pd.concat({s: x[cols] for s, x in scores.items()}, axis=1)
            goodness = wide.rank(axis=1, pct=True, ascending=True).max(axis=1, level=1) if False else None
            for symbol, score_frame in scores.items():
                g = pd.DataFrame({c: wide.xs(c, axis=1, level=1).rank(axis=1, pct=True, ascending=True)[symbol] for c in cols}).max(axis=1)
                g = g.reindex(score_frame.index).dropna()
                pseudo_price = 1 + ((1 - g) if rank_mode == "normal" else g)
                pseudo = pd.DataFrame({"date": g.index, "open": pseudo_price, "high": pseudo_price, "low": pseudo_price, "close": pseudo_price}).dropna()
                entries = dp_entries(pseudo, side)
                for date in entries:
                    hits_rank_rows.append({"symbol": symbol, "date": date, "collapsed_label": f"oracle_{side}"})
        for symbol, score_frame in scores.items():
            g = price_goodness[symbol].reindex(score_frame.index).dropna()
            pseudo_price = 1 + ((1 - g) if rank_mode == "normal" else g)
            pseudo = pd.DataFrame({"date": g.index, "open": pseudo_price, "high": pseudo_price, "low": pseudo_price, "close": pseudo_price}).dropna()
            for side in ("long", "short"):
                for date in dp_entries(pseudo, side):
                    price_rank_rows.append({"symbol": symbol, "date": date, "collapsed_label": f"oracle_{side}"})
    hits_rank = pd.DataFrame(hits_rank_rows).drop_duplicates() if hits_rank_rows else pd.DataFrame(columns=["symbol", "date", "collapsed_label"])
    price_rank = pd.DataFrame(price_rank_rows).drop_duplicates() if price_rank_rows else pd.DataFrame(columns=["symbol", "date", "collapsed_label"])
    return hits_rank, price_rank


def cache_dir(tier: str) -> Path:
    return REPO_ROOT / "artifacts" / "trading_app_v2" / BASE_CACHE[tier] / f"mcap_{TIERS[tier]}_train_2020-12-31_seed_20260707" / "feature_family_panels"


def run_tier(tier: str, rank_mode: str) -> pd.DataFrame:
    started = perf_counter(); warehouse = Warehouse(); idx = pd.read_csv(cache_dir(tier) / "index.csv")
    first = pd.read_parquet(idx.iloc[0].panel_path); symbols = sorted(first.symbol.astype(str).str.upper().unique()); prices = {}
    for symbol in symbols:
        raw = warehouse.read_prices(symbol, provider="fmp", start="1900-01-01", end=f"{LAST_TEST_YEAR}-12-31")
        if raw is not None and not raw.empty:
            try: prices[symbol] = normalize_prices(raw)
            except Exception: pass
    rank_targets, dp_targets = build_targets(prices, rank_mode); close = pd.DataFrame({s: f.set_index("date").close for s, f in prices.items()}).sort_index().ffill(); next_returns = close.pct_change().shift(-1)
    dates = pd.DatetimeIndex(next_returns.index[(next_returns.index >= "2021-01-01") & (next_returns.index <= "2025-12-31")]); results = []
    for _, meta in idx.iterrows():
        panel = pd.read_parquet(meta.panel_path); metadata = pd.read_parquet(meta.metadata_path); source, family = str(meta.source), str(meta.family); folds = []
        for label_name, targets in ((f"hits_rank_dp_{rank_mode}", rank_targets), (f"price_rank_dp_{rank_mode}", dp_targets)):
            fold_scores = []
            for test_year in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1):
                train, usable = prepare_family_dataset(panel, metadata.assign(source=source, family=family), targets, source=source, family=family, min_feature_coverage=.50); train = train.loc[pd.to_datetime(train.date) < pd.Timestamp(f"2020-01-01") if test_year == FIRST_TEST_YEAR else pd.to_datetime(train.date) < pd.Timestamp(f"{test_year}-01-01")]
                if len(train) < 50 or train.collapsed_label.nunique() < 2: continue
                med = train[usable].median().replace([np.inf, -np.inf], np.nan).fillna(0); train[usable] = train[usable].replace([np.inf, -np.inf], np.nan).fillna(med).astype("float32")
                model = RapidsRandomForestClassifier.fit(train, features=usable, target_col="collapsed_label", random_state=20260715 + test_year, params={"n_estimators": RF_ESTIMATORS, "max_depth": 16, "max_features": "sqrt", "n_bins": 128, "n_streams": 8})
                pred = build_family_prediction_frame(panel, usable, min_feature_coverage=.50); pred = pred.loc[pd.to_datetime(pred.date).between(pd.Timestamp(f"{test_year}-01-01"), pd.Timestamp(f"{test_year}-12-31"))].copy(); pred[usable] = pred[usable].replace([np.inf, -np.inf], np.nan).fillna(med).astype("float32")
                proba = model.predict_proba_frame(pred, usable); fold_scores.append(build_strategy_score_frame(source=source, family=family, prediction_frame=pred[["symbol", "date"]], probability_frame=proba, apply_ae_to_exits=False)); del model, train, pred, proba
            if not fold_scores: continue
            scores = pd.concat(fold_scores, ignore_index=True); summary, _, _ = run_shared_book_framework_comparison(scores=scores, next_returns=next_returns, symbols=tuple(prices), dates=dates, variants=("long_short",), top_k_values=(20,), entry_threshold=.5, exit_threshold=.5, cost_models={"family_common": SharedBookCostModel(.5, 5.0)})
            if not summary.empty:
                row = summary.iloc[0].to_dict(); row.update({"tier": tier, "label_source": label_name, "source": source, "family": family}); results.append(row)
    out = pd.DataFrame(results); out.to_parquet(OUT / f"{tier.lower()}_rank_space_{rank_mode}_wfo_results.parquet", index=False); print({"tier": tier, "rank_mode": rank_mode, "rows": len(out), "seconds": round(perf_counter() - started, 1)}); return out


RANK_MODES = tuple(x.strip().lower() for x in os.getenv("GRAPH_ORACLE_RANK_MODES", "normal,reverse").split(",") if x.strip())
all_results = [run_tier(t, mode) for mode in RANK_MODES for t in TIERS]
combined = pd.concat(all_results, ignore_index=True); combined.to_csv(OUT / "all_rank_space_hits_wfo_results.csv", index=False)
print(combined.groupby(["tier", "label_source"]).total_return.agg(["count", "mean", "median"]).round(4))
