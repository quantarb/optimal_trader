"""Model-coupling ablation for the 2024 -> 2025 entry/exit experiment.

Compares combined versus separate oracle classifiers and two-model versus
four-model HITS systems.  HITS models use top/bottom sparse training rows.
"""
from __future__ import annotations

import os
import hashlib
from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import run_oracle_vs_hits_entry_exit_2024_2025 as base

from quant_orchestrator.platforms.ml_frameworks.rapids.random_forest import RapidsRandomForestClassifier

OUT = REPO_ROOT / "artifacts" / "oracle_hits_model_coupling_ablation_2024_2025"
OUT.mkdir(parents=True, exist_ok=True)
WEIGHTING = os.getenv("ABLATION_HITS_WEIGHTING", "clip")


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return base.SEED + int.from_bytes(digest[:4], "little") % 10000


def score_frame(test: pd.DataFrame, long_entry, long_exit, short_entry, short_exit) -> pd.DataFrame:
    out = test[["symbol", "date"]].copy()
    out["long_score"], out["short_score"] = long_entry, short_entry
    out["long_exit_score"], out["short_exit_score"] = long_exit, short_exit
    return out


def oracle_model(train: pd.DataFrame, test: pd.DataFrame, features: list[str], labels: list[str], target: str, seed: int):
    rows = train.loc[train[target].notna()].copy()
    if len(rows) < 10 or rows[target].nunique() < 2:
        return None
    model = RapidsRandomForestClassifier.fit(rows, features=features, target_col=target, random_state=seed, params={"n_estimators": base.RF_ESTIMATORS, "max_depth": 16, "max_features": "sqrt", "n_bins": 128, "n_streams": 8})
    return model.predict_proba_frame(test, features)


def prob(proba: pd.DataFrame | None, label: str, index: pd.Index) -> np.ndarray:
    if proba is None:
        return np.zeros(len(index), dtype=float)
    col = f"prob__{label}"
    return proba[col].to_numpy() if col in proba else np.zeros(len(index), dtype=float)


def run_family(row: pd.Series, prices: pd.DataFrame, labels: pd.DataFrame, close: pd.DataFrame, test_dates: pd.DatetimeIndex) -> list[dict]:
    panel = pd.read_parquet(row.panel_path)
    meta = pd.read_parquet(row.metadata_path)
    family, source = str(row.family), str(row.source)
    features = [c for c in meta.feature.astype(str) if c in panel.columns]
    base_frame = panel[["symbol", "date", *features]].copy()
    base_frame.symbol = base_frame.symbol.astype(str).str.upper(); base_frame.date = pd.to_datetime(base_frame.date).dt.normalize()
    base_frame = base_frame.merge(labels, on=["symbol", "date"], how="inner").loc[lambda x: x.date.between(base.TRAIN_START, base.TEST_END)].reset_index(drop=True)
    train = base_frame.loc[base_frame.date.between(base.TRAIN_START, base.TRAIN_END)].copy()
    test = base_frame.loc[base_frame.date.between(base.TEST_START, base.TEST_END)].copy()
    if train.empty or test.empty or not features:
        return []
    med = train[features].apply(pd.to_numeric, errors="coerce").median().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train = base.clean_features(train, features, med); test = base.clean_features(test, features, med)
    results = []

    # Combined oracle: one four-class model, still trained only on oracle dates.
    event_cols = ["buy", "sell", "short", "cover"]
    combined = train.loc[train[event_cols].sum(axis=1).eq(1)].copy()
    combined["target"] = combined[event_cols].idxmax(axis=1)
    p = oracle_model(combined, test, features, event_cols, "target", base.SEED)
    scores = score_frame(test, prob(p, "buy", test.index), prob(p, "sell", test.index), prob(p, "short", test.index), prob(p, "cover", test.index))
    for variant in ("long", "short"):
        m = base.backtest_scores(scores, close, variant, test_dates, base.ORACLE_THRESHOLD)
        m.update({"tier": str(row.tier), "model": "oracle_combined", "weighting": "dp", "variant": variant, "family": family, "source": source, "train_rows": len(combined)})
        results.append(m)

    # Separate oracle: one long buy/sell model and one short/cover model.
    for side, events in (("long", ["buy", "sell"]), ("short", ["short", "cover"])):
        event = train.loc[train[events].sum(axis=1).eq(1)].copy(); event["target"] = event[events].idxmax(axis=1)
        p = oracle_model(event, test, features, events, "target", base.SEED + 1)
        if side == "long":
            scores = score_frame(test, prob(p, "buy", test.index), 0.0, 0.0, prob(p, "sell", test.index))
        else:
            scores = score_frame(test, 0.0, prob(p, "short", test.index), prob(p, "cover", test.index), 0.0)
        m = base.backtest_scores(scores, close, side, test_dates, base.ORACLE_THRESHOLD)
        m.update({"tier": str(row.tier), "model": "oracle_separate", "weighting": "dp", "variant": side, "family": family, "source": source, "train_rows": len(event)})
        results.append(m)

    # HITS components use the sparse top/bottom tails.  Two-model mode trains
    # only the long hub/authority pair; four-model mode trains all four.
    roles = [("long", "hub"), ("long", "authority")] if os.getenv("ABLATION_HITS_MODE", "both") == "two_and_four" else [("long", "hub"), ("long", "authority"), ("short", "hub"), ("short", "authority")]
    # Always build both modes explicitly for a direct long-side comparison.
    for mode, mode_roles in (("hits_two_model", [("long", "hub"), ("long", "authority")]), ("hits_four_model", [("long", "hub"), ("long", "authority"), ("short", "hub"), ("short", "authority")])):
        predictions = {}
        for side, role in mode_roles:
            target = f"{side}_{role}" if WEIGHTING == "clip" else f"{side}_{role}_rank"
            sparse = base.select_top_bottom_rows(train, target)
            # Keep shared long models bit-for-bit comparable between the
            # two-model and four-model variants.  The mode must not enter the
            # seed; otherwise this ablation mixes model coupling with RF
            # randomness.
            values = base.reg_predict(sparse, test, features, target, stable_seed(family, target))
            predictions[(side, role)] = pd.Series(values, index=test.index).groupby(test.date).rank(pct=True, method="average").to_numpy()
        if ("long", "hub") not in predictions or ("long", "authority") not in predictions:
            continue
        scores = score_frame(test, predictions[("long", "hub")], 0.0, 0.0, predictions[("long", "authority")])
        m = base.backtest_scores(scores, close, "long", test_dates, base.HITS_THRESHOLD)
        m.update({"tier": str(row.tier), "model": mode, "weighting": WEIGHTING, "variant": "long", "family": family, "source": source})
        results.append(m)
        if mode == "hits_four_model" and ("short", "hub") in predictions and ("short", "authority") in predictions:
            scores = score_frame(test, 0.0, predictions[("short", "hub")], predictions[("short", "authority")], 0.0)
            m = base.backtest_scores(scores, close, "short", test_dates, base.HITS_THRESHOLD)
            m.update({"tier": str(row.tier), "model": mode, "weighting": WEIGHTING, "variant": "short", "family": family, "source": source})
            results.append(m)
    return results


def run_tier(tier: str) -> pd.DataFrame:
    index, prices, labels = base.load_data(tier)
    index["tier"] = tier
    requested = tuple(x.strip() for x in os.getenv("ORACLE_HITS_FAMILIES", "").split(",") if x.strip())
    if requested:
        index = index.loc[index.family.astype(str).isin(requested)]
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    dates = pd.DatetimeIndex(close.index[(close.index >= base.TEST_START) & (close.index <= base.TEST_END)])
    rows = []
    for _, row in index.iterrows():
        rows.extend(run_family(row, prices, labels, close, dates))
        print({"tier": tier, "family": str(row.family), "rows": len(rows)}, flush=True)
    out = pd.DataFrame(rows); out.to_parquet(OUT / f"{tier.lower()}_results.parquet", index=False); return out


def main() -> None:
    tiers = tuple(x.strip().upper() for x in os.getenv("ORACLE_HITS_ABLATION_TIERS", "1T,100B,10B").split(",") if x.strip())
    all_rows = [run_tier(t) for t in tiers]
    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    out.to_csv(OUT / "all_results.csv", index=False)
    summary = out.groupby(["tier", "model", "weighting", "variant"], as_index=False).agg(families=("family", "nunique"), mean_return=("total_return", "mean"), median_return=("total_return", "median"), min_return=("total_return", "min"), max_return=("total_return", "max"), mean_sharpe=("sharpe", "mean")) if not out.empty else out
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
