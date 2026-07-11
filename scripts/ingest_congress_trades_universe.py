#!/usr/bin/env python3
"""Refresh ownership_government_trades in Arctic for every expanded-universe symbol.

Then build congress event-pair labels for the full universe and write
oracle+congress label tables for the ablation.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path[:0] = [str(REPO), str(ROOT / "quant-warehouse"), str(ROOT / "quant-orchestrator")]

from app.quant_warehouse_storage import ensure_quant_warehouse_storage

ensure_quant_warehouse_storage()

from quant_warehouse.platforms.data_providers.fmp.target_engineering.event_pairs.store import (
    build_event_pairs_from_historical_data,
)
from quant_warehouse.warehouse.api import Warehouse

BASE_MANIFEST = (
    REPO
    / "artifacts/trading_app_v2/equity_meta_model_10b"
    / "mcap_10000000000_train_2020-12-31_seed_20260707"
    / "oracle_labels"
    / "manifest.json"
)
OUT = (
    REPO
    / "artifacts/trading_app_v2/equity_meta_model_10b"
    / "label_ablation_oracle_vs_congress"
    / "expanded_universe_run"
    / "congress_ingest"
)
MEGA = ("AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA")
MAX_WORKERS = 8
SECTION = "ownership_government_trades"
PROVIDER = "fmp"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _universe() -> list[str]:
    base = [str(s).strip().upper() for s in json.loads(BASE_MANIFEST.read_text())["symbols"] if str(s).strip()]
    return sorted(dict.fromkeys([*base, *MEGA]))


def _refresh_one(symbol: str) -> dict:
    # each worker gets its own Warehouse to avoid shared-state issues
    wh = Warehouse()
    try:
        stats = wh.fundamentals.refresh_section(symbol, SECTION, provider=PROVIDER)
        df = wh.fundamentals.read(symbol, section=SECTION, provider=PROVIDER)
        n = 0 if df is None else int(len(df))
        n_tt = 0
        if df is not None and not df.empty and "transaction_type" in df.columns:
            n_tt = int(df["transaction_type"].notna().sum())
        return {
            "symbol": symbol,
            "status": "ok",
            "rows": int(stats.get("rows", n) if isinstance(stats, dict) else n),
            "transaction_type_nonnull": n_tt,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 — collect per-symbol failures
        return {
            "symbol": symbol,
            "status": "error",
            "rows": 0,
            "transaction_type_nonnull": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _combine_labels(*frames: pd.DataFrame) -> pd.DataFrame:
    usable = [f for f in frames if f is not None and not f.empty]
    if not usable:
        return pd.DataFrame(columns=["symbol", "date", "collapsed_label", "label_source"])
    labels = pd.concat(usable, ignore_index=True, sort=False)
    labels["symbol"] = labels["symbol"].astype(str).str.upper()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    labels = labels.dropna(subset=["symbol", "date", "collapsed_label"])
    side_n = labels.groupby(["symbol", "date"])["collapsed_label"].nunique()
    conflicts = side_n.loc[side_n.gt(1)].index
    if len(conflicts):
        conflict_df = pd.DataFrame(list(conflicts), columns=["symbol", "date"]).assign(_c=True)
        labels = labels.merge(conflict_df, on=["symbol", "date"], how="left")
        labels = labels.loc[~labels["_c"].fillna(False)].drop(columns=["_c"])
    return (
        labels.assign(label_source=labels["label_source"].fillna("unknown").astype(str))
        .groupby(["symbol", "date", "collapsed_label"], as_index=False)
        .agg(label_source=("label_source", lambda v: "|".join(dict.fromkeys(sorted(map(str, v))))))
        .sort_values(["date", "symbol", "collapsed_label"])
        .reset_index(drop=True)
    )


def _build_congress_labels(symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    wh = Warehouse()
    pair_frames = []
    diag = []
    for i, sym in enumerate(symbols, start=1):
        pairs = build_event_pairs_from_historical_data(
            sym,
            fundamentals=wh.fundamentals,
            event_families=("congress",),
            provider=PROVIDER,
        )
        n = 0 if pairs is None or pairs.empty else len(pairs)
        diag.append({"symbol": sym, "congress_pairs": n})
        if pairs is not None and not pairs.empty:
            pair_frames.append(pairs)
        if i % 50 == 0 or i == len(symbols):
            _log(f"[labels] {i}/{len(symbols)} with_pairs={len(pair_frames)} total_pair_rows={sum(len(x) for x in pair_frames)}")
    if not pair_frames:
        empty = pd.DataFrame(columns=["symbol", "date", "collapsed_label", "label_source"])
        return empty, pd.DataFrame(diag)
    congress = pd.concat(pair_frames, ignore_index=True)
    etype = congress["event_type"].astype(str)
    mapped = etype.map({"congress_buy": "oracle_long", "congress_sell": "oracle_short"})
    out = pd.DataFrame(
        {
            "symbol": congress["symbol"].astype(str).str.upper(),
            "date": pd.to_datetime(congress["event_date"], errors="coerce").dt.normalize(),
            "collapsed_label": mapped,
            "label_source": "event_congress_" + etype.str.replace("congress_", "", regex=False),
        }
    ).dropna(subset=["symbol", "date", "collapsed_label"])
    nlab = out.groupby(["symbol", "date"])["collapsed_label"].nunique()
    bad = nlab.loc[nlab.gt(1)].index
    if len(bad):
        bad_df = pd.DataFrame(list(bad), columns=["symbol", "date"]).assign(_b=True)
        out = out.merge(bad_df, on=["symbol", "date"], how="left")
        out = out.loc[~out["_b"].fillna(False)].drop(columns=["_b"])
    labels = (
        out.groupby(["symbol", "date", "collapsed_label"], as_index=False)
        .agg(label_source=("label_source", lambda v: "|".join(dict.fromkeys(sorted(map(str, v))))))
        .reset_index(drop=True)
    )
    return labels, pd.DataFrame(diag)


def main() -> None:
    started = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    symbols = _universe()
    _log(f"universe={len(symbols)} workers={MAX_WORKERS}")

    # Optional resume: skip symbols already usable if --force not set
    force = "--force" in sys.argv
    need = list(symbols)
    if not force:
        wh = Warehouse()
        skip = []
        for sym in symbols:
            df = wh.fundamentals.read(sym, section=SECTION, provider=PROVIDER)
            if (
                df is not None
                and not df.empty
                and "transaction_type" in df.columns
                and df["transaction_type"].notna().any()
            ):
                skip.append(sym)
        need = [s for s in symbols if s not in set(skip)]
        _log(f"already_usable={len(skip)} need_refresh={len(need)} (pass --force to re-fetch all)")
    else:
        _log("force refresh all symbols")

    rows = []
    if need:
        done = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {pool.submit(_refresh_one, sym): sym for sym in need}
            for fut in as_completed(futs):
                row = fut.result()
                rows.append(row)
                done += 1
                if done % 25 == 0 or done == len(need):
                    ok = sum(1 for r in rows if r["status"] == "ok")
                    usable = sum(1 for r in rows if r["transaction_type_nonnull"] > 0)
                    err = sum(1 for r in rows if r["status"] != "ok")
                    _log(
                        f"[ingest] {done}/{len(need)} ok={ok} usable_tt={usable} err={err} "
                        f"elapsed={perf_counter()-started:.0f}s last={row['symbol']} rows={row['rows']} tt={row['transaction_type_nonnull']}"
                    )
    else:
        _log("nothing to refresh")

    ingest_df = pd.DataFrame(rows)
    if not ingest_df.empty:
        ingest_df.to_csv(OUT / "ingest_results.csv", index=False)
        _log(
            "ingest summary: "
            f"ok={(ingest_df.status.eq('ok').sum())} "
            f"usable_tt={(ingest_df.transaction_type_nonnull.gt(0).sum())} "
            f"err={(ingest_df.status.ne('ok').sum())} "
            f"total_rows={int(ingest_df.rows.sum())}"
        )

    # Full-universe usability audit after ingest
    wh = Warehouse()
    audit = []
    for sym in symbols:
        df = wh.fundamentals.read(sym, section=SECTION, provider=PROVIDER)
        n = 0 if df is None else len(df)
        n_tt = 0
        if df is not None and not df.empty and "transaction_type" in df.columns:
            n_tt = int(df["transaction_type"].notna().sum())
        audit.append({"symbol": sym, "rows": n, "transaction_type_nonnull": n_tt, "usable": n_tt > 0})
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(OUT / "post_ingest_coverage.csv", index=False)
    _log(
        f"post-ingest coverage: usable={int(audit_df.usable.sum())}/{len(audit_df)} "
        f"rows_sum={int(audit_df.rows.sum())} tt_sum={int(audit_df.transaction_type_nonnull.sum())}"
    )

    # Build congress labels for full universe
    _log("building congress labels for full universe from Arctic")
    congress_labels, pair_diag = _build_congress_labels(symbols)
    congress_labels.to_parquet(OUT / "labels_congress_full_universe.parquet", index=False)
    pair_diag.to_csv(OUT / "congress_pair_diag.csv", index=False)
    # also write to ablation root for downstream
    ablation = OUT.parent.parent
    congress_labels.to_parquet(ablation / "labels_congress_only.parquet", index=False)
    _log(
        f"congress labels: rows={len(congress_labels)} symbols={congress_labels['symbol'].nunique() if not congress_labels.empty else 0} "
        f"sources={congress_labels['label_source'].value_counts().to_dict() if not congress_labels.empty else {}}"
    )

    # Combine with expanded oracle-only if present
    oracle_path = OUT.parent / "labels_oracle_only_expanded.parquet"
    if not oracle_path.exists():
        oracle_path = ablation / "labels_oracle_only.parquet"
    if oracle_path.exists():
        oracle = pd.read_parquet(oracle_path)
        combined = _combine_labels(oracle, congress_labels)
        combined.to_parquet(OUT.parent / "labels_oracle_plus_congress_full_universe.parquet", index=False)
        combined.to_parquet(ablation / "labels_oracle_plus_congress.parquet", index=False)
        _log(
            f"combined oracle+congress: rows={len(combined)} "
            f"oracle_in={len(oracle)} congress_in={len(congress_labels)} "
            f"sources_head={combined['label_source'].value_counts().head(10).to_dict()}"
        )
    else:
        _log(f"oracle labels not found at {oracle_path}; skipped combine")

    summary = {
        "universe": len(symbols),
        "refreshed": len(need),
        "post_ingest_usable_symbols": int(audit_df.usable.sum()),
        "post_ingest_tt_rows": int(audit_df.transaction_type_nonnull.sum()),
        "congress_label_rows": int(len(congress_labels)),
        "congress_label_symbols": int(congress_labels["symbol"].nunique()) if not congress_labels.empty else 0,
        "elapsed_seconds": perf_counter() - started,
        "force": force,
    }
    (OUT / "ingest_summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"DONE {json.dumps(summary)}")


if __name__ == "__main__":
    main()
