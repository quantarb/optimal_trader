#!/usr/bin/env python3
"""Execute trading_app_v2_equity_meta_model.ipynb for a given min market cap.

Uses the notebook's own cells (all feature families incl. technicals). Does not
reimplement training — only drives the existing notebook path.

Examples:
  python scripts/run_equity_meta_notebook.py --min-market-cap 1000000000000 --tag 1t --smoke
  python scripts/run_equity_meta_notebook.py --min-market-cap 100000000000 --tag 100b
  python scripts/run_equity_meta_notebook.py --min-market-cap 10000000000 --tag 10b
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-market-cap", type=int, required=True)
    parser.add_argument("--tag", type=str, required=True, help="artifact folder suffix, e.g. 1t / 100b / 10b")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Faster path: skip FMP refresh, symbol-level backtesting.py, reuse caches when valid",
    )
    parser.add_argument(
        "--skip-anchored-wfo",
        action="store_true",
        help="Skip anchored WFO (faster smoke)",
    )
    parser.add_argument(
        "--rebuild-feature-family-cache",
        action="store_true",
    )
    parser.add_argument(
        "--rebuild-family-score-cache",
        action="store_true",
    )
    parser.add_argument(
        "--label-mode",
        choices=("all_events", "oracle_only", "congress_buy_only"),
        default="all_events",
        help=(
            "all_events: notebook default (oracle + all event families incl. congress buy/sell). "
            "oracle_only: no event labels. "
            "congress_buy_only: oracle + congress buy events only (no sells, no other events)."
        ),
    )
    parser.add_argument(
        "--skip-symbol-level-bt",
        action="store_true",
        help="Skip symbol-level backtesting.py (faster A/B).",
    )
    parser.add_argument(
        "--oracle-ye-k-max",
        type=int,
        default=12,
        help="Use YE oracle k=1..K (inclusive). Default 12 (notebook default).",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=str,
        default="",
        help="Reuse an existing feature_family_panels directory (labels/k sweeps).",
    )
    parser.add_argument(
        "--run-name-suffix",
        type=str,
        default="",
        help="Optional artifact dir suffix, e.g. kmax03.",
    )
    args = parser.parse_args()

    started = perf_counter()
    repo = Path(__file__).resolve().parents[1]
    # Notebook cell 1 walks Path.cwd() for REPO_ROOT / default_paths.
    import os

    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    workspace = repo.parent
    for dirname in ("quant-orchestrator", "quant-warehouse", "fmpsdk"):
        p = (workspace / dirname).resolve()
        if p.exists():
            sys.path = [str(p)] + [
                e for e in sys.path if str(Path(e or ".").expanduser().resolve()) != str(p)
            ]

    # Prefer non-interactive display
    def display(obj=None, *a, **k):
        if obj is None:
            return
        if isinstance(obj, pd.DataFrame):
            print(obj.head(12).to_string(), flush=True)
            if len(obj) > 12:
                print(f"... ({len(obj)} rows)", flush=True)
        else:
            print(obj, flush=True)

    # IPython.display used by notebook
    import types

    fake_ipython = types.ModuleType("IPython")
    fake_display_mod = types.ModuleType("IPython.display")
    fake_display_mod.display = display
    sys.modules["IPython"] = fake_ipython
    sys.modules["IPython.display"] = fake_display_mod

    g: dict = {
        "__name__": "__main__",
        "display": display,
        "pd": pd,
        "np": np,
        "json": json,
        "Path": Path,
        "sys": sys,
        "perf_counter": perf_counter,
    }

    nb_path = repo / "notebooks" / "trading_app_v2_equity_meta_model.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    def exec_cell(idx: int, *, label: str | None = None, source_override: str | None = None) -> None:
        cell = nb["cells"][idx]
        if cell.get("cell_type") != "code":
            print(f"\n===== SKIP non-code CELL {idx} {label or ''} =====", flush=True)
            return
        src = source_override
        if src is None:
            src = cell["source"]
            if isinstance(src, list):
                src = "".join(src)
        print(f"\n===== CELL {idx} {label or ''} =====", flush=True)
        t0 = perf_counter()
        try:
            exec(compile(src, f"{nb_path.name}:cell{idx}", "exec"), g, g)
        except Exception:
            print(f"!!!!! cell {idx} FAILED after {perf_counter()-t0:.1f}s !!!!!", flush=True)
            traceback.print_exc()
            raise
        print(f"----- cell {idx} done in {perf_counter()-t0:.1f}s -----", flush=True)

    # Cell 1: imports / paths (hardcodes equity_meta_model_10b — override after)
    exec_cell(1, label="imports")
    tag = str(args.tag).strip().lower()
    meta_dir = g["paths"].artifact_root / f"equity_meta_model_{tag}"
    meta_dir.mkdir(parents=True, exist_ok=True)
    g["META_ARTIFACT_DIR"] = meta_dir
    print({"META_ARTIFACT_DIR": str(meta_dir)}, flush=True)

    exec_cell(3, label="storage")
    exec_cell(5, label="config")

    # Overrides after config cell
    g["MIN_MARKET_CAP"] = int(args.min_market_cap)
    g["RUN_FMP_REFRESH"] = False
    g["INCLUDE_TECHNICAL_FEATURE_FAMILIES"] = True
    g["REQUIRE_ALL_REQUESTED_FEATURE_FAMILIES"] = True
    g["RUN_EQUITY_META_EXPERIMENT"] = True
    g["RUN_BACKTESTS"] = True
    g["REBUILD_FEATURE_FAMILY_CACHE"] = bool(args.rebuild_feature_family_cache)
    g["REBUILD_FAMILY_SCORE_CACHE"] = bool(args.rebuild_family_score_cache)
    g["REUSE_FAMILY_SCORE_CACHE"] = not bool(args.rebuild_family_score_cache)

    label_mode = str(args.label_mode)
    if label_mode == "oracle_only":
        g["INCLUDE_EVENT_LABELS_IN_ORACLE_LABELS"] = False
        g["EVENT_LABEL_FAMILIES"] = ()
    elif label_mode == "congress_buy_only":
        # Only congress family; sell rows filtered after target-engineering cell.
        g["INCLUDE_EVENT_LABELS_IN_ORACLE_LABELS"] = True
        g["EVENT_LABEL_FAMILIES"] = ("congress",)
    else:
        g["INCLUDE_EVENT_LABELS_IN_ORACLE_LABELS"] = True
        # keep notebook EVENT_LABEL_FAMILIES

    # Label-mode / k-sweep runs need their own artifact trees so caches don't clash.
    tag = str(args.tag).strip().lower()
    k_max = int(args.oracle_ye_k_max)
    if k_max < 1 or k_max > 12:
        raise ValueError("--oracle-ye-k-max must be in 1..12")
    ye_ks = list(range(1, k_max + 1))
    g["ORACLE_YE_K_MAX"] = k_max
    g["ORACLE_YE_KS"] = tuple(ye_ks)

    dir_parts = [f"equity_meta_model_{tag}"]
    if label_mode != "all_events":
        dir_parts[0] = f"equity_meta_model_{tag}_{label_mode}"
    if args.run_name_suffix:
        dir_parts[0] = f"{dir_parts[0]}_{args.run_name_suffix}"
    elif k_max != 12 or label_mode == "oracle_only":
        # Distinguish k sweeps / oracle-only trees from default all-events k=1..12 runs.
        if k_max != 12:
            dir_parts[0] = f"{dir_parts[0]}_kmax{k_max:02d}"
    meta_dir = g["paths"].artifact_root / dir_parts[0]
    meta_dir.mkdir(parents=True, exist_ok=True)
    g["META_ARTIFACT_DIR"] = meta_dir
    if label_mode != "all_events" or k_max != 12:
        # Labels change ⇒ family scores must retrain.
        g["REBUILD_FAMILY_SCORE_CACHE"] = True
        g["REUSE_FAMILY_SCORE_CACHE"] = False

    if args.smoke or args.skip_anchored_wfo:
        g["RUN_ANCHORED_WFO"] = False
    if args.smoke or args.skip_symbol_level_bt:
        g["RUN_SYMBOL_LEVEL_BACKTESTING_PY"] = False
        g["SAVE_LARGE_INTERMEDIATE_CSVS"] = False

    # Rebuild artifact key/dir after MIN_MARKET_CAP override
    g["RUN_ARTIFACT_KEY"] = f"mcap_{int(g['MIN_MARKET_CAP'])}_train_{g['BASE_TRAIN_END']}_seed_{g['RANDOM_SEED']}"
    g["RUN_ARTIFACT_DIR"] = g["META_ARTIFACT_DIR"] / g["RUN_ARTIFACT_KEY"]
    g["RUN_ARTIFACT_DIR"].mkdir(parents=True, exist_ok=True)

    # Optional shared feature cache (k sweeps share FE within a universe).
    feature_cache = str(args.feature_cache_dir or "").strip()
    if feature_cache:
        shared = Path(feature_cache).expanduser().resolve()
        if not shared.exists():
            raise FileNotFoundError(f"--feature-cache-dir not found: {shared}")
        target = g["RUN_ARTIFACT_DIR"] / "feature_family_panels"
        if target.exists() or target.is_symlink():
            if target.is_symlink() or target.is_file():
                target.unlink()
            else:
                # Keep existing local cache if present.
                pass
        if not target.exists():
            target.symlink_to(shared, target_is_directory=True)
            print({"feature_cache_symlink": str(target), "->": str(shared)}, flush=True)
        g["REBUILD_FEATURE_FAMILY_CACHE"] = False

    # Rebuild EXPERIMENT_STRATEGY_SOURCES in case technicals flag changed
    tech = g["TECHNICAL_STRATEGY_SOURCES"] if g["INCLUDE_TECHNICAL_FEATURE_FAMILIES"] else ()
    g["EXPERIMENT_STRATEGY_SOURCES"] = tuple(
        dict.fromkeys((*g["DEFAULT_STRATEGY_SOURCES"], *tech))
    )

    print(
        {
            "tag": tag,
            "label_mode": label_mode,
            "oracle_ye_k_max": k_max,
            "oracle_ye_ks": ye_ks,
            "min_market_cap": g["MIN_MARKET_CAP"],
            "run_artifact_dir": str(g["RUN_ARTIFACT_DIR"]),
            "n_strategy_sources": len(g["EXPERIMENT_STRATEGY_SOURCES"]),
            "include_event_labels": g.get("INCLUDE_EVENT_LABELS_IN_ORACLE_LABELS"),
            "event_label_families": list(g.get("EVENT_LABEL_FAMILIES") or ()),
            "include_technicals": g["INCLUDE_TECHNICAL_FEATURE_FAMILIES"],
            "run_anchored_wfo": g["RUN_ANCHORED_WFO"],
            "run_symbol_level_bt": g.get("RUN_SYMBOL_LEVEL_BACKTESTING_PY"),
            "rebuild_family_score_cache": g.get("REBUILD_FAMILY_SCORE_CACHE"),
            "feature_cache_dir": feature_cache or None,
            "smoke": bool(args.smoke),
        },
        flush=True,
    )

    # Pipeline cells
    exec_cell(7, label="universe_screen")
    symbols = list(g.get("screened_equity_symbols") or [])
    print({"n_symbols": len(symbols), "symbols_head": symbols[:25]}, flush=True)
    if not symbols:
        raise RuntimeError("Universe screen returned 0 symbols")

    # Patch cell 9 to use the requested YE k set (notebook hardcodes 1..12).
    cell9 = nb["cells"][9]["source"]
    if isinstance(cell9, list):
        cell9 = "".join(cell9)
    cell9 = cell9.replace(
        'oracle_trade_k_by_frequency={"YE": tuple(range(1, 13))}',
        f"oracle_trade_k_by_frequency={{'YE': {tuple(ye_ks)!r}}}",
    )
    cell9 = cell9.replace(
        'manifest.get("oracle_trade_k_by_frequency") == {"YE": list(range(1, 13))}',
        f'manifest.get("oracle_trade_k_by_frequency") == {{"YE": {list(ye_ks)!r}}}',
    )
    cell9 = cell9.replace(
        '"oracle_trade_k_by_frequency": {"YE": list(range(1, 13))},',
        f'"oracle_trade_k_by_frequency": {{"YE": {list(ye_ks)!r}}},',
    )
    exec_cell(9, label="target_engineering", source_override=cell9)

    if label_mode == "congress_buy_only":
        # Keep oracle trades + congress_buy event rows only (drop sells and non-congress events).
        labels = g["oracle_label_rows"].copy()
        events = g.get("event_label_rows", pd.DataFrame())
        labels["label_source"] = labels["label_source"].astype(str)
        pure_oracle = labels.loc[labels["label_source"].eq("oracle_trade"), ["symbol", "date", "collapsed_label", "label_source"]].copy()
        if events is not None and not events.empty:
            ev = events.copy()
            if "event_type" in ev.columns:
                buy = ev.loc[ev["event_type"].astype(str).eq("congress_buy")].copy()
            else:
                buy = ev.loc[ev["label_source"].astype(str).str.contains("congress_buy", na=False)].copy()
            if not buy.empty:
                buy = buy[["symbol", "date", "collapsed_label", "label_source"]].copy()
                buy["collapsed_label"] = "oracle_long"
                buy["label_source"] = "event_congress_buy"
            else:
                buy = pd.DataFrame(columns=["symbol", "date", "collapsed_label", "label_source"])
        else:
            # Fallback: filter combined table
            buy = labels.loc[
                labels["label_source"].astype(str).str.fullmatch(r"event_congress_buy")
                | labels["label_source"].astype(str).eq("event_congress_buy"),
                ["symbol", "date", "collapsed_label", "label_source"],
            ].copy()
            buy["collapsed_label"] = "oracle_long"
            buy["label_source"] = "event_congress_buy"

        combined = pd.concat([pure_oracle, buy], ignore_index=True, sort=False)
        combined["symbol"] = combined["symbol"].astype(str).str.upper()
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.normalize()
        combined = combined.dropna(subset=["symbol", "date", "collapsed_label"])
        # Drop symbol/date conflicts (long vs short)
        nlab = combined.groupby(["symbol", "date"])["collapsed_label"].nunique()
        bad = nlab.loc[nlab.gt(1)].index
        if len(bad):
            bad_df = pd.DataFrame(list(bad), columns=["symbol", "date"]).assign(_c=True)
            combined = combined.merge(bad_df, on=["symbol", "date"], how="left")
            combined = combined.loc[~combined["_c"].fillna(False)].drop(columns=["_c"])
        combined = (
            combined.groupby(["symbol", "date", "collapsed_label"], as_index=False)
            .agg(label_source=("label_source", lambda v: "|".join(dict.fromkeys(sorted(map(str, v))))))
            .sort_values(["date", "symbol"])
            .reset_index(drop=True)
        )
        g["oracle_label_rows"] = combined
        g["event_label_rows"] = buy
        # Invalidate any label cache written by cell 9 so save/meta use filtered labels.
        cache_dir = g.get("ORACLE_LABEL_CACHE_DIR")
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            for name, frame in (
                ("oracle_label_rows.parquet", combined),
                ("event_label_rows.parquet", buy if buy is not None else pd.DataFrame()),
            ):
                path = cache_dir / name
                try:
                    frame.to_parquet(path, index=False)
                except Exception as exc:  # noqa: BLE001
                    print(f"[label-mode] warn write {path}: {exc}", flush=True)
        print(
            {
                "label_mode": "congress_buy_only",
                "pure_oracle_rows": len(pure_oracle),
                "congress_buy_rows": len(buy),
                "combined_rows": len(combined),
                "sources": combined["label_source"].value_counts().head(10).to_dict(),
            },
            flush=True,
        )

    print(
        {
            "oracle_label_rows": len(g.get("oracle_label_rows", [])),
            "event_label_rows": len(g.get("event_label_rows", [])),
            "label_classes": sorted(g["oracle_label_rows"]["collapsed_label"].dropna().unique())
            if "oracle_label_rows" in g and len(g["oracle_label_rows"])
            else [],
            "label_mode": label_mode,
        },
        flush=True,
    )

    exec_cell(11, label="feature_engineering")
    idx = g.get("feature_family_index")
    if idx is None or len(idx) == 0:
        raise RuntimeError("feature_family_index empty after FE")
    print(
        {
            "feature_families": len(idx),
            "families_with_features": int(idx["features"].gt(0).sum()),
            "strategy_sources": idx.loc[idx["features"].gt(0), "strategy_source"].astype(str).tolist(),
        },
        flush=True,
    )
    tech_rows = idx.loc[idx["strategy_source"].astype(str).str.contains("technical", case=False, na=False)]
    if tech_rows.empty or not tech_rows["features"].gt(0).any():
        raise RuntimeError("No technical feature families with features>0 — notebook FE path incomplete")

    exec_cell(13, label="experiment_plan")
    exec_cell(15, label="train_base_family_models")
    print(
        {
            "base_model_rows": len(g.get("base_model_results", [])),
            "trained_ok": int((g["base_model_results"]["status"] == "ok").sum())
            if "base_model_results" in g and "status" in g["base_model_results"].columns
            else None,
        },
        flush=True,
    )

    exec_cell(17, label="meta_training_matrix")
    exec_cell(19, label="train_meta_classifier")
    exec_cell(21, label="secondary_diagnostics")

    if g.get("RUN_BACKTESTS", True):
        exec_cell(23, label="primary_trading_performance")

    if g.get("RUN_ANCHORED_WFO", False):
        exec_cell(25, label="anchored_wfo")
    else:
        for name in ("anchored_wfo_stitched_summary", "anchored_wfo_fold_summary"):
            g.setdefault(name, pd.DataFrame())

    if g.get("RUN_SYMBOL_LEVEL_BACKTESTING_PY", False):
        exec_cell(27, label="symbol_level_backtesting_py")
    else:
        # Cell 29 always writes these; stub when symbol-level BT is skipped.
        for name in (
            "symbol_beat_summary",
            "symbol_beat_by_group",
            "symbol_level_backtesting_py",
        ):
            g.setdefault(name, pd.DataFrame())

    # Ensure optional frames exist for save cell.
    for name in (
        "comparison_strategy_scores",
        "trade_log",
        "regime_year_importance_summary",
    ):
        g.setdefault(name, pd.DataFrame())

    exec_cell(29, label="save_artifacts")

    summary = {
        "tag": tag,
        "label_mode": label_mode,
        "oracle_ye_k_max": k_max,
        "oracle_ye_ks": ye_ks,
        "min_market_cap": int(g["MIN_MARKET_CAP"]),
        "n_symbols": len(symbols),
        "n_strategy_sources": len(g["EXPERIMENT_STRATEGY_SOURCES"]),
        "feature_families": int(len(idx)),
        "oracle_label_rows": int(len(g.get("oracle_label_rows", []))),
        "label_source_counts": (
            g["oracle_label_rows"]["label_source"].astype(str).value_counts().head(20).to_dict()
            if "oracle_label_rows" in g and len(g["oracle_label_rows"])
            else {}
        ),
        "run_artifact_dir": str(g["RUN_ARTIFACT_DIR"]),
        "elapsed_seconds": perf_counter() - started,
        "smoke": bool(args.smoke),
    }
    # Capture best trading metrics for k-sweep tables.
    lb_path = g["RUN_ARTIFACT_DIR"] / "trading_performance_leaderboard.csv"
    if lb_path.exists() and lb_path.stat().st_size > 10:
        lb = pd.read_csv(lb_path)
        if not lb.empty and "sharpe" in lb.columns:
            best = lb.sort_values("sharpe", ascending=False).iloc[0]
            summary["best_strategy"] = str(best.get("strategy_source"))
            summary["best_total_return"] = float(best.get("total_return"))
            summary["best_sharpe"] = float(best.get("sharpe"))
            summary["best_max_drawdown"] = float(best.get("max_drawdown"))
            summary["leaderboard"] = lb.to_dict("records")
    out = g["RUN_ARTIFACT_DIR"] / "notebook_runner_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\n===== RUN COMPLETE =====", flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
