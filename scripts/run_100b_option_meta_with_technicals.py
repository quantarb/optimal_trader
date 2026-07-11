#!/usr/bin/env python3
"""$100B run: universe → oracle labels → FE (fundamentals + technicals) → Stage A/B.

Reuses cached option_candidate_panel.parquet (filtered to screened symbols; no ThetaData rebuild).
Writes artifacts under option_family_ranker/option_meta_stack_100b_tech/.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


def main() -> int:
    started = perf_counter()
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    workspace = repo.parent
    for pkg, dirname in [
        ("quant_orchestrator", "quant-orchestrator"),
        ("quant_warehouse", "quant-warehouse"),
        ("fmpsdk", "fmpsdk"),
    ]:
        p = (workspace / dirname).resolve()
        if p.exists():
            sys.path = [str(p)] + [
                e
                for e in sys.path
                if str(Path(e or ".").expanduser().resolve()) != str(p)
            ]

    def display(obj=None, *args, **kwargs):
        if obj is None:
            return
        if isinstance(obj, pd.DataFrame):
            print(obj.head(20).to_string())
            if len(obj) > 20:
                print(f"... ({len(obj)} rows)")
        else:
            print(obj)

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

    nb_path = repo / "notebooks" / "trading_app_v2_option_ml_ranker.ipynb"
    nb = json.loads(nb_path.read_text())

    def exec_cell(idx: int, *, label: str | None = None) -> None:
        src = "".join(nb["cells"][idx]["source"])
        print(f"\n===== CELL {idx} {label or ''} =====", flush=True)
        t0 = perf_counter()
        exec(compile(src, f"{nb_path.name}:cell{idx}", "exec"), g, g)
        print(f"----- cell {idx} done in {perf_counter() - t0:.1f}s -----", flush=True)

    exec_cell(1, label="imports")
    exec_cell(3, label="storage")
    exec_cell(5, label="config")

    # $100B production-style settings
    g["MIN_MARKET_CAP"] = 100_000_000_000
    g["RUN_UNIVERSE_SCREEN"] = True
    g["RUN_ORACLE_TRADE_LABELS"] = True
    g["RUN_OPTION_COVERAGE"] = False
    g["RUN_OPTION_LABEL_PANEL"] = False
    g["RUN_EQUITY_FAMILY_MODELS"] = True
    g["RUN_OPTION_META_STACK"] = True
    g["RUN_OPTION_RANKER_TRAINING"] = True
    g["INCLUDE_TECHNICAL_FEATURE_FAMILIES"] = True
    g["ALL_FEATURE_FAMILIES"] = True
    g["REQUIRE_ALL_REQUESTED_FEATURE_FAMILIES"] = True
    g["N_ESTIMATORS"] = 400
    g["TRAIN_TOP_K_BY_RETURN"] = 100

    paths = g["paths"]
    full_panel = paths.artifact_root / "option_candidate_panel.parquet"
    if not full_panel.exists():
        raise FileNotFoundError(f"Missing option panel: {full_panel}")

    meta_dir = paths.option_artifact_dir / "option_meta_stack_100b_tech"
    meta_dir.mkdir(parents=True, exist_ok=True)
    g["_SMOKE_META_DIR"] = meta_dir  # reused inject name from 1T script pattern

    filtered_panel = meta_dir / "option_candidate_panel_100b.parquet"
    g["OPTION_PANEL_PATH"] = filtered_panel

    print(
        {
            "min_market_cap": g["MIN_MARKET_CAP"],
            "full_panel": str(full_panel),
            "filtered_panel": str(filtered_panel),
            "meta_dir": str(meta_dir),
            "include_technical": g["INCLUDE_TECHNICAL_FEATURE_FAMILIES"],
            "feature_families": len(g.get("FEATURE_FAMILIES", ())),
            "n_estimators": g["N_ESTIMATORS"],
        },
        flush=True,
    )

    exec_cell(7, label="universe")
    symbols = tuple(str(s).strip().upper() for s in g["screened_equity_symbols"])
    g["screened_equity_symbols"] = symbols
    print(f"100B symbols={len(symbols)}", flush=True)

    # Filter cached option panel to screened universe
    print(f"[panel] filtering {full_panel} to {len(symbols)} symbols...", flush=True)
    t0 = perf_counter()
    panel = pd.read_parquet(full_panel)
    panel["symbol"] = panel["symbol"].astype(str).str.upper()
    before = len(panel)
    panel = panel.loc[panel["symbol"].isin(set(symbols))].copy()
    if panel.empty:
        raise RuntimeError("Filtered option panel is empty for $100B universe")
    panel.to_parquet(filtered_panel, index=False)
    print(
        {
            "panel_rows_before": before,
            "panel_rows_after": len(panel),
            "panel_symbols": int(panel["symbol"].nunique()),
            "panel_trades": int(panel["trade_id"].nunique()) if "trade_id" in panel.columns else None,
            "filter_seconds": round(perf_counter() - t0, 1),
            "saved": str(filtered_panel),
        },
        flush=True,
    )
    del panel

    exec_cell(9, label="oracle_labels")
    print({"oracle_label_rows": len(g["oracle_label_rows"])}, flush=True)

    exec_cell(11, label="feature_engineering")
    fam = (
        g["selected_feature_metadata"]
        .assign(
            strategy_source=lambda d: d["source"].astype(str) + "." + d["family"].astype(str)
        )[["strategy_source", "family"]]
        .drop_duplicates()
        .sort_values("strategy_source")
    )
    tech = fam.loc[fam["family"].astype(str).str.startswith("technical_")]
    print(
        {
            "feature_panel_rows": len(g["feature_panel"]),
            "selected_features": len(g["selected_features"]),
            "n_families": int(fam["strategy_source"].nunique()),
            "n_technical_families": int(tech["strategy_source"].nunique()),
            "technical_families": tech["strategy_source"].tolist(),
        },
        flush=True,
    )
    if tech.empty:
        raise RuntimeError("No technical families selected — FE failed")

    src17 = "".join(nb["cells"][17]["source"])
    inject = """
META_DIR = _SMOKE_META_DIR
META_DIR.mkdir(parents=True, exist_ok=True)
TEMPORAL_PANEL_PATH = META_DIR / "option_candidate_panel_temporal_is_oos.parquet"
TEMPORAL_SPLIT_SUMMARY_PATH = META_DIR / "option_ranker_temporal_split_summary.json"
EQUITY_SCORE_PATH = META_DIR / "equity_family_scores_on_option_entry_dates.parquet"
OPTION_STACK_PATH = META_DIR / "option_rows_with_equity_scores.parquet"
META_MODEL_PATH = META_DIR / "meta_stack_ranker.pkl"
META_SUMMARY_PATH = META_DIR / "meta_stack_summary.json"
OOS_COMPARISON_PATH = META_DIR / "option_ranker_oos_vs_baseline.csv"
EQUITY_MODEL_RESULTS_PATH = META_DIR / "equity_family_model_results.csv"
print({"META_DIR": str(META_DIR), "OPTION_PANEL_PATH": str(OPTION_PANEL_PATH)}, flush=True)
"""
    marker = 'EQUITY_MODEL_RESULTS_PATH = META_DIR / "equity_family_model_results.csv"'
    if marker not in src17:
        raise RuntimeError("Could not find META path marker in cell 17")
    src17 = src17.replace(marker, marker + "\n" + inject, 1)

    print("\n===== CELL 17 stage A/B ($100B) =====", flush=True)
    t0 = perf_counter()
    exec(compile(src17, f"{nb_path.name}:cell17", "exec"), g, g)
    print(f"----- cell 17 done in {perf_counter() - t0:.1f}s -----", flush=True)

    summary_path = meta_dir / "meta_stack_summary.json"
    selector_path = meta_dir / "selector_summary.csv"
    result = {
        "elapsed_s": round(perf_counter() - started, 1),
        "min_market_cap": g["MIN_MARKET_CAP"],
        "n_symbols": len(symbols),
        "n_families": int(fam["strategy_source"].nunique()),
        "n_technical_families": int(tech["strategy_source"].nunique()),
        "technical_families": tech["strategy_source"].tolist(),
        "meta_dir": str(meta_dir),
        "summary_exists": summary_path.exists(),
        "selector_exists": selector_path.exists(),
    }
    if summary_path.exists():
        result["summary"] = json.loads(summary_path.read_text())
    if selector_path.exists():
        result["selector"] = pd.read_csv(selector_path).to_dict(orient="records")

    out = meta_dir / "run_100b_tech_summary.json"
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print("\n===== 100B RESULT =====", flush=True)
    print(json.dumps({k: v for k, v in result.items() if k != "summary"}, indent=2, default=str), flush=True)
    if "summary" in result and isinstance(result["summary"], dict):
        for key in ("headline", "fixed_near_atm", "ml_options", "head_to_head"):
            if key in result["summary"]:
                print(key, ":", json.dumps(result["summary"][key], indent=2, default=str), flush=True)
    print(f"Wrote {out}", flush=True)

    if not summary_path.exists():
        print("FAIL: missing meta_stack_summary.json", flush=True)
        return 1
    print("PASS: $100B run with technicals completed", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
