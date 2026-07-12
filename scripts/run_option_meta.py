#!/usr/bin/env python3
"""Run the canonical equity-family and option-meta workflow at any scale."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-market-cap", type=int, default=1_000_000_000_000)
    parser.add_argument("--tag", default="", help="Artifact tag; defaults to the numeric threshold.")
    parser.add_argument("--n-estimators", type=int, default=400)
    args = parser.parse_args()
    min_market_cap = int(args.min_market_cap)
    tag = str(args.tag).strip().lower() or f"mcap_{min_market_cap}"
    started = perf_counter()

    repo = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repo), str(repo.parent / "quant-warehouse"), str(repo.parent / "quant-orchestrator")]

    def display(obj=None, *_args, **_kwargs):
        if obj is not None:
            print(obj.head(20).to_string() if isinstance(obj, pd.DataFrame) else obj)

    g = {
        "__name__": "__main__",
        "display": display,
        "pd": pd,
        "np": np,
        "json": json,
        "Path": Path,
        "sys": sys,
        "perf_counter": perf_counter,
    }
    notebook_path = repo / "notebooks" / "trading_app_v2_option_ml_ranker.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    def exec_cell(index: int, label: str) -> None:
        print(f"\n===== {label} =====", flush=True)
        then = perf_counter()
        source = "".join(notebook["cells"][index]["source"])
        exec(compile(source, f"{notebook_path.name}:cell{index}", "exec"), g, g)
        print(f"----- {label} completed in {perf_counter() - then:.1f}s -----", flush=True)

    exec_cell(1, "imports")
    exec_cell(3, "storage")
    exec_cell(5, "configuration")
    g.update(
        {
            "MIN_MARKET_CAP": min_market_cap,
            "ORACLE_YE_K": (1, 2, 3),
            "RUN_UNIVERSE_SCREEN": True,
            "RUN_ORACLE_TRADE_LABELS": True,
            "RUN_OPTION_COVERAGE": True,
            "RUN_OPTION_LABEL_PANEL": True,
            "RUN_EQUITY_FAMILY_MODELS": True,
            "RUN_OPTION_META_STACK": True,
            "RUN_OPTION_RANKER_TRAINING": True,
            "INCLUDE_TECHNICAL_FEATURE_FAMILIES": True,
            "ALL_FEATURE_FAMILIES": True,
            "REQUIRE_ALL_REQUESTED_FEATURE_FAMILIES": True,
            "N_ESTIMATORS": int(args.n_estimators),
            "TRAIN_TOP_K_BY_RETURN": 128,
            "OPTION_MAX_DTE": None,
        }
    )
    paths = g["paths"]
    output_dir = paths.option_artifact_dir / f"option_meta_stack_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    option_panel_path = output_dir / "option_candidate_panel_unified.parquet"
    g["OPTION_PANEL_PATH"] = option_panel_path
    g["_RUN_OUTPUT_DIR"] = output_dir
    if option_panel_path.exists() and {"rank_y", "label_basis"}.issubset(
        set(pq.ParquetFile(option_panel_path).schema.names)
    ):
        g["RUN_OPTION_LABEL_PANEL"] = False
        g["RUN_OPTION_COVERAGE"] = False

    exec_cell(7, "universe")
    symbols = tuple(str(value).strip().upper() for value in g["screened_equity_symbols"])
    g["screened_equity_symbols"] = symbols
    exec_cell(9, "oracle equity labels")
    exec_cell(13, "option coverage")
    exec_cell(15, "unified option labels")

    option_panel = g["option_candidate_panel"]
    required = {"rank_y", "label_basis"}
    if missing := required.difference(option_panel.columns):
        raise RuntimeError(f"Unified option panel missing columns: {sorted(missing)}")
    if pd.to_numeric(option_panel["rank_y"], errors="coerce").isna().any():
        raise RuntimeError("Unified option panel contains missing rank_y")
    basis_counts = option_panel["label_basis"].value_counts().to_dict()
    if not basis_counts.get("realized_exit_return") or not basis_counts.get("expiration_closeness"):
        raise RuntimeError(f"Unified option panel missing a target basis: {basis_counts}")

    labels = g["oracle_label_rows"].copy()
    labels["symbol"] = labels["symbol"].astype(str).str.upper()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    option_dates = option_panel[["symbol", "entry_date"]].rename(columns={"entry_date": "date"})
    observation_dates = pd.concat(
        [labels[["symbol", "date"]], option_dates], ignore_index=True
    ).dropna().drop_duplicates()

    print("\n===== optimized warehouse feature panels =====", flush=True)
    feature_started = perf_counter()
    from quant_warehouse.research_tools import (
        FamilyEvaluationConfig,
        build_fundamental_feature_panel,
        build_technical_feature_panel,
        cap_features_by_quality,
    )
    from quant_warehouse.platforms.data_providers.fmp.feature_engineering import TA_CLASSIC_FAMILY_PREFIXES
    from quant_warehouse.warehouse.api import Warehouse

    warehouse = g.get("warehouse") or Warehouse()
    feature_config = FamilyEvaluationConfig(
        provider="fmp",
        market_cap_min=min_market_cap,
        start_date=g["DATA_START"],
        end_date=g["optional_date"](g["DATA_END"]),
    )
    fundamental, fundamental_meta, _, fundamental_timing = build_fundamental_feature_panel(
        symbols, feature_config, warehouse=warehouse, observation_dates=observation_dates
    )
    technical, technical_meta, _, technical_timing = build_technical_feature_panel(
        symbols,
        feature_config,
        strategy_sources=tuple(f"fmp.{family}" for family in TA_CLASSIC_FAMILY_PREFIXES),
        warehouse=warehouse,
        observation_dates=observation_dates,
        max_workers=8,
    )
    if technical.empty or technical_meta.empty:
        raise RuntimeError("Curated technical feature families are mandatory but the panel is empty")
    keys = ["symbol", "date"]
    overlap = set(fundamental.columns).intersection(technical.columns).difference(keys)
    technical = technical.drop(columns=sorted(overlap), errors="ignore")
    raw_panel = fundamental.merge(technical, on=keys, how="outer", sort=False)
    raw_meta = (
        pd.concat([fundamental_meta, technical_meta], ignore_index=True, sort=False)
        .drop_duplicates("feature", keep="first")
        .reset_index(drop=True)
    )
    selected, selected_meta, quality = cap_features_by_quality(raw_panel, raw_meta, max_features=None)
    selected = [column for column in selected if column in raw_panel.columns]
    feature_panel = raw_panel[[*keys, *selected]].copy()
    families = selected_meta.assign(
        strategy_source=lambda frame: frame["source"].astype(str) + "." + frame["family"].astype(str)
    )
    technical_family_count = int(
        families.loc[families["family"].astype(str).str.startswith("technical_")]["strategy_source"].nunique()
    )
    if technical_family_count != 6:
        raise RuntimeError(f"Expected all six curated technical families, found {technical_family_count}")
    g.update(
        {
            "warehouse": warehouse,
            "engineering_warehouse": warehouse,
            "feature_panel": feature_panel,
            "selected_features": selected,
            "selected_feature_metadata": selected_meta,
            "feature_quality": quality,
            "raw_feature_metadata": raw_meta,
        }
    )
    print(
        {
            "feature_rows": len(feature_panel),
            "features": len(selected),
            "families": int(families["strategy_source"].nunique()),
            "technical_families": technical_family_count,
            "feature_seconds": round(perf_counter() - feature_started, 2),
            **fundamental_timing,
            **technical_timing,
        },
        flush=True,
    )
    del raw_panel, fundamental, technical
    gc.collect()

    stage_source = "".join(notebook["cells"][17]["source"])
    marker = 'EQUITY_MODEL_RESULTS_PATH = META_DIR / "equity_family_model_results.csv"'
    injection = '''
META_DIR = _RUN_OUTPUT_DIR
META_DIR.mkdir(parents=True, exist_ok=True)
TEMPORAL_PANEL_PATH = META_DIR / "option_candidate_panel_temporal_is_oos.parquet"
TEMPORAL_SPLIT_SUMMARY_PATH = META_DIR / "option_ranker_temporal_split_summary.json"
EQUITY_SCORE_PATH = META_DIR / "equity_family_scores_on_option_entry_dates.parquet"
OPTION_STACK_PATH = META_DIR / "option_rows_with_equity_scores.parquet"
META_MODEL_PATH = META_DIR / "meta_stack_ranker.pkl"
META_SUMMARY_PATH = META_DIR / "meta_stack_summary.json"
OOS_COMPARISON_PATH = META_DIR / "option_ranker_oos_vs_baseline.csv"
EQUITY_MODEL_RESULTS_PATH = META_DIR / "equity_family_model_results.csv"
'''
    if marker not in stage_source:
        raise RuntimeError("Option meta notebook artifact-path contract changed")
    stage_source = stage_source.replace(marker, marker + "\n" + injection, 1)
    print("\n===== CUDA equity families and option meta-ranker =====", flush=True)
    exec(compile(stage_source, f"{notebook_path.name}:cell17", "exec"), g, g)

    result = {
        "status": "ok",
        "elapsed_seconds": round(perf_counter() - started, 2),
        "min_market_cap": min_market_cap,
        "tag": tag,
        "symbols": len(symbols),
        "label_basis_rows": basis_counts,
        "feature_families": int(families["strategy_source"].nunique()),
        "technical_feature_families": technical_family_count,
        "output_dir": str(output_dir),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
