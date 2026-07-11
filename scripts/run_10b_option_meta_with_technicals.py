#!/usr/bin/env python3
"""$10B run: universe → oracle labels → sparse FE (fundamentals + technicals) → Stage A/B.

Memory-safe feature engineering: per-symbol build, keep only oracle/option entry dates.
Reuses cached option_candidate_panel.parquet (filtered; no ThetaData rebuild).
Writes artifacts under option_family_ranker/option_meta_stack_10b_tech/.
"""
from __future__ import annotations

import gc
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
    for dirname in ("quant-orchestrator", "quant-warehouse", "fmpsdk"):
        p = (workspace / dirname).resolve()
        if p.exists():
            sys.path = [str(p)] + [
                e for e in sys.path if str(Path(e or ".").expanduser().resolve()) != str(p)
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

    g["MIN_MARKET_CAP"] = 10_000_000_000
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

    meta_dir = paths.option_artifact_dir / "option_meta_stack_10b_tech"
    meta_dir.mkdir(parents=True, exist_ok=True)
    g["_SMOKE_META_DIR"] = meta_dir
    filtered_panel = meta_dir / "option_candidate_panel_10b.parquet"
    g["OPTION_PANEL_PATH"] = filtered_panel

    print(
        {
            "min_market_cap": g["MIN_MARKET_CAP"],
            "full_panel": str(full_panel),
            "filtered_panel": str(filtered_panel),
            "meta_dir": str(meta_dir),
            "include_technical": True,
            "n_estimators": g["N_ESTIMATORS"],
        },
        flush=True,
    )

    exec_cell(7, label="universe")
    symbols = tuple(str(s).strip().upper() for s in g["screened_equity_symbols"])
    g["screened_equity_symbols"] = symbols
    print(f"10B symbols={len(symbols)}", flush=True)

    print(f"[panel] filtering {full_panel} to {len(symbols)} symbols...", flush=True)
    t0 = perf_counter()
    panel = pd.read_parquet(full_panel)
    panel["symbol"] = panel["symbol"].astype(str).str.upper()
    before = len(panel)
    panel = panel.loc[panel["symbol"].isin(set(symbols))].copy()
    if panel.empty:
        raise RuntimeError("Filtered option panel is empty for $10B universe")
    panel.to_parquet(filtered_panel, index=False)
    print(
        {
            "panel_rows_before": before,
            "panel_rows_after": len(panel),
            "panel_symbols": int(panel["symbol"].nunique()),
            "panel_trades": int(panel["trade_id"].nunique()),
            "filter_seconds": round(perf_counter() - t0, 1),
        },
        flush=True,
    )

    option_keys = (
        panel[["symbol", "entry_date"]]
        .assign(
            symbol=lambda d: d["symbol"].astype(str).str.upper(),
            entry_date=lambda d: pd.to_datetime(d["entry_date"], errors="coerce").dt.normalize(),
        )
        .dropna()
        .drop_duplicates()
        .rename(columns={"entry_date": "date"})
    )
    del panel
    gc.collect()

    exec_cell(9, label="oracle_labels")
    # Drop heavy multi-k windows if present (unique is enough downstream)
    for heavy in ("oracle_trade_windows", "oracle_price_frames", "oracle_trade_result"):
        if heavy in g:
            del g[heavy]
    gc.collect()
    print({"oracle_label_rows": len(g["oracle_label_rows"])}, flush=True)

    # ---------- sparse FE ----------
    print("\n===== SPARSE FEATURE ENGINEERING (10B) =====", flush=True)
    from quant_warehouse.research_tools.feature_family_eval import (
        FamilyEvaluationConfig,
        FeatureSpec,
        _add_cross_symbol_context_features,
        _add_macro_context_features,
        _add_time_calendar_features,
        _build_symbol_fundamental_panel,
        cap_features_by_quality,
    )
    from quant_warehouse.platforms.data_providers.fmp.feature_engineering import (
        TA_CLASSIC_FAMILY_PREFIXES,
        build_price_ta_classic_feature_families,
    )
    from quant_warehouse.warehouse.api import Warehouse

    warehouse = g.get("warehouse") or Warehouse()
    g["warehouse"] = warehouse
    feature_config = FamilyEvaluationConfig(
        provider="fmp",
        market_cap_min=g["MIN_MARKET_CAP"],
        start_date=g["DATA_START"],
        end_date=g["optional_date"](g["DATA_END"]),
    )

    labels = g["oracle_label_rows"].copy()
    labels["symbol"] = labels["symbol"].astype(str).str.upper()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    label_keys = labels[["symbol", "date"]].dropna().drop_duplicates()
    needed_keys = pd.concat([label_keys, option_keys], ignore_index=True).drop_duplicates()
    needed_by_symbol = {
        str(sym): set(pd.to_datetime(grp["date"]).tolist())
        for sym, grp in needed_keys.groupby("symbol")
    }
    print(
        {
            "label_keys": len(label_keys),
            "option_keys": len(option_keys),
            "needed_keys": len(needed_keys),
            "symbols_with_keys": len(needed_by_symbol),
        },
        flush=True,
    )

    def _slice_prices(frame: pd.DataFrame, start, end) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()
        out = frame.copy()
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out.loc[out.index.notna()].sort_index()
        if start is not None:
            out = out.loc[out.index >= pd.Timestamp(start)]
        if end is not None:
            out = out.loc[out.index <= pd.Timestamp(end)]
        return out

    frames: list[pd.DataFrame] = []
    all_specs: list[FeatureSpec] = []
    tech_specs: list[FeatureSpec] = []
    diagnostics: list[dict] = []
    t_fe = perf_counter()
    n_sym = len(symbols)
    for i, symbol in enumerate(symbols, start=1):
        need_dates = needed_by_symbol.get(symbol)
        if not need_dates:
            diagnostics.append({"symbol": symbol, "status": "no_needed_dates"})
            continue
        t_sym = perf_counter()
        fund_frame, fund_specs, diag = _build_symbol_fundamental_panel(warehouse, symbol, feature_config)
        status = str(diag.get("status", "ok"))
        if fund_frame is None or fund_frame.empty:
            diagnostics.append({"symbol": symbol, "status": f"fund_{status}", "seconds": perf_counter() - t_sym})
            continue
        fund_frame = fund_frame.copy()
        fund_frame["symbol"] = symbol
        fund_frame["date"] = pd.to_datetime(fund_frame["date"], errors="coerce").dt.normalize()
        fund_frame = fund_frame.loc[fund_frame["date"].isin(need_dates)].copy()
        if fund_frame.empty:
            diagnostics.append({"symbol": symbol, "status": "fund_no_overlap_dates", "seconds": perf_counter() - t_sym})
            continue

        # technicals
        try:
            prices = warehouse.read_prices(symbol, provider=feature_config.provider)
            prices = _slice_prices(prices, feature_config.start_date, feature_config.end_date)
            built = build_price_ta_classic_feature_families(symbol, prices) if not prices.empty else {}
        except Exception as exc:
            built = {}
            diagnostics.append({"symbol": symbol, "status": f"tech_err:{type(exc).__name__}"})

        tech_cols = {}
        for family_name, built_set in (built or {}).items():
            if built_set is None or not getattr(built_set, "feature_cols", None):
                continue
            df = built_set.df
            if df is None or df.empty:
                continue
            work = df.reset_index()
            if "date" not in work.columns:
                for c in work.columns:
                    if c != "symbol" and pd.api.types.is_datetime64_any_dtype(work[c]):
                        work = work.rename(columns={c: "date"})
                        break
            if "date" not in work.columns:
                continue
            work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
            work = work.loc[work["date"].isin(need_dates)]
            for col in built_set.feature_cols:
                if col not in work.columns:
                    continue
                s = pd.Series(pd.to_numeric(work[col], errors="coerce").to_numpy(), index=work["date"].to_numpy())
                tech_cols[col] = s
                tech_specs.append(
                    FeatureSpec(
                        feature=col,
                        family=str(family_name),
                        source="fmp",
                        source_column=col,
                        expected_direction="higher_is_better",
                    )
                )

        if tech_cols:
            tech_df = pd.DataFrame(tech_cols)
            tech_df["date"] = tech_df.index
            tech_df = tech_df.reset_index(drop=True)
            tech_df["date"] = pd.to_datetime(tech_df["date"], errors="coerce").dt.normalize()
            fund_frame = fund_frame.merge(tech_df, on="date", how="left")

        # float32 for numeric cols to save RAM
        for col in fund_frame.columns:
            if col in {"symbol", "date"}:
                continue
            if pd.api.types.is_float_dtype(fund_frame[col]) or pd.api.types.is_integer_dtype(fund_frame[col]):
                fund_frame[col] = pd.to_numeric(fund_frame[col], errors="coerce").astype("float32")

        frames.append(fund_frame)
        all_specs.extend(fund_specs)
        diagnostics.append(
            {
                "symbol": symbol,
                "status": "ok",
                "rows": len(fund_frame),
                "seconds": round(perf_counter() - t_sym, 2),
            }
        )
        if i % 25 == 0 or i == n_sym:
            print(
                f"[sparse-fe] {i}/{n_sym} symbols frames={len(frames)} "
                f"rows={sum(len(x) for x in frames)} elapsed={perf_counter()-t_fe:.0f}s",
                flush=True,
            )
            gc.collect()

    if not frames:
        raise RuntimeError("Sparse FE produced no frames")

    raw_feature_panel = pd.concat(frames, ignore_index=True, sort=False)
    del frames
    gc.collect()
    raw_feature_panel["symbol"] = raw_feature_panel["symbol"].astype(str).str.upper()
    raw_feature_panel["date"] = pd.to_datetime(raw_feature_panel["date"], errors="coerce").dt.normalize()

    # shared families on sparse panel
    all_specs.extend(_add_time_calendar_features(raw_feature_panel))
    all_specs.extend(_add_macro_context_features(warehouse, raw_feature_panel, feature_config))
    all_specs.extend(_add_cross_symbol_context_features(warehouse, raw_feature_panel, feature_config))
    all_specs.extend(tech_specs)

    raw_feature_metadata = (
        pd.DataFrame([s.__dict__ for s in all_specs])
        .drop_duplicates(subset=["feature"], keep="first")
        .sort_values(["source", "family", "feature"])
        .reset_index(drop=True)
    )
    # keep metadata only for columns present
    present = set(raw_feature_panel.columns)
    raw_feature_metadata = raw_feature_metadata.loc[raw_feature_metadata["feature"].isin(present)].reset_index(drop=True)

    selected_features, selected_feature_metadata, feature_quality = cap_features_by_quality(
        raw_feature_panel,
        raw_feature_metadata,
        max_features=None,
    )

    raw_metadata = raw_feature_metadata.copy()
    raw_metadata["strategy_source"] = raw_metadata["source"].astype(str) + "." + raw_metadata["family"].astype(str)
    raw_available = set(raw_metadata["strategy_source"].astype(str))
    required_sources = {str(s).strip() for s in g["FEATURE_FAMILIES"]}
    if g["ALL_FEATURE_FAMILIES"]:
        wanted = set(raw_available)
    else:
        wanted = set(required_sources)

    missing_raw = sorted(required_sources.difference(raw_available))
    meta = selected_feature_metadata.copy()
    meta["strategy_source"] = meta["source"].astype(str) + "." + meta["family"].astype(str)
    selected_feature_metadata = (
        meta.loc[meta["strategy_source"].isin(wanted)].drop(columns=["strategy_source"]).reset_index(drop=True)
    )
    selected_available = set(
        selected_feature_metadata["source"].astype(str) + "." + selected_feature_metadata["family"].astype(str)
    )
    missing_sel = sorted(required_sources.difference(selected_available))
    if g["REQUIRE_ALL_REQUESTED_FEATURE_FAMILIES"] and (missing_raw or missing_sel):
        raise RuntimeError(f"Missing families raw={missing_raw} selected={missing_sel}")

    selected_features = [
        f
        for f in selected_features
        if f in set(selected_feature_metadata["feature"].astype(str)) and f in raw_feature_panel.columns
    ]
    feature_panel = raw_feature_panel[["symbol", "date", *selected_features]].copy()
    del raw_feature_panel
    gc.collect()

    fam = (
        selected_feature_metadata.assign(
            strategy_source=lambda d: d["source"].astype(str) + "." + d["family"].astype(str)
        )[["strategy_source", "family"]]
        .drop_duplicates()
        .sort_values("strategy_source")
    )
    tech = fam.loc[fam["family"].astype(str).str.startswith("technical_")]
    print(
        {
            "feature_panel_rows": len(feature_panel),
            "selected_features": len(selected_features),
            "n_families": int(fam["strategy_source"].nunique()),
            "n_technical_families": int(tech["strategy_source"].nunique()),
            "technical_families": tech["strategy_source"].tolist(),
            "fe_seconds": round(perf_counter() - t_fe, 1),
            "missing_raw": missing_raw,
            "missing_selected": missing_sel,
        },
        flush=True,
    )
    display(fam)
    if tech.empty:
        raise RuntimeError("No technical families in sparse FE")

    g["feature_panel"] = feature_panel
    g["selected_features"] = selected_features
    g["selected_feature_metadata"] = selected_feature_metadata
    g["feature_quality"] = feature_quality
    g["raw_feature_metadata"] = raw_feature_metadata
    g["engineering_warehouse"] = warehouse

    # Stage A/B
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

    print("\n===== CELL 17 stage A/B ($10B) =====", flush=True)
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
        "feature_panel_rows": len(feature_panel),
        "meta_dir": str(meta_dir),
        "summary_exists": summary_path.exists(),
        "selector_exists": selector_path.exists(),
        "fe_mode": "sparse_entry_dates",
    }
    if summary_path.exists():
        result["summary"] = json.loads(summary_path.read_text())
    if selector_path.exists():
        result["selector"] = pd.read_csv(selector_path).to_dict(orient="records")

    out = meta_dir / "run_10b_tech_summary.json"
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print("\n===== 10B RESULT =====", flush=True)
    print(json.dumps({k: v for k, v in result.items() if k != "summary"}, indent=2, default=str), flush=True)
    if "summary" in result:
        for key in ("headline", "fixed_near_atm", "ml_options", "head_to_head"):
            if key in result["summary"]:
                print(key, ":", json.dumps(result["summary"][key], indent=2, default=str), flush=True)
    print(f"Wrote {out}", flush=True)

    if not summary_path.exists():
        print("FAIL: missing meta_stack_summary.json", flush=True)
        return 1
    print("PASS: $10B run with technicals completed", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
