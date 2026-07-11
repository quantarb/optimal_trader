#!/usr/bin/env python3
"""Expand 10B equity-meta universe with mega-caps that have Arctic congress data.

Runs oracle-only vs oracle+congress ablation:
  1) Add AAPL/MSFT/NVDA/GOOGL/AMZN/META/TSLA to the 10B symbol set
  2) Rebuild oracle labels for the added symbols
  3) Build congress labels from Arctic ownership_government_trades
  4) Build fundamental feature families for the added symbols and stream-train
     family RFs on (base 10B panels + mega-cap panels)
  5) Shared-book OOS backtests (2021+) for both label variants
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path[:0] = [str(REPO), str(ROOT / "quant-warehouse"), str(ROOT / "quant-orchestrator")]

from app.quant_warehouse_storage import ensure_quant_warehouse_storage

ensure_quant_warehouse_storage()

from quant_orchestrator.platforms.ml_frameworks.rapids.random_forest import RapidsRandomForestClassifier
from quant_orchestrator.research_tools.ml_trading import (
    build_family_prediction_frame,
    build_strategy_score_frame,
    prepare_family_dataset,
    write_ml_trading_artifact_files,
)
from quant_orchestrator.research_tools.ml_trading_experiment import (
    MLTradingExperimentConfig,
    _build_oracle_trade_label_rows_sparse,
    _load_price_frames,
    _run_shared_book_backtests,
)
from quant_warehouse.config import WarehouseConfig
from quant_warehouse.platforms.data_providers.fmp.target_engineering.event_pairs.store import (
    build_event_pairs_from_historical_data,
)
from quant_warehouse.research_tools import BinaryTargetConfig, FamilyEvaluationConfig
from quant_warehouse.research_tools.feature_family_eval import _build_symbol_fundamental_panel
from quant_warehouse.warehouse.api import Warehouse

MEGA_CAPS = ("AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA")
BASE_RUN = (
    REPO
    / "artifacts/trading_app_v2/equity_meta_model_10b"
    / "mcap_10000000000_train_2020-12-31_seed_20260707"
)
ABLATION_ROOT = REPO / "artifacts/trading_app_v2/equity_meta_model_10b/label_ablation_oracle_vs_congress"
OUT = ABLATION_ROOT / "expanded_universe_run"

# Fundamental families present in the 10B cache (exclude technicals for runtime).
CLEAN_FAMILY_SOURCES = (
    "fmp.fmp_income_mcap",
    "fmp.fmp_balance_mcap",
    "fmp.fmp_cash_mcap",
    "fmp.fmp_daily_mcap_multiple",
    "fmp.fmp_daily_mcap_yield",
    "fmp.fmp_daily_ev_multiple",
    "fmp.fmp_daily_ev_yield",
    "financetoolkit.ft_growth_income",
    "financetoolkit.ft_growth_balance",
    "financetoolkit.ft_growth_cash",
    "financetoolkit.ft_ratios_profitability",
    "financetoolkit.ft_ratios_efficiency",
    "financetoolkit.ft_ratios_valuation",
    "financetoolkit.ft_ratios_solvency",
    "financetoolkit.ft_ratios_liquidity",
)

BASE_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 16,
    "max_features": "sqrt",
    "n_bins": 128,
    "n_streams": 8,
}
TRAIN_END = "2020-12-31"
OOS_START = "2021-01-01"
SCORE_START = "2021-01-01"
RANDOM_SEED = 20260707
MIN_FEATURE_COVERAGE = 0.50
MIN_TRAIN_ROWS = 250


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_base_symbols() -> list[str]:
    manifest = json.loads((BASE_RUN / "oracle_labels" / "manifest.json").read_text())
    return [str(s).strip().upper() for s in manifest["symbols"] if str(s).strip()]


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


def _congress_labels_from_arctic(warehouse: Warehouse, symbols: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for sym in symbols:
        pairs = build_event_pairs_from_historical_data(
            sym,
            fundamentals=warehouse.fundamentals,
            event_families=("congress",),
            provider="fmp",
        )
        if pairs is None or pairs.empty:
            _log(f"[congress] {sym}: 0 pairs")
            continue
        _log(f"[congress] {sym}: {len(pairs)} pairs")
        frames.append(pairs)
    if not frames:
        return pd.DataFrame(columns=["symbol", "date", "collapsed_label", "label_source"])
    congress = pd.concat(frames, ignore_index=True)
    etype = congress["event_type"].astype(str)
    mapped = etype.map({"congress_buy": "oracle_long", "congress_sell": "oracle_short"})
    out = pd.DataFrame(
        {
            "symbol": congress["symbol"].astype(str).str.upper(),
            "date": pd.to_datetime(congress["event_date"], errors="coerce").dt.normalize(),
            "collapsed_label": mapped,
            "label_source": "event_congress_" + etype.str.replace("congress_", "", regex=False),
        }
    )
    out = out.dropna(subset=["symbol", "date", "collapsed_label"])
    # day-collapse: drop mixed buy/sell days
    nlab = out.groupby(["symbol", "date"])["collapsed_label"].nunique()
    bad = nlab.loc[nlab.gt(1)].index
    if len(bad):
        bad_df = pd.DataFrame(list(bad), columns=["symbol", "date"]).assign(_b=True)
        out = out.merge(bad_df, on=["symbol", "date"], how="left")
        out = out.loc[~out["_b"].fillna(False)].drop(columns=["_b"])
    return (
        out.groupby(["symbol", "date", "collapsed_label"], as_index=False)
        .agg(label_source=("label_source", lambda v: "|".join(dict.fromkeys(sorted(map(str, v))))))
        .reset_index(drop=True)
    )


def _build_mega_family_panels(
    warehouse: Warehouse,
    symbols: tuple[str, ...],
    *,
    out_dir: Path,
) -> dict[str, Path]:
    """Build per-family parquet panels for mega-cap symbols only."""
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_config = FamilyEvaluationConfig(
        provider="fmp",
        market_cap_min=10_000_000_000,
        start_date="1900-01-01",
        end_date=None,
        max_features_per_family=50,
    )
    family_parts: dict[str, list[pd.DataFrame]] = {src: [] for src in CLEAN_FAMILY_SOURCES}
    family_meta_parts: dict[str, list[pd.DataFrame]] = {src: [] for src in CLEAN_FAMILY_SOURCES}

    for i, sym in enumerate(symbols, start=1):
        frame, specs, diag = _build_symbol_fundamental_panel(warehouse, sym, feature_config)
        _log(
            f"[mega-fe] {i}/{len(symbols)} {sym} rows={len(frame)} specs={len(specs)} "
            f"status={diag.get('status', diag)}"
        )
        if frame is None or frame.empty or not specs:
            continue
        meta = pd.DataFrame([s.__dict__ for s in specs]).drop_duplicates()
        meta["source"] = meta["source"].astype(str)
        meta["family"] = meta["family"].astype(str)
        for strategy_source in CLEAN_FAMILY_SOURCES:
            source, family = strategy_source.split(".", 1)
            fam_meta = meta.loc[meta["source"].eq(source) & meta["family"].eq(family)]
            cols = [c for c in fam_meta["feature"].tolist() if c in frame.columns]
            if not cols:
                continue
            part = frame[["symbol", "date", *cols]].copy()
            part["symbol"] = part["symbol"].astype(str).str.upper()
            part["date"] = pd.to_datetime(part["date"], errors="coerce").dt.normalize()
            for c in cols:
                part[c] = pd.to_numeric(part[c], errors="coerce").astype("float32")
            family_parts[strategy_source].append(part)
            family_meta_parts[strategy_source].append(fam_meta)

    paths: dict[str, Path] = {}
    for strategy_source, parts in family_parts.items():
        panel_path = out_dir / f"{strategy_source}.parquet"
        meta_path = out_dir / f"{strategy_source}.metadata.parquet"
        if not parts:
            pd.DataFrame(columns=["symbol", "date"]).to_parquet(panel_path, index=False)
            pd.DataFrame(columns=["source", "family", "feature"]).to_parquet(meta_path, index=False)
            paths[strategy_source] = panel_path
            continue
        panel = pd.concat(parts, ignore_index=True, sort=False)
        meta = pd.concat(family_meta_parts[strategy_source], ignore_index=True, sort=False).drop_duplicates()
        # keep only features present
        feat_cols = [c for c in meta["feature"].tolist() if c in panel.columns]
        meta = meta.loc[meta["feature"].isin(feat_cols)].copy()
        panel = panel[["symbol", "date", *feat_cols]]
        panel.to_parquet(panel_path, index=False)
        meta.to_parquet(meta_path, index=False)
        paths[strategy_source] = panel_path
        _log(f"[mega-fe] wrote {strategy_source}: rows={len(panel)} features={len(feat_cols)} syms={panel['symbol'].nunique()}")
    return paths


def _load_combined_family_panel(
    strategy_source: str,
    *,
    base_index: pd.DataFrame,
    mega_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source, family = strategy_source.split(".", 1)
    base_row = base_index.loc[
        base_index["source"].astype(str).eq(source) & base_index["family"].astype(str).eq(family)
    ]
    if base_row.empty:
        raise KeyError(f"missing base family {strategy_source}")
    base_panel_path = Path(base_row.iloc[0]["panel_path"])
    base_meta_path = Path(base_row.iloc[0]["metadata_path"])
    base_panel = pd.read_parquet(base_panel_path)
    base_meta = pd.read_parquet(base_meta_path)
    mega_panel_path = mega_dir / f"{strategy_source}.parquet"
    mega_meta_path = mega_dir / f"{strategy_source}.metadata.parquet"
    if mega_panel_path.exists() and mega_panel_path.stat().st_size > 0:
        mega_panel = pd.read_parquet(mega_panel_path)
        mega_meta = pd.read_parquet(mega_meta_path) if mega_meta_path.exists() else pd.DataFrame()
    else:
        mega_panel = pd.DataFrame()
        mega_meta = pd.DataFrame()

    if mega_panel is not None and not mega_panel.empty:
        # align columns to base feature set (training metadata)
        base_feats = [c for c in base_meta["feature"].tolist() if c in base_panel.columns]
        for c in base_feats:
            if c not in mega_panel.columns:
                mega_panel[c] = np.nan
        mega_use = mega_panel[["symbol", "date", *base_feats]].copy()
        base_use = base_panel[["symbol", "date", *base_feats]].copy()
        panel = pd.concat([base_use, mega_use], ignore_index=True, sort=False)
        # drop duplicate symbol/date preferring mega rows last
        panel = panel.drop_duplicates(subset=["symbol", "date"], keep="last")
    else:
        panel = base_panel.copy()
    panel["symbol"] = panel["symbol"].astype(str).str.upper()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    return panel, base_meta


def _train_and_score_variant(
    *,
    name: str,
    labels: pd.DataFrame,
    base_index: pd.DataFrame,
    mega_dir: Path,
    config: MLTradingExperimentConfig,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = labels.copy()
    labels["symbol"] = labels["symbol"].astype(str).str.upper()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    labels.to_parquet(out_dir / "labels.parquet", index=False)

    train_end = pd.Timestamp(TRAIN_END)
    oos_start = pd.Timestamp(OOS_START)
    score_start = pd.Timestamp(SCORE_START)

    model_rows = []
    score_frames = []
    models = {}
    t0 = perf_counter()
    for job_i, strategy_source in enumerate(CLEAN_FAMILY_SOURCES, start=1):
        source, family = strategy_source.split(".", 1)
        panel, metadata = _load_combined_family_panel(strategy_source, base_index=base_index, mega_dir=mega_dir)
        family_frame, features = prepare_family_dataset(
            panel,
            metadata,
            labels,
            source=source,
            family=family,
            min_feature_coverage=MIN_FEATURE_COVERAGE,
        )
        train = family_frame.loc[pd.to_datetime(family_frame["date"], errors="coerce").le(train_end)].copy()
        oos = family_frame.loc[pd.to_datetime(family_frame["date"], errors="coerce").ge(oos_start)].copy()
        _log(
            f"[{name}] base-train {job_i}/{len(CLEAN_FAMILY_SOURCES)} {strategy_source} "
            f"panel={len(panel):,} train={len(train):,} feats={len(features)}"
        )
        if len(train) < MIN_TRAIN_ROWS or train["collapsed_label"].nunique() < 2:
            model_rows.append(
                {
                    "strategy_source": strategy_source,
                    "source": source,
                    "family": family,
                    "status": "skipped_sparse_train",
                    "features": len(features),
                    "train_rows": len(train),
                    "oos_rows": len(oos),
                }
            )
            del panel, metadata, family_frame, train, oos
            gc.collect()
            continue

        fit_t0 = perf_counter()
        classifier = RapidsRandomForestClassifier.fit(
            train,
            features=features,
            target_col="collapsed_label",
            random_state=RANDOM_SEED,
            params=BASE_RF_PARAMS,
        )
        fit_s = perf_counter() - fit_t0
        payload = {
            "classifier": classifier,
            "autoencoder": None,
            "feature_autoencoder": None,
            "features": features,
            "raw_features": features,
            "representation": "raw",
        }
        models[(source, family)] = payload

        # Score OOS calendar for shared-book backtests (score_start+)
        pred_input = build_family_prediction_frame(panel, features, min_feature_coverage=MIN_FEATURE_COVERAGE)
        pred_input = pred_input.loc[pd.to_datetime(pred_input["date"], errors="coerce").ge(score_start)].copy()
        if not pred_input.empty:
            # chunk score by year to limit GPU/host memory
            year_frames = []
            years = sorted(pd.to_datetime(pred_input["date"]).dt.year.unique())
            for year in years:
                chunk = pred_input.loc[pd.to_datetime(pred_input["date"]).dt.year.eq(int(year))].copy()
                if chunk.empty:
                    continue
                proba = classifier.predict_proba_frame(chunk, features)
                scores = build_strategy_score_frame(
                    source=source,
                    family=family,
                    prediction_frame=chunk,
                    probability_frame=proba,
                    ae_familiarity_frame=None,
                    apply_ae_to_exits=False,
                )
                year_frames.append(scores)
                del chunk, proba, scores
            if year_frames:
                score_frames.append(pd.concat(year_frames, ignore_index=True, sort=False))
            del year_frames

        model_rows.append(
            {
                "strategy_source": strategy_source,
                "source": source,
                "family": family,
                "status": "ok",
                "features": len(features),
                "train_rows": len(train),
                "oos_rows": len(oos),
                "classes": int(family_frame["collapsed_label"].nunique()),
                "classifier_fit_seconds": fit_s,
                "classifier_backend": classifier.gpu_info.get("backend"),
                "gpu_device_name": classifier.gpu_info.get("device_name"),
            }
        )
        del panel, metadata, family_frame, train, oos, pred_input
        gc.collect()

    single = pd.concat(score_frames, ignore_index=True, sort=False) if score_frames else pd.DataFrame()
    if single.empty:
        mean_scores = pd.DataFrame()
        strategy_scores = pd.DataFrame()
    else:
        mean_scores = (
            single.groupby(["symbol", "date"], as_index=False)
            .agg(
                long_score=("long_score", "mean"),
                short_score=("short_score", "mean"),
                long_exit_score=("long_exit_score", "mean"),
                short_exit_score=("short_exit_score", "mean"),
                classifier_long_score=("classifier_long_score", "mean"),
                classifier_short_score=("classifier_short_score", "mean"),
                long_agree_count=("long_agree_count", "sum"),
                short_agree_count=("short_agree_count", "sum"),
                ae_familiarity=("ae_familiarity", "mean"),
                ae_recon_error=("ae_recon_error", "mean"),
                ae_latent_distance=("ae_latent_distance", "mean"),
                model_count=("strategy_source", "nunique"),
            )
        )
        mean_scores["source"] = "ensemble"
        mean_scores["family"] = "mean"
        mean_scores["strategy_source"] = "ensemble_mean"
        mean_scores["net_score"] = mean_scores["long_score"] - mean_scores["short_score"]
        score_cols = [
            "strategy_source",
            "source",
            "family",
            "symbol",
            "date",
            "long_score",
            "short_score",
            "long_exit_score",
            "short_exit_score",
            "classifier_long_score",
            "classifier_short_score",
            "long_agree_count",
            "short_agree_count",
            "ae_familiarity",
            "ae_recon_error",
            "ae_latent_distance",
            "net_score",
            "model_count",
        ]
        strategy_scores = pd.concat([mean_scores[score_cols], single[score_cols]], ignore_index=True, sort=False)

    model_results = pd.DataFrame(model_rows)
    _log(
        f"[{name}] train+score done in {perf_counter()-t0:.1f}s "
        f"ok_models={(model_results['status'].eq('ok').sum() if not model_results.empty else 0)} "
        f"score_rows={len(strategy_scores):,}"
    )

    # Prices for all labeled/score symbols
    symbols = tuple(sorted(set(labels["symbol"].astype(str)) | set(strategy_scores["symbol"].astype(str) if not strategy_scores.empty else [])))
    warehouse = Warehouse()
    price_frames = _load_price_frames(warehouse, symbols, provider="fmp", start="2019-01-01", end=None)
    backtest_summary, trade_log, yearly_backtest_summary, _audit = _run_shared_book_backtests(
        config,
        strategy_scores,
        price_frames,
        oos_start=oos_start,
    )

    write_ml_trading_artifact_files(
        model_results=model_results,
        strategy_scores=strategy_scores,
        backtest_summary=backtest_summary,
        trade_log=trade_log if trade_log is not None else pd.DataFrame(),
        model_vs_trading=pd.DataFrame(),
        metric_correlations=pd.DataFrame(),
        yearly_backtest_summary=yearly_backtest_summary if yearly_backtest_summary is not None else pd.DataFrame(),
        symbol_strategy_summary=pd.DataFrame(),
        symbol_robustness_summary=pd.DataFrame(),
        backtesting_py_symbol_validation=pd.DataFrame(),
        phase_timings=pd.DataFrame([{"phase": "train_score_backtest", "seconds": perf_counter() - t0}]),
        analysis_markdown=f"# {name}\n\nExpanded-universe oracle vs congress ablation.\n",
        directory=out_dir,
    )
    labels.to_parquet(out_dir / "labels.parquet", index=False)

    best = None
    if backtest_summary is not None and not backtest_summary.empty:
        ens = backtest_summary.loc[backtest_summary["strategy_source"].astype(str).eq("ensemble_mean")]
        if not ens.empty and "sharpe" in ens.columns:
            best = ens.sort_values("sharpe", ascending=False).iloc[0].to_dict()
        else:
            best = backtest_summary.sort_values("sharpe", ascending=False).iloc[0].to_dict() if "sharpe" in backtest_summary.columns else None

    summary = {
        "variant": name,
        "n_labels": int(len(labels)),
        "n_label_symbols": int(labels["symbol"].nunique()),
        "label_sources": labels["label_source"].value_counts().head(20).to_dict(),
        "trained_ok": int(model_results["status"].eq("ok").sum()) if not model_results.empty else 0,
        "score_rows": int(len(strategy_scores)),
        "score_symbols": int(strategy_scores["symbol"].nunique()) if not strategy_scores.empty else 0,
        "best_ensemble_row": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in (best or {}).items() if k in (
            "strategy_source", "variant", "top_k", "sharpe", "total_return", "max_drawdown", "n_trades", "score_symbols"
        )},
    }
    (out_dir / "variant_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _log(f"[{name}] summary: {json.dumps(summary, default=str)}")
    return summary


def main() -> None:
    started = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)

    base_symbols = _load_base_symbols()
    expanded = tuple(sorted(dict.fromkeys([*base_symbols, *MEGA_CAPS])))
    added = [s for s in MEGA_CAPS if s not in set(base_symbols)]
    _log(f"base_symbols={len(base_symbols)} added={added} expanded={len(expanded)}")

    warehouse = Warehouse()
    target_config = BinaryTargetConfig(
        provider="fmp",
        start_date="1900-01-01",
        end_date=None,
        event_families=("congress",),
        oracle_trade_k_by_frequency={"YE": tuple(range(1, 13))},
    )

    # --- Labels ---
    oracle_only_path = ABLATION_ROOT / "labels_oracle_only.parquet"
    if oracle_only_path.exists():
        base_oracle = pd.read_parquet(oracle_only_path)
        _log(f"loaded base oracle-only labels: {len(base_oracle):,}")
    else:
        base_oracle = pd.read_parquet(BASE_RUN / "oracle_labels" / "oracle_label_rows.parquet")
        base_oracle = base_oracle.loc[base_oracle["label_source"].astype(str).eq("oracle_trade")].copy()
        base_oracle.to_parquet(oracle_only_path, index=False)

    _log(f"building oracle labels for added mega-caps: {added}")
    mega_oracle, mega_diag, mega_secs, mega_windows = _build_oracle_trade_label_rows_sparse(
        tuple(added),
        target_config,
        warehouse=warehouse,
    )
    _log(
        f"mega oracle rows={len(mega_oracle)} unique_windows={len(mega_windows)} seconds={mega_secs:.1f}"
    )
    mega_oracle.to_parquet(OUT / "mega_oracle_labels.parquet", index=False)
    mega_diag.to_parquet(OUT / "mega_oracle_diagnostics.parquet", index=False)
    mega_windows.to_parquet(OUT / "mega_oracle_trade_windows_unique.parquet", index=False)

    labels_oracle_only = _combine_labels(base_oracle, mega_oracle)
    labels_oracle_only.to_parquet(OUT / "labels_oracle_only_expanded.parquet", index=False)

    congress_path = ABLATION_ROOT / "labels_congress_only.parquet"
    if congress_path.exists():
        congress_labels = pd.read_parquet(congress_path)
        # ensure only mega symbols
        congress_labels = congress_labels.loc[congress_labels["symbol"].astype(str).str.upper().isin(MEGA_CAPS)].copy()
        _log(f"loaded congress labels: {len(congress_labels)}")
    else:
        congress_labels = _congress_labels_from_arctic(warehouse, MEGA_CAPS)
        congress_labels.to_parquet(ABLATION_ROOT / "labels_congress_only.parquet", index=False)

    # refresh event pairs artifact
    pairs_frames = []
    for sym in MEGA_CAPS:
        p = build_event_pairs_from_historical_data(
            sym, fundamentals=warehouse.fundamentals, event_families=("congress",), provider="fmp"
        )
        if p is not None and not p.empty:
            pairs_frames.append(p)
    if pairs_frames:
        pd.concat(pairs_frames, ignore_index=True).to_parquet(OUT / "congress_event_pairs.parquet", index=False)

    labels_plus = _combine_labels(labels_oracle_only, congress_labels)
    labels_plus.to_parquet(OUT / "labels_oracle_plus_congress_expanded.parquet", index=False)
    _log(
        f"labels oracle_only={len(labels_oracle_only):,} plus_congress={len(labels_plus):,} "
        f"delta={len(labels_plus)-len(labels_oracle_only):,}"
    )

    # --- Mega feature panels ---
    mega_dir = OUT / "mega_feature_panels"
    if not (mega_dir / "manifest.json").exists():
        _log("building mega-cap feature family panels")
        paths = _build_mega_family_panels(warehouse, tuple(added), out_dir=mega_dir)
        (mega_dir / "manifest.json").write_text(
            json.dumps({"symbols": list(added), "families": list(paths), "n_families": len(paths)}, indent=2)
        )
    else:
        _log(f"reusing mega feature panels at {mega_dir}")

    base_index = pd.read_csv(BASE_RUN / "feature_family_panels" / "index.csv")

    config = MLTradingExperimentConfig(
        experiment_name="equity_meta_10b_plus_mega_congress_ablation",
        min_market_cap=10_000_000_000,
        symbols=expanded,
        start_date="1900-01-01",
        train_end=TRAIN_END,
        oos_start=OOS_START,
        score_start=SCORE_START,
        top_k_values=(5, 10, 20, 40),
        strategy_sources=CLEAN_FAMILY_SOURCES,
        target_label_mode="oracle_only",
        oracle_frequencies=("YE",),
        oracle_k_min=1,
        oracle_k_max=12,
        rf_params=BASE_RF_PARAMS,
        log_mlflow=False,
        run_zipline_backtests=False,  # vectorized shared-book path only
        run_model_diagnostics=False,
        random_seed=RANDOM_SEED,
        quant_warehouse_root=str(ROOT / "quant-warehouse"),
    )

    summaries = []
    for name, labels in (
        ("oracle_only", labels_oracle_only),
        ("oracle_plus_congress", labels_plus),
    ):
        summaries.append(
            _train_and_score_variant(
                name=name,
                labels=labels,
                base_index=base_index,
                mega_dir=mega_dir,
                config=config,
                out_dir=OUT / name,
            )
        )

    comparison = {
        "expanded_symbols": len(expanded),
        "added_symbols": list(added),
        "base_symbols": len(base_symbols),
        "families": list(CLEAN_FAMILY_SOURCES),
        "train_end": TRAIN_END,
        "oos_start": OOS_START,
        "variants": summaries,
        "elapsed_seconds": perf_counter() - started,
    }
    # side-by-side best ensemble sharpe
    rows = []
    for s in summaries:
        row = {"variant": s["variant"], "n_labels": s["n_labels"], "trained_ok": s["trained_ok"]}
        row.update({f"best_{k}": v for k, v in (s.get("best_ensemble_row") or {}).items()})
        rows.append(row)
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(OUT / "comparison_summary.csv", index=False)
    (OUT / "comparison_summary.json").write_text(json.dumps(comparison, indent=2, default=str))
    _log(f"DONE elapsed={perf_counter()-started:.1f}s")
    _log(cmp_df.to_string(index=False))
    _log(f"artifacts: {OUT}")


if __name__ == "__main__":
    main()
