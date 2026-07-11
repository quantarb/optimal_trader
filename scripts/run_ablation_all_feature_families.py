#!/usr/bin/env python3
"""Train oracle-only vs oracle+congress_buy on EVERY feature family in the 10B cache.

Reports per-family and ensemble shared-book OOS metrics.
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
    _load_price_frames,
    _run_shared_book_backtests,
)
from quant_warehouse.warehouse.api import Warehouse

BASE_RUN = (
    REPO
    / "artifacts/trading_app_v2/equity_meta_model_10b"
    / "mcap_10000000000_train_2020-12-31_seed_20260707"
)
OUT = (
    REPO
    / "artifacts/trading_app_v2/equity_meta_model_10b"
    / "label_ablation_oracle_vs_congress"
    / "expanded_universe_run"
)
ALL_FAM_OUT = OUT / "all_feature_families"
MEGA_DIR = OUT / "mega_feature_panels"

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


def _load_family_list() -> list[str]:
    idx = pd.read_csv(BASE_RUN / "feature_family_panels" / "index.csv")
    idx = idx.loc[idx["features"].fillna(0).astype(int).gt(0)].copy()
    return idx["strategy_source"].astype(str).tolist()


def _load_panel(strategy_source: str, base_index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source, family = strategy_source.split(".", 1)
    row = base_index.loc[
        base_index["source"].astype(str).eq(source) & base_index["family"].astype(str).eq(family)
    ]
    if row.empty:
        raise KeyError(strategy_source)
    row = row.iloc[0]
    base_panel = pd.read_parquet(row["panel_path"])
    base_meta = pd.read_parquet(row["metadata_path"])
    base_feats = [c for c in base_meta["feature"].tolist() if c in base_panel.columns]
    base_use = base_panel[["symbol", "date", *base_feats]].copy()

    mega_path = MEGA_DIR / f"{strategy_source}.parquet"
    if mega_path.exists():
        mega = pd.read_parquet(mega_path)
        if mega is not None and not mega.empty:
            for c in base_feats:
                if c not in mega.columns:
                    mega[c] = np.nan
            mega_use = mega[["symbol", "date", *base_feats]].copy()
            panel = pd.concat([base_use, mega_use], ignore_index=True, sort=False)
            panel = panel.drop_duplicates(subset=["symbol", "date"], keep="last")
        else:
            panel = base_use
    else:
        panel = base_use

    panel["symbol"] = panel["symbol"].astype(str).str.upper()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    return panel, base_meta


def _train_variant(
    *,
    name: str,
    labels: pd.DataFrame,
    families: list[str],
    base_index: pd.DataFrame,
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

    model_rows: list[dict] = []
    score_frames: list[pd.DataFrame] = []
    t0 = perf_counter()

    for job_i, strategy_source in enumerate(families, start=1):
        source, family = strategy_source.split(".", 1)
        try:
            panel, metadata = _load_panel(strategy_source, base_index)
        except Exception as exc:  # noqa: BLE001
            _log(f"[{name}] {job_i}/{len(families)} {strategy_source} LOAD_FAIL {exc}")
            model_rows.append(
                {
                    "strategy_source": strategy_source,
                    "source": source,
                    "family": family,
                    "status": "load_failed",
                    "error": str(exc),
                }
            )
            continue

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
            f"[{name}] {job_i}/{len(families)} {strategy_source} "
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

        pred_input = build_family_prediction_frame(panel, features, min_feature_coverage=MIN_FEATURE_COVERAGE)
        pred_input = pred_input.loc[pd.to_datetime(pred_input["date"], errors="coerce").ge(score_start)].copy()
        year_frames: list[pd.DataFrame] = []
        if not pred_input.empty:
            for year in sorted(pd.to_datetime(pred_input["date"]).dt.year.unique()):
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
            }
        )
        del panel, metadata, family_frame, train, oos, pred_input, year_frames, classifier
        gc.collect()

    single = pd.concat(score_frames, ignore_index=True, sort=False) if score_frames else pd.DataFrame()
    if single.empty:
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
        f"[{name}] train+score {perf_counter()-t0:.1f}s "
        f"ok={(model_results['status'].eq('ok').sum() if not model_results.empty else 0)} "
        f"scores={len(strategy_scores):,}"
    )

    config = MLTradingExperimentConfig(
        experiment_name=f"all_families_{name}",
        min_market_cap=10_000_000_000,
        start_date="1900-01-01",
        train_end=TRAIN_END,
        oos_start=OOS_START,
        score_start=SCORE_START,
        top_k_values=(5, 10, 20, 40),
        strategy_sources=tuple(families),
        target_label_mode="oracle_only",
        rf_params=BASE_RF_PARAMS,
        log_mlflow=False,
        run_zipline_backtests=False,
        run_model_diagnostics=False,
        random_seed=RANDOM_SEED,
        quant_warehouse_root=str(ROOT / "quant-warehouse"),
    )
    symbols = tuple(
        sorted(
            set(labels["symbol"].astype(str))
            | (set(strategy_scores["symbol"].astype(str)) if not strategy_scores.empty else set())
        )
    )
    warehouse = Warehouse()
    price_frames = _load_price_frames(warehouse, symbols, provider="fmp", start="2019-01-01", end=None)
    backtest_summary, trade_log, yearly, audit = _run_shared_book_backtests(
        config,
        strategy_scores,
        price_frames,
        oos_start=oos_start,
    )
    write_ml_trading_artifact_files(
        model_results=model_results,
        strategy_scores=strategy_scores,
        backtest_summary=backtest_summary if backtest_summary is not None else pd.DataFrame(),
        trade_log=trade_log if trade_log is not None else pd.DataFrame(),
        model_vs_trading=pd.DataFrame(),
        metric_correlations=pd.DataFrame(),
        yearly_backtest_summary=yearly if yearly is not None else pd.DataFrame(),
        symbol_strategy_summary=pd.DataFrame(),
        symbol_robustness_summary=pd.DataFrame(),
        backtesting_py_symbol_validation=pd.DataFrame(),
        phase_timings=pd.DataFrame([{"phase": name, "seconds": perf_counter() - t0}]),
        analysis_markdown=f"# {name} — all feature families\n",
        directory=out_dir,
    )

    # Compact per-family long_only metrics
    family_cmp = pd.DataFrame()
    if yearly is not None and not yearly.empty:
        focus = yearly.loc[
            yearly["variant"].astype(str).eq("long_only")
            & yearly["strategy_source"].astype(str).ne("")
        ].copy()
        rows = []
        for (src, k), g in focus.groupby(["strategy_source", "top_k"]):
            g = g.sort_values("year")
            rows.append(
                {
                    "strategy_source": src,
                    "top_k": int(k),
                    "mean_sharpe": float(g["sharpe"].mean()),
                    "compound_total_return": float(np.prod(1.0 + g["total_return"].astype(float).values) - 1.0),
                    "mean_max_dd": float(g["max_drawdown"].mean()),
                    "n_years": int(g["year"].nunique()),
                }
            )
        family_cmp = pd.DataFrame(rows).sort_values(["top_k", "mean_sharpe"], ascending=[True, False])
        family_cmp.to_csv(out_dir / "per_family_long_only_metrics.csv", index=False)

    summary = {
        "variant": name,
        "n_labels": int(len(labels)),
        "n_label_symbols": int(labels["symbol"].nunique()),
        "families_requested": len(families),
        "trained_ok": int(model_results["status"].eq("ok").sum()) if not model_results.empty else 0,
        "score_rows": int(len(strategy_scores)),
        "elapsed_seconds": perf_counter() - t0,
        "label_sources": labels["label_source"].value_counts().head(20).to_dict(),
        "ensemble_long_only": (
            family_cmp.loc[family_cmp["strategy_source"].eq("ensemble_mean")]
            .to_dict("records")
            if not family_cmp.empty
            else []
        ),
        "top_families_k20": (
            family_cmp.loc[family_cmp["top_k"].eq(20)]
            .head(15)
            .to_dict("records")
            if not family_cmp.empty
            else []
        ),
    }
    (out_dir / "variant_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _log(f"[{name}] summary trained_ok={summary['trained_ok']} ensemble={summary['ensemble_long_only']}")
    return summary


def main() -> None:
    started = perf_counter()
    ALL_FAM_OUT.mkdir(parents=True, exist_ok=True)
    families = _load_family_list()
    base_index = pd.read_csv(BASE_RUN / "feature_family_panels" / "index.csv")
    _log(f"families={len(families)}: {families}")

    oracle_path = OUT / "labels_oracle_only_expanded.parquet"
    buy_path = OUT / "labels_oracle_plus_congress_buy_only.parquet"
    full_path = OUT / "labels_oracle_plus_congress_full_universe.parquet"

    if not buy_path.exists() and full_path.exists():
        # derive buy-only from full if needed
        full = pd.read_parquet(full_path)
        oracle = pd.read_parquet(oracle_path)
        buy_events = full.loc[full["label_source"].astype(str).str.contains("congress_buy", na=False)].copy()
        # better: use congress buy labels file if present
        cong_buy = OUT / "congress_ingest" / "labels_congress_buy_only.parquet"
        if cong_buy.exists():
            buy_events = pd.read_parquet(cong_buy)
        # combine was already done in prior run ideally
        labels_buy = full.loc[
            ~full["label_source"].astype(str).str.contains("congress_sell", na=False)
        ].copy()
        # simpler reconstruct
        from quant_orchestrator.research_tools.ml_trading_experiment import _combine_label_rows  # may not exist

        buy_events = pd.read_parquet(OUT / "congress_ingest" / "labels_congress_full_universe.parquet")
        buy_events = buy_events.loc[buy_events["label_source"].astype(str).eq("event_congress_buy")].copy()
        labels_buy = pd.concat(
            [oracle[["symbol", "date", "collapsed_label", "label_source"]], buy_events],
            ignore_index=True,
        )
        labels_buy["symbol"] = labels_buy["symbol"].astype(str).str.upper()
        labels_buy["date"] = pd.to_datetime(labels_buy["date"], errors="coerce").dt.normalize()
        side_n = labels_buy.groupby(["symbol", "date"])["collapsed_label"].nunique()
        conflicts = side_n.loc[side_n.gt(1)].index
        if len(conflicts):
            cdf = pd.DataFrame(list(conflicts), columns=["symbol", "date"]).assign(_c=True)
            labels_buy = labels_buy.merge(cdf, on=["symbol", "date"], how="left")
            labels_buy = labels_buy.loc[~labels_buy["_c"].fillna(False)].drop(columns=["_c"])
        labels_buy = (
            labels_buy.groupby(["symbol", "date", "collapsed_label"], as_index=False)
            .agg(label_source=("label_source", lambda v: "|".join(dict.fromkeys(sorted(map(str, v))))))
            .reset_index(drop=True)
        )
        labels_buy.to_parquet(buy_path, index=False)
        _log(f"wrote buy-only labels {len(labels_buy)}")

    variants = [
        ("oracle_only", oracle_path),
        ("oracle_plus_congress_buy_only", buy_path),
    ]
    # optional full buy+sell if present
    if full_path.exists():
        variants.append(("oracle_plus_congress_full", full_path))

    summaries = []
    for name, path in variants:
        if not path.exists():
            _log(f"skip {name}: missing {path}")
            continue
        labels = pd.read_parquet(path)
        _log(f"=== {name} labels={len(labels):,} sources={labels['label_source'].value_counts().head(5).to_dict()}")
        summaries.append(
            _train_variant(
                name=name,
                labels=labels,
                families=families,
                base_index=base_index,
                out_dir=ALL_FAM_OUT / name,
            )
        )

    # Cross-variant comparison: ensemble + each family at k=20
    comp_rows = []
    fam_rows = []
    for s in summaries:
        name = s["variant"]
        ypath = ALL_FAM_OUT / name / "per_family_long_only_metrics.csv"
        if not ypath.exists():
            continue
        m = pd.read_csv(ypath)
        m["variant_run"] = name
        fam_rows.append(m)
        for _, r in m.loc[m["strategy_source"].eq("ensemble_mean")].iterrows():
            comp_rows.append(
                {
                    "variant_run": name,
                    "top_k": int(r["top_k"]),
                    "mean_sharpe": float(r["mean_sharpe"]),
                    "compound_total_return": float(r["compound_total_return"]),
                    "mean_max_dd": float(r["mean_max_dd"]),
                }
            )
    if fam_rows:
        fam_all = pd.concat(fam_rows, ignore_index=True)
        fam_all.to_csv(ALL_FAM_OUT / "per_family_all_variants.csv", index=False)
        # pivot k=20
        k20 = fam_all.loc[fam_all["top_k"].eq(20)].pivot_table(
            index="strategy_source",
            columns="variant_run",
            values=["mean_sharpe", "compound_total_return"],
        )
        k20.columns = [f"{a}__{b}" for a, b in k20.columns]
        k20 = k20.reset_index()
        if "mean_sharpe__oracle_only" in k20.columns and "mean_sharpe__oracle_plus_congress_buy_only" in k20.columns:
            k20["sharpe_delta_buy_vs_oracle"] = (
                k20["mean_sharpe__oracle_plus_congress_buy_only"] - k20["mean_sharpe__oracle_only"]
            )
        if "mean_sharpe__oracle_only" in k20.columns and "mean_sharpe__oracle_plus_congress_full" in k20.columns:
            k20["sharpe_delta_full_vs_oracle"] = (
                k20["mean_sharpe__oracle_plus_congress_full"] - k20["mean_sharpe__oracle_only"]
            )
        k20 = k20.sort_values(
            [c for c in k20.columns if c.startswith("sharpe_delta")][:1] or ["strategy_source"],
            ascending=False,
        )
        k20.to_csv(ALL_FAM_OUT / "per_family_k20_comparison.csv", index=False)
        _log("\n=== per-family k=20 comparison (head) ===")
        _log(k20.head(35).to_string(index=False))

    if comp_rows:
        comp = pd.DataFrame(comp_rows).sort_values(["top_k", "variant_run"])
        comp.to_csv(ALL_FAM_OUT / "ensemble_comparison.csv", index=False)
        _log("\n=== ensemble long_only ===")
        _log(comp.to_string(index=False))

    final = {
        "families": families,
        "n_families": len(families),
        "variants": summaries,
        "elapsed_seconds": perf_counter() - started,
    }
    (ALL_FAM_OUT / "run_summary.json").write_text(json.dumps(final, indent=2, default=str))
    _log(f"DONE elapsed={perf_counter()-started:.1f}s artifacts={ALL_FAM_OUT}")


if __name__ == "__main__":
    main()
