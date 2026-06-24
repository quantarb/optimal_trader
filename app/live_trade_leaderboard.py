from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from urllib.parse import quote_plus

import numpy as np
import pandas as pd

from app.optimal_trade_lookup import bootstrap_django, save_trade_vector_artifacts

LEADERBOARD_FRAME_FILENAME = "leaderboard_latest.pkl"
LEADERBOARD_JSON_FILENAME = "leaderboard_latest.json"
LEADERBOARD_META_FILENAME = "leaderboard_latest_meta.json"
LATEST_SCORED_FRAME_FILENAME = "latest_scored_latest.pkl"
MODEL_ARTIFACT_FILENAMES = (
    "clf_raw.pkl",
    "reg_raw.pkl",
    "reg_trade_return_raw.pkl",
    "reg_duration_raw.pkl",
    "ae_raw.pkl",
)


@dataclass(frozen=True)
class LiveTradeLeaderboardArtifacts:
    config: dict[str, Any]
    artifact_dir: Path
    universe: tuple[str, ...]
    latest_date: pd.Timestamp
    leaderboard: pd.DataFrame
    latest_scored: pd.DataFrame
    reference_trade_count: int
    vector_metadata: dict[str, Any]


def default_live_trade_config() -> dict[str, Any]:
    return {
        "dates": {
            "data_start": "1990-01-01",
            "data_end": pd.Timestamp.today().strftime("%Y-%m-%d"),
        },
        "universe": {
            "source": "auto",
            "symbols": [],
            "country": "US",
            "exchanges": ["NASDAQ", "NYSE", "AMEX"],
            "min_market_cap": 10_000_000_000.0,
            "exclude_pooled_vehicles": True,
            "size": None,
        },
        "labels": {
            "k_params": {"YE": [1, 2, 4, 8]},
            "use_sample_weight": True,
            "alpha": 4.0,
            "r_clip": 0.10,
            "horizon_balance": True,
        },
        "costs": {
            "fee_bps": 5.0,
            "slippage_bps": 5.0,
        },
        "runtime": {
            "rf_split_ratio": 1.0,
            "artifact_dir": os.path.abspath("artifacts/raw_stack"),
        },
        "fmp_refresh": {
            "enabled": True,
            "refresh_symbol_sections_before_build": True,
            "refresh_macro_before_build": True,
            "repair_symbol_metadata_before_build": False,
            "mode": "scoring_ready",
            "skip_cached_inactive_symbols": True,
            "skip_recent_price_attempts": True,
            "existing_historical_sections_only": True,
            "max_symbols": None,
            "verbose": False,
        },
        "probability_columns": {
            "buy_col": "clf__prob_1",
            "short_col": None,
            "infer_short_from_buy": True,
        },
        "strategy": {
            "score_col": "buy_score_mean_raw_pct6",
            "component_threshold": 0.50,
            "price_col": "close",
            "instrument": "equity",
        },
    }


def run_live_trade_leaderboard_build(
    *,
    config: Mapping[str, Any] | None = None,
    progress_logger: Any | None = None,
) -> LiveTradeLeaderboardArtifacts:
    log = _coerce_progress_logger(progress_logger)
    log("Bootstrapping app modules and importing live-trade pipeline modules")
    bootstrap_django()
    from backtest.latest import make_autoencoder_familiarity_predictor, run_latest_prediction_custom
    from backtest.raw_stack import ProbabilityColumnConfig, enrich_scored_panel
    from data.preparation import MLDatasetConfig, prepare_ml_dataset
    from fmp.workflows import run_scoring_data_refresh_from_fmp
    from features.macro import MacroFeatureConfig
    from ml.raw_stack import save_raw_stack_artifacts, train_ae, train_rf_models
    from pipeline.api import build_fundamental_dataframe, build_label_dataframe, build_macro_dataframe
    from pipeline.symbol_filters import select_top_symbols_by_latest_market_cap
    from pipeline.notebook_universe import resolve_notebook_universe
    from trading.live_trade import (
        build_technical_dataframe_from_django,
        expected_latest_price_date_from_market_clock,
        resolve_fmp_api_key,
    )

    cfg = default_live_trade_config()
    if config:
        cfg = _deep_merge(cfg, dict(config))

    requested_start_date = pd.Timestamp(str(cfg["dates"]["data_start"])).normalize()
    requested_end_date = pd.Timestamp(str(cfg["dates"]["data_end"])).normalize()
    completed_market_date = pd.Timestamp(expected_latest_price_date_from_market_clock()).normalize()
    effective_end_date = min(requested_end_date, completed_market_date)
    START_DATE = requested_start_date.strftime("%Y-%m-%d")
    END_DATE = effective_end_date.strftime("%Y-%m-%d")
    if requested_end_date > effective_end_date:
        log(
            "Requested build end date "
            f"{requested_end_date.date().isoformat()} is after the latest completed market date; "
            f"capping refresh/build end date to {effective_end_date.date().isoformat()}"
        )
    log(f"Resolving symbol universe from {START_DATE} to {END_DATE}")

    ctx = SimpleNamespace(api_key=resolve_fmp_api_key(required=False))
    resolved_universe = resolve_notebook_universe(
        cfg["universe"],
        api_key=str(ctx.api_key or ""),
        progress_logger=log,
    )
    universe = resolved_universe.symbols
    universe_source = resolved_universe.source
    if not universe:
        raise RuntimeError("No symbols remained after universe resolution.")
    log(f"Resolved {len(universe):,} symbols for the training universe from {universe_source}")

    fmp_refresh_cfg = dict(cfg.get("fmp_refresh", {}))
    macro_feature_config = MacroFeatureConfig()
    auto_refresh_enabled = bool(fmp_refresh_cfg.get("enabled", False))
    has_fmp_api_key = bool(str(ctx.api_key or "").strip())
    try:
        from data.warehouse_refresh import use_warehouse_refresh

        refresh_backend_ready = bool(use_warehouse_refresh() or has_fmp_api_key)
    except Exception:
        refresh_backend_ready = has_fmp_api_key

    if auto_refresh_enabled and bool(fmp_refresh_cfg.get("refresh_symbol_sections_before_build", False)):
        if refresh_backend_ready:
            run_scoring_data_refresh_from_fmp(
                symbols=universe,
                target_start_date=START_DATE,
                target_end_date=END_DATE,
                refresh_mode=str(fmp_refresh_cfg.get("mode") or "prices_only"),
                refresh_symbol_sections_before_build=True,
                repair_symbol_metadata_before_build=bool(
                    fmp_refresh_cfg.get("repair_symbol_metadata_before_build", False)
                ),
                refresh_macro_before_build=bool(fmp_refresh_cfg.get("refresh_macro_before_build", False)),
                skip_cached_inactive_symbols=bool(fmp_refresh_cfg.get("skip_cached_inactive_symbols", True)),
                skip_recent_price_attempts=bool(fmp_refresh_cfg.get("skip_recent_price_attempts", True)),
                max_symbols=fmp_refresh_cfg.get("max_symbols"),
                existing_historical_sections_only=bool(
                    fmp_refresh_cfg.get("existing_historical_sections_only", True)
                ),
                macro_config=macro_feature_config,
                verbose=bool(fmp_refresh_cfg.get("verbose", False)),
                progress_logger=log,
            )
        else:
            log("Skipping warehouse symbol refresh because quant-warehouse/OpenBB credentials are not configured")

    log("Building technical feature panel from quant-warehouse price history")
    technical_df, _technical_cols = build_technical_dataframe_from_django(
        symbols=universe,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    if technical_df.empty:
        raise RuntimeError("No technical feature rows were built.")
    log(f"Technical panel ready with {len(technical_df):,} rows")
    technical_dates = pd.DatetimeIndex(technical_df.index.get_level_values("date")).normalize()
    technical_symbols = pd.Index(technical_df.index.get_level_values("symbol")).astype(str).str.upper()
    recent_date_counts = []
    for raw_date in sorted(pd.unique(technical_dates))[-3:]:
        mask = technical_dates == raw_date
        recent_date_counts.append(f"{pd.Timestamp(raw_date).date().isoformat()}={technical_symbols[mask].nunique():,}")
    if recent_date_counts:
        log("Recent technical coverage by date | " + " | ".join(recent_date_counts))
    target_technical_rows = int((technical_dates == effective_end_date).sum())
    target_technical_symbols = int(technical_symbols[technical_dates == effective_end_date].nunique())
    log(
        "Target scoring-date technical coverage"
        f" | date {effective_end_date.date().isoformat()}"
        f" | rows {target_technical_rows:,}"
        f" | symbols {target_technical_symbols:,}/{len(universe):,}"
    )
    if target_technical_symbols < len(universe):
        stale_latest = (
            pd.DataFrame({"symbol": technical_symbols, "date": technical_dates})
            .sort_values(["symbol", "date"])
            .groupby("symbol", as_index=False, sort=False)
            .tail(1)
        )
        stale_latest = stale_latest.loc[stale_latest["date"].lt(effective_end_date)].copy()
        present_symbols = set(stale_latest["symbol"].astype(str).str.upper())
        present_symbols.update(str(symbol).strip().upper() for symbol in technical_symbols)
        missing_symbols = [symbol for symbol in universe if str(symbol).strip().upper() not in present_symbols]
        stale_sample = ", ".join(
            f"{row.symbol}:{pd.Timestamp(row.date).date().isoformat()}"
            for row in stale_latest.head(20).itertuples(index=False)
        )
        missing_sample = ", ".join(missing_symbols[:20])
        log(
            f"FMP/local price coverage is incomplete for {effective_end_date.date().isoformat()}; "
            "symbols without target-date data will use their own latest available scored date. "
            f"{target_technical_symbols:,}/{len(universe):,} symbols have target-date technical rows. "
            f"First stale symbols: {stale_sample or '<none>'}. "
            f"First missing symbols: {missing_sample or '<none>'}"
        )

    log("Building aligned fundamental feature panel")
    fund_df, _fund_cols = build_fundamental_dataframe(
        ctx=ctx,
        symbols=universe,
        start_date=START_DATE,
        end_date=END_DATE,
        target_index=technical_df.index,
        daily_prices=technical_df,
        verbose=False,
    )
    log("Building aligned macro feature panel")
    macro_df, _macro_cols = build_macro_dataframe(
        ctx=ctx,
        start_date=START_DATE,
        end_date=END_DATE,
        config=macro_feature_config,
        target_index=technical_df.index,
        verbose=False,
    )
    final_df = pd.concat([technical_df, fund_df, macro_df], axis=1).sort_index()
    top_market_cap_n = cfg.get("universe", {}).get("top_market_cap_n")
    if top_market_cap_n not in (None, ""):
        market_cap_selection = select_top_symbols_by_latest_market_cap(
            final_df,
            end_date=END_DATE,
            top_n=int(top_market_cap_n),
            symbols=universe,
        )
        selected_universe = tuple(
            str(symbol).strip().upper()
            for symbol in market_cap_selection.get("selected_symbols", [])
            if str(symbol).strip()
        )
        if selected_universe:
            universe = selected_universe
            technical_df = technical_df.loc[technical_df.index.get_level_values("symbol").isin(universe)].copy()
            fund_df = fund_df.loc[fund_df.index.get_level_values("symbol").isin(universe)].copy()
            macro_df = macro_df.loc[macro_df.index.get_level_values("symbol").isin(universe)].copy()
            final_df = final_df.loc[final_df.index.get_level_values("symbol").isin(universe)].copy()
            log(
                f"Applied historical market-cap filter | top_n={int(top_market_cap_n):,} | selected {len(universe):,} symbols"
            )
        else:
            log(
                f"Historical market-cap filter returned no symbols; continuing with resolved universe of {len(universe):,}"
            )

    panel_dates = pd.DatetimeIndex(final_df.index.get_level_values("date")).normalize()
    allowed_mask = panel_dates <= completed_market_date
    if not allowed_mask.any():
        raise RuntimeError(
            "No feature rows were available on or before the latest completed market date "
            f"({completed_market_date.date().isoformat()})."
        )
    final_df = final_df.loc[allowed_mask].copy()
    scoring_date = effective_end_date
    log(
        "Combined feature panel ready with "
        f"{len(final_df):,} rows x {final_df.shape[1]:,} columns"
        f" | scoring date {scoring_date.date().isoformat()}"
    )

    execution_params = {
        "price_col": "close",
        "fee_bps": float(cfg["costs"]["fee_bps"]),
        "slippage_bps": float(cfg["costs"]["slippage_bps"]),
    }
    weighting_params = {
        "use_sample_weight": bool(cfg["labels"]["use_sample_weight"]),
        "alpha": float(cfg["labels"]["alpha"]),
        "r_clip": float(cfg["labels"]["r_clip"]),
        "horizon_balance": bool(cfg["labels"]["horizon_balance"]),
    }

    symbols_in_panel = set(technical_df.index.get_level_values("symbol"))
    daily_map_all = {
        symbol: technical_df.xs(symbol, level="symbol").copy()
        for symbol in universe
        if symbol in symbols_in_panel
    }
    log("Building optimal-trade label dataframe")
    label_df_all = build_label_dataframe(
        daily_by_symbol=daily_map_all,
        k_params=dict(cfg["labels"]["k_params"]),
        execution_params=execution_params,
        weighting=weighting_params,
        add_rank_labels=True,
        verbose=False,
    )
    if label_df_all.empty:
        raise RuntimeError("No labels were generated from the available history.")
    log(f"Label dataframe ready with {len(label_df_all):,} rows")

    log("Preparing joined ML training dataset")
    train_df, raw_feature_list, _ = prepare_ml_dataset(
        features_df=final_df,
        labels_df=label_df_all,
        target_cols=["target", "trade_return", "trade_duration_days"],
        weight_col="sample_weight",
        config=MLDatasetConfig(drop_nan_features=False),
        verbose=True,
    )
    if train_df.empty:
        raise RuntimeError("The joined training dataset is empty.")
    log(f"Training dataset ready with {len(train_df):,} rows and {len(raw_feature_list):,} raw features")

    trade_return_values = pd.to_numeric(train_df["trade_return"], errors="coerce")
    train_df["trade_return_pct_target"] = trade_return_values.rank(pct=True, method="average")

    log("Training random-forest classifier and regressors")
    rf_bundle = train_rf_models(
        train_df,
        raw_feature_list,
        split_ratio=float(cfg["runtime"]["rf_split_ratio"]),
        classifier_target_col="target",
        ranking_target_col="rank_y",
        classifier_market_position_col=None,
        train_trade_return_model=True,
        trade_return_target_col="trade_return_pct_target",
        train_duration_model=False,
    )
    log(
        "Random-forest training complete"
        f" | trade return regressor: {'yes' if rf_bundle.trade_return_reg is not None else 'no'}"
        f" | duration regressor: {'yes' if rf_bundle.duration_reg is not None else 'no'}"
    )
    clf_raw = rf_bundle.clf
    reg_raw = rf_bundle.trade_return_reg if rf_bundle.trade_return_reg is not None else rf_bundle.ranking_reg
    log("Training autoencoder embedding model")
    ae_raw, ae_numeric_cols = train_ae(train_df, raw_feature_list)
    log(f"Autoencoder training complete with {len(ae_numeric_cols):,} numeric embedding features")

    log("Saving raw-stack model artifacts")
    artifact_dir = save_raw_stack_artifacts(
        clf_raw=clf_raw,
        reg_trade_return_raw=reg_raw,
        reg_duration_raw=rf_bundle.duration_reg,
        ae_raw=ae_raw,
        raw_feature_list=raw_feature_list,
        ae_raw_numeric_cols=ae_numeric_cols,
        artifact_dir=str(cfg["runtime"]["artifact_dir"]),
    )
    log(f"Artifacts saved to {artifact_dir}")

    prob_cfg = ProbabilityColumnConfig(**cfg["probability_columns"])
    ae_predict = make_autoencoder_familiarity_predictor(ae_numeric_cols)
    scoring_panel, scoring_panel_stats = build_latest_scoring_panel(
        feature_df=final_df,
        scoring_date=scoring_date,
        allow_carry_forward=False,
    )
    inactive_symbol_count = max(int(len(universe)) - int(len(scoring_panel)), 0)
    log(
        "Latest scoring panel ready with "
        f"{len(scoring_panel):,} symbols"
        f" | exact-date rows {int(scoring_panel_stats['exact_date_count']):,}"
        f" | inactive symbols {inactive_symbol_count:,}"
    )
    log("Scoring the latest cross-section with the trained models")
    latest_date, latest_scored = run_latest_prediction_custom(
        train_data=scoring_panel,
        model_specs=[
            {"model": clf_raw, "pred_col": "clf", "include_class_probs": True},
            {"model": reg_raw, "pred_col": "ranking"},
            {"model": ae_raw, "pred_col": "ae_familiarity", "predict_fn": lambda df, m: ae_predict(df, m)},
        ],
        market_position_value=None,
        combine_scores_fn=lambda df: pd.to_numeric(df[prob_cfg.buy_col], errors="coerce").fillna(0.0)
        * pd.to_numeric(df["ranking"], errors="coerce").fillna(0.0)
        * pd.to_numeric(df["ae_familiarity"], errors="coerce").fillna(1.0),
        row_filter_fn=None,
        round_decimals=None,
    )
    latest_scored = enrich_scored_panel(latest_scored, prob_config=prob_cfg)
    log(f"Latest scoring complete for {pd.Timestamp(latest_date).date().isoformat()} with {len(latest_scored):,} symbols")

    log("Building historical entry-trade reference catalog")
    reference_catalog = build_live_trade_reference_catalog(
        label_df=label_df_all,
        feature_df=final_df,
    )
    log(f"Reference catalog ready with {len(reference_catalog):,} entry trades")
    log("Embedding historical entry trades and saving vector artifacts")
    vector_metadata = save_trade_vector_artifacts(
        reference_catalog=reference_catalog,
        ae_model=ae_raw,
        numeric_cols=ae_numeric_cols,
        categorical_cols=[],
        artifact_dir=str(artifact_dir),
    )
    log(f"Trade vector artifacts saved with backend={vector_metadata.get('backend')}")

    log("Building latest leaderboard frame")
    leaderboard = build_leaderboard_frame(
        latest_scored=latest_scored,
        latest_date=latest_date,
        score_col=str(cfg["strategy"]["score_col"]),
        eligibility_threshold=float(cfg["strategy"].get("component_threshold", 0.50)),
    )
    log(f"Leaderboard ready with {len(leaderboard):,} ranked symbols")
    save_leaderboard_artifacts(
        artifact_dir=artifact_dir,
        leaderboard=leaderboard,
        latest_scored=latest_scored,
        latest_date=latest_date,
        config=cfg,
        vector_metadata=vector_metadata,
        universe_size=len(universe),
        inactive_symbol_count=inactive_symbol_count,
    )
    log("Leaderboard artifacts saved")
    return LiveTradeLeaderboardArtifacts(
        config=cfg,
        artifact_dir=Path(artifact_dir),
        universe=universe,
        latest_date=pd.Timestamp(latest_date),
        leaderboard=leaderboard,
        latest_scored=latest_scored.copy(),
        reference_trade_count=int(len(reference_catalog)),
        vector_metadata=vector_metadata,
    )


def build_latest_scoring_panel(
    *,
    feature_df: pd.DataFrame,
    scoring_date: pd.Timestamp,
    allow_carry_forward: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if feature_df.empty:
        raise RuntimeError("Feature panel is empty; cannot build latest scoring panel.")

    scoring_ts = pd.Timestamp(scoring_date).normalize()
    work = feature_df.reset_index().copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work["symbol"] = work["symbol"].astype(str).str.strip().str.upper()
    work = work.loc[work["date"].notna() & work["symbol"].ne("") & work["date"].le(scoring_ts)].copy()
    if work.empty:
        raise RuntimeError(
            f"No feature rows were available on or before the scoring date {scoring_ts.date().isoformat()}."
        )

    work = work.sort_values(["symbol", "date"])
    latest_rows = work.groupby("symbol", as_index=False, sort=False).tail(1).copy()
    latest_rows["feature_as_of_date"] = latest_rows["date"]
    exact_date_mask = latest_rows["feature_as_of_date"].eq(scoring_ts)
    inactive_count = int((~exact_date_mask).sum())
    if not allow_carry_forward:
        latest_rows = latest_rows.loc[exact_date_mask].copy()
        exact_date_mask = pd.Series(True, index=latest_rows.index, dtype=bool)
        if latest_rows.empty:
            raise RuntimeError(
                f"No symbols had feature rows exactly on the scoring date {scoring_ts.date().isoformat()}."
            )
    latest_rows["date"] = scoring_ts

    scoring_panel = latest_rows.set_index(["date", "symbol"]).sort_index()
    stats = {
        "symbol_count": int(len(latest_rows)),
        "exact_date_count": int(exact_date_mask.sum()),
        "carry_forward_count": int((~exact_date_mask).sum()),
        "inactive_count": inactive_count,
    }
    return scoring_panel, stats


def build_live_trade_reference_catalog(
    *,
    label_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    work = label_df.copy()
    if "symbol" not in work.columns or "date" not in work.columns:
        work = work.reset_index()
    work["symbol"] = work["symbol"].astype(str).str.strip().str.upper()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    action_label = work.get("action_label", work.get("label", pd.Series([""] * len(work), index=work.index))).astype(str).str.strip().str.lower()
    work = work.loc[action_label.isin(["buy", "short"])].copy()
    if work.empty:
        raise RuntimeError("No entry trades were available to build the live-trade reference catalog.")

    work["hold_days"] = pd.to_numeric(
        work.get("hold_days", work.get("trade_duration_days")),
        errors="coerce",
    )
    work["entry_date"] = pd.to_datetime(work.get("entry_date", work["date"]), errors="coerce")
    work["exit_date"] = pd.to_datetime(work.get("exit_date"), errors="coerce")
    missing_exit = work["exit_date"].isna()
    work.loc[missing_exit, "exit_date"] = work.loc[missing_exit, "entry_date"] + pd.to_timedelta(
        work.loc[missing_exit, "hold_days"].fillna(0.0),
        unit="D",
    )
    work["ret_dec"] = pd.to_numeric(work.get("ret_dec", work.get("trade_return")), errors="coerce")
    if "freq" not in work.columns and "horizon" in work.columns:
        work["freq"] = work["horizon"].astype(str).str.split("_k").str[0]
    if "k" not in work.columns and "horizon" in work.columns:
        extracted_k = work["horizon"].astype(str).str.extract(r"k(\d+)", expand=False)
        work["k"] = pd.to_numeric(extracted_k, errors="coerce").astype("Int64")
    work["entry_px"] = pd.to_numeric(work.get("entry_px"), errors="coerce")
    work["exit_px"] = pd.to_numeric(work.get("exit_px"), errors="coerce")
    work = work.dropna(subset=["symbol", "date", "entry_date"]).copy()

    merged = work.merge(
        feature_df.reset_index(),
        on=["date", "symbol"],
        how="inner",
        suffixes=("", "_feature"),
    )
    if merged.empty:
        raise RuntimeError("No entry trades could be aligned to the feature panel.")
    merged = merged.sort_values(["entry_date", "symbol", "ret_dec"], ascending=[True, True, False])
    merged = merged.drop_duplicates(subset=["symbol", "entry_date", "side", "freq", "k"], keep="first")
    return merged.reset_index(drop=True)


def _component_cols_for_score(score_col: str) -> list[str]:
    mapping = {
        "buy_score_mean_raw3": ["prob_buy", "pred_rf_reg", "ae_familiarity"],
        "buy_score_mean_raw_pct6": [
            "prob_buy",
            "pred_rf_reg",
            "ae_familiarity",
            "prob_buy_pct",
            "pred_rf_reg_pct",
            "ae_familiarity_pct",
        ],
        "buy_score_pct_mean": ["prob_buy_pct", "pred_rf_reg_pct", "ae_familiarity_pct"],
        "buy_score_pct_product": ["prob_buy_pct", "pred_rf_reg_pct", "ae_familiarity_pct"],
        "buy_score_raw": ["prob_buy", "pred_rf_reg", "ae_familiarity"],
        "buy_score": ["prob_buy", "pred_rf_reg", "ae_familiarity"],
        "short_score_mean_raw3": ["prob_short", "pred_rf_reg", "ae_familiarity"],
        "short_score_mean_raw_pct6": [
            "prob_short",
            "pred_rf_reg",
            "ae_familiarity",
            "prob_short_pct",
            "pred_rf_reg_pct",
            "ae_familiarity_pct",
        ],
        "short_score_pct_mean": ["prob_short_pct", "pred_rf_reg_pct", "ae_familiarity_pct"],
        "short_score_pct_product": ["prob_short_pct", "pred_rf_reg_pct", "ae_familiarity_pct"],
        "short_score_raw": ["prob_short", "pred_rf_reg", "ae_familiarity"],
        "short_score": ["prob_short", "pred_rf_reg", "ae_familiarity"],
    }
    if score_col not in mapping:
        raise KeyError(f"No component mapping configured for score column: {score_col}")
    return list(mapping[score_col])


def _short_score_col_for_score(score_col: str) -> str:
    mapping = {
        "buy_score_mean_raw3": "short_score_mean_raw3",
        "buy_score_mean_raw_pct6": "short_score_mean_raw_pct6",
        "buy_score_pct_mean": "short_score_pct_mean",
        "buy_score_pct_product": "short_score_pct_product",
        "buy_score_raw": "short_score_raw",
        "buy_score": "short_score",
    }
    key = str(score_col)
    if key in mapping:
        return str(mapping[key])
    if key.startswith("buy_"):
        return "short_" + key[len("buy_") :]
    raise KeyError(f"No short-score mapping configured for score column: {score_col}")


def build_leaderboard_frame(
    *,
    latest_scored: pd.DataFrame,
    latest_date: pd.Timestamp,
    score_col: str,
    eligibility_threshold: float = 0.50,
) -> pd.DataFrame:
    work = latest_scored.copy()
    work.index = pd.Index([str(value).strip().upper() for value in work.index], name="symbol")
    work = work.reset_index().rename(columns={"index": "symbol"})
    work["prob_buy"] = pd.to_numeric(work.get("prob_buy", work.get("clf__prob_1")), errors="coerce")
    work["prob_short"] = pd.to_numeric(work.get("prob_short"), errors="coerce")
    missing_prob_short = work["prob_short"].isna()
    work.loc[missing_prob_short, "prob_short"] = 1.0 - work.loc[missing_prob_short, "prob_buy"].fillna(0.0)
    work["ranking_score"] = pd.to_numeric(work.get("pred_rf_reg", work.get("ranking")), errors="coerce")
    work["ae_familiarity"] = pd.to_numeric(work.get("ae_familiarity"), errors="coerce")
    work["price"] = pd.to_numeric(work.get("close"), errors="coerce")
    long_score_col = str(score_col)
    short_score_col = _short_score_col_for_score(long_score_col)
    long_component_cols = list(_component_cols_for_score(long_score_col))
    short_component_cols = list(_component_cols_for_score(short_score_col))
    component_cols = sorted(set(long_component_cols + short_component_cols))
    for component_col in component_cols:
        work[component_col] = pd.to_numeric(work.get(component_col), errors="coerce")
    work["long_score"] = pd.to_numeric(work.get(long_score_col), errors="coerce")
    work["short_score"] = pd.to_numeric(work.get(short_score_col), errors="coerce")
    long_rank_value = work["long_score"].fillna(-np.inf)
    short_rank_value = work["short_score"].fillna(-np.inf)
    work["direction"] = np.where(short_rank_value.gt(long_rank_value), "Short", "Long")
    work["classifier_score"] = np.where(work["direction"].eq("Short"), work["prob_short"], work["prob_buy"])
    work["combined_score"] = np.where(work["direction"].eq("Short"), work["short_score"], work["long_score"])
    work["__score_col_name"] = np.where(work["direction"].eq("Short"), short_score_col, long_score_col)
    threshold = float(eligibility_threshold)
    work["eligible"] = work["combined_score"].notna() & np.isfinite(work["combined_score"])
    selected_component_cols: list[str] = []
    for long_col, short_col in zip(long_component_cols, short_component_cols):
        selected_col = f"__selected_component__{len(selected_component_cols)}"
        selected_component_cols.append(selected_col)
        work[selected_col] = np.where(work["direction"].eq("Short"), work[short_col], work[long_col])
        work["eligible"] &= pd.to_numeric(work[selected_col], errors="coerce").fillna(-np.inf).gt(threshold)
    for idx, component_col in enumerate(selected_component_cols):
        work[f"__component__{idx}"] = pd.to_numeric(work[component_col], errors="coerce")
    work["eligible_score"] = work["combined_score"] * work["eligible"].astype(float)
    feature_as_of_date = pd.Series(
        pd.to_datetime(work.get("feature_as_of_date"), errors="coerce"),
        index=work.index,
    )
    fallback_scoring_date = pd.Timestamp(latest_date).normalize()
    work["scored_date"] = feature_as_of_date.fillna(fallback_scoring_date).dt.date.astype(str)
    work = work.sort_values(["eligible_score", "combined_score"], ascending=[False, False]).reset_index(drop=True)
    work["rank"] = range(1, len(work) + 1)
    work["similar_trades_url"] = work["symbol"].map(lambda symbol: f"/?symbol={quote_plus(str(symbol))}")
    return work.rename(
        columns={
            "rank": "Rank",
            "scored_date": "Scored Date",
            "symbol": "Symbol",
            "direction": "Direction",
            "eligible": "Eligible",
            "classifier_score": "Classifier Score",
            "ranking_score": "Regressor Score",
            "ae_familiarity": "Autoencoder Score",
            "combined_score": "Combined Score",
            "similar_trades_url": "Similar Trades",
            "price": "__price",
            "long_score": "__long_score",
            "short_score": "__short_score",
        }
    )[
        [
            "Rank",
            "Scored Date",
            "Symbol",
            "Direction",
            "Eligible",
            "Classifier Score",
            "Regressor Score",
            "Autoencoder Score",
            "Combined Score",
            "Similar Trades",
            "__price",
            "__long_score",
            "__short_score",
            "__score_col_name",
            *[f"__component__{idx}" for idx in range(len(selected_component_cols))],
        ]
    ]


def build_robinhood_trade_sheet(
    *,
    leaderboard: pd.DataFrame,
    top_k: int,
    account_size: float,
    eligible_only: bool = True,
    include_shorts: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = leaderboard.copy()
    if work.empty:
        return pd.DataFrame(), {
            "selected_count": 0,
            "long_count": 0,
            "short_count": 0,
            "target_weight_per_trade": 0.0,
            "account_size": float(max(account_size, 0.0)),
        }

    if "Rank" in work.columns:
        work["Rank"] = pd.to_numeric(work["Rank"], errors="coerce")
        work = work.sort_values(["Rank"], ascending=[True], kind="stable")
    elif "Combined Score" in work.columns:
        work["Combined Score"] = pd.to_numeric(work["Combined Score"], errors="coerce")
        work = work.sort_values(["Combined Score"], ascending=[False], kind="stable")

    if eligible_only and "Eligible" in work.columns:
        eligible_mask = pd.Series(work["Eligible"], dtype="boolean").fillna(False)
        work = work.loc[eligible_mask].copy()

    if not include_shorts and "Direction" in work.columns:
        work = work.loc[work["Direction"].astype(str).str.strip().str.lower().ne("short")].copy()

    top_k_value = max(int(top_k), 0)
    if top_k_value <= 0:
        return pd.DataFrame(), {
            "selected_count": 0,
            "long_count": 0,
            "short_count": 0,
            "target_weight_per_trade": 0.0,
            "account_size": float(max(account_size, 0.0)),
        }

    work = work.head(top_k_value).copy()
    selected_count = int(len(work))
    safe_account_size = float(max(account_size, 0.0))
    target_weight = (1.0 / float(selected_count)) if selected_count > 0 else 0.0

    work["Direction"] = work.get("Direction", "").astype(str)
    work["Combined Score"] = pd.to_numeric(work.get("Combined Score"), errors="coerce")
    work["Classifier Score"] = pd.to_numeric(work.get("Classifier Score"), errors="coerce")
    work["Regressor Score"] = pd.to_numeric(work.get("Regressor Score"), errors="coerce")
    work["Autoencoder Score"] = pd.to_numeric(work.get("Autoencoder Score"), errors="coerce")
    work["__price"] = pd.to_numeric(work.get("__price"), errors="coerce")
    work["Robinhood Action"] = np.where(
        work["Direction"].str.strip().str.lower().eq("short"),
        "Short",
        "Buy",
    )
    work["Target Weight"] = target_weight
    work["Target Dollars"] = safe_account_size * target_weight
    work["Estimated Shares"] = np.where(
        work["__price"].gt(0.0),
        np.floor(work["Target Dollars"] / work["__price"]),
        np.nan,
    )
    work["Robinhood URL"] = work["Symbol"].map(
        lambda symbol: f"https://robinhood.com/stocks/{quote_plus(str(symbol).strip().upper())}"
    )

    out = work[
        [
            "Rank",
            "Symbol",
            "Scored Date",
            "Direction",
            "Robinhood Action",
            "Classifier Score",
            "Regressor Score",
            "Autoencoder Score",
            "Combined Score",
            "__price",
            "Target Weight",
            "Target Dollars",
            "Estimated Shares",
            "Robinhood URL",
        ]
    ].rename(columns={"__price": "Price"})

    long_count = int(out["Direction"].astype(str).str.strip().str.lower().eq("long").sum())
    short_count = int(out["Direction"].astype(str).str.strip().str.lower().eq("short").sum())
    summary = {
        "selected_count": selected_count,
        "long_count": long_count,
        "short_count": short_count,
        "target_weight_per_trade": target_weight,
        "account_size": safe_account_size,
    }
    return out.reset_index(drop=True), summary


def save_leaderboard_artifacts(
    *,
    artifact_dir: Path,
    leaderboard: pd.DataFrame,
    latest_scored: pd.DataFrame,
    latest_date: pd.Timestamp,
    config: Mapping[str, Any],
    vector_metadata: Mapping[str, Any],
    universe_size: int | None = None,
    inactive_symbol_count: int | None = None,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    leaderboard.to_pickle(artifact_dir / LEADERBOARD_FRAME_FILENAME)
    latest_scored.to_pickle(artifact_dir / LATEST_SCORED_FRAME_FILENAME)
    records_payload = {
        "columns": list(leaderboard.columns),
        "rows": leaderboard.to_dict(orient="records"),
    }
    (artifact_dir / LEADERBOARD_JSON_FILENAME).write_text(json.dumps(records_payload, indent=2, default=str), encoding="utf-8")
    payload = {
        "latest_date": str(pd.Timestamp(latest_date).date()),
        "rows": int(len(leaderboard)),
        "universe_size": None if universe_size is None else int(universe_size),
        "scored_symbol_count": int(len(leaderboard)),
        "inactive_symbol_count": (
            max(int(universe_size) - int(len(leaderboard)), 0)
            if inactive_symbol_count is None and universe_size is not None
            else int(inactive_symbol_count or 0)
        ),
        "leaderboard_scope": "all_scored",
        "config": dict(config),
        "vector_metadata": dict(vector_metadata),
    }
    (artifact_dir / LEADERBOARD_META_FILENAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_saved_leaderboard(
    *,
    artifact_dir: str | os.PathLike[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    if artifact_dir is None:
        artifact_dir = default_live_trade_config()["runtime"]["artifact_dir"]
    artifact_path = Path(str(artifact_dir)).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = Path(__file__).resolve().parents[1] / artifact_path
    frame_path = artifact_path / LEADERBOARD_FRAME_FILENAME
    json_path = artifact_path / LEADERBOARD_JSON_FILENAME
    meta_path = artifact_path / LEADERBOARD_META_FILENAME
    if not frame_path.exists() and not json_path.exists():
        return None
    leaderboard = pd.DataFrame()
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8")) or {}
            leaderboard = pd.DataFrame(payload.get("rows") or [])
            ordered_columns = [str(col) for col in payload.get("columns") or [] if str(col)]
            if ordered_columns:
                existing_columns = [col for col in ordered_columns if col in leaderboard.columns]
                remaining_columns = [col for col in leaderboard.columns if col not in existing_columns]
                leaderboard = leaderboard[existing_columns + remaining_columns]
        except Exception:
            leaderboard = pd.DataFrame()
    if leaderboard.empty and frame_path.exists():
        try:
            leaderboard = pd.read_pickle(frame_path)
        except Exception:
            return None
    metadata = {}
    if meta_path.exists():
        try:
            metadata = dict(json.loads(meta_path.read_text(encoding="utf-8")) or {})
        except Exception:
            metadata = {}
    return leaderboard, metadata


def _resolve_artifact_dir_path(artifact_dir: str | os.PathLike[str] | None = None) -> Path:
    if artifact_dir is None:
        artifact_dir = default_live_trade_config()["runtime"]["artifact_dir"]
    artifact_path = Path(str(artifact_dir)).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = Path(__file__).resolve().parents[1] / artifact_path
    return artifact_path


def latest_scored_staleness_reason(
    *,
    artifact_dir: str | os.PathLike[str] | None = None,
) -> str | None:
    artifact_path = _resolve_artifact_dir_path(artifact_dir)
    latest_scored_path = artifact_path / LATEST_SCORED_FRAME_FILENAME
    if not latest_scored_path.exists():
        return None

    model_paths = [artifact_path / filename for filename in MODEL_ARTIFACT_FILENAMES]
    existing_model_paths = [path for path in model_paths if path.exists()]
    if not existing_model_paths:
        return None

    latest_scored_mtime = latest_scored_path.stat().st_mtime
    newest_model_path = max(existing_model_paths, key=lambda path: path.stat().st_mtime)
    newest_model_mtime = newest_model_path.stat().st_mtime
    if latest_scored_mtime >= newest_model_mtime:
        return None
    return (
        f"{LATEST_SCORED_FRAME_FILENAME} is older than {newest_model_path.name}; "
        "the latest scores need to be regenerated with the current model artifacts."
    )


def load_saved_latest_scored(
    *,
    artifact_dir: str | os.PathLike[str] | None = None,
    require_fresh: bool = True,
) -> pd.DataFrame | None:
    if require_fresh and latest_scored_staleness_reason(artifact_dir=artifact_dir):
        return None
    artifact_path = _resolve_artifact_dir_path(artifact_dir)
    frame_path = artifact_path / LATEST_SCORED_FRAME_FILENAME
    if not frame_path.exists():
        return None
    try:
        latest_scored = pd.read_pickle(frame_path)
    except Exception:
        return None
    return latest_scored if isinstance(latest_scored, pd.DataFrame) else None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


def _coerce_progress_logger(progress_logger: Any | None):
    if callable(progress_logger):
        return progress_logger

    def _default_logger(message: str) -> None:
        print(message, flush=True)

    return _default_logger
