from __future__ import annotations

import json
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from app.quant_warehouse_storage import ensure_quant_warehouse_storage
from platforms.brokers.option_pricing import normalize_option_limit_price


DEFAULT_STRATEGY_SOURCES = (
    "fmp.fmp_income_mcap",
    "fmp.fmp_balance_mcap",
    "fmp.fmp_cash_mcap",
    "fmp.fmp_daily_mcap_multiple",
    "fmp.fmp_daily_mcap_yield",
    "fmp.fmp_daily_ev_multiple",
    "fmp.fmp_daily_ev_yield",
    "fmp.time_calendar",
    "fmp.economic_indicators",
    "fmp.treasury_rates",
    "fmp.sector_performance",
    "fmp.industry_performance",
    "fmp.sector_pe",
    "fmp.industry_pe",
    "financetoolkit.ft_growth_income",
    "financetoolkit.ft_growth_balance",
    "financetoolkit.ft_growth_cash",
    "financetoolkit.ft_ratios_profitability",
    "financetoolkit.ft_ratios_efficiency",
    "financetoolkit.ft_ratios_valuation",
    "financetoolkit.ft_ratios_solvency",
    "financetoolkit.ft_ratios_liquidity",
)

OPTION_MODEL_FEATURES = (
    "dte",
    "dte_gap",
    "moneyness",
    "abs_moneyness",
    "spread_pct",
    "volume",
    "open_interest",
    "liquidity_score",
    "delta",
    "abs_delta",
    "gamma",
    "abs_gamma",
    "theta",
    "abs_theta",
    "vega",
    "abs_vega",
    "rho",
    "abs_rho",
    "theta_to_mid",
    "vega_to_mid",
    "iv",
    "iv_expiration_z",
    "iv_times_sqrt_dte",
)


@dataclass(frozen=True)
class TradingAppV2Paths:
    repo_root: Path
    artifact_root: Path
    equity_artifact_dir: Path
    option_artifact_dir: Path
    live_artifact_dir: Path


@dataclass(frozen=True)
class SubmissionSafetyPolicy:
    """Hard limits applied immediately before an order reaches a broker."""

    max_plan_age_hours: float = 24.0
    max_orders: int = 100
    max_quantity_per_order: int = 10_000
    max_option_contracts_per_order: int = 100


def find_repo_root(start: Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "app").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate optimal_trader repo root from {current}")


def default_paths(repo_root: Path | None = None) -> TradingAppV2Paths:
    root = find_repo_root(repo_root)
    artifact_root = root / "artifacts" / "trading_app_v2"
    return TradingAppV2Paths(
        repo_root=root,
        artifact_root=artifact_root,
        equity_artifact_dir=artifact_root / "equity_moe",
        option_artifact_dir=artifact_root / "option_family_ranker",
        live_artifact_dir=artifact_root / "live",
    )


def load_equity_artifacts(artifact_dir: Path) -> dict[str, pd.DataFrame]:
    artifact_dir = Path(artifact_dir)
    return {
        "strategy_scores": pd.read_csv(artifact_dir / "strategy_scores.csv"),
        "backtest_summary": _read_csv_if_exists(artifact_dir / "backtest_summary.csv"),
        "model_results": _read_csv_if_exists(artifact_dir / "model_results.csv"),
    }


def latest_prices_from_quant_warehouse(
    symbols: Sequence[str],
    *,
    provider: str = "fmp",
    lookback_days: int = 30,
) -> dict[str, float]:
    ensure_quant_warehouse_storage()
    from quant_warehouse import Warehouse

    warehouse = Warehouse()
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=int(lookback_days))
    prices: dict[str, float] = {}
    for symbol in _normalize_symbols(symbols):
        frame = warehouse.read_prices(symbol, provider=provider, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        if frame is None or frame.empty or "close" not in frame.columns:
            continue
        close = pd.to_numeric(frame["close"], errors="coerce").dropna()
        if not close.empty and float(close.iloc[-1]) > 0:
            prices[symbol] = float(close.iloc[-1])
    return prices


def build_latest_equity_leaderboard(
    strategy_scores: pd.DataFrame,
    *,
    top_k: int,
    min_long_score: float = 0.50,
    price_provider: str = "fmp",
) -> pd.DataFrame:
    required = {"date", "symbol", "strategy_source", "long_score", "short_score"}
    missing = required.difference(strategy_scores.columns)
    if missing:
        raise KeyError(f"strategy_scores missing required columns: {sorted(missing)}")

    scores = strategy_scores.copy()
    scores["symbol"] = scores["symbol"].astype(str).str.strip().str.upper()
    scores["date"] = pd.to_datetime(scores["date"], errors="coerce").dt.normalize()
    scores["long_score"] = pd.to_numeric(scores["long_score"], errors="coerce")
    scores["short_score"] = pd.to_numeric(scores["short_score"], errors="coerce")
    scores = scores.dropna(subset=["date", "symbol", "long_score", "short_score"])
    latest_by_source = (
        scores.sort_values(["strategy_source", "symbol", "date"])
        .groupby(["strategy_source", "symbol"], as_index=False, sort=False)
        .tail(1)
    )
    latest_by_symbol = (
        latest_by_source.groupby("symbol", as_index=False)
        .agg(
            score_date=("date", "max"),
            prob_buy=("long_score", "mean"),
            prob_short=("short_score", "mean"),
            model_count=("strategy_source", "nunique"),
            best_family_score=("long_score", "max"),
        )
        .sort_values(["prob_buy", "best_family_score"], ascending=[False, False], kind="stable")
        .reset_index(drop=True)
    )
    latest_by_symbol["rank"] = latest_by_symbol.index + 1
    latest_by_symbol["selected"] = latest_by_symbol["rank"].le(int(top_k)) & latest_by_symbol["prob_buy"].ge(float(min_long_score))
    price_map = latest_prices_from_quant_warehouse(latest_by_symbol["symbol"], provider=price_provider)
    latest_by_symbol["close"] = latest_by_symbol["symbol"].map(price_map)
    latest_by_symbol["eligible"] = latest_by_symbol["selected"] & latest_by_symbol["close"].gt(0)
    return latest_by_symbol


def build_symbol_score_table(strategy_scores: pd.DataFrame, leaderboard: pd.DataFrame | None = None) -> pd.DataFrame:
    required = {"date", "symbol", "strategy_source", "long_score", "short_score"}
    missing = required.difference(strategy_scores.columns)
    if missing:
        raise KeyError(f"strategy_scores missing required columns: {sorted(missing)}")

    scores = strategy_scores.copy()
    scores["symbol"] = scores["symbol"].astype(str).str.strip().str.upper()
    scores["strategy_source"] = scores["strategy_source"].astype(str).str.strip()
    scores["date"] = pd.to_datetime(scores["date"], errors="coerce").dt.normalize()
    scores["long_score"] = pd.to_numeric(scores["long_score"], errors="coerce")
    scores["short_score"] = pd.to_numeric(scores["short_score"], errors="coerce")
    scores = scores.dropna(subset=["date", "symbol", "strategy_source", "long_score", "short_score"])
    latest = (
        scores.sort_values(["strategy_source", "symbol", "date"])
        .groupby(["strategy_source", "symbol"], as_index=False, sort=False)
        .tail(1)
        .copy()
    )
    if latest.empty:
        return pd.DataFrame(columns=["rank", "symbol"])

    ensemble = latest.loc[latest["strategy_source"].eq("ensemble_mean")].copy()
    if ensemble.empty:
        ensemble = (
            latest.groupby("symbol", as_index=False)
            .agg(
                date=("date", "max"),
                long_score=("long_score", "mean"),
                short_score=("short_score", "mean"),
                model_count=("strategy_source", "nunique"),
            )
            .assign(strategy_source="ensemble_mean")
        )
    if "model_count" not in ensemble.columns:
        ensemble["model_count"] = pd.NA
    base = ensemble[["symbol", "date", "long_score", "short_score", "model_count"]].rename(
        columns={
            "date": "score_date",
            "long_score": "ensemble_long_score",
            "short_score": "ensemble_short_score",
            "model_count": "ensemble_model_count",
        }
    )
    base["ensemble_net_score"] = base["ensemble_long_score"] - base["ensemble_short_score"]

    family_scores = latest.loc[~latest["strategy_source"].eq("ensemble_mean")].copy()
    if not family_scores.empty:
        long_wide = family_scores.pivot(index="symbol", columns="strategy_source", values="long_score")
        short_wide = family_scores.pivot(index="symbol", columns="strategy_source", values="short_score")
        long_wide = long_wide.rename(columns={col: f"long__{col}" for col in long_wide.columns})
        short_wide = short_wide.rename(columns={col: f"short__{col}" for col in short_wide.columns})
        family_wide = long_wide.join(short_wide, how="outer").reset_index()
        families = sorted(family_scores["strategy_source"].dropna().astype(str).unique())
        family_cols = ["symbol"]
        for family in families:
            family_cols.extend([f"long__{family}", f"short__{family}"])
        family_wide = family_wide.reindex(columns=[col for col in family_cols if col in family_wide.columns])
        table = base.merge(family_wide, on="symbol", how="outer")
    else:
        table = base

    if leaderboard is not None and not leaderboard.empty:
        lead = leaderboard.copy()
        lead["symbol"] = lead["symbol"].astype(str).str.strip().str.upper()
        lead_cols = [
            col
            for col in ("symbol", "rank", "prob_buy", "prob_short", "best_family_score", "selected", "close", "eligible")
            if col in lead.columns
        ]
        table = lead[lead_cols].merge(table, on="symbol", how="outer")
    if "rank" not in table.columns:
        table["rank"] = pd.NA
    missing_rank = pd.to_numeric(table["rank"], errors="coerce").isna()
    if missing_rank.any():
        max_rank = pd.to_numeric(table["rank"], errors="coerce").max()
        start_rank = int(max_rank) + 1 if pd.notna(max_rank) else 1
        fallback = table.loc[missing_rank].sort_values(
            ["ensemble_long_score", "symbol"],
            ascending=[False, True],
            kind="stable",
        )
        table.loc[fallback.index, "rank"] = range(start_rank, start_rank + len(fallback))
    table["rank"] = pd.to_numeric(table["rank"], errors="coerce").astype("Int64")
    ordered_prefix = [
        col
        for col in (
            "rank",
            "symbol",
            "score_date",
            "ensemble_long_score",
            "ensemble_short_score",
            "ensemble_net_score",
            "prob_buy",
            "prob_short",
            "best_family_score",
            "ensemble_model_count",
            "selected",
            "eligible",
            "close",
        )
        if col in table.columns
    ]
    remaining = [col for col in table.columns if col not in ordered_prefix]
    return table.reindex(columns=[*ordered_prefix, *remaining]).sort_values(["rank", "symbol"], kind="stable").reset_index(drop=True)


def build_option_ml_ranking_table(
    option_ranker_dir: Path,
    *,
    symbols: Sequence[str] | None = None,
    selected_only: bool = True,
    one_per_symbol: bool = True,
    tradable_as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    root = Path(option_ranker_dir)
    if not root.exists():
        return pd.DataFrame()
    scored_paths = sorted(path for path in root.glob("*/eval_scored.parquet") if path.is_file())
    if not scored_paths:
        return pd.DataFrame()
    selected_symbols = {
        str(symbol).strip().upper()
        for symbol in (symbols or ())
        if str(symbol).strip()
    }

    base_cols = [
        "trade_id",
        "symbol",
        "side",
        "equity_signal_side",
        "entry_date",
        "equity_exit_date",
        "option_exit_date",
        "expiration",
        "contract_symbol",
        "option_type",
        "option_action",
        "strike",
        "dte",
        "moneyness",
        "abs_moneyness",
        "spread_pct",
        "bid",
        "ask",
        "entry_mid",
        "option_return",
        "rank_y",
    ]
    key_cols: list[str] | None = None
    table: pd.DataFrame | None = None
    score_cols: list[str] = []

    for path in scored_paths:
        family = path.parent.name
        frame = pd.read_parquet(path)
        prediction_cols = [
            col
            for col in frame.columns
            if str(col).startswith("pred_") and not str(col).endswith("_pairwise")
        ]
        if not prediction_cols:
            continue
        score_col = prediction_cols[0]
        if key_cols is None:
            if "contract_symbol" in frame.columns:
                key_cols = [col for col in ("trade_id", "symbol", "entry_date", "contract_symbol") if col in frame.columns]
            else:
                key_cols = [
                    col
                    for col in ("trade_id", "symbol", "entry_date", "expiration", "strike", "option_type", "option_action")
                    if col in frame.columns
                ]
            if not key_cols:
                return pd.DataFrame()
            keep_cols = [col for col in base_cols if col in frame.columns]
            table = frame[keep_cols].copy()
            table = table.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)

        family_score_col = f"family_score__{family}"
        score_frame = frame[[*key_cols, score_col]].copy()
        score_frame[score_col] = pd.to_numeric(score_frame[score_col], errors="coerce")
        score_frame = score_frame.drop_duplicates(subset=key_cols, keep="first").rename(columns={score_col: family_score_col})
        table = table.merge(score_frame, on=key_cols, how="outer")
        score_cols.append(family_score_col)

    if table is None or not score_cols:
        return pd.DataFrame()
    if selected_symbols and "symbol" in table.columns:
        table["symbol"] = table["symbol"].astype(str).str.strip().str.upper()
        table = table.loc[table["symbol"].isin(selected_symbols)].copy()
        if table.empty:
            return pd.DataFrame()
    if "expiration" in table.columns:
        as_of = pd.Timestamp.today().normalize() if tradable_as_of is None else pd.Timestamp(tradable_as_of).normalize()
        expirations = pd.to_datetime(table["expiration"], errors="coerce").dt.normalize()
        table = table.loc[expirations.ge(as_of)].copy()
        if table.empty:
            return pd.DataFrame()
    table["option_ensemble_mean_score"] = table[score_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    table["option_family_score_count"] = table[score_cols].notna().sum(axis=1)
    sort_cols = ["trade_id", "option_ensemble_mean_score"]
    ascending = [True, False]
    if "option_return" in table.columns:
        table["option_return"] = pd.to_numeric(table["option_return"], errors="coerce")
        sort_cols.append("option_return")
        ascending.append(False)
    if "contract_symbol" in table.columns:
        sort_cols.append("contract_symbol")
        ascending.append(True)
    table = table.sort_values(sort_cols, ascending=ascending, kind="stable").reset_index(drop=True)
    table["option_ensemble_rank"] = table.groupby("trade_id", sort=False).cumcount() + 1
    table["selected_by_option_ensemble"] = table["option_ensemble_rank"].eq(1)
    if selected_only:
        table = table.loc[table["selected_by_option_ensemble"]].copy()
    if one_per_symbol and "symbol" in table.columns:
        symbol_sort_cols = ["symbol"]
        symbol_ascending = [True]
        if "entry_date" in table.columns:
            table["_entry_date_sort"] = pd.to_datetime(table["entry_date"], errors="coerce")
            symbol_sort_cols.append("_entry_date_sort")
            symbol_ascending.append(False)
        symbol_sort_cols.append("option_ensemble_mean_score")
        symbol_ascending.append(False)
        if "contract_symbol" in table.columns:
            symbol_sort_cols.append("contract_symbol")
            symbol_ascending.append(True)
        table = (
            table.sort_values(symbol_sort_cols, ascending=symbol_ascending, kind="stable")
            .groupby("symbol", as_index=False, sort=False)
            .head(1)
            .drop(columns=["_entry_date_sort"], errors="ignore")
            .reset_index(drop=True)
        )
    if "entry_date" in table.columns:
        table["entry_date"] = pd.to_datetime(table["entry_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "expiration" in table.columns:
        table["expiration"] = pd.to_datetime(table["expiration"], errors="coerce").dt.strftime("%Y-%m-%d")
    ordered_prefix = [
        col
        for col in (
            "selected_by_option_ensemble",
            "option_ensemble_rank",
            "option_ensemble_mean_score",
            "option_family_score_count",
            "trade_id",
            "symbol",
            "side",
            "equity_signal_side",
            "entry_date",
            "option_type",
            "option_action",
            "contract_symbol",
            "expiration",
            "strike",
            "dte",
            "moneyness",
            "spread_pct",
            "bid",
            "ask",
            "entry_mid",
            "option_return",
            "rank_y",
        )
        if col in table.columns
    ]
    remaining = [col for col in table.columns if col not in ordered_prefix]
    return table.reindex(columns=[*ordered_prefix, *remaining]).sort_values(
        ["trade_id", "option_ensemble_rank"],
        kind="stable",
    ).reset_index(drop=True)


def backfill_thetadata_eod_for_score_date(
    *,
    symbols: Sequence[str],
    score_date: str | pd.Timestamp,
    max_workers: int = 1,
    overwrite: bool = False,
    request_sleep: float = 0.0,
) -> dict[str, Any]:
    from quant_warehouse.migrate.backfill_thetadata_options import backfill_thetadata_options

    clean_symbols = _normalize_symbols(symbols)
    if not clean_symbols:
        return {"symbols_requested": 0, "symbols_completed": 0, "symbols_failed": 0, "results": []}
    date_text = pd.Timestamp(score_date).date().isoformat()
    return backfill_thetadata_options(
        symbols=clean_symbols,
        start_date=date_text,
        end_date=date_text,
        backfill_window_days=1,
        fallback_window_days=1,
        max_workers=max(1, int(max_workers)),
        overwrite=bool(overwrite),
        skip_existing=not bool(overwrite),
        request_sleep=float(request_sleep),
        progress_logger=print,
    )


def backfill_thetadata_for_oracle_trade_windows(
    oracle_trades: Path | pd.DataFrame,
    *,
    symbols: Sequence[str] | None = None,
    max_trades: int | None = None,
    backfill_window_days: int = 7,
    fallback_window_days: int = 1,
    overwrite: bool = False,
    request_sleep: float = 0.0,
) -> dict[str, Any]:
    from quant_warehouse.migrate.backfill_thetadata_options import backfill_thetadata_options_for_oracle_trades

    if isinstance(oracle_trades, pd.DataFrame):
        trades = oracle_trades.copy()
    else:
        path = Path(oracle_trades).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Oracle trades file not found: {path}")
        if path.suffix.lower() == ".parquet":
            trades = pd.read_parquet(path)
        elif path.suffix.lower() == ".csv":
            trades = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported oracle trades file type: {path.suffix}")
    return backfill_thetadata_options_for_oracle_trades(
        trades,
        symbols=symbols,
        max_trades=max_trades,
        backfill_window_days=int(backfill_window_days),
        fallback_window_days=int(fallback_window_days),
        skip_existing=not bool(overwrite),
        overwrite=bool(overwrite),
        request_sleep=float(request_sleep),
        progress_logger=print,
    )


def build_score_date_option_candidate_panel(
    *,
    leaderboard: pd.DataFrame,
    score_date: str | pd.Timestamp | None = None,
    symbols: Sequence[str] | None = None,
    target_dte: int = 90,
) -> pd.DataFrame:
    from quant_warehouse.platforms.data_providers.thetadata.feature_engineering.option_features import (
        build_option_contract_features,
    )
    from quant_warehouse.platforms.data_providers.thetadata.options import read_thetadata_eod_option_chain

    if leaderboard is None or leaderboard.empty:
        return pd.DataFrame()
    lead = leaderboard.copy()
    lead["symbol"] = lead["symbol"].astype(str).str.strip().str.upper()
    if symbols:
        wanted = set(_normalize_symbols(symbols))
        lead = lead.loc[lead["symbol"].isin(wanted)].copy()
    if "selected" in lead.columns:
        lead = lead.loc[lead["selected"].astype(bool)].copy()
    if lead.empty:
        return pd.DataFrame()
    if score_date is None:
        score_date = pd.to_datetime(lead["score_date"], errors="coerce").max() if "score_date" in lead.columns else pd.Timestamp.today()
    score_ts = pd.Timestamp(score_date).normalize()
    frames: list[pd.DataFrame] = []

    for row in lead.to_dict("records"):
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        try:
            chain = read_thetadata_eod_option_chain(
                symbol,
                start_date=score_ts,
                end_date=score_ts,
                require_rich_columns=False,
            )
        except Exception:
            continue
        if chain.empty:
            continue
        spot = _number(row.get("close"))
        featured = build_option_contract_features(
            chain,
            underlying_price=spot if spot > 0 else None,
            target_dte=int(target_dte),
        ).df
        if featured.empty:
            continue
        featured["snapshot_date"] = pd.to_datetime(featured.get("snapshot_date"), errors="coerce").dt.normalize()
        featured["expiration"] = pd.to_datetime(featured.get("expiration"), errors="coerce").dt.normalize()
        featured = featured.loc[featured["snapshot_date"].eq(score_ts) & featured["expiration"].ge(score_ts)].copy()
        if featured.empty:
            continue
        prob_buy = _number(row.get("prob_buy"))
        prob_short = _number(row.get("prob_short"))
        side = "short" if prob_short > prob_buy else "long"
        option_type = "put" if side == "short" else "call"
        option_action = "buy_put" if side == "short" else "buy_call"
        featured["option_type"] = featured["option_type"].astype(str).str.lower().str.strip()
        featured = featured.loc[featured["option_type"].eq(option_type)].copy()
        if featured.empty:
            continue
        featured["trade_id"] = f"live|{symbol}|{score_ts.date()}|{side}"
        featured["symbol"] = symbol
        featured["side"] = side
        featured["equity_signal_side"] = side
        featured["entry_date"] = score_ts
        featured["option_action"] = option_action
        featured["entry_mid"] = pd.to_numeric(featured.get("mid"), errors="coerce")
        frames.append(featured)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def build_score_date_option_ml_ranking_table(
    option_ranker_dir: Path,
    *,
    leaderboard: pd.DataFrame,
    score_date: str | pd.Timestamp | None = None,
    symbols: Sequence[str] | None = None,
    target_dte: int = 90,
    min_market_cap: int = 1_000_000_000_000,
    start_date: str = "1900-01-01",
) -> pd.DataFrame:
    root = Path(option_ranker_dir)
    family_dirs = sorted(path for path in root.iterdir() if path.is_dir()) if root.exists() else []
    if not family_dirs:
        return pd.DataFrame()
    candidates = build_score_date_option_candidate_panel(
        leaderboard=leaderboard,
        score_date=score_date,
        symbols=symbols,
        target_dte=target_dte,
    )
    if candidates.empty:
        return pd.DataFrame()
    score_ts = pd.to_datetime(candidates["entry_date"], errors="coerce").max()
    symbol_list = tuple(sorted(candidates["symbol"].dropna().astype(str).str.upper().unique()))
    from quant_warehouse.research_tools.feature_family_eval import FamilyEvaluationConfig, build_fundamental_feature_panel

    feature_panel, _metadata, _diagnostics, _timings = build_fundamental_feature_panel(
        symbol_list,
        FamilyEvaluationConfig(
            market_cap_min=int(min_market_cap),
            start_date=str(start_date),
            end_date=pd.Timestamp(score_ts).date().isoformat(),
        ),
    )
    table = candidates.copy()
    key_cols = [col for col in ("trade_id", "symbol", "entry_date", "contract_symbol") if col in table.columns]
    score_cols: list[str] = []
    for family_dir in family_dirs:
        summary_path = family_dir / "summary.json"
        ranker_path = family_dir / "ranker.pkl"
        if not summary_path.exists() or not ranker_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        option_features = [col for col in summary.get("option_features", OPTION_MODEL_FEATURES) if str(col)]
        family_features = [col for col in summary.get("family_features", []) if str(col)]
        joined = _join_score_date_family_features(table, feature_panel, family_features)
        model_features = [*option_features, *family_features]
        for col in model_features:
            if col not in joined.columns:
                joined[col] = pd.NA
        with ranker_path.open("rb") as handle:
            model = pickle.load(handle)
        score_col = f"family_score__{family_dir.name}"
        feature_frame = joined[model_features].apply(pd.to_numeric, errors="coerce")
        joined[score_col] = model.predict(feature_frame)
        table = table.merge(
            joined[[*key_cols, score_col]].drop_duplicates(subset=key_cols, keep="first"),
            on=key_cols,
            how="left",
        )
        score_cols.append(score_col)
    if not score_cols:
        return pd.DataFrame()
    return _finalize_option_ml_score_table(table, score_cols, tradable_as_of=score_ts)


def _join_score_date_family_features(option_frame: pd.DataFrame, feature_panel: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    if option_frame.empty or feature_panel.empty or not feature_cols:
        return option_frame.copy()
    left = option_frame.copy()
    left["_row_id"] = range(len(left))
    left["symbol"] = left["symbol"].astype(str).str.upper()
    left["entry_date"] = pd.to_datetime(left["entry_date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    right = feature_panel.rename(columns={"date": "feature_date"}).copy()
    right["symbol"] = right["symbol"].astype(str).str.upper()
    right["feature_date"] = pd.to_datetime(right["feature_date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    keep_cols = ["symbol", "feature_date", *[col for col in feature_cols if col in right.columns]]
    merged_parts: list[pd.DataFrame] = []
    for symbol, group in left.sort_values(["symbol", "entry_date", "_row_id"]).groupby("symbol", sort=False):
        family = right.loc[right["symbol"].eq(symbol), keep_cols].sort_values("feature_date")
        if family.empty:
            merged_parts.append(group)
            continue
        merged_parts.append(
            pd.merge_asof(
                group.sort_values("entry_date"),
                family,
                by="symbol",
                left_on="entry_date",
                right_on="feature_date",
                direction="backward",
            )
        )
    return pd.concat(merged_parts, ignore_index=True, sort=False).sort_values("_row_id").drop(columns=["_row_id"]).reset_index(drop=True)


def _finalize_option_ml_score_table(
    table: pd.DataFrame,
    score_cols: Sequence[str],
    *,
    tradable_as_of: str | pd.Timestamp,
) -> pd.DataFrame:
    if table.empty or not score_cols:
        return pd.DataFrame()
    work = table.copy()
    as_of = max(pd.Timestamp(tradable_as_of).normalize(), pd.Timestamp.today().normalize())
    if "expiration" in work.columns:
        expirations = pd.to_datetime(work["expiration"], errors="coerce").dt.normalize()
        work = work.loc[expirations.ge(as_of)].copy()
        if work.empty:
            return pd.DataFrame()
    work["option_ensemble_mean_score"] = work[list(score_cols)].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    work["option_family_score_count"] = work[list(score_cols)].notna().sum(axis=1)
    sort_cols = ["trade_id", "option_ensemble_mean_score"]
    ascending = [True, False]
    if "contract_symbol" in work.columns:
        sort_cols.append("contract_symbol")
        ascending.append(True)
    work = work.sort_values(sort_cols, ascending=ascending, kind="stable").reset_index(drop=True)
    work["option_ensemble_rank"] = work.groupby("trade_id", sort=False).cumcount() + 1
    work["selected_by_option_ensemble"] = work["option_ensemble_rank"].eq(1)
    work = work.loc[work["selected_by_option_ensemble"]].copy()
    if "symbol" in work.columns:
        work = (
            work.sort_values(["symbol", "option_ensemble_mean_score", "contract_symbol"], ascending=[True, False, True], kind="stable")
            .groupby("symbol", as_index=False, sort=False)
            .head(1)
            .reset_index(drop=True)
        )
    if "entry_date" in work.columns:
        work["entry_date"] = pd.to_datetime(work["entry_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "expiration" in work.columns:
        work["expiration"] = pd.to_datetime(work["expiration"], errors="coerce").dt.strftime("%Y-%m-%d")
    ordered_prefix = [
        col
        for col in (
            "selected_by_option_ensemble",
            "option_ensemble_rank",
            "option_ensemble_mean_score",
            "option_family_score_count",
            "trade_id",
            "symbol",
            "side",
            "equity_signal_side",
            "entry_date",
            "option_type",
            "option_action",
            "contract_symbol",
            "expiration",
            "strike",
            "dte",
            "moneyness",
            "spread_pct",
            "bid",
            "ask",
            "entry_mid",
        )
        if col in work.columns
    ]
    remaining = [col for col in work.columns if col not in ordered_prefix]
    return work.reindex(columns=[*ordered_prefix, *remaining]).sort_values(["symbol"], kind="stable").reset_index(drop=True)


def save_live_artifacts(
    *,
    live_dir: Path,
    leaderboard: pd.DataFrame,
    symbol_scores: pd.DataFrame | None = None,
    option_ml_rankings: pd.DataFrame | None = None,
    orders: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, str]:
    live_dir = Path(live_dir)
    live_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "leaderboard": str(live_dir / "leaderboard_latest.csv"),
        "symbol_scores": str(live_dir / "symbol_scores.csv"),
        "option_ml_rankings": str(live_dir / "option_ml_rankings.csv"),
        "metadata": str(live_dir / "metadata.json"),
    }
    leaderboard.to_csv(paths["leaderboard"], index=False)
    score_frame = symbol_scores.copy() if symbol_scores is not None else leaderboard.copy()
    score_frame.to_csv(paths["symbol_scores"], index=False)
    option_frame = option_ml_rankings.copy() if option_ml_rankings is not None else pd.DataFrame()
    option_frame.to_csv(paths["option_ml_rankings"], index=False)
    metadata = {
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "rows": int(len(leaderboard)),
        "selected": int(leaderboard.get("selected", pd.Series(dtype=bool)).sum()),
    }
    (live_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    plan_created_at = pd.Timestamp.now(tz="UTC").isoformat()
    for name, frame in dict(orders or {}).items():
        order_path = live_dir / f"{name}_orders.csv"
        stamped = _stamp_order_plan(frame, created_at=plan_created_at)
        stamped.to_csv(order_path, index=False)
        paths[f"{name}_orders"] = str(order_path)
    return paths


def leaderboard_to_ranked_scores(leaderboard: pd.DataFrame) -> pd.DataFrame:
    frame = leaderboard.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    out = frame.set_index("symbol")[["close", "prob_buy", "prob_short", "selected"]].copy()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["prob_buy"] = pd.to_numeric(out["prob_buy"], errors="coerce")
    out["prob_short"] = pd.to_numeric(out["prob_short"], errors="coerce")
    out["selected"] = out["selected"].astype(bool)
    return out


def alpaca_client_from_env(prefix: str):
    from platforms.brokers.alpaca import AlpacaPaperClient

    clean = str(prefix).strip().upper()
    key = os.getenv(f"{clean}_ALPACA_PAPER_API_KEY") or os.getenv(f"ALPACA_{clean}_PAPER_API_KEY")
    secret = os.getenv(f"{clean}_ALPACA_PAPER_API_SECRET") or os.getenv(f"ALPACA_{clean}_PAPER_API_SECRET")
    if not key or not secret:
        key = os.getenv("ALPACA_PAPER_API_KEY")
        secret = os.getenv("ALPACA_PAPER_API_SECRET")
    if not key or not secret:
        raise RuntimeError(f"Missing Alpaca paper credentials for prefix={prefix!r}.")
    return AlpacaPaperClient(api_key=str(key), api_secret=str(secret))


def build_alpaca_equity_orders(
    *,
    leaderboard: pd.DataFrame,
    account_prefix: str,
    gross_exposure: float = 0.95,
    liquidate_unselected: bool = True,
) -> pd.DataFrame:
    from platforms.brokers.alpaca import build_equal_weight_order_plan

    client = alpaca_client_from_env(account_prefix)
    account = client.get_account()
    open_orders = client.get_open_orders()
    positions = {
        str(row.get("symbol") or "").strip().upper(): float(row.get("qty") or 0.0)
        for row in client.get_positions()
        if str(row.get("asset_class") or "us_equity").lower() in {"us_equity", "equity", ""}
    }
    selected = leaderboard.loc[leaderboard["eligible"], "symbol"].astype(str).str.upper().tolist()
    prices = dict(zip(leaderboard["symbol"].astype(str).str.upper(), pd.to_numeric(leaderboard["close"], errors="coerce")))
    orders = build_equal_weight_order_plan(
        selected,
        prices,
        positions,
        portfolio_value=float(account.get("portfolio_value") or account.get("equity") or 0.0),
        gross_exposure=float(gross_exposure),
        liquidate_unselected=bool(liquidate_unselected),
    )
    cancel_orders = _build_open_order_cancel_rows(open_orders, asset_classes={"", "us_equity", "equity"})
    return pd.DataFrame([*cancel_orders, *orders])


def _build_open_order_cancel_rows(
    open_orders: Sequence[Mapping[str, Any]],
    *,
    asset_classes: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in open_orders or []:
        order = dict(raw)
        asset_class = str(order.get("asset_class") or "").strip().lower()
        if asset_class not in asset_classes:
            continue
        order_id = str(order.get("id") or order.get("order_id") or "").strip()
        symbol = str(order.get("symbol") or "").strip().upper()
        if not order_id:
            continue
        rows.append(
            {
                "symbol": symbol,
                "action": "cancel_open_order",
                "side": "cancel",
                "qty": 0,
                "order_id": order_id,
                "order_type": "cancel",
                "time_in_force": str(order.get("time_in_force") or ""),
                "reason": "Cancel existing open order before creating the refreshed trading_app_v2 plan.",
            }
        )
    return rows


def build_alpaca_option_orders(
    *,
    leaderboard: pd.DataFrame,
    account_prefix: str,
    strategy_allocation: float,
    option_bucket: str = "otm_option",
    tenor_days: int = 90,
    max_contracts_per_position: int | None = None,
) -> dict[str, Any]:
    client = alpaca_client_from_env(account_prefix)
    ranked = leaderboard_to_ranked_scores(leaderboard)
    selected_symbols = ranked.loc[ranked["selected"]].index.astype(str).tolist()
    as_of = pd.Timestamp.today().normalize()
    target_expiration = as_of + pd.Timedelta(days=int(tenor_days))
    expiration_lte = target_expiration + pd.Timedelta(days=45)
    option_contracts: dict[str, list[dict[str, Any]]] = {}
    selected_contract_symbols: list[str] = []
    for symbol in selected_symbols:
        contracts = client.get_option_contracts(
            symbol,
            option_type="call",
            expiration_date_gte=str(as_of.date()),
            expiration_date_lte=str(expiration_lte.date()),
        )
        option_contracts[symbol] = contracts
        contract = select_alpaca_option_contract(
            contracts,
            underlying_price=float(ranked.loc[symbol, "close"]),
            target_expiration=target_expiration.date(),
            option_bucket=option_bucket,
        )
        if contract:
            selected_contract_symbols.append(str(contract.get("symbol") or ""))
    current_positions = _enrich_alpaca_option_records(client, client.get_positions())
    open_orders = _enrich_alpaca_option_records(client, client.get_open_orders())
    position_contract_symbols = [str(row.get("symbol") or "").strip().upper() for row in current_positions if str(row.get("symbol") or "").strip()]
    option_snapshots = client.get_option_snapshots([*selected_contract_symbols, *position_contract_symbols])
    plan = build_alpaca_option_trade_plan(
        ranked_scores=ranked,
        current_option_positions=current_positions,
        open_orders=open_orders,
        option_contracts=option_contracts,
        option_snapshots=option_snapshots,
        strategy_allocation=float(strategy_allocation),
        as_of_date=as_of.date(),
        option_bucket=option_bucket,
        tenor_days=int(tenor_days),
        max_contracts_per_position=max_contracts_per_position,
    )
    plan["client"] = client
    return plan


def _number(value: Any, default: float = 0.0) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else float(default)


def _option_quote(snapshot: Mapping[str, Any]) -> tuple[float, float, float]:
    quote = dict(snapshot.get("latestQuote") or snapshot.get("latest_quote") or {})
    trade = dict(snapshot.get("latestTrade") or snapshot.get("latest_trade") or {})
    bid = _number(quote.get("bp", quote.get("bid_price")))
    ask = _number(quote.get("ap", quote.get("ask_price")))
    trade_price = _number(trade.get("p", trade.get("price")))
    mark = (bid + ask) / 2.0 if bid > 0 and ask > 0 else ask or bid or trade_price
    return bid, ask, mark


def select_alpaca_option_contract(
    contracts: Sequence[Mapping[str, Any]],
    *,
    underlying_price: float,
    target_expiration: Any,
    option_bucket: str,
) -> dict[str, Any] | None:
    strike_multiplier = {
        "atm_option": 1.0,
        "otm_option": 1.05,
        "ditm_option": 0.90,
    }.get(str(option_bucket), 1.05)
    target_strike = float(underlying_price) * strike_multiplier
    target_date = pd.Timestamp(target_expiration).date()
    candidates: list[tuple[int, float, str, dict[str, Any]]] = []
    for raw_contract in contracts:
        contract = dict(raw_contract)
        expiration = pd.to_datetime(contract.get("expiration_date"), errors="coerce")
        strike = _number(contract.get("strike_price"), default=float("nan"))
        symbol = str(contract.get("symbol") or "").strip().upper()
        if pd.isna(expiration) or not math.isfinite(strike) or strike <= 0 or not symbol:
            continue
        candidates.append((abs((expiration.date() - target_date).days), abs(strike - target_strike), symbol, contract))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][3]


def build_alpaca_option_trade_plan(
    *,
    ranked_scores: pd.DataFrame,
    current_option_positions: Sequence[Mapping[str, Any]],
    open_orders: Sequence[Mapping[str, Any]],
    option_contracts: Mapping[str, Sequence[Mapping[str, Any]]],
    option_snapshots: Mapping[str, Mapping[str, Any]],
    strategy_allocation: float,
    as_of_date: Any,
    option_bucket: str,
    tenor_days: int,
    max_contracts_per_position: int | None = None,
) -> dict[str, pd.DataFrame]:
    selected = ranked_scores.loc[ranked_scores["selected"]].copy()
    target_symbols = set(selected.index.astype(str))
    slot_budget = float(strategy_allocation) / len(target_symbols) if target_symbols else 0.0
    target_date = pd.Timestamp(as_of_date).date() + pd.Timedelta(days=int(tenor_days))

    held_underlyings: set[str] = set()
    action_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    contracts_to_close = 0
    for raw_position in current_option_positions:
        position = dict(raw_position)
        symbol = str(position.get("symbol") or "").strip().upper()
        underlying = str(position.get("underlying_symbol") or "").strip().upper()
        option_type = str(position.get("option_type") or position.get("type") or "").lower()
        qty = int(abs(_number(position.get("qty", position.get("quantity")))))
        if not symbol or not underlying or qty <= 0:
            continue
        position_rows.append(position)
        if option_type in {"", "call"}:
            held_underlyings.add(underlying)
        if underlying not in target_symbols or option_type == "put":
            bid, ask, mark = _option_quote(option_snapshots.get(symbol, {}))
            contracts_to_close += qty
            action_rows.append(
                {
                    "symbol": symbol,
                    "underlying_symbol": underlying,
                    "action": "sell_to_close_put" if option_type == "put" else "sell_to_close_call",
                    "side": "sell",
                    "qty": qty,
                    "quantity": qty,
                    "order_type": "limit",
                    "time_in_force": "day",
                    "bid_price": bid,
                    "ask_price": ask,
                    "mark_price": mark,
                    "reason": "Underlying is no longer selected by trading_app_v2.",
                }
            )

    pending_buy_underlyings: set[str] = set()
    pending_cancel_rows: list[dict[str, Any]] = []
    normalized_orders: list[dict[str, Any]] = []
    for raw_order in open_orders:
        order = dict(raw_order)
        underlying = str(order.get("underlying_symbol") or "").strip().upper()
        side = str(order.get("side") or "").strip().lower()
        option_type = str(order.get("option_type") or order.get("type") or "").strip().lower()
        qty = _number(order.get("qty", order.get("quantity")))
        filled_qty = _number(order.get("filled_qty", order.get("filled_quantity")))
        remaining = max(qty - filled_qty, 0.0)
        normalized = {
            "order_id": str(order.get("id") or order.get("order_id") or ""),
            "symbol": str(order.get("symbol") or "").strip().upper(),
            "underlying_symbol": underlying,
            "side": side,
            "option_type": option_type,
            "remaining_qty": remaining,
            "status": str(order.get("status") or ""),
        }
        normalized_orders.append(normalized)
        if side == "buy" and remaining > 0 and underlying:
            if underlying not in target_symbols or option_type == "put":
                pending_cancel_rows.append(
                    {
                        "symbol": normalized["symbol"],
                        "underlying_symbol": underlying,
                        "action": "cancel_buy_to_open_put" if option_type == "put" else "cancel_buy_to_open_call",
                        "side": "cancel",
                        "qty": remaining,
                        "quantity": remaining,
                        "order_id": normalized["order_id"],
                        "order_type": "cancel",
                        "time_in_force": "day",
                        "reason": "Open option order is no longer selected by trading_app_v2.",
                    }
                )
            else:
                pending_buy_underlyings.add(underlying)

    target_contract_rows: list[dict[str, Any]] = []
    desired_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for underlying in selected.index.astype(str):
        contract = select_alpaca_option_contract(
            option_contracts.get(underlying, []),
            underlying_price=float(selected.loc[underlying, "close"]),
            target_expiration=target_date,
            option_bucket=option_bucket,
        )
        if contract is None:
            skipped_rows.append({"symbol": underlying, "reason": "No matching active Alpaca call contract."})
            continue
        contract_symbol = str(contract.get("symbol") or "").strip().upper()
        bid, ask, mark = _option_quote(option_snapshots.get(contract_symbol, {}))
        limit_price = bid or mark
        contract_value = limit_price * 100.0
        quantity = int(slot_budget // contract_value) if contract_value > 0 else 0
        if max_contracts_per_position is not None:
            quantity = min(quantity, max(int(max_contracts_per_position), 0))
        desired = {
            "symbol": underlying,
            "option_symbol": contract_symbol,
            "option_type": "call",
            "expiry_date": contract.get("expiration_date"),
            "strike_price": _number(contract.get("strike_price")),
            "underlying_price": float(selected.loc[underlying, "close"]),
            "bid_price": bid,
            "ask_price": ask,
            "mark_price": mark,
            "limit_price": limit_price,
            "contract_value": contract_value,
            "target_dollars": slot_budget,
            "quantity": quantity,
            "combined_score": float(selected.loc[underlying, "prob_buy"]),
        }
        target_contract_rows.append(desired)
        if underlying in held_underlyings or underlying in pending_buy_underlyings:
            continue
        desired_rows.append(desired)
        if quantity <= 0:
            skipped_rows.append({**desired, "reason": "One contract exceeds the per-position option budget."})
            continue
        action_rows.append(
            {
                **desired,
                "symbol": contract_symbol,
                "underlying_symbol": underlying,
                "action": "buy_to_open_call",
                "side": "buy",
                "qty": quantity,
                "order_type": "limit",
                "time_in_force": "day",
                "reason": "New current top-K trading_app_v2 option position.",
            }
        )

    actions = apply_option_limit_policy(pd.DataFrame([*pending_cancel_rows, *action_rows]))
    summary = pd.DataFrame(
        [
            {
                "target_positions": len(target_symbols),
                "calls_to_open": int(actions.get("action", pd.Series(dtype=str)).eq("buy_to_open_call").sum()),
                "contracts_to_close": contracts_to_close,
                "orders_to_cancel": int(actions.get("action", pd.Series(dtype=str)).astype(str).str.startswith("cancel_").sum()),
                "strategy_allocation": float(strategy_allocation),
                "occupied_slots": len(held_underlyings & target_symbols),
                "pending_buy_underlyings": len(pending_buy_underlyings & target_symbols),
            }
        ]
    )
    return {
        "summary": summary,
        "target_contracts": pd.DataFrame(target_contract_rows),
        "desired_contracts": pd.DataFrame(desired_rows),
        "current_option_positions": pd.DataFrame(position_rows),
        "pending_option_orders": pd.DataFrame(normalized_orders),
        "actions": actions,
        "actionable_orders": actions.copy(),
        "skipped_symbols": pd.DataFrame(skipped_rows),
    }


def build_robinhood_option_orders(
    *,
    target_contracts: pd.DataFrame,
    gate_discount_pct: float,
    account_number: str | None = None,
    current_option_positions: pd.DataFrame | None = None,
    pending_option_orders: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Reconcile Robinhood option account state before creating new live orders."""

    from platforms.brokers import robinhood

    current = (
        current_option_positions.copy()
        if current_option_positions is not None
        else robinhood.load_robinhood_option_positions(account_number=account_number)
    )
    pending = (
        pending_option_orders.copy()
        if pending_option_orders is not None
        else robinhood.load_robinhood_open_option_orders(account_number=account_number)
    )
    targets = _normalize_robinhood_target_contracts(target_contracts)
    target_by_symbol = {
        str(row["symbol"]).strip().upper(): row
        for _, row in targets.iterrows()
        if str(row.get("symbol") or "").strip()
    }
    target_symbols = set(target_by_symbol)

    action_rows: list[dict[str, Any]] = []
    held_target_symbols: set[str] = set()
    pending_buy_symbols: set[str] = set()
    pending_sell_contracts: set[tuple[str, str, float, str]] = set()

    if current is not None and not current.empty:
        for _, raw_position in current.iterrows():
            position = raw_position.to_dict()
            symbol = str(position.get("symbol") or position.get("underlying_symbol") or "").strip().upper()
            quantity = int(abs(round(_number(position.get("quantity", position.get("qty"))))))
            if not symbol or quantity <= 0:
                continue
            target = target_by_symbol.get(symbol)
            if target is not None and _same_option_contract(position, target) and str(position.get("option_type") or "").lower() == "call":
                held_target_symbols.add(symbol)
                continue
            sell_row = {
                "symbol": symbol,
                "action": "sell_to_close_put" if str(position.get("option_type") or "").lower() == "put" else "sell_to_close_call",
                "reason": "Existing Robinhood option position is no longer the target contract.",
                "quantity": quantity,
                "expiry_date": str(position.get("expiry_date") or ""),
                "strike_price": _number(position.get("strike_price")),
                "option_type": str(position.get("option_type") or "call").strip().lower(),
                "order_type": "limit",
                "time_in_force": "gtc",
                "bid_price": position.get("bid_price"),
                "ask_price": position.get("ask_price"),
                "mark_price": position.get("mark_price"),
                "average_price": position.get("average_price"),
            }
            priced_sell = apply_option_limit_policy(pd.DataFrame([sell_row]), time_in_force="gtc")
            action_rows.extend(priced_sell.to_dict(orient="records"))

    if pending is not None and not pending.empty:
        for _, raw_order in pending.iterrows():
            order = raw_order.to_dict()
            symbol = str(order.get("symbol") or order.get("underlying_symbol") or "").strip().upper()
            action = str(order.get("action") or "").strip().lower()
            if action.startswith("sell_to_close") and symbol:
                pending_sell_contracts.add(_option_contract_key(order))
                continue
            if not action.startswith("buy_to_open") or not symbol:
                continue
            target = target_by_symbol.get(symbol)
            if target is not None and _same_option_contract(order, target) and action == "buy_to_open_call":
                pending_buy_symbols.add(symbol)
                continue
            action_rows.append(
                {
                    "symbol": symbol,
                    "action": "cancel_buy_to_open_put" if action == "buy_to_open_put" else "cancel_buy_to_open_call",
                    "reason": "Open Robinhood option order is no longer the target contract.",
                    "quantity": order.get("contract_quantity", order.get("quantity", 0)),
                    "expiry_date": str(order.get("expiry_date") or ""),
                    "strike_price": order.get("strike_price"),
                    "option_type": str(order.get("option_type") or "call").strip().lower(),
                    "order_type": "cancel",
                    "order_id": str(order.get("order_id") or ""),
                    "cancel_url": str(order.get("cancel_url") or ""),
                    "price": order.get("price"),
                }
            )

    for _, target in targets.sort_values(["combined_score", "symbol"], ascending=[False, True], kind="stable").iterrows():
        symbol = str(target.get("symbol") or "").strip().upper()
        if not symbol or symbol in held_target_symbols or symbol in pending_buy_symbols:
            continue
        quantity = int(_number(target.get("quantity", target.get("target_contracts"))))
        if quantity <= 0:
            continue
        buy_row = {
            **target.to_dict(),
            "symbol": symbol,
            "action": "buy_to_open_call",
            "reason": "New current top-K trading_app_v2 Robinhood option target.",
            "quantity": quantity,
            "qty": quantity,
            "option_type": "call",
            "order_type": "limit",
            "time_in_force": "gtc",
        }
        priced = apply_option_limit_policy(pd.DataFrame([buy_row]), time_in_force="gtc")
        priced["gate_discount_pct"] = float(gate_discount_pct)
        if float(gate_discount_pct) >= 100.0:
            priced["skip_submit"] = True
            priced["skip_reason"] = "gate_discount_pct_100_blocks_orders"
        action_rows.extend(priced.to_dict(orient="records"))

    actions = pd.DataFrame(action_rows)
    if not actions.empty:
        if "combined_score" not in actions.columns:
            actions["combined_score"] = pd.NA
        if "skip_submit" not in actions.columns:
            actions["skip_submit"] = False
        sell_mask = actions["action"].astype(str).str.startswith("sell_to_close")
        if sell_mask.any():
            duplicate_sell = actions.loc[sell_mask].apply(lambda row: _option_contract_key(row.to_dict()) in pending_sell_contracts, axis=1)
            actions.loc[actions.loc[sell_mask].index[duplicate_sell], "skip_submit"] = True
            actions.loc[actions.loc[sell_mask].index[duplicate_sell], "skip_reason"] = "pending_sell_to_close_exists"
        skip_submit = actions.get("skip_submit", pd.Series(False, index=actions.index))
        actions["skip_submit"] = skip_submit.map(lambda value: bool(value) if pd.notna(value) else False)
        priority = {
            "cancel_buy_to_open_call": 0,
            "cancel_buy_to_open_put": 1,
            "sell_to_close_call": 2,
            "sell_to_close_put": 3,
            "buy_to_open_call": 4,
        }
        actions["_priority"] = actions["action"].map(priority).fillna(99)
        actions = actions.sort_values(["_priority", "combined_score", "symbol"], ascending=[True, False, True], kind="stable").drop(columns=["_priority"])
    else:
        actions["skip_submit"] = pd.Series(dtype=bool)

    summary = pd.DataFrame(
        [
            {
                "target_positions": int(len(target_symbols)),
                "positions_seen": int(0 if current is None else len(current)),
                "open_orders_seen": int(0 if pending is None else len(pending)),
                "positions_kept": int(len(held_target_symbols)),
                "pending_buys_kept": int(len(pending_buy_symbols)),
                "orders_to_cancel": int(actions["action"].astype(str).str.startswith("cancel_").sum()) if not actions.empty else 0,
                "positions_to_exit": int(actions["action"].astype(str).str.startswith("sell_to_close").sum()) if not actions.empty else 0,
                "orders_to_open": int(actions["action"].astype(str).str.startswith("buy_to_open").sum()) if not actions.empty else 0,
                "gate_discount_pct": float(gate_discount_pct),
            }
        ]
    )
    return {
        "summary": summary,
        "current_option_positions": pd.DataFrame() if current is None else current.reset_index(drop=True),
        "pending_option_orders": pd.DataFrame() if pending is None else pending.reset_index(drop=True),
        "target_contracts": targets.reset_index(drop=True),
        "actions": actions.reset_index(drop=True),
        "actionable_orders": actions.reset_index(drop=True),
    }


def _normalize_robinhood_target_contracts(target_contracts: pd.DataFrame) -> pd.DataFrame:
    if target_contracts is None or target_contracts.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "option_type",
                "expiry_date",
                "strike_price",
                "quantity",
                "limit_price",
                "combined_score",
            ]
        )
    out = target_contracts.copy()
    if "underlying_symbol" in out.columns:
        symbol = out["underlying_symbol"]
    else:
        symbol = out.get("symbol", pd.Series("", index=out.index))
    out["symbol"] = symbol.astype(str).str.strip().str.upper()
    out["option_type"] = out.get("option_type", "call")
    out["option_type"] = out["option_type"].astype(str).str.strip().str.lower().replace({"": "call"})
    if "expiry_date" not in out.columns and "expiration_date" in out.columns:
        out["expiry_date"] = out["expiration_date"]
    if "quantity" not in out.columns and "target_contracts" in out.columns:
        out["quantity"] = out["target_contracts"]
    if "limit_price" not in out.columns:
        if "limit_order_price" in out.columns:
            out["limit_price"] = out["limit_order_price"]
        elif "bid_price" in out.columns:
            out["limit_price"] = out["bid_price"]
        elif "mark_price" in out.columns:
            out["limit_price"] = out["mark_price"]
        else:
            out["limit_price"] = pd.NA
    if "combined_score" not in out.columns:
        out["combined_score"] = pd.NA
    out["strike_price"] = pd.to_numeric(out.get("strike_price"), errors="coerce")
    out["quantity"] = pd.to_numeric(out.get("quantity"), errors="coerce").fillna(0).astype("int64")
    out["limit_price"] = pd.to_numeric(out["limit_price"], errors="coerce")
    return out.dropna(subset=["symbol", "expiry_date", "strike_price"]).loc[out["quantity"].gt(0)].reset_index(drop=True)


def _option_contract_key(row: Mapping[str, Any]) -> tuple[str, str, float, str]:
    strike = pd.to_numeric(pd.Series([row.get("strike_price")]), errors="coerce").iloc[0]
    return (
        str(row.get("symbol") or row.get("underlying_symbol") or "").strip().upper(),
        str(row.get("expiry_date") or row.get("expiration_date") or "").strip(),
        float(strike) if pd.notna(strike) else 0.0,
        str(row.get("option_type") or "").strip().lower(),
    )


def _same_option_contract(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return _option_contract_key(left) == _option_contract_key(right)


def _first_positive(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = _number(row.get(key), default=float("nan"))
        if math.isfinite(value) and value > 0:
            return float(value)
    return None


def build_llm_review_orders(
    *,
    leaderboard: pd.DataFrame,
    top_k: int,
    account_prefix: str,
    as_of_date: str | None = None,
    trading_agents_config: Any | None = None,
) -> pd.DataFrame:
    from platforms.agents.trading_agents import approved_symbols, review_trade_candidates

    candidates = leaderboard.head(int(top_k)).copy()
    reviewed = review_trade_candidates(candidates, as_of_date=as_of_date, config=trading_agents_config)
    symbols = approved_symbols(reviewed)
    if not symbols:
        return pd.DataFrame(columns=["symbol", "side", "qty", "reason"])
    reviewed_leaderboard = leaderboard.loc[leaderboard["symbol"].astype(str).str.upper().isin(symbols)].copy()
    reviewed_leaderboard["eligible"] = True
    orders = build_alpaca_equity_orders(leaderboard=reviewed_leaderboard, account_prefix=account_prefix)
    if not orders.empty and not reviewed.empty:
        review_cols = [col for col in ("symbol", "llm_decision", "llm_rating", "llm_reason", "llm_review_date") if col in reviewed.columns]
        orders = orders.merge(reviewed[review_cols], on="symbol", how="left")
    return orders


def apply_option_limit_policy(
    orders: pd.DataFrame,
    *,
    time_in_force: str | None = None,
) -> pd.DataFrame:
    """Set option limit prices from the executable side of the quote.

    Buy-to-open orders bid. Sell-to-close orders ask. Cancels pass through.
    """

    if orders is None or orders.empty:
        return pd.DataFrame() if orders is None else orders.copy()
    work = orders.copy()
    if "skip_submit" not in work.columns:
        work["skip_submit"] = False
    if "skip_reason" not in work.columns:
        work["skip_reason"] = ""
    for idx, row in work.iterrows():
        action = str(row.get("action") or "").strip().lower()
        if action.startswith("cancel_") or action == "cancel_open_order":
            work.at[idx, "skip_submit"] = False
            continue
        if action.startswith("buy_to_open") or str(row.get("side") or "").strip().lower() == "buy":
            price = _first_positive(row.to_dict(), ("bid_price",))
            source = "bid_price"
            pricing_side = "buy"
        elif action.startswith("sell_to_close") or str(row.get("side") or "").strip().lower() == "sell":
            price = _first_positive(row.to_dict(), ("ask_price",))
            source = "ask_price"
            pricing_side = "sell"
        else:
            continue
        work.at[idx, "order_type"] = "limit"
        if time_in_force is not None:
            work.at[idx, "time_in_force"] = str(time_in_force)
        if price is None:
            work.at[idx, "skip_submit"] = True
            work.at[idx, "skip_reason"] = f"missing_{source}"
            continue
        limit_price = normalize_option_limit_price(float(price), side=pricing_side)
        if limit_price is None:
            work.at[idx, "skip_submit"] = True
            work.at[idx, "skip_reason"] = f"invalid_{source}"
            continue
        work.at[idx, "limit_price"] = float(limit_price)
        work.at[idx, "limit_order_price"] = float(limit_price)
        work.at[idx, "price"] = float(limit_price)
        work.at[idx, "limit_price_source"] = source
        work.at[idx, "skip_submit"] = False
        work.at[idx, "skip_reason"] = ""
    return work


def _stamp_order_plan(orders: pd.DataFrame, *, created_at: str | None = None) -> pd.DataFrame:
    stamped = orders.copy()
    if not stamped.empty and "plan_created_at" not in stamped.columns:
        stamped["plan_created_at"] = created_at or pd.Timestamp.now(tz="UTC").isoformat()
    return stamped


def validate_order_plan_for_submission(
    orders: pd.DataFrame,
    *,
    asset_type: str,
    policy: SubmissionSafetyPolicy | None = None,
    now: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return actionable orders or raise before any broker side effect occurs."""

    if orders is None or orders.empty:
        return pd.DataFrame()
    limits = policy or SubmissionSafetyPolicy()
    actionable = orders.loc[~orders.get("skip_submit", pd.Series(False, index=orders.index)).astype(bool)].copy()
    if actionable.empty:
        return actionable
    if len(actionable) > int(limits.max_orders):
        raise ValueError(f"Refusing order plan with {len(actionable)} rows; limit is {limits.max_orders}.")

    if "plan_created_at" not in actionable.columns:
        raise ValueError("Refusing unstamped order plan: missing plan_created_at.")
    created = pd.to_datetime(actionable["plan_created_at"], errors="coerce", utc=True)
    if created.isna().any():
        raise ValueError("Refusing order plan with an invalid plan_created_at.")
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    ages = current - created
    if ages.lt(pd.Timedelta(0)).any():
        raise ValueError("Refusing order plan dated in the future.")
    if ages.gt(pd.Timedelta(hours=float(limits.max_plan_age_hours))).any():
        raise ValueError(f"Refusing stale order plan older than {limits.max_plan_age_hours:g} hours.")

    cancel_mask = actionable.get("action", pd.Series("", index=actionable.index)).astype(str).str.lower().str.startswith("cancel_")
    cancel_mask |= actionable.get("side", pd.Series("", index=actionable.index)).astype(str).str.lower().eq("cancel")
    if cancel_mask.any():
        order_ids = actionable.loc[cancel_mask].get("order_id", pd.Series("", index=actionable.index[cancel_mask])).astype(str).str.strip()
        if order_ids.eq("").any():
            raise ValueError("Refusing cancellation row without an order_id.")

    trade_rows = actionable.loc[~cancel_mask]
    if not trade_rows.empty:
        required = {"symbol", "side", "qty"}
        missing = required.difference(trade_rows.columns)
        if missing:
            raise ValueError(f"Refusing order plan missing required columns: {sorted(missing)}.")
        symbols = trade_rows["symbol"].astype(str).str.strip()
        if symbols.eq("").any():
            raise ValueError("Refusing order plan with an empty symbol.")
        sides = trade_rows["side"].astype(str).str.lower().str.strip()
        if not sides.isin({"buy", "sell"}).all():
            raise ValueError("Refusing order plan with a side other than buy or sell.")
        quantities = pd.to_numeric(trade_rows["qty"], errors="coerce")
        if quantities.isna().any() or quantities.le(0).any() or quantities.mod(1).ne(0).any():
            raise ValueError("Refusing order plan with a non-positive or non-integer quantity.")
        quantity_limit = (
            limits.max_option_contracts_per_order if str(asset_type).lower() == "option" else limits.max_quantity_per_order
        )
        if quantities.gt(int(quantity_limit)).any():
            raise ValueError(f"Refusing order quantity above the {int(quantity_limit)} per-order limit.")

    duplicate_cols = [col for col in ("action", "symbol", "side", "qty", "order_id") if col in actionable.columns]
    if duplicate_cols and actionable.duplicated(subset=duplicate_cols, keep=False).any():
        raise ValueError(f"Refusing duplicate order rows keyed by {duplicate_cols}.")
    return actionable.reset_index(drop=True)


def submit_alpaca_orders(
    client: Any,
    orders: pd.DataFrame,
    *,
    asset_type: str = "equity",
) -> pd.DataFrame:
    """Validate and submit an Alpaca plan using limits for its asset class."""

    if orders is None or orders.empty:
        return pd.DataFrame()
    actionable = validate_order_plan_for_submission(orders, asset_type=asset_type)
    responses = client.submit_orders(actionable.to_dict(orient="records"))
    return pd.DataFrame(responses)


def submit_robinhood_option_orders(orders: pd.DataFrame, *, account_number: str | None = None) -> pd.DataFrame:
    if orders is None or orders.empty:
        return pd.DataFrame()
    actionable = validate_order_plan_for_submission(orders, asset_type="option")
    from platforms.brokers.robinhood import submit_robinhood_option_orders as _submit

    return _submit(orders_df=actionable, account_number=account_number, time_in_force="gtc")


def write_streamlit_leaderboard_app(
    *,
    live_dir: Path,
    output_path: Path | None = None,
    leaderboard: pd.DataFrame | None = None,
    symbol_scores: pd.DataFrame | None = None,
    option_ml_rankings: pd.DataFrame | None = None,
    orders: Mapping[str, pd.DataFrame] | None = None,
) -> Path:
    live_dir = Path(live_dir).resolve()
    repo_root = find_repo_root(Path(__file__).resolve())
    output = Path(output_path or (live_dir / "streamlit_trading_app_v2.py"))
    output.parent.mkdir(parents=True, exist_ok=True)
    if leaderboard is not None:
        def _json_frame(frame: pd.DataFrame | None) -> str:
            source = frame.copy() if frame is not None else pd.DataFrame()
            return source.to_json(orient="split", date_format="iso")

        payload = {
            "leaderboard": _json_frame(leaderboard),
            "symbol_scores": _json_frame(symbol_scores if symbol_scores is not None else leaderboard),
            "option_ml_rankings": _json_frame(option_ml_rankings),
            "orders": {str(name): _json_frame(_stamp_order_plan(frame)) for name, frame in dict(orders or {}).items()},
        }
        payload_literal = json.dumps(payload)
        script = f'''from __future__ import annotations

from io import StringIO
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

REPO_ROOT = Path(r"{str(repo_root)}")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.trading_app_v2_runtime import alpaca_client_from_env, submit_alpaca_orders, submit_robinhood_option_orders

EMBEDDED_DATA = {payload_literal}


def read_embedded_frame(name: str) -> pd.DataFrame:
    raw = EMBEDDED_DATA.get(name)
    if not raw:
        return pd.DataFrame()
    return pd.read_json(StringIO(raw), orient="split")


def read_embedded_orders() -> dict[str, pd.DataFrame]:
    raw_orders = EMBEDDED_DATA.get("orders") or {{}}
    return {{
        str(name): pd.read_json(StringIO(raw), orient="split") if raw else pd.DataFrame()
        for name, raw in raw_orders.items()
    }}


st.set_page_config(page_title="Trading App V2", layout="wide")
st.title("Trading App V2 Leaderboard")

leaderboard = read_embedded_frame("leaderboard")
if leaderboard.empty:
    st.warning("Leaderboard is empty for this generated app snapshot.")
    st.stop()
selected = int(leaderboard.get("selected", pd.Series(dtype=bool)).sum())
eligible = int(leaderboard.get("eligible", pd.Series(dtype=bool)).sum())
cols = st.columns(4)
cols[0].metric("Rows", f"{{len(leaderboard):,}}")
cols[1].metric("Selected", f"{{selected:,}}")
cols[2].metric("Eligible", f"{{eligible:,}}")
cols[3].metric("Latest Score Date", str(leaderboard.get("score_date", pd.Series([""])).max()))

symbol_tab, option_tab, orders_tab = st.tabs(["Symbol Scores", "Option ML Rankings", "Orders"])

with symbol_tab:
    score_table = read_embedded_frame("symbol_scores")
    if score_table.empty:
        st.warning("Symbol score view is empty. Showing leaderboard only.")
        score_table = leaderboard.copy()
    st.subheader("Scores By Symbol")
    st.dataframe(score_table.sort_values(["rank", "symbol"], kind="stable"), width="stretch", hide_index=True)

with option_tab:
    option_rankings = read_embedded_frame("option_ml_rankings")
    if option_rankings.empty:
        st.info("No tradable, unexpired option ML ranking rows are embedded in this app snapshot.")
    else:
        st.subheader("Selected Option ML Rankings")
        st.dataframe(option_rankings, width="stretch", hide_index=True)

with orders_tab:
    order_frames = read_embedded_orders()
    for name, frame in sorted(order_frames.items()):
        st.subheader(name.replace("_", " ").title())
        st.dataframe(frame, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Submit Orders By Account")
    account_prefixes = {{
        "alpaca_equity_paper": "EQUITY",
        "alpaca_option_paper": "OPTION",
        "alpaca_llm_paper": "LLM",
    }}
    submitters = {{
        **{{name: "alpaca" for name in account_prefixes}},
        "robinhood_option_real": "robinhood_option",
    }}
    for name in sorted(order_frames):
        orders = order_frames[name]
        account_label = name.replace("_", " ").title()
        with st.expander(account_label, expanded=False):
            if orders.empty:
                st.info("No orders for this account.")
                continue
            st.dataframe(orders, width="stretch", hide_index=True)
            confirm = st.checkbox(
                f"I have reviewed only the {{account_label}} orders.",
                key=f"confirm_{{name}}",
            )
            disabled = not confirm or name not in submitters
            if name not in submitters:
                st.warning("No submitter is configured for this account.")
            if st.button(f"Submit {{account_label}} Orders", type="primary", disabled=disabled, key=f"submit_{{name}}"):
                if submitters[name] == "alpaca":
                    client = alpaca_client_from_env(account_prefixes[name])
                    result = submit_alpaca_orders(client, orders)
                elif submitters[name] == "robinhood_option":
                    result = submit_robinhood_option_orders(orders)
                else:
                    result = pd.DataFrame()
                st.write(f"{{name}}: {{len(result)}} response row(s)")
                st.dataframe(result, width="stretch", hide_index=True)
'''
        output.write_text(script, encoding="utf-8")
        return output

    script = f'''from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import streamlit as st

LIVE_DIR = Path(r"{str(live_dir)}")
REPO_ROOT = Path(r"{str(repo_root)}")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.trading_app_v2_runtime import alpaca_client_from_env, submit_alpaca_orders, submit_robinhood_option_orders


def read_csv_if_nonempty(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


st.set_page_config(page_title="Trading App V2", layout="wide")
st.title("Trading App V2 Leaderboard")

leaderboard_path = LIVE_DIR / "leaderboard_latest.csv"
if not leaderboard_path.exists():
    st.error(f"Missing leaderboard: {{leaderboard_path}}")
    st.stop()

leaderboard = read_csv_if_nonempty(leaderboard_path)
if leaderboard.empty:
    st.warning(f"Leaderboard is empty: {{leaderboard_path}}")
    st.stop()
selected = int(leaderboard.get("selected", pd.Series(dtype=bool)).sum())
eligible = int(leaderboard.get("eligible", pd.Series(dtype=bool)).sum())
cols = st.columns(4)
cols[0].metric("Rows", f"{{len(leaderboard):,}}")
cols[1].metric("Selected", f"{{selected:,}}")
cols[2].metric("Eligible", f"{{eligible:,}}")
cols[3].metric("Latest Score Date", str(leaderboard.get("score_date", pd.Series([""])).max()))

symbol_tab, option_tab, orders_tab = st.tabs(["Symbol Scores", "Option ML Rankings", "Orders"])

with symbol_tab:
    symbol_scores_path = LIVE_DIR / "symbol_scores.csv"
    score_table = read_csv_if_nonempty(symbol_scores_path)
    if score_table.empty:
        st.warning(f"Missing or empty symbol score view: {{symbol_scores_path}}. Showing leaderboard only.")
        score_table = leaderboard.copy()
    st.subheader("Scores By Symbol")
    st.dataframe(score_table.sort_values(["rank", "symbol"], kind="stable"), width="stretch", hide_index=True)

with option_tab:
    option_rankings_path = LIVE_DIR / "option_ml_rankings.csv"
    option_rankings = read_csv_if_nonempty(option_rankings_path)
    if option_rankings.empty:
        st.info(f"No tradable, unexpired option ML ranking rows found at {{option_rankings_path}}.")
    else:
        st.subheader("Selected Option ML Rankings")
        st.dataframe(option_rankings, width="stretch", hide_index=True)

with orders_tab:
    order_frames = {{}}
    for path in sorted(LIVE_DIR.glob("*_orders.csv")):
        st.subheader(path.stem.replace("_", " ").title())
        frame = read_csv_if_nonempty(path)
        order_frames[path.stem.removesuffix("_orders")] = frame
        st.dataframe(frame, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Submit Orders By Account")
    account_prefixes = {{
        "alpaca_equity_paper": "EQUITY",
        "alpaca_option_paper": "OPTION",
        "alpaca_llm_paper": "LLM",
    }}
    submitters = {{
        **{{name: "alpaca" for name in account_prefixes}},
        "robinhood_option_real": "robinhood_option",
    }}
    for name in sorted(order_frames):
        orders = order_frames[name]
        account_label = name.replace("_", " ").title()
        with st.expander(account_label, expanded=False):
            if orders.empty:
                st.info("No orders for this account.")
                continue
            st.dataframe(orders, width="stretch", hide_index=True)
            confirm = st.checkbox(
                f"I have reviewed only the {{account_label}} orders.",
                key=f"confirm_{{name}}",
            )
            disabled = not confirm or name not in submitters
            if name not in submitters:
                st.warning("No submitter is configured for this account.")
            if st.button(f"Submit {{account_label}} Orders", type="primary", disabled=disabled, key=f"submit_{{name}}"):
                if submitters[name] == "alpaca":
                    client = alpaca_client_from_env(account_prefixes[name])
                    result = submit_alpaca_orders(client, orders)
                elif submitters[name] == "robinhood_option":
                    result = submit_robinhood_option_orders(orders)
                else:
                    result = pd.DataFrame()
                st.write(f"{{name}}: {{len(result)}} response row(s)")
                st.dataframe(result, width="stretch", hide_index=True)
'''
    output.write_text(script, encoding="utf-8")
    return output


def _normalize_symbols(symbols: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()))


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    return pd.read_csv(path)


def _enrich_alpaca_option_records(client: Any, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    cache: dict[str, dict[str, Any]] = {}
    for raw in records:
        record = dict(raw)
        symbol = str(record.get("symbol") or "").strip().upper()
        asset_class = str(record.get("asset_class") or "").strip().lower()
        if not symbol or (asset_class and asset_class not in {"us_option", "option"}):
            continue
        try:
            if symbol not in cache:
                cache[symbol] = client.get_option_contract(symbol)
            contract = cache[symbol]
        except Exception:
            contract = {}
        record["underlying_symbol"] = str(contract.get("underlying_symbol") or "").strip().upper()
        record["option_type"] = str(contract.get("type") or "").strip().lower()
        record["expiry_date"] = contract.get("expiration_date")
        record["strike_price"] = contract.get("strike_price")
        enriched.append(record)
    return enriched
