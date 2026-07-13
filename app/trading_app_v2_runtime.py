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


def resolve_option_training_panel(artifact_dir: Path, *, min_market_cap: int) -> Path:
    """Return the newest successful unified option panel for an exact universe."""

    root = Path(artifact_dir).expanduser().resolve()
    matches: list[tuple[int, Path]] = []
    for summary_path in root.glob("*/run_summary.json"):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if summary.get("status") != "ok":
            continue
        if int(summary.get("min_market_cap", -1)) != int(min_market_cap):
            continue
        panel = summary_path.parent / "option_candidate_panel_unified.parquet"
        if not panel.exists():
            continue
        try:
            # Inspect the Parquet schema without materializing the panel.
            import pyarrow.parquet as pq

            columns = set(pq.ParquetFile(panel).schema.names)
        except (OSError, ValueError):
            continue
        if {"rank_y", "label_basis"}.issubset(columns):
            matches.append((summary_path.stat().st_mtime_ns, panel))
    if not matches:
        raise FileNotFoundError(
            "No successful unified option candidate panel found for "
            f"min_market_cap={int(min_market_cap)} under {root}. "
            "Run scripts/run_option_meta.py with this --min-market-cap first."
        )
    return max(matches, key=lambda item: item[0])[1]


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
    )
    latest_by_symbol["direction"] = latest_by_symbol["prob_buy"].ge(latest_by_symbol["prob_short"]).map({True: "long", False: "short"})
    latest_by_symbol["confidence"] = latest_by_symbol[["prob_buy", "prob_short"]].max(axis=1)
    latest_by_symbol = latest_by_symbol.sort_values(
        ["confidence", "best_family_score"], ascending=[False, False], kind="stable"
    ).reset_index(drop=True)
    latest_by_symbol["rank"] = latest_by_symbol.index + 1
    price_map = latest_prices_from_quant_warehouse(latest_by_symbol["symbol"], provider=price_provider)
    latest_by_symbol["close"] = latest_by_symbol["symbol"].map(price_map)
    latest_by_symbol["eligible"] = latest_by_symbol["close"].gt(0) & latest_by_symbol["confidence"].ge(float(min_long_score))
    latest_by_symbol["capacity_rank"] = pd.NA
    eligible_index = latest_by_symbol.index[latest_by_symbol["eligible"]]
    latest_by_symbol.loc[eligible_index, "capacity_rank"] = range(1, len(eligible_index) + 1)
    latest_by_symbol["selected"] = latest_by_symbol["eligible"] & pd.to_numeric(
        latest_by_symbol["capacity_rank"], errors="coerce"
    ).le(int(top_k))
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
        # Signals are formed after the score-date close and executed on a later
        # session, so a contract expiring on the score date is already unusable.
        featured = featured.loc[featured["snapshot_date"].eq(score_ts) & featured["expiration"].gt(score_ts)].copy()
        if featured.empty:
            continue
        prob_buy = _number(row.get("prob_buy"))
        prob_short = _number(row.get("prob_short"))
        featured["option_type"] = featured["option_type"].astype(str).str.lower().str.strip()
        featured = featured.loc[featured["option_type"].isin({"call", "put"})].copy()
        if featured.empty:
            continue
        featured["symbol"] = symbol
        featured["side"] = featured["option_type"].map({"call": "long", "put": "short"})
        featured["equity_signal_side"] = featured["side"]
        featured["trade_id"] = featured["option_type"].map(
            lambda option_type: f"live|{symbol}|{score_ts.date()}|{option_type}"
        )
        featured["entry_date"] = score_ts
        featured["option_action"] = featured["option_type"].map({"call": "buy_call", "put": "buy_put"})
        featured["entry_mid"] = pd.to_numeric(featured.get("mid"), errors="coerce")
        frames.append(featured)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def select_optionable_leaderboard(
    leaderboard: pd.DataFrame,
    *,
    score_date: str | pd.Timestamp,
    top_k: int = 20,
) -> pd.DataFrame:
    """Select the highest-ranked underlyings with a locally available score-date chain."""

    from quant_warehouse.platforms.data_providers.thetadata.options import read_thetadata_eod_option_chain

    selected_rows = []
    score_ts = pd.Timestamp(score_date).normalize()
    for row in leaderboard.sort_values("rank", kind="stable").to_dict("records"):
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
        if chain is None or chain.empty:
            continue
        selected_rows.append(row)
        if len(selected_rows) >= int(top_k):
            break
    out = pd.DataFrame(selected_rows)
    if not out.empty:
        out["selected"] = True
        out["eligible"] = True
        out["option_rank"] = range(1, len(out) + 1)
    return out


def build_score_date_option_ml_ranking_table(
    option_ranker_dir: Path,
    *,
    leaderboard: pd.DataFrame,
    score_date: str | pd.Timestamp | None = None,
    symbols: Sequence[str] | None = None,
    target_dte: int = 90,
    min_market_cap: int = 1_000_000_000_000,
    start_date: str = "1900-01-01",
    max_underlyings: int = 20,
    equity_family_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    requested_symbols = _normalize_symbols(symbols or ())
    if len(requested_symbols) > int(max_underlyings):
        raise ValueError(
            f"Option score-date refresh requested {len(requested_symbols)} underlyings; "
            f"limit is {int(max_underlyings)}. Score equities and select top-K first."
        )
    root = Path(option_ranker_dir)
    candidates = build_score_date_option_candidate_panel(
        leaderboard=leaderboard,
        score_date=score_date,
        symbols=symbols,
        target_dte=target_dte,
    )
    if candidates.empty:
        return pd.DataFrame()
    meta_model_path = root / "option_meta_stack" / "meta_stack_ranker.pkl"
    if meta_model_path.exists():
        if equity_family_scores is None or equity_family_scores.empty:
            raise ValueError("Meta-stack option scoring requires current equity_family_scores")
        from quant_orchestrator.research_tools import score_option_meta_ranker

        return score_option_meta_ranker(meta_model_path, candidates, equity_family_scores)
    family_dirs = sorted(path for path in root.iterdir() if path.is_dir()) if root.exists() else []
    if not family_dirs:
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
    out = frame.set_index("symbol")[["close", "prob_buy", "prob_short", "direction", "selected"]].copy()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["prob_buy"] = pd.to_numeric(out["prob_buy"], errors="coerce")
    out["prob_short"] = pd.to_numeric(out["prob_short"], errors="coerce")
    out["selected"] = out["selected"].astype(bool)
    return out


def alpaca_client_from_env(prefix: str):
    from platforms.brokers.alpaca import AlpacaPaperClient

    if "PYTEST_CURRENT_TEST" not in os.environ:
        try:
            from dotenv import load_dotenv

            load_dotenv(find_repo_root(Path(__file__).resolve()) / ".env", override=False)
        except Exception:
            pass

    clean = str(prefix).strip().upper()
    key = os.getenv(f"{clean}_ALPACA_PAPER_API_KEY") or os.getenv(f"ALPACA_{clean}_PAPER_API_KEY")
    secret = os.getenv(f"{clean}_ALPACA_PAPER_API_SECRET") or os.getenv(f"ALPACA_{clean}_PAPER_API_SECRET")
    if not key or not secret:
        raise RuntimeError(
            f"Missing dedicated Alpaca paper credentials for prefix={prefix!r}; "
            "generic ALPACA_PAPER credentials are not accepted for multi-account trading."
        )
    return AlpacaPaperClient(api_key=str(key), api_secret=str(secret))


def load_distinct_alpaca_paper_accounts(
    prefixes: Sequence[str] = ("EQUITY", "OPTION", "LLM"),
) -> dict[str, Any]:
    """Load dedicated paper clients and fail closed unless every account is distinct."""

    normalized = [str(prefix).strip().upper() for prefix in prefixes]
    if len(normalized) != 3 or len(set(normalized)) != 3 or any(not prefix for prefix in normalized):
        raise ValueError("Exactly three distinct Alpaca account prefixes are required.")

    clients = {prefix: alpaca_client_from_env(prefix) for prefix in normalized}
    credential_pairs = {(client.api_key, client.api_secret) for client in clients.values()}
    if len(credential_pairs) != len(clients):
        raise RuntimeError("Alpaca paper credentials must be distinct for EQUITY, OPTION, and LLM accounts.")

    account_ids: dict[str, str] = {}
    for prefix, client in clients.items():
        account_id = str(client.get_account().get("id") or "").strip()
        if not account_id:
            raise RuntimeError(f"Alpaca paper account for prefix={prefix!r} did not return an account ID.")
        account_ids[prefix] = account_id
    if len(set(account_ids.values())) != len(account_ids):
        raise RuntimeError("EQUITY, OPTION, and LLM credentials must resolve to distinct Alpaca account IDs.")
    return clients


def build_alpaca_equity_orders(
    *,
    leaderboard: pd.DataFrame,
    account_prefix: str,
    gross_exposure: float = 0.95,
    liquidate_unselected: bool = True,
) -> pd.DataFrame:
    from platforms.brokers.alpaca import build_directional_equity_order_plan

    client = alpaca_client_from_env(account_prefix)
    account = client.get_account()
    open_orders = client.get_open_orders()
    positions = {
        str(row.get("symbol") or "").strip().upper(): float(row.get("qty") or 0.0)
        for row in client.get_positions()
        if str(row.get("asset_class") or "us_equity").lower() in {"us_equity", "equity", ""}
    }
    eligible_rows = leaderboard.loc[
        leaderboard.get("eligible", pd.Series(True, index=leaderboard.index)).astype(bool)
    ]
    directions = eligible_rows[["symbol", "direction"]].to_dict(orient="records")
    scored_symbols = {str(row["symbol"]).strip().upper() for row in directions}
    directions.extend(
        {"symbol": symbol, "direction": "exit"}
        for symbol in positions
        if symbol not in scored_symbols
    )
    prices = dict(zip(eligible_rows["symbol"].astype(str).str.upper(), pd.to_numeric(eligible_rows["close"], errors="coerce")))
    orders = build_directional_equity_order_plan(
        directions,
        prices,
        positions,
        portfolio_value=float(account.get("portfolio_value") or account.get("equity") or 0.0),
        max_positions=20,
        gross_exposure=float(gross_exposure),
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
                    "time_in_force": "gtc",
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
                "time_in_force": "gtc",
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
    discount_pct: float,
    account_number: str | None = None,
    current_option_positions: pd.DataFrame | None = None,
    pending_option_orders: pd.DataFrame | None = None,
    strategy_allocation: float = 100_000.0,
) -> dict[str, pd.DataFrame]:
    """Reconcile Robinhood option account state before creating new live orders."""

    from platforms.brokers import robinhood

    if current_option_positions is None or pending_option_orders is None:
        robinhood.robinhood_login()

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
            if target is not None and _same_option_contract(position, target):
                held_target_symbols.add(symbol)
                continue
            sell_row = {
                "symbol": symbol,
                "underlying_symbol": str(position.get("underlying_symbol") or symbol).strip().upper(),
                "action": "sell_to_close_put" if str(position.get("option_type") or "").lower() == "put" else "sell_to_close_call",
                "reason": "Existing Robinhood option position is no longer the target contract.",
                "quantity": quantity,
                "qty": quantity,
                "side": "sell",
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
            target_type = str(target.get("option_type") or "call").lower() if target is not None else ""
            if target is not None and _same_option_contract(order, target) and action == f"buy_to_open_{target_type}":
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
        bid = _first_positive(target, ("bid_price", "bid"))
        discounted_bid = bid * (1.0 - float(discount_pct) / 100.0) if bid is not None else None
        quantity = option_contract_quantity(
            account_value=float(strategy_allocation),
            option_price=discounted_bid,
            max_underlyings=20,
        )
        if quantity <= 0:
            # Preserve an explicitly sized target only when no live bid is
            # available; otherwise never fall back to an oversized one-lot.
            quantity = int(_number(target.get("quantity", target.get("target_contracts")))) if bid is None else 0
        if quantity <= 0:
            continue
        option_type = str(target.get("option_type") or "call").strip().lower()
        buy_row = {
            **target.to_dict(),
            "symbol": symbol,
            "underlying_symbol": symbol,
            "action": f"buy_to_open_{option_type}",
            "reason": "New current top-K trading_app_v2 Robinhood option target.",
            "quantity": quantity,
            "qty": quantity,
            "side": "buy",
            "option_type": option_type,
            "order_type": "limit",
            "time_in_force": "gtc",
        }
        priced = apply_option_limit_policy(
            pd.DataFrame([buy_row]),
            time_in_force="gtc",
            discount_pct=min(float(discount_pct), 99.99),
        )
        priced = apply_robinhood_submission_gate(priced, discount_pct=discount_pct)
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
            "buy_to_open_put": 5,
        }
        actions["_priority"] = actions["action"].map(priority).fillna(99)
        actions = actions.sort_values(["_priority", "combined_score", "symbol"], ascending=[True, False, True], kind="stable").drop(columns=["_priority"])
    else:
        actions["skip_submit"] = pd.Series(dtype=bool)

    # Keep the Robinhood view aligned with Alpaca: execution-critical fields
    # come first, while the wide research/features payload remains available to
    # the right for auditability.
    # Normalize the exported schema to Alpaca's order schema. Robinhood's API
    # still receives its underlying chain symbol through the adapter below.
    if "underlying_symbol" not in actions.columns:
        actions["underlying_symbol"] = actions.get("symbol", "")
    if "contract_symbol" in actions.columns:
        actions["symbol"] = actions["contract_symbol"].where(
            actions["contract_symbol"].astype(str).str.strip().ne(""), actions["symbol"]
        )
    if "qty" not in actions.columns and "quantity" in actions.columns:
        actions["qty"] = actions["quantity"]
    actions = actions.drop(columns=[column for column in ("contract_symbol", "quantity", "expiration", "strike") if column in actions.columns])
    display_priority = [
        "symbol", "underlying_symbol", "option_type", "action", "side", "qty",
        "bid_price", "ask_price", "skip_submit", "skip_reason", "order_type", "time_in_force",
        "limit_price", "limit_order_price", "price", "limit_price_source", "live_quote_priced_at",
        "expiry_date", "dte", "strike_price",
        "discount_pct", "skip_submit", "skip_reason", "reason",
    ]
    ordered_columns = [column for column in display_priority if column in actions.columns]
    ordered_columns.extend(column for column in actions.columns if column not in ordered_columns)
    actions = actions.loc[:, ordered_columns]

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
                "discount_pct": float(discount_pct),
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


def apply_robinhood_submission_gate(
    orders: pd.DataFrame,
    *,
    discount_pct: float,
) -> pd.DataFrame:
    """Apply a Robinhood-only gate without mutating any paper-account plan."""

    gate = float(discount_pct)
    if not 0.0 <= gate <= 100.0:
        raise ValueError("discount_pct must be in [0, 100]")
    out = pd.DataFrame() if orders is None else orders.copy()
    if out.empty:
        return out
    out["discount_pct"] = gate
    if "skip_submit" not in out.columns:
        out["skip_submit"] = False
    if "skip_reason" not in out.columns:
        out["skip_reason"] = ""
    trade_mask = ~out.get("action", pd.Series("", index=out.index)).astype(str).str.startswith("cancel_")
    if gate >= 100.0:
        out.loc[trade_mask, "skip_submit"] = True
        out.loc[trade_mask, "skip_reason"] = "robinhood_gate_100_blocks_orders"
    return out


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
    if "quantity" not in out.columns:
        # Quantity is recomputed from the live discounted bid by the planner;
        # retain the target rows through normalization with a neutral placeholder.
        out["quantity"] = pd.Series(1, index=out.index, dtype="int64")
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


def _option_contract_quantity(
    *,
    account_value: float,
    option_price: float | None,
    max_underlyings: int,
    contract_multiplier: int = 100,
) -> int:
    """Return contracts for one equal-dollar option sleeve.

    Option quotes are per share, whereas Alpaca quantities are contracts. The
    standard 100-share multiplier is applied so each entry targets
    ``account_value / max_underlyings`` dollars of premium.
    """
    value = _number(account_value, default=float("nan"))
    price = _number(option_price, default=float("nan"))
    capacity = int(max_underlyings)
    multiplier = int(contract_multiplier)
    if capacity <= 0 or multiplier <= 0:
        raise ValueError("max_underlyings and contract_multiplier must be positive")
    if not math.isfinite(value) or value <= 0 or not math.isfinite(price) or price <= 0:
        return 0
    return max(0, int(math.floor((value / capacity) / (price * multiplier))))


def option_contract_quantity(
    *,
    account_value: float,
    option_price: float | None,
    max_underlyings: int = 20,
    contract_multiplier: int = 100,
) -> int:
    """Public sizing helper for Alpaca and discounted Robinhood targets."""
    return _option_contract_quantity(
        account_value=account_value,
        option_price=option_price,
        max_underlyings=max_underlyings,
        contract_multiplier=contract_multiplier,
    )


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


def build_ranked_alpaca_option_orders(
    *,
    option_rankings: pd.DataFrame,
    decisions: pd.DataFrame,
    account_prefix: str,
    decision_col: str = "direction",
    llm: bool = False,
    max_underlyings: int = 20,
    strategy_allocation: float | None = 100_000.0,
) -> pd.DataFrame:
    """Reconcile prior-day symbol/contract selections, then price with live Alpaca quotes."""

    from platforms.brokers.alpaca import build_directional_option_order_plan, build_llm_option_order_plan

    if option_rankings is None or option_rankings.empty:
        return pd.DataFrame()
    ranked = option_rankings.copy()
    selected_mask = ranked.get("selected_by_option_ensemble", pd.Series(False, index=ranked.index)).astype(bool)
    selected = ranked.loc[selected_mask].copy()
    selected["underlying_symbol"] = selected["symbol"].astype(str).str.upper()
    if "contract_symbol" not in selected.columns:
        raise KeyError("option rankings require contract_symbol")
    client = alpaca_client_from_env(account_prefix)
    account = client.get_account()
    account_value = float(account.get("equity") or account.get("portfolio_value") or account.get("cash") or 0.0)
    if account_value <= 0:
        raise ValueError(f"Alpaca account {account_prefix!r} has no positive equity/portfolio value")
    if strategy_allocation is not None:
        strategy_allocation = float(strategy_allocation)
        if strategy_allocation <= 0:
            raise ValueError("strategy_allocation must be positive when provided")
        account_value = min(account_value, strategy_allocation)
    current_positions = _enrich_alpaca_option_records(client, client.get_positions())
    normalized_decisions = decisions.copy()
    held_underlyings = {
        str(row.get("underlying_symbol") or "").strip().upper()
        for row in current_positions
        if str(row.get("underlying_symbol") or "").strip()
    }
    normalized_decisions["symbol"] = normalized_decisions["symbol"].astype(str).str.upper()
    if decision_col != "direction":
        normalized_decisions["decision"] = normalized_decisions[decision_col]
    planner = build_llm_option_order_plan if llm else build_directional_option_order_plan
    contract_symbols = list(dict.fromkeys(
        [str(value).strip().upper() for value in selected["contract_symbol"].tolist()]
        + [str(row.get("contract_symbol") or row.get("symbol") or "").strip().upper() for row in current_positions]
    ))
    snapshots = client.get_option_snapshots(contract_symbols)
    quote_rows = []
    for contract_symbol in contract_symbols:
        bid, ask, _mark = _option_quote(snapshots.get(contract_symbol, {}))
        quote_rows.append({"symbol": contract_symbol, "bid_price": bid, "ask_price": ask})
    quote_frame = pd.DataFrame(quote_rows)
    quote_by_symbol = quote_frame.set_index("symbol").to_dict(orient="index") if not quote_frame.empty else {}
    sized_rows: list[dict[str, Any]] = []
    for row in selected.to_dict(orient="records"):
        quote = quote_by_symbol.get(str(row["contract_symbol"]).upper(), {})
        # Use ask for a conservative budget; bid is only a missing-ask fallback.
        price = _first_positive(quote, ("ask_price", "bid_price"))
        quantity = _option_contract_quantity(
            account_value=account_value,
            option_price=price,
            max_underlyings=int(max_underlyings),
        )
        if quantity <= 0:
            continue
        row["qty"] = quantity
        row["target_notional"] = float(account_value) / int(max_underlyings)
        row["estimated_entry_notional"] = float(price) * 100.0 * quantity
        sized_rows.append(row)
    selected_contracts = sized_rows
    available_underlyings = {str(row["underlying_symbol"]).upper() for row in sized_rows}
    normalized_decisions = normalized_decisions.loc[
        normalized_decisions["symbol"].isin(available_underlyings | held_underlyings)
    ].copy()
    # Re-run reconciliation after sizing so an unpriceable/over-budget contract
    # is omitted rather than accidentally falling back to a one-contract order.
    raw_orders = planner(
        normalized_decisions.to_dict(orient="records"),
        selected_contracts,
        current_positions,
        max_underlyings=int(max_underlyings),
    )
    if not raw_orders:
        return pd.DataFrame()
    intents = pd.DataFrame(raw_orders)
    contract_symbols = intents.loc[
        ~intents["action"].astype(str).str.startswith("cancel_"), "symbol"
    ].astype(str).str.upper().unique().tolist()
    quote_frame = quote_frame.loc[quote_frame["symbol"].isin(contract_symbols)].copy()
    return generate_live_option_limit_prices(
        intents,
        quote_frame,
        time_in_force="gtc",
    )


def build_llm_ranked_option_orders(
    *,
    leaderboard: pd.DataFrame,
    option_rankings: pd.DataFrame,
    account_prefix: str,
    top_k: int = 20,
    as_of_date: str | None = None,
    trading_agents_config: Any | None = None,
    strategy_allocation: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from platforms.agents.trading_agents import review_trade_candidates

    candidates = leaderboard.head(int(top_k)).copy()
    reviewed = review_trade_candidates(candidates, as_of_date=as_of_date, config=trading_agents_config)
    rating = reviewed.get("llm_rating", pd.Series("Hold", index=reviewed.index)).astype(str).str.lower()
    reviewed["decision"] = rating.map(
        {
            "buy": "buy",
            "overweight": "buy",
            "sell": "sell",
            "underweight": "sell",
            "hold": "hold",
        }
    ).fillna("hold")
    orders = build_ranked_alpaca_option_orders(
        option_rankings=option_rankings,
        decisions=reviewed[["symbol", "decision"]],
        account_prefix=account_prefix,
        decision_col="decision",
        llm=True,
        max_underlyings=int(top_k),
        strategy_allocation=strategy_allocation,
    )
    return orders, reviewed


def apply_option_limit_policy(
    orders: pd.DataFrame,
    *,
    time_in_force: str | None = "gtc",
    discount_pct: float = 0.0,
    priced_at: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Set option limit prices from the executable side of the quote.

    Buy-to-open orders bid. Sell-to-close orders ask. Cancels pass through.
    """

    discount = float(discount_pct)
    if not 0.0 <= discount < 100.0:
        raise ValueError("discount_pct must be in [0, 100)")
    if orders is None or orders.empty:
        return pd.DataFrame() if orders is None else orders.copy()
    work = orders.copy()
    price_timestamp = pd.Timestamp.now(tz="UTC") if priced_at is None else pd.Timestamp(priced_at)
    price_timestamp = (
        price_timestamp.tz_localize("UTC")
        if price_timestamp.tzinfo is None
        else price_timestamp.tz_convert("UTC")
    )
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
            if price is not None:
                price *= 1.0 - (discount / 100.0)
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
        work.at[idx, "live_quote_priced_at"] = price_timestamp.isoformat()
        work.at[idx, "discount_pct"] = discount if pricing_side == "buy" else 0.0
        work.at[idx, "skip_submit"] = False
        work.at[idx, "skip_reason"] = ""
    return work


def generate_live_option_limit_prices(
    order_intents: pd.DataFrame,
    live_quotes: pd.DataFrame,
    *,
    discount_pct: float = 0.0,
    time_in_force: str = "gtc",
    priced_at: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Join live quotes to already-selected contracts and generate limit prices.

    The input intents own symbol and contract selection. This function only adds
    the current bid/ask and derives an executable limit; it never reranks.
    """

    if order_intents is None or order_intents.empty:
        return pd.DataFrame() if order_intents is None else order_intents.copy()
    intents = order_intents.copy()
    quotes = live_quotes.copy() if live_quotes is not None else pd.DataFrame()
    key = "contract_symbol" if "contract_symbol" in intents.columns else "symbol"
    quote_key = "contract_symbol" if "contract_symbol" in quotes.columns else "symbol"
    if key not in intents.columns:
        raise KeyError("option order intents require symbol or contract_symbol")
    required_quotes = {quote_key, "bid_price", "ask_price"}
    missing = required_quotes.difference(quotes.columns)
    if missing:
        raise KeyError(f"live option quotes missing columns: {sorted(missing)}")
    if quotes[quote_key].astype(str).duplicated().any():
        raise ValueError("live option quotes must contain one row per contract")
    quote_columns = quotes[[quote_key, "bid_price", "ask_price"]].rename(columns={quote_key: key})
    intents = intents.drop(columns=[col for col in ("bid_price", "ask_price") if col in intents.columns])
    priced = intents.merge(quote_columns, on=key, how="left", validate="many_to_one")
    return apply_option_limit_policy(
        priced,
        time_in_force=time_in_force,
        discount_pct=discount_pct,
        priced_at=priced_at,
    )


def validate_prior_day_selection(
    frame: pd.DataFrame,
    *,
    selection_date_col: str,
    live_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fail closed unless every model/contract selection predates live pricing."""

    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    if selection_date_col not in frame.columns:
        raise KeyError(f"selection frame missing {selection_date_col!r}")
    selected = pd.to_datetime(frame[selection_date_col], errors="coerce").dt.normalize()
    if selected.isna().any():
        raise ValueError(f"selection frame contains an invalid {selection_date_col}")
    current = pd.Timestamp.now(tz="UTC") if live_date is None else pd.Timestamp(live_date)
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    if selected.ge(current.tz_localize(None).normalize()).any():
        raise ValueError("symbol and option selections must use completed prior-day data")
    return frame.copy()


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

    duplicate_cols = [col for col in ("action", "symbol", "side", "qty", "order_id") if col in actionable.columns]
    if duplicate_cols and actionable.duplicated(subset=duplicate_cols, keep=False).any():
        raise ValueError(f"Refusing duplicate order rows keyed by {duplicate_cols}.")

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
        order_types = trade_rows.get("order_type", pd.Series("", index=trade_rows.index)).astype(str).str.lower().str.strip()
        if str(asset_type).lower() == "option":
            if not order_types.eq("limit").all():
                raise ValueError("Refusing an option order that is not a limit order.")
            limit_prices = pd.to_numeric(trade_rows.get("limit_price"), errors="coerce")
            if limit_prices.isna().any() or limit_prices.le(0).any():
                raise ValueError("Refusing an option order without a positive limit_price.")
        elif not order_types.eq("market").all():
            raise ValueError("Refusing an equity order that is not a market order.")

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
    existing_orders = list(client.get_open_orders()) if hasattr(client, "get_open_orders") else []
    existing_positions = list(client.get_positions()) if hasattr(client, "get_positions") else []
    open_keys = {
        (str(row.get("symbol") or "").upper(), str(row.get("side") or "").lower(), str(row.get("type") or row.get("order_type") or "").lower())
        for row in existing_orders
    }
    held_symbols = {str(row.get("symbol") or "").upper() for row in existing_positions}
    candidate = orders.copy()
    duplicate_mask = candidate.apply(
        lambda row: (
            (str(row.get("symbol") or "").upper(), str(row.get("side") or "").lower(), str(row.get("order_type") or "").lower()) in open_keys
            or (str(row.get("symbol") or "").upper() in held_symbols and str(row.get("side") or "").lower() == "buy")
        ),
        axis=1,
    )
    candidate = candidate.loc[~duplicate_mask].copy()
    actionable = validate_order_plan_for_submission(candidate, asset_type=asset_type)
    responses = client.submit_orders(actionable.to_dict(orient="records"))
    return pd.DataFrame(responses)


def submit_robinhood_option_orders(orders: pd.DataFrame, *, account_number: str | None = None) -> pd.DataFrame:
    if orders is None or orders.empty:
        return pd.DataFrame()
    # Robinhood caps each option order at 100 contracts. Split only at the
    # submission boundary so the displayed plan still shows the full target.
    validated_parts: list[pd.DataFrame] = []
    for _, row in orders.iterrows():
        row_frame = pd.DataFrame([row.to_dict()])
        if str(row.get("action") or "").lower().startswith("cancel_"):
            chunks = [row_frame]
        else:
            total = int(pd.to_numeric(pd.Series([row.get("qty")]), errors="coerce").iloc[0])
            chunks = []
            while total > 0:
                chunk = row_frame.copy()
                size = min(total, 100)
                chunk.loc[:, "qty"] = size
                chunks.append(chunk)
                total -= size
        for chunk in chunks:
            validated_parts.append(validate_order_plan_for_submission(chunk, asset_type="option"))
    actionable = pd.concat(validated_parts, ignore_index=True) if validated_parts else pd.DataFrame()
    if actionable.empty:
        return actionable
    broker_orders = actionable.copy()
    # The displayed plan follows Alpaca's schema; translate to Robinhood's
    # chain-symbol/quantity fields only at the broker boundary.
    if "underlying_symbol" in broker_orders.columns:
        broker_orders["symbol"] = broker_orders["underlying_symbol"]
    if "qty" in broker_orders.columns:
        broker_orders["quantity"] = broker_orders["qty"]
    from platforms.brokers.robinhood import submit_robinhood_option_orders as _submit
    from platforms.brokers import robinhood

    # Authenticate only at the explicit submit boundary; plan regeneration is
    # intentionally non-blocking and does not perform Robinhood login.
    robinhood.robinhood_login()

    return _submit(orders_df=broker_orders, account_number=account_number, time_in_force="gtc")


def regenerate_order_plan_from_account_state(
    order_frames: Mapping[str, pd.DataFrame],
    *,
    max_positions: int = 20,
    account_state: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Refresh a displayed plan without submitting orders.

    Alpaca plans are suppressed when the account already has capacity occupied
    by open orders/positions. Robinhood rows are retained but repriced from
    fresh Robinhood bid/ask quotes.
    """
    refreshed = {str(name): frame.copy() for name, frame in order_frames.items()}
    account_prefixes = {
        "alpaca_equity_paper": "EQUITY",
        "alpaca_option_paper": "OPTION",
        "alpaca_llm_paper": "LLM",
    }
    for name, prefix in account_prefixes.items():
        if name not in refreshed:
            continue
        client = alpaca_client_from_env(prefix)
        if account_state is not None:
            occupied = len(account_state.get(f"{name}_orders", pd.DataFrame())) + len(account_state.get(f"{name}_positions", pd.DataFrame()))
        else:
            occupied = len(client.get_open_orders()) + len(client.get_positions())
        # Regeneration is a reconciliation pass, not a second entry pass:
        # any existing Alpaca order/position means there is nothing new to
        # submit for that account in this snapshot.
        if occupied > 0:
            refreshed[name] = refreshed[name].iloc[0:0].copy()

    robinhood_name = "robinhood_option_real"
    if robinhood_name in refreshed and not refreshed[robinhood_name].empty:
        fresh = refreshed[robinhood_name].copy()
        discount = float(pd.to_numeric(fresh.get("discount_pct", 90.0), errors="coerce").dropna().iloc[0]) if "discount_pct" in fresh.columns and pd.to_numeric(fresh["discount_pct"], errors="coerce").notna().any() else 90.0
        refreshed[robinhood_name] = apply_option_limit_policy(fresh, time_in_force="gtc", discount_pct=discount)
    return refreshed


def load_account_state_snapshot() -> dict[str, pd.DataFrame]:
    """Load existing broker orders and positions for frontend reconciliation."""
    state: dict[str, pd.DataFrame] = {}
    for name, prefix in {
        "alpaca_equity_paper": "EQUITY",
        "alpaca_option_paper": "OPTION",
        "alpaca_llm_paper": "LLM",
    }.items():
        client = alpaca_client_from_env(prefix)
        state[f"{name}_orders"] = pd.DataFrame(client.get_open_orders())
        state[f"{name}_positions"] = pd.DataFrame(client.get_positions())
    # Robinhood authentication can block for an extended period. Keep its
    # state out of this synchronous refresh; its displayed plan is repriced
    # from the cached quote fields and submission remains an explicit action.
    state["robinhood_option_real_orders"] = pd.DataFrame()
    state["robinhood_option_real_positions"] = pd.DataFrame()
    return state


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

from app.trading_app_v2_runtime import alpaca_client_from_env, load_account_state_snapshot, regenerate_order_plan_from_account_state, submit_alpaca_orders, submit_robinhood_option_orders

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

symbol_tab, option_tab, orders_tab = st.tabs(["Symbol Scores", "Option ML Rankings", "Orders / Positions"])

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
    if "regenerated_order_frames" in st.session_state:
        order_frames = st.session_state["regenerated_order_frames"]
    existing_state = st.session_state.get("existing_account_state", {{}})
    if existing_state:
        st.subheader("Existing Broker State")
        for account_name in sorted({{name.removesuffix("_orders").removesuffix("_positions") for name in existing_state}}):
            with st.expander(account_name.replace("_", " ").title(), expanded=False):
                st.caption("Existing open orders")
                st.dataframe(existing_state.get(f"{{account_name}}_orders", pd.DataFrame()), width="stretch", hide_index=True)
                st.caption("Existing positions")
                st.dataframe(existing_state.get(f"{{account_name}}_positions", pd.DataFrame()), width="stretch", hide_index=True)
    account_names = sorted(order_frames)
    account_tabs = st.tabs([name.replace("_", " ").title() for name in account_names]) if account_names else []
    for account_tab, name in zip(account_tabs, account_names):
        with account_tab:
            frame = order_frames[name]
            st.subheader(name.replace("_", " ").title())
            if frame.empty:
                st.info("No orders for this account.")
            else:
                st.dataframe(frame, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Submit All Orders")
    account_prefixes = {{
        "alpaca_equity_paper": "EQUITY",
        "alpaca_option_paper": "OPTION",
        "alpaca_llm_paper": "LLM",
    }}
    alpaca_asset_types = {{
        "alpaca_equity_paper": "equity",
        "alpaca_option_paper": "option",
        "alpaca_llm_paper": "option",
    }}
    submitters = {{
        **{{name: "alpaca" for name in account_prefixes}},
        "robinhood_option_real": "robinhood_option",
    }}
    if st.button("Regenerate Plan", key="regenerate_plan"):
        with st.spinner("Refreshing account state and Robinhood quotes..."):
            state = load_account_state_snapshot()
            st.session_state["existing_account_state"] = state
            st.session_state["regenerated_order_frames"] = regenerate_order_plan_from_account_state(order_frames, account_state=state)
        st.success("Plan regenerated. The refreshed state will be shown on the next interaction.")
    confirm_all = st.checkbox(
        "I have reviewed all displayed orders and want to submit them to every configured account.",
        key="confirm_all_accounts",
    )
    if st.button("Submit All Account Orders", type="primary", disabled=not confirm_all, key="submit_all_accounts"):
        # Reconcile one final time immediately before any broker side effect;
        # this prevents a stale embedded plan from duplicating Alpaca orders.
        with st.spinner("Reconciling current account state before submission..."):
            submission_state = load_account_state_snapshot()
            order_frames = regenerate_order_plan_from_account_state(order_frames, account_state=submission_state)
        submission_results = {{}}
        for name in sorted(order_frames):
            orders = order_frames[name]
            if orders.empty or name not in submitters:
                continue
            try:
                if submitters[name] == "alpaca":
                    result = submit_alpaca_orders(
                        alpaca_client_from_env(account_prefixes[name]),
                        orders,
                        asset_type=alpaca_asset_types[name],
                    )
                else:
                    result = submit_robinhood_option_orders(orders)
                submission_results[name] = result
            except Exception as exc:
                submission_results[name] = pd.DataFrame([{{"error": f"{{type(exc).__name__}}: {{exc}}"}}])
        st.success("Submission attempt completed for all configured accounts.")
        for name, result in submission_results.items():
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

from app.trading_app_v2_runtime import alpaca_client_from_env, load_account_state_snapshot, regenerate_order_plan_from_account_state, submit_alpaca_orders, submit_robinhood_option_orders


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

symbol_tab, option_tab, orders_tab = st.tabs(["Symbol Scores", "Option ML Rankings", "Orders / Positions"])

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
        order_frames[path.stem.removesuffix("_orders")] = read_csv_if_nonempty(path)
    if "regenerated_order_frames" in st.session_state:
        order_frames = st.session_state["regenerated_order_frames"]
    existing_state = st.session_state.get("existing_account_state", {{}})
    if existing_state:
        st.subheader("Existing Broker State")
        for account_name in sorted({{name.removesuffix("_orders").removesuffix("_positions") for name in existing_state}}):
            with st.expander(account_name.replace("_", " ").title(), expanded=False):
                st.caption("Existing open orders")
                st.dataframe(existing_state.get(f"{{account_name}}_orders", pd.DataFrame()), width="stretch", hide_index=True)
                st.caption("Existing positions")
                st.dataframe(existing_state.get(f"{{account_name}}_positions", pd.DataFrame()), width="stretch", hide_index=True)
    account_names = sorted(order_frames)
    account_tabs = st.tabs([name.replace("_", " ").title() for name in account_names]) if account_names else []
    for account_tab, name in zip(account_tabs, account_names):
        with account_tab:
            frame = order_frames[name]
            st.subheader(name.replace("_", " ").title())
            if frame.empty:
                st.info("No orders for this account.")
            else:
                st.dataframe(frame, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Submit All Orders")
    account_prefixes = {{
        "alpaca_equity_paper": "EQUITY",
        "alpaca_option_paper": "OPTION",
        "alpaca_llm_paper": "LLM",
    }}
    alpaca_asset_types = {{
        "alpaca_equity_paper": "equity",
        "alpaca_option_paper": "option",
        "alpaca_llm_paper": "option",
    }}
    submitters = {{
        **{{name: "alpaca" for name in account_prefixes}},
        "robinhood_option_real": "robinhood_option",
    }}
    if st.button("Regenerate Plan", key="regenerate_plan"):
        with st.spinner("Refreshing account state and Robinhood quotes..."):
            state = load_account_state_snapshot()
            st.session_state["existing_account_state"] = state
            st.session_state["regenerated_order_frames"] = regenerate_order_plan_from_account_state(order_frames, account_state=state)
        st.success("Plan regenerated. The refreshed state will be shown on the next interaction.")
    confirm_all = st.checkbox(
        "I have reviewed all displayed orders and want to submit them to every configured account.",
        key="confirm_all_accounts",
    )
    if st.button("Submit All Account Orders", type="primary", disabled=not confirm_all, key="submit_all_accounts"):
        with st.spinner("Reconciling current account state before submission..."):
            submission_state = load_account_state_snapshot()
            order_frames = regenerate_order_plan_from_account_state(order_frames, account_state=submission_state)
        submission_results = {{}}
        for name in sorted(order_frames):
            orders = order_frames[name]
            if orders.empty or name not in submitters:
                continue
            try:
                if submitters[name] == "alpaca":
                    result = submit_alpaca_orders(
                        alpaca_client_from_env(account_prefixes[name]),
                        orders,
                        asset_type=alpaca_asset_types[name],
                    )
                else:
                    result = submit_robinhood_option_orders(orders)
                submission_results[name] = result
            except Exception as exc:
                submission_results[name] = pd.DataFrame([{{"error": f"{{type(exc).__name__}}: {{exc}}"}}])
        st.success("Submission attempt completed for all configured accounts.")
        for name, result in submission_results.items():
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
