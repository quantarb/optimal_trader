# modules/data/build.py
# ============================================================
# Data build layer extracted from modules/pipeline.py
#
# Responsibility:
#   - prices (SQLite-first) via DataContext.store + FMP fetch
#   - features (computed on-the-fly)
#   - events + labels per symbol
#   - training_df (concatenated across symbols)
#   - optional: global rank labels + expand into task rows
#
# This file MUST NOT:
#   - train models
#   - run backtests
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import os
import numpy as np
import pandas as pd

from modules.utils.normalize import normalize_cols
from modules.schema import (
    validate_daily_df,
    validate_labels_df,
    validate_feature_cols,
    align_dict_to_columns,
)

from modules.data.context import DataContext
from modules.data.prices_sqlite import load_or_fetch_prices_daily, load_or_fetch_prices_daily_fast
from modules.data.quality import DataQualityConfig, assess_and_clean_prices_daily
from features.technical import load_or_compute_features_daily
from modules.data.dataset_rows import build_event_training_dataset

from labels.strategy_solver import solve_joint_trades_by_frequency, solve_longs_by_frequency, solve_shorts_by_frequency
from labels.events import generate_optimal_events

# IMPORTANT:
# Your pasted modules/labeler.py contains add_rank_regression_labels,
# but pipeline.py currently imports from modules.ml.labeler.
# To preserve behavior, we mirror pipeline.py here.
from labels.directional import add_binary_classification_labels
from labels.ranking import add_rank_regression_labels


# --------------------------
# Types
# --------------------------
@dataclass
class SkipRecord:
    symbol: str
    stage: str
    error_type: str
    message: str


@dataclass
class SymbolResult:
    symbol: str
    daily_df: Optional[pd.DataFrame]
    training_df: Optional[pd.DataFrame]
    feature_cols: List[str]
    skipped: Optional[SkipRecord] = None


@dataclass
class BuildResult:
    daily_by_symbol: Dict[str, pd.DataFrame]
    training_df: pd.DataFrame
    feature_cols: List[str]
    skipped: List[SkipRecord]


# --------------------------
# Debug helper
# --------------------------
def _print_weight_debug(symbol: str, labels: pd.DataFrame) -> None:
    if "sample_weight" not in labels.columns:
        return

    cols = [c for c in ["trade_return", "sample_weight"] if c in labels.columns]
    print(f"\n[DEBUG:{symbol}] sample_weight summary:")
    if cols:
        print(labels[cols].describe())

    if "trade_return" in labels.columns:
        print(f"\n[DEBUG:{symbol}] top 5 trade_return rows:")
        print(labels.sort_values("trade_return").tail(5)[cols])

    if {"side", "horizon", "sample_weight"} <= set(labels.columns):
        print(f"\n[DEBUG:{symbol}] total weight mass per (side,horizon):")
        print(
            labels.groupby(["side", "horizon"])["sample_weight"]
            .sum()
            .sort_values(ascending=False)
        )


def _summarize_skips(skipped: List[SkipRecord], universe: List[str]) -> None:
    print("\n" + "=" * 90)
    print("[ERROR] No training data produced. All symbols were skipped.")
    print("=" * 90)

    print(f"[ERROR] total skipped: {len(skipped)} / {len(universe)}")

    if not skipped:
        print("[ERROR] No SkipRecords captured (unexpected).")
        return

    s = pd.Series([(r.stage, r.error_type) for r in skipped], dtype="object")
    counts = s.value_counts().head(20)

    print("\n[ERROR] Top skip categories (stage, error_type):")
    for (stage, etype), n in counts.items():
        print(f" - {stage:10s} | {etype:18s} | {n}")

    print("\n[ERROR] First 20 skip details:")
    for r in skipped[:20]:
        print(f" - {r.symbol} | stage={r.stage} | {r.error_type}: {r.message}")


# --------------------------
# REFACTOR: Shared Helper for Prices + Features
# --------------------------
def _process_symbol_features(
        symbol: str,
        ctx: DataContext,
        last_dt_hint: Optional[pd.Timestamp],
        debug_data_quality: bool,
        data_quality_overrides: Optional[Dict[str, Any]],
        use_fast_prices: bool,
        execution_params: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Core logic: Prices -> Clean -> Features.
    Returns (df_daily, feature_cols, stage_reached)
    """
    stage = "load"

    # default execution params if None
    ep = execution_params or {}
    price_col = str(ep.get("price_col", "close"))

    # 1) Prices (SQLite-first)
    if use_fast_prices:
        df_prices = load_or_fetch_prices_daily_fast(symbol, ctx=ctx, last_dt_hint=last_dt_hint)
    else:
        df_prices = load_or_fetch_prices_daily(symbol, ctx=ctx)

    df_prices = normalize_cols(df_prices)

    # 1.5) Data Quality
    dq_cfg = DataQualityConfig()
    if data_quality_overrides:
        allowed = set(dq_cfg.__dict__.keys())
        safe = {k: v for k, v in dict(data_quality_overrides).items() if k in allowed}
        if safe:
            dq_cfg = DataQualityConfig(**{**dq_cfg.__dict__, **safe})

    try:
        df_prices, _ = assess_and_clean_prices_daily(
            df_prices,
            symbol=str(symbol),
            price_col=price_col,
            cfg=dq_cfg,
            debug=bool(debug_data_quality),
        )
    except RuntimeError as e:
        raise RuntimeError(f"data_quality_reject: {e}") from e

    # 2) Features
    stage = "features"
    fr = load_or_compute_features_daily(symbol, df_prices=df_prices)
    df_daily = normalize_cols(fr.df_daily)

    # Validation
    if not isinstance(df_daily.index, pd.DatetimeIndex):
        if "date" in df_daily.columns:
            df_daily["date"] = pd.to_datetime(df_daily["date"], errors="coerce")
            df_daily = df_daily.set_index("date")
        else:
            raise RuntimeError("df_daily must have a DatetimeIndex or a 'date' column")

    df_daily = df_daily.sort_index()

    # Keep symbol column for panels
    df_daily["symbol"] = symbol
    validate_daily_df(df_daily, ctx=f"{symbol}:daily")

    return df_daily, list(fr.feature_cols), stage


# --------------------------
# Step 2a: One symbol (Full Pipeline)
# --------------------------
def process_symbol(
        symbol: str,
        *,
        ctx: DataContext,
        last_dt_hint: Optional[pd.Timestamp],
        k_params: Dict[str, int],
        execution_params: Dict[str, Any],
        weighting: Dict[str, Any],
        debug_data_quality: bool = False,
        data_quality_overrides: Optional[Dict[str, Any]] = None,
        skip_on_error: bool = True,
        debug_weights: bool = False,
        use_fast_prices: bool = True,
) -> SymbolResult:
    """
    Full pipeline: Prices -> Features -> Events -> Labels -> Training Data
    """
    stage = "init"
    try:
        # --- REUSED LOGIC START ---
        df_daily, feature_cols, stage = _process_symbol_features(
            symbol=symbol,
            ctx=ctx,
            last_dt_hint=last_dt_hint,
            debug_data_quality=debug_data_quality,
            data_quality_overrides=data_quality_overrides,
            use_fast_prices=use_fast_prices,
            execution_params=execution_params,
        )
        # --- REUSED LOGIC END ---

        # 3) Events
        stage = "events"
        events = generate_optimal_events(
            df_daily=df_daily,
            solve_longs_by_frequency_fn=solve_longs_by_frequency,
            solve_shorts_by_frequency_fn=solve_shorts_by_frequency,
            solve_joint_by_frequency_fn=solve_joint_trades_by_frequency,
            k_params=k_params,
            price_col=execution_params["price_col"],
            fee_bps=execution_params["fee_bps"],
            slippage_bps=execution_params["slippage_bps"],
        )

        if events is None or events.empty:
            raise RuntimeError("no events produced (solver inputs / columns issue)")

        # 4) Labels (directional + sample_weight)
        stage = "labels"
        labels = add_binary_classification_labels(events, **weighting)
        if labels is None or labels.empty:
            raise RuntimeError("no labels produced")

        labels = validate_labels_df(labels, ctx=f"{symbol}:labels")

        if debug_weights:
            _print_weight_debug(symbol, labels)

        # 5) Training dataset (join labels onto event dates)
        stage = "training"
        ds = build_event_training_dataset(
            df_features=df_daily,
            labels=labels,
            symbol=symbol,
        )

        return SymbolResult(
            symbol=symbol,
            daily_df=df_daily,
            training_df=ds.training_df,
            feature_cols=feature_cols,
            skipped=None,
        )

    except Exception as e:
        if not skip_on_error:
            raise

        rec = SkipRecord(
            symbol=symbol,
            stage=stage,
            error_type=type(e).__name__,
            message=str(e)[:800],
        )
        return SymbolResult(
            symbol=symbol,
            daily_df=None,
            training_df=None,
            feature_cols=[],
            skipped=rec,
        )


# --------------------------
# Step 2b: Multi-symbol builder (Full Pipeline)
# --------------------------
def build_training_and_daily(
        universe: List[str],
        *,
        api_key: str,
        data_dir: str,
        k_params: Dict[str, int],
        execution_params: Dict[str, Any],
        weighting: Dict[str, Any],
        db_name: str = "quant.db",
        sleep_s: float = 0.0,
        debug_first_symbol: bool = True,
        skip_on_error: bool = True,
        verbose_data: bool = True,
        history_years: Optional[int] = None,
        debug_data_quality: bool = False,
        data_quality_overrides: Optional[Dict[str, Any]] = None,
        add_rank_labels: bool = True,
        add_rank_tasks_to_mtl: bool = True,
) -> BuildResult:
    """
    Returns:
      - daily_by_symbol: dict[symbol]->daily features dataframe
      - training_df: global event-based training dataframe (optionally expanded to MTL tasks)
      - feature_cols: global feature columns
      - skipped: per-symbol skip records
    """
    os.makedirs(data_dir, exist_ok=True)

    ctx = DataContext.from_data_dir(
        api_key=api_key,
        data_dir=data_dir,
        db_name=db_name,
        sleep_s=sleep_s,
        verbose=verbose_data,
        history_years=history_years,
    )

    # Bulk query last_dt per symbol (for incremental price fetches)
    ctx.store.init_schema()
    last_df = ctx.store.get_last_price_dates_from_prices_for_symbols(universe)

    last_dt_map: Dict[str, Optional[pd.Timestamp]] = {s: None for s in universe}
    if last_df is not None and not last_df.empty:
        last_df = last_df.dropna(subset=["last_price_date"]).copy()
        for _, row in last_df.iterrows():
            sym = str(row["symbol"])
            dt = pd.to_datetime(row["last_price_date"], errors="coerce")
            if pd.notna(dt):
                last_dt_map[sym] = pd.Timestamp(dt).normalize()

    # If we want earliest-history backfill, force canonical loader for this run
    use_fast_prices = False if history_years else True

    daily_by_symbol: Dict[str, pd.DataFrame] = {}
    training_rows: List[pd.DataFrame] = []
    feature_cols_set: set[str] = set()
    skipped: List[SkipRecord] = []

    printed_debug = False

    for symbol in universe:
        r = process_symbol(
            symbol,
            ctx=ctx,
            last_dt_hint=last_dt_map.get(symbol),
            k_params=k_params,
            execution_params=execution_params,
            weighting=weighting,
            debug_data_quality=debug_data_quality,
            data_quality_overrides=data_quality_overrides,
            skip_on_error=skip_on_error,
            debug_weights=(debug_first_symbol and not printed_debug),
            use_fast_prices=use_fast_prices,
        )

        if r.skipped is not None:
            skipped.append(r.skipped)
            continue

        if debug_first_symbol and not printed_debug:
            printed_debug = True

        daily_by_symbol[symbol] = r.daily_df  # type: ignore[assignment]
        training_rows.append(r.training_df)  # type: ignore[arg-type]
        feature_cols_set.update(r.feature_cols)

    if not training_rows:
        _summarize_skips(skipped, universe)
        raise RuntimeError("No training data produced")

    # --------------------------------------------------------
    # (A) Global training_df (all symbols)
    # --------------------------------------------------------
    training_df = pd.concat(training_rows).sort_index()
    training_df = normalize_cols(training_df)

    feature_cols = sorted(feature_cols_set)
    validate_feature_cols(feature_cols)

    # Ensure all features exist in training_df
    for c in feature_cols:
        if c not in training_df.columns:
            training_df[c] = np.nan

    # --------------------------------------------------------
    # (B) Add global rank regression label
    # --------------------------------------------------------
    if add_rank_labels:
        training_df = add_rank_regression_labels(training_df)
        training_df = normalize_cols(training_df)

    # --------------------------------------------------------
    # (C) Expand into MTL task rows (dir + rank_reg)
    # --------------------------------------------------------
    if add_rank_tasks_to_mtl:
        # Direction rows: task="dir:<horizon>" with binary target
        dir_df = training_df.copy()
        dir_df["task"] = "dir:" + dir_df["horizon"].astype(str)

        # Rank rows: task="rank_reg:<horizon>" with float target=rank_y
        if "rank_y" not in training_df.columns:
            raise RuntimeError(
                "add_rank_tasks_to_mtl=True but 'rank_y' not found (set add_rank_labels=True)."
            )

        rank_df = training_df.loc[training_df["rank_y"].notna()].copy()
        rank_df["target"] = pd.to_numeric(rank_df["rank_y"], errors="coerce")
        rank_df["sample_weight"] = 1.0
        rank_df["task"] = "rank_reg:" + rank_df["horizon"].astype(str)

        training_df = pd.concat([dir_df, rank_df], axis=0).sort_index()
        training_df = normalize_cols(training_df)

    # --------------------------------------------------------
    # (D) Align daily feature frames to the global feature column set
    # --------------------------------------------------------
    daily_by_symbol = align_dict_to_columns(
        daily_by_symbol,
        feature_cols,
        fill_value=np.nan,
    )

    return BuildResult(
        daily_by_symbol=daily_by_symbol,
        training_df=training_df,
        feature_cols=feature_cols,
        skipped=skipped,
    )


# ============================================================
# NEW: Build ONLY Technicals (Prices + Features)
# ============================================================
def build_technical_panel(
        universe: List[str],
        *,
        api_key: str,
        data_dir: str,
        db_name: str = "quant.db",
        sleep_s: float = 0.0,
        skip_on_error: bool = True,
        verbose_data: bool = True,
        debug_data_quality: bool = False,
        data_quality_overrides: Optional[Dict[str, Any]] = None,
        # Execution params removed from required args
        execution_params: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, List[str], List[SkipRecord]]:
    """
    Builds a MultiIndex DataFrame (date, symbol) containing ONLY prices and technical features.
    No labels, no events, no targets.
    """
    os.makedirs(data_dir, exist_ok=True)
    ctx = DataContext.from_data_dir(
        api_key=api_key,
        data_dir=data_dir,
        db_name=db_name,
        sleep_s=sleep_s,
        verbose=verbose_data
    )

    # Pre-fetch last dates for incremental speed
    ctx.store.init_schema()
    last_df = ctx.store.get_last_price_dates_from_prices_for_symbols(universe)
    last_dt_map = {s: None for s in universe}
    if last_df is not None and not last_df.empty:
        last_df = last_df.dropna(subset=["last_price_date"])
        for _, row in last_df.iterrows():
            last_dt_map[str(row["symbol"])] = pd.to_datetime(row["last_price_date"]).normalize()

    frames = []
    feature_cols_set = set()
    skipped = []

    for symbol in universe:
        try:
            df, fcols, _ = _process_symbol_features(
                symbol=symbol,
                ctx=ctx,
                last_dt_hint=last_dt_map.get(symbol),
                debug_data_quality=debug_data_quality,
                data_quality_overrides=data_quality_overrides,
                use_fast_prices=True,
                execution_params=execution_params,  # Pass it if present, otherwise None
            )
            frames.append(df)
            feature_cols_set.update(fcols)

        except Exception as e:
            if not skip_on_error:
                raise
            skipped.append(SkipRecord(symbol, "features_only", type(e).__name__, str(e)[:200]))

    if not frames:
        return pd.DataFrame(), [], skipped

    # Create global panel
    panel = pd.concat(frames)
    panel = normalize_cols(panel)

    # Set MultiIndex
    if "date" in panel.columns and "symbol" in panel.columns:
        # Reset index if date is index to ensure we have columns to set
        if panel.index.name == "date":
            panel = panel.reset_index()
        panel = panel.set_index(["date", "symbol"]).sort_index()
    elif panel.index.name == "date" and "symbol" in panel.columns:
        panel = panel.reset_index().set_index(["date", "symbol"]).sort_index()

    feature_cols = sorted(list(feature_cols_set))
    return panel, feature_cols, skipped


# ============================================================
# Canonical dataset builder (explicit train/infer windows)
# ============================================================

def build_dataset(
        *,
        ctx: DataContext,
        symbols: List[str],
        train_start: str,
        train_end: str,
        infer_start: str,
        infer_end: str,
        k_params: Dict[str, int],
        execution_params: Dict[str, Any],
        weighting: Dict[str, Any],
        add_rank_labels: bool = True,
        add_rank_tasks_to_mtl: bool = True,
        debug_data_quality: bool = False,
        data_quality_overrides: Optional[Dict[str, Any]] = None,
        skip_on_error: bool = True,
        verbose_data: bool = True,
) -> Dict[str, Any]:
    """Build dataset artifacts with explicit temporal boundaries.

    This is now the only supported entrypoint for dataset construction.

    Returns:
      - daily_by_symbol: dict[symbol] -> daily dataframe (full history used during build)
      - training_df: training dataframe filtered to [train_start, train_end)
      - inference_panel: MultiIndex(date,symbol) panel filtered to [infer_start, infer_end)
      - feature_cols: global feature column list
      - skipped: per-symbol skips
      - meta: build metadata (windows, symbols)
    """
    # Derive data_dir from the sqlite db_path.
    data_dir = os.path.dirname(ctx.store.db_path) or "."
    api_key = ctx.api_key

    out = build_training_and_daily(
        list(symbols),
        api_key=api_key,
        data_dir=data_dir,
        k_params=k_params,
        execution_params=execution_params,
        weighting=weighting,
        db_name=os.path.basename(ctx.store.db_path),
        sleep_s=ctx.sleep_s,
        debug_first_symbol=False,
        skip_on_error=skip_on_error,
        verbose_data=verbose_data,
        history_years=ctx.history_years,
        debug_data_quality=debug_data_quality,
        data_quality_overrides=data_quality_overrides,
        add_rank_labels=add_rank_labels,
        add_rank_tasks_to_mtl=add_rank_tasks_to_mtl,
    )

    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    infer_start_ts = pd.Timestamp(infer_start)
    infer_end_ts = pd.Timestamp(infer_end)

    training_df = out.training_df.copy()
    if "date" in training_df.columns:
        training_df["date"] = pd.to_datetime(training_df["date"])
        training_df = training_df[(training_df["date"] >= train_start_ts) & (training_df["date"] < train_end_ts)]
    else:
        # If date is the index, coerce and filter
        idx = pd.to_datetime(training_df.index)
        training_df = training_df[(idx >= train_start_ts) & (idx < train_end_ts)]

    # Build inference panel from daily_by_symbol (features already present there).
    frames = []
    for sym, df in out.daily_by_symbol.items():
        if df is None or len(df) == 0:
            continue
        d = df.copy()
        d["symbol"] = sym
        if "date" not in d.columns:
            d = d.reset_index()
        d["date"] = pd.to_datetime(d["date"])
        d = d[(d["date"] >= infer_start_ts) & (d["date"] < infer_end_ts)]
        frames.append(d)

    if frames:
        panel = pd.concat(frames, axis=0, ignore_index=True)
        panel = panel.sort_values(["date", "symbol"]).set_index(["date", "symbol"])
        panel.index = panel.index.set_names(["date", "symbol"])
    else:
        panel = pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"]))

    meta = {
        "symbols": list(symbols),
        "train_window": {"start": str(train_start_ts.date()), "end": str(train_end_ts.date())},
        "infer_window": {"start": str(infer_start_ts.date()), "end": str(infer_end_ts.date())},
        "k_params": dict(k_params),
        "execution_params": dict(execution_params),
        "weighting": dict(weighting),
    }

    return {
        "daily_by_symbol": out.daily_by_symbol,
        "training_df": training_df,
        "inference_panel": panel,
        "feature_cols": list(out.feature_cols),
        "skipped": out.skipped,
        "meta": meta,
    }
