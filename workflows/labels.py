from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import os
from typing import Any

import pandas as pd
from django.utils import timezone

from data.historical_prices import load_adjusted_price_frames
from domain.labels.specs import LabelBuildSpec
from domain.trades import (
    TradeGenerationResult,
    apply_trade_deduplication,
    build_label_rows_from_completed_trades,
    build_label_statistics,
    solve_joint_trades_by_frequency,
    trade_return_pct,
)
from fmp.models import Symbol, SymbolSectionHistorical
from infra.fmp import FMPClient


@dataclass(frozen=True)
class OracleLabelWorkflowResult:
    """Canonical label rows and diagnostics from oracle trade generation."""

    label_rows: list[dict[str, Any]]
    statistics: dict[str, Any]
    completed_trades: list[dict[str, Any]]


def build_trade_results(
    symbols: list[str],
    *,
    spec: LabelBuildSpec,
    progress_callback=None,
    price_frames: dict[str, pd.DataFrame] | None = None,
) -> TradeGenerationResult:
    """Build raw oracle trade candidates for a symbol set."""

    api_key = os.getenv("FMP_API_KEY") or ""
    trade_rows: list[dict[str, Any]] = []
    completed_trades: list[dict[str, Any]] = []
    normalized_symbols = [str(sym).strip().upper() for sym in list(symbols or []) if str(sym).strip()]
    symbol_map = {
        str(symbol.symbol).strip().upper(): symbol
        for symbol in Symbol.objects.filter(symbol__in=normalized_symbols).only("id", "symbol")
    }
    if price_frames is None:
        price_frames = load_adjusted_price_frames(normalized_symbols, start_date=spec.start_date, end_date=spec.end_date)
    total_symbols = len(normalized_symbols)
    if callable(progress_callback):
        progress_callback(completed=0, total=total_symbols, current_symbol="")

    for idx, symbol_code in enumerate(normalized_symbols, start=1):
        if callable(progress_callback):
            progress_callback(completed=max(0, idx - 1), total=total_symbols, current_symbol=symbol_code)
        symbol_obj = symbol_map.get(symbol_code)
        if not symbol_obj:
            symbol_obj = Symbol.objects.create(symbol=symbol_code)
        daily_prices = _load_adjusted_daily(
            symbol_obj,
            price_frames=price_frames,
            start_date=spec.start_date,
            end_date=spec.end_date,
        )
        if daily_prices.empty:
            if api_key and spec.download_missing_prices:
                _download_and_store_adjusted_prices(symbol_obj, api_key)
                daily_prices = _load_adjusted_daily(symbol_obj, start_date=spec.start_date, end_date=spec.end_date)
            if daily_prices.empty:
                if callable(progress_callback):
                    progress_callback(completed=idx, total=total_symbols, current_symbol=symbol_code)
                continue

        for freq, ks in spec.k_params.items():
            for k in ks:
                joint_trades = solve_joint_trades_by_frequency(
                    daily_prices,
                    k=int(k),
                    freq=freq,
                    min_profit_pct=spec.min_profit_pct,
                    long_entry_price_col=spec.buy_execution,
                    long_exit_price_col=spec.sell_execution,
                    short_entry_price_col=spec.short_execution,
                    short_exit_price_col=spec.cover_execution,
                )
                for trade in joint_trades:
                    entry_dt = pd.Timestamp(trade["entry_row"].name)
                    exit_dt = pd.Timestamp(trade["exit_row"].name)
                    entry_px = float(trade["entry_price"])
                    exit_px = float(trade["exit_price"])
                    side = str(trade.get("side") or "").strip().lower()
                    if side not in {"long", "short"}:
                        continue
                    ret_dec = trade_return_pct(side, entry_px, exit_px)
                    trade_rows.append(
                        {
                            "symbol": symbol_code,
                            "side": side,
                            "freq": freq,
                            "k": int(k),
                            "entry_date": entry_dt.strftime("%Y-%m-%d"),
                            "exit_date": exit_dt.strftime("%Y-%m-%d"),
                            "entry_px": f"{entry_px:,.4f}",
                            "exit_px": f"{exit_px:,.4f}",
                            "ret_pct": f"{ret_dec * 100:.2f}%",
                        }
                    )
                    completed_trades.append(
                        {
                            "symbol": symbol_code,
                            "side": side,
                            "freq": freq,
                            "k": int(k),
                            "entry_date": entry_dt.strftime("%Y-%m-%d"),
                            "exit_date": exit_dt.strftime("%Y-%m-%d"),
                            "entry_px": f"{entry_px:,.4f}",
                            "exit_px": f"{exit_px:,.4f}",
                            "ret_dec": ret_dec,
                            "hold_days": int((exit_dt - entry_dt).days),
                        }
                    )
        if callable(progress_callback):
            progress_callback(completed=idx, total=total_symbols, current_symbol=symbol_code)
    return TradeGenerationResult(trade_rows=trade_rows, completed_trades=completed_trades)


def build_oracle_labels(
    symbols: list[str],
    *,
    spec: LabelBuildSpec,
    progress_callback=None,
    price_frames: dict[str, pd.DataFrame] | None = None,
) -> OracleLabelWorkflowResult:
    """Build canonical label rows and summary statistics from oracle trades."""

    generated = build_trade_results(symbols, spec=spec, progress_callback=progress_callback, price_frames=price_frames)
    _, completed = apply_trade_deduplication(generated.trade_rows, generated.completed_trades, mode=spec.trade_dedup_mode)
    label_rows = build_label_rows_from_completed_trades(completed)
    return OracleLabelWorkflowResult(
        label_rows=label_rows,
        statistics=build_label_statistics(label_rows),
        completed_trades=completed,
    )


def _stable_record_key(record: dict[str, Any]) -> str:
    blob = json.dumps(record, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _download_and_store_adjusted_prices(symbol_obj: Symbol, api_key: str) -> int:
    client = FMPClient(api_key=api_key, timeout_s=30.0, max_retries=2)
    section_key = "prices_div_adj"
    start = timezone.datetime(1999, 1, 22).date()
    end = timezone.now().date()
    cur = start
    saved = 0
    while cur <= end:
        nxt = min(end, cur + timedelta(days=365 * 10))
        raw = client.get_json(
            "/stable/historical-price-eod/dividend-adjusted",
            params={"symbol": symbol_obj.symbol, "from": cur.isoformat(), "to": nxt.isoformat()},
        )
        rows = raw if isinstance(raw, list) else []
        for rec in rows:
            if not isinstance(rec, dict):
                continue
            date_raw = str(rec.get("date") or "")[:10]
            try:
                record_date = pd.to_datetime(date_raw).date() if date_raw else None
            except Exception:
                record_date = None
            SymbolSectionHistorical.objects.update_or_create(
                symbol=symbol_obj,
                section_key=section_key,
                record_key=_stable_record_key(rec),
                defaults={"record_date": record_date, "payload": rec},
            )
            saved += 1
        cur = nxt + timedelta(days=1)
    return saved


def _load_adjusted_daily(
    symbol_obj: Symbol,
    *,
    price_frames: dict[str, pd.DataFrame] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    if price_frames is not None:
        cached = price_frames.get(str(symbol_obj.symbol).strip().upper())
        if cached is not None:
            return cached
    qs = (
        SymbolSectionHistorical.objects.filter(symbol=symbol_obj, section_key="prices_div_adj")
        .order_by("record_date", "updated_at")
        .values_list("payload", "record_date")
    )
    if start_date:
        qs = qs.filter(record_date__gte=pd.to_datetime(start_date).date())
    if end_date:
        qs = qs.filter(record_date__lte=pd.to_datetime(end_date).date())
    rows = []
    for payload_value, record_date in qs.iterator(chunk_size=5000):
        payload = payload_value if isinstance(payload_value, dict) else {}
        dt = payload.get("date") or (record_date.isoformat() if record_date else None)
        if not dt:
            continue
        rows.append(
            {
                "date": str(dt)[:10],
                "adj_open": payload.get("adjOpen"),
                "adj_high": payload.get("adjHigh"),
                "adj_low": payload.get("adjLow"),
                "adj_close": payload.get("adjClose"),
                "volume": payload.get("volume"),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("adj_open", "adj_high", "adj_low", "adj_close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    return df[~df.index.duplicated(keep="last")]
