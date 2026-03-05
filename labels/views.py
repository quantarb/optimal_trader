from __future__ import annotations

import json
import os
import hashlib
import re
from datetime import timedelta
from urllib.parse import urlencode
import pandas as pd

from django.shortcuts import render
from django.urls import reverse
from django.utils import timezone

from .forms import LabelingConfigForm
from fmp.models import Symbol
from fmp.models import SymbolSectionHistorical
from labels.strategy_solver import solve_joint_trades_by_frequency
from modules.data.fmp_client import FMPClient


def _non_empty(value, default):
    return default if value in (None, "") else value


def _parse_k_list(raw: str | None) -> list[int]:
    if raw in (None, ""):
        return []
    out: list[int] = []
    seen: set[int] = set()
    for part in str(raw).split(","):
        p = part.strip()
        if not p:
            continue
        try:
            k = int(p)
        except Exception:
            continue
        if k > 0 and k not in seen:
            out.append(k)
            seen.add(k)
    return out


def _symbol_choices():
    rows = Symbol.objects.order_by("symbol").values_list("symbol", "company_name")
    out = []
    for symbol, company_name in rows:
        s = str(symbol or "").strip().upper()
        if not s:
            continue
        label = f"{s} - {company_name}" if company_name else s
        out.append((s, label))
    return out


def _stable_record_key(record: dict) -> str:
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
            params={
                "symbol": symbol_obj.symbol,
                "from": cur.isoformat(),
                "to": nxt.isoformat(),
            },
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
                defaults={
                    "record_date": record_date,
                    "payload": rec,
                },
            )
            saved += 1
        cur = nxt + timedelta(days=1)
    return saved


def _load_adjusted_daily(symbol_obj: Symbol) -> pd.DataFrame:
    qs = (
        SymbolSectionHistorical.objects.filter(symbol=symbol_obj, section_key="prices_div_adj")
        .order_by("record_date", "updated_at")
        .only("payload", "record_date")
    )
    rows = []
    for item in qs.iterator():
        payload = item.payload if isinstance(item.payload, dict) else {}
        dt = payload.get("date") or (item.record_date.isoformat() if item.record_date else None)
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
    df = df[~df.index.duplicated(keep="last")]
    return df


def _ret_pct(side: str, entry_px: float, exit_px: float) -> float:
    if not entry_px:
        return 0.0
    if side == "long":
        return (float(exit_px) - float(entry_px)) / float(entry_px)
    return (float(entry_px) - float(exit_px)) / float(entry_px)


def _build_normalized_config(cleaned_data: dict, symbols: list[str], k_params: dict[str, list[int]]) -> dict:
    return {
        "symbols": symbols,
        "k_params": k_params,
        "solver": {
            "min_profit_pct": float(_non_empty(cleaned_data.get("min_profit_pct"), 0.01)),
        },
        "execution_params": {
            "buy_execution": cleaned_data.get("buy_execution") or "adj_high",
            "sell_execution": cleaned_data.get("sell_execution") or "adj_low",
            "short_execution": cleaned_data.get("short_execution") or "adj_low",
            "cover_execution": cleaned_data.get("cover_execution") or "adj_high",
            "fee_bps": float(_non_empty(cleaned_data.get("fee_bps"), 0.0)),
            "slippage_bps": float(_non_empty(cleaned_data.get("slippage_bps"), 0.0)),
        },
        "options": {
            "trade_dedup_mode": str(cleaned_data.get("trade_dedup_mode") or "exact"),
        },
    }


def _build_detail_query_string(normalized: dict) -> str:
    query: dict[str, str] = {
        "min_profit_pct": str(normalized["solver"]["min_profit_pct"]),
        "buy_execution": str(normalized["execution_params"]["buy_execution"]),
        "sell_execution": str(normalized["execution_params"]["sell_execution"]),
        "short_execution": str(normalized["execution_params"]["short_execution"]),
        "cover_execution": str(normalized["execution_params"]["cover_execution"]),
        "fee_bps": str(normalized["execution_params"]["fee_bps"]),
        "slippage_bps": str(normalized["execution_params"]["slippage_bps"]),
        "trade_dedup_mode": str(normalized["options"]["trade_dedup_mode"]),
    }
    freq_map = {
        "W": "k_w_list",
        "M": "k_m_list",
        "QE": "k_qe_list",
        "YE": "k_ye_list",
    }
    for freq, param_name in freq_map.items():
        ks = normalized["k_params"].get(freq) or []
        if ks:
            query[param_name] = ",".join(str(int(k)) for k in ks)
    return urlencode(query)


def _build_trade_results(
    symbols: list[str],
    k_params: dict[str, list[int]],
    min_profit_pct: float,
    buy_col: str,
    sell_col: str,
    short_col: str,
    cover_col: str,
) -> tuple[list[dict], list[dict]]:
    api_key = os.getenv("FMP_API_KEY") or ""
    trades_rows: list[dict] = []
    completed_trades: list[dict] = []

    for sym in symbols:
        symbol_obj = Symbol.objects.filter(symbol__iexact=sym).first()
        if not symbol_obj:
            continue
        df_daily = _load_adjusted_daily(symbol_obj)
        if df_daily.empty:
            if api_key:
                _download_and_store_adjusted_prices(symbol_obj, api_key)
                df_daily = _load_adjusted_daily(symbol_obj)
            if df_daily.empty:
                continue

        for freq, ks in k_params.items():
            for k in ks:
                joint_trades = solve_joint_trades_by_frequency(
                    df_daily,
                    k=int(k),
                    freq=freq,
                    min_profit_pct=min_profit_pct,
                    long_entry_price_col=buy_col,
                    long_exit_price_col=sell_col,
                    short_entry_price_col=short_col,
                    short_exit_price_col=cover_col,
                )
                for t in joint_trades:
                    entry_dt = pd.Timestamp(t["entry_row"].name)
                    exit_dt = pd.Timestamp(t["exit_row"].name)
                    entry_px = float(t["entry_price"])
                    exit_px = float(t["exit_price"])
                    side = str(t.get("side") or "").strip().lower()
                    if side not in {"long", "short"}:
                        continue
                    ret_dec = _ret_pct(side, entry_px, exit_px)
                    trades_rows.append(
                        {
                            "symbol": sym,
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
                            "symbol": sym,
                            "side": side,
                            "freq": freq,
                            "k": int(k),
                            "entry_date": entry_dt.strftime("%Y-%m-%d"),
                            "exit_date": exit_dt.strftime("%Y-%m-%d"),
                            "ret_dec": ret_dec,
                            "hold_days": int((exit_dt - entry_dt).days),
                        }
                    )
    return trades_rows, completed_trades


def _sort_trade_rows(trades_rows: list[dict]) -> list[dict]:
    return sorted(trades_rows, key=lambda r: (r["symbol"], r["entry_date"], r["exit_date"]), reverse=True)


def _apply_trade_deduplication(
    trades_rows: list[dict],
    completed_trades: list[dict],
    mode: str,
) -> tuple[list[dict], list[dict]]:
    mode_value = str(mode or "exact").strip().lower()
    if mode_value not in {"exact", "entry_date"}:
        return trades_rows, completed_trades

    if mode_value == "exact":
        kept_rows: list[dict] = []
        kept_completed: list[dict] = []
        seen: set[tuple[str, str, str, str]] = set()
        for row, completed in zip(trades_rows, completed_trades):
            key = (
                str(row.get("symbol") or ""),
                str(row.get("side") or ""),
                str(row.get("entry_date") or ""),
                str(row.get("exit_date") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            kept_rows.append(row)
            kept_completed.append(completed)
        return kept_rows, kept_completed

    best_by_key: dict[tuple[str, str, str], tuple[int, float]] = {}
    for idx, completed in enumerate(completed_trades):
        key = (
            str(completed.get("symbol") or ""),
            str(completed.get("side") or ""),
            str(completed.get("entry_date") or ""),
        )
        ret_dec = float(completed.get("ret_dec") or 0.0)
        prev = best_by_key.get(key)
        if prev is None or ret_dec > prev[1]:
            best_by_key[key] = (idx, ret_dec)
    keep_indices = {idx for idx, _ in best_by_key.values()}
    kept_rows = [row for idx, row in enumerate(trades_rows) if idx in keep_indices]
    kept_completed = [row for idx, row in enumerate(completed_trades) if idx in keep_indices]
    return kept_rows, kept_completed


def _build_symbol_trade_groups(trades_rows: list[dict]) -> list[dict]:
    if not trades_rows:
        return []
    grouped_rows: dict[str, list[dict]] = {}
    for row in trades_rows:
        grouped_rows.setdefault(str(row["symbol"]), []).append(row)
    out = []
    for symbol in sorted(grouped_rows.keys()):
        out.append(
            {
                "symbol": symbol,
                "anchor": "symbol-trades-" + re.sub(r"[^a-zA-Z0-9_-]+", "-", symbol).strip("-").lower(),
                "rows": grouped_rows[symbol],
            }
        )
    return out


def _build_trade_aggregates(completed_trades: list[dict], detail_query_string: str = "") -> tuple[dict, list[dict], list[dict]]:
    if not completed_trades:
        return {}, [], []
    rets = pd.Series([t["ret_dec"] for t in completed_trades], dtype=float)
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    breakevens = rets[rets == 0]
    hold_days = pd.Series([t["hold_days"] for t in completed_trades], dtype=float)
    long_count = int(sum(1 for t in completed_trades if t["side"] == "long"))
    short_count = int(sum(1 for t in completed_trades if t["side"] == "short"))
    trade_stats = {
        "total_trades": int(len(completed_trades)),
        "long_trades": long_count,
        "short_trades": short_count,
        "winning_trades": int(len(wins)),
        "losing_trades": int(len(losses)),
        "breakeven_trades": int(len(breakevens)),
        "win_rate_pct": float((rets > 0).mean() * 100.0),
        "loss_rate_pct": float((rets < 0).mean() * 100.0),
        "avg_return_pct": float(rets.mean() * 100.0),
        "median_return_pct": float(rets.median() * 100.0),
        "avg_win_pct": float(wins.mean() * 100.0) if len(wins) else None,
        "avg_loss_pct": float(losses.mean() * 100.0) if len(losses) else None,
        "profit_factor": (
            float(wins.sum() / abs(losses.sum()))
            if len(losses) and abs(float(losses.sum())) > 1e-12
            else None
        ),
        "avg_holding_days": float(hold_days.mean()) if len(hold_days) else None,
    }
    gdf = pd.DataFrame(completed_trades)
    grouped_trade_stats: list[dict] = []
    grouped = (
        gdf.groupby(["side", "freq", "k"], as_index=False)
        .agg(
            trades=("ret_dec", "count"),
            trade_return_mean=("ret_dec", "mean"),
            trade_return_std=("ret_dec", "std"),
            trade_duration_mean=("hold_days", "mean"),
            trade_duration_std=("hold_days", "std"),
        )
        .sort_values(["side", "freq", "k"], ascending=[True, True, True])
    )
    for _, row in grouped.iterrows():
        ret_mean = float(row["trade_return_mean"])
        ret_std = float(row["trade_return_std"]) if pd.notna(row["trade_return_std"]) else None
        sharpe = None
        if ret_std is not None and abs(ret_std) > 1e-12:
            sharpe = ret_mean / ret_std
        grouped_trade_stats.append(
            {
                "side": str(row["side"]),
                "freq": str(row["freq"]),
                "k": int(row["k"]),
                "trades": int(row["trades"]),
                "trade_return_mean_pct": ret_mean * 100.0,
                "trade_return_std_pct": (ret_std * 100.0) if ret_std is not None else None,
                "trade_duration_mean": float(row["trade_duration_mean"]),
                "trade_duration_std": float(row["trade_duration_std"]) if pd.notna(row["trade_duration_std"]) else None,
                "sharpe": sharpe,
            }
        )
    symbol_grouped_trade_stats: list[dict] = []
    symbol_grouped = (
        gdf.groupby(["symbol"], as_index=False)
        .agg(
            trades=("ret_dec", "count"),
            trade_return_mean=("ret_dec", "mean"),
            trade_return_std=("ret_dec", "std"),
            trade_duration_mean=("hold_days", "mean"),
            trade_duration_std=("hold_days", "std"),
        )
        .sort_values(["symbol"], ascending=[True])
    )
    for _, row in symbol_grouped.iterrows():
        symbol = str(row["symbol"])
        ret_mean = float(row["trade_return_mean"])
        ret_std = float(row["trade_return_std"]) if pd.notna(row["trade_return_std"]) else None
        sharpe = None
        if ret_std is not None and abs(ret_std) > 1e-12:
            sharpe = ret_mean / ret_std
        detail_href = reverse("labels-symbol-detail", args=[symbol])
        if detail_query_string:
            detail_href += "?" + detail_query_string
        symbol_grouped_trade_stats.append(
            {
                "symbol": symbol,
                "anchor": "symbol-trades-" + re.sub(r"[^a-zA-Z0-9_-]+", "-", symbol).strip("-").lower(),
                "detail_href": detail_href,
                "trades": int(row["trades"]),
                "trade_return_mean_pct": ret_mean * 100.0,
                "trade_return_std_pct": (ret_std * 100.0) if ret_std is not None else None,
                "trade_duration_mean": float(row["trade_duration_mean"]),
                "trade_duration_std": float(row["trade_duration_std"]) if pd.notna(row["trade_duration_std"]) else None,
                "sharpe": sharpe,
            }
        )
    return trade_stats, grouped_trade_stats, symbol_grouped_trade_stats


def _build_labeling_chart_context(symbol_obj: Symbol | None, trades_rows: list[dict]) -> dict:
    empty = {
        "labels_json": "[]",
        "opens_json": "[]",
        "highs_json": "[]",
        "lows_json": "[]",
        "closes_json": "[]",
        "volumes_json": "[]",
        "entry_markers_json": "[]",
        "exit_markers_json": "[]",
        "trade_lines_json": "[]",
        "points_count": 0,
        "trade_markers_count": 0,
    }
    if not symbol_obj:
        return empty
    df_daily = _load_adjusted_daily(symbol_obj)
    if df_daily.empty:
        return empty

    labels = [idx.strftime("%Y-%m-%d") for idx in df_daily.index]
    opens = [None if pd.isna(v) else float(v) for v in df_daily["adj_open"].tolist()]
    highs = [None if pd.isna(v) else float(v) for v in df_daily["adj_high"].tolist()]
    lows = [None if pd.isna(v) else float(v) for v in df_daily["adj_low"].tolist()]
    closes = [None if pd.isna(v) else float(v) for v in df_daily["adj_close"].tolist()]
    volumes = [None if pd.isna(v) else float(v) for v in df_daily["volume"].tolist()]

    entry_markers: list[dict] = []
    exit_markers: list[dict] = []
    trade_lines: list[dict] = []
    for row in trades_rows:
        try:
            entry_px = float(str(row.get("entry_px") or "").replace(",", ""))
            exit_px = float(str(row.get("exit_px") or "").replace(",", ""))
        except Exception:
            continue
        entry_date = str(row.get("entry_date") or "")[:10]
        exit_date = str(row.get("exit_date") or "")[:10]
        if not entry_date or not exit_date:
            continue
        side = str(row.get("side") or "").strip().lower()
        freq = str(row.get("freq") or "")
        k = row.get("k")
        ret_pct = str(row.get("ret_pct") or "")
        entry_markers.append(
            {
                "x": entry_date,
                "y": entry_px,
                "type": "Long Entry" if side == "long" else "Short Entry",
                "details": [f"Freq: {freq}", f"k: {k}", f"Return: {ret_pct}"],
            }
        )
        exit_markers.append(
            {
                "x": exit_date,
                "y": exit_px,
                "type": "Long Exit" if side == "long" else "Cover",
                "details": [f"Freq: {freq}", f"k: {k}", f"Return: {ret_pct}"],
            }
        )
        trade_lines.append(
            {
                "entry_x": entry_date,
                "entry_y": entry_px,
                "exit_x": exit_date,
                "exit_y": exit_px,
                "side": "Long" if side == "long" else "Short",
                "ret_pct": ret_pct,
            }
        )

    return {
        "labels_json": json.dumps(labels),
        "opens_json": json.dumps(opens),
        "highs_json": json.dumps(highs),
        "lows_json": json.dumps(lows),
        "closes_json": json.dumps(closes),
        "volumes_json": json.dumps(volumes),
        "entry_markers_json": json.dumps(entry_markers),
        "exit_markers_json": json.dumps(exit_markers),
        "trade_lines_json": json.dumps(trade_lines),
        "points_count": len(labels),
        "trade_markers_count": len(entry_markers) + len(exit_markers),
    }


def labeling_config_form(request):
    normalized = None
    trades_rows = []
    trades_error = ""
    trade_stats = {}
    grouped_trade_stats = []
    symbol_grouped_trade_stats = []
    symbol_trade_groups = []
    force_default_exec = request.method != "POST"
    symbol_choices = _symbol_choices()
    if request.method == "POST":
        form = LabelingConfigForm(request.POST, symbol_choices=symbol_choices)
        if form.is_valid():
            cd = form.cleaned_data
            k_params = {
                "W": _parse_k_list(cd.get("k_w_list")),
                "M": _parse_k_list(cd.get("k_m_list")),
                "QE": _parse_k_list(cd.get("k_qe_list")),
                "YE": _parse_k_list(cd.get("k_ye_list")),
            }
            k_params = {freq: ks for freq, ks in k_params.items() if ks}

            symbols = [str(s).strip().upper() for s in (cd.get("symbols") or []) if str(s).strip()]

            normalized = _build_normalized_config(cd, symbols, k_params)
            try:
                trades_rows, completed_trades = _build_trade_results(
                    symbols=symbols,
                    k_params=k_params,
                    min_profit_pct=float(normalized["solver"]["min_profit_pct"]),
                    buy_col=str(normalized["execution_params"]["buy_execution"]),
                    sell_col=str(normalized["execution_params"]["sell_execution"]),
                    short_col=str(normalized["execution_params"]["short_execution"]),
                    cover_col=str(normalized["execution_params"]["cover_execution"]),
                )
                trades_rows, completed_trades = _apply_trade_deduplication(
                    trades_rows,
                    completed_trades,
                    mode=str(normalized["options"]["trade_dedup_mode"]),
                )
                trades_rows = _sort_trade_rows(trades_rows)
                symbol_trade_groups = _build_symbol_trade_groups(trades_rows)
                trade_stats, grouped_trade_stats, symbol_grouped_trade_stats = _build_trade_aggregates(
                    completed_trades,
                    detail_query_string=_build_detail_query_string(normalized),
                )
            except Exception as exc:
                trades_error = str(exc)
    else:
        form = LabelingConfigForm(
            symbol_choices=symbol_choices,
            initial={
                "buy_execution": "adj_high",
                "sell_execution": "adj_low",
                "short_execution": "adj_low",
                "cover_execution": "adj_high",
            },
        )

    return render(
        request,
        "labels/config_form.html",
        {
            "form": form,
            "normalized": json.dumps(normalized, indent=2, sort_keys=True) if normalized else "",
            "trades_rows": trades_rows,
            "trades_error": trades_error,
            "trade_stats": trade_stats,
            "grouped_trade_stats": grouped_trade_stats,
            "symbol_grouped_trade_stats": symbol_grouped_trade_stats,
            "symbol_trade_groups": symbol_trade_groups,
            "force_default_exec": force_default_exec,
        },
    )


def labeling_symbol_detail(request, symbol: str):
    symbol_value = str(symbol or "").strip().upper()
    symbol_obj = Symbol.objects.filter(symbol__iexact=symbol_value).first()
    k_params = {
        "W": _parse_k_list(request.GET.get("k_w_list")),
        "M": _parse_k_list(request.GET.get("k_m_list")),
        "QE": _parse_k_list(request.GET.get("k_qe_list")),
        "YE": _parse_k_list(request.GET.get("k_ye_list")),
    }
    k_params = {freq: ks for freq, ks in k_params.items() if ks}
    cleaned_data = {
        "min_profit_pct": request.GET.get("min_profit_pct", 0.01),
        "buy_execution": request.GET.get("buy_execution", "adj_high"),
        "sell_execution": request.GET.get("sell_execution", "adj_low"),
        "short_execution": request.GET.get("short_execution", "adj_low"),
        "cover_execution": request.GET.get("cover_execution", "adj_high"),
        "fee_bps": request.GET.get("fee_bps", 10.0),
        "slippage_bps": request.GET.get("slippage_bps", 10.0),
        "trade_dedup_mode": request.GET.get("trade_dedup_mode", "exact"),
    }
    normalized = _build_normalized_config(cleaned_data, [symbol_value], k_params)
    trades_rows: list[dict] = []
    completed_trades: list[dict] = []
    trades_error = ""
    try:
        trades_rows, completed_trades = _build_trade_results(
            symbols=[symbol_value],
            k_params=k_params,
            min_profit_pct=float(normalized["solver"]["min_profit_pct"]),
            buy_col=str(normalized["execution_params"]["buy_execution"]),
            sell_col=str(normalized["execution_params"]["sell_execution"]),
            short_col=str(normalized["execution_params"]["short_execution"]),
            cover_col=str(normalized["execution_params"]["cover_execution"]),
        )
        trades_rows, completed_trades = _apply_trade_deduplication(
            trades_rows,
            completed_trades,
            mode=str(normalized["options"]["trade_dedup_mode"]),
        )
        trades_rows = _sort_trade_rows(trades_rows)
    except Exception as exc:
        trades_error = str(exc)
    trade_stats, grouped_trade_stats, _ = _build_trade_aggregates(completed_trades)
    chart_context = _build_labeling_chart_context(symbol_obj, trades_rows)
    return render(
        request,
        "labels/symbol_detail.html",
        {
            "symbol": symbol_value,
            "normalized": json.dumps(normalized, indent=2, sort_keys=True),
            "trades_rows": trades_rows,
            "trades_error": trades_error,
            "trade_stats": trade_stats,
            "grouped_trade_stats": grouped_trade_stats,
            **chart_context,
        },
    )
