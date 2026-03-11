from __future__ import annotations

import json
import os
import hashlib
import re
from datetime import timedelta
from urllib.parse import urlencode
import pandas as pd

from django.http import Http404, JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.urls import reverse
from django.views.decorators.clickjacking import xframe_options_exempt
from django.views.decorators.http import require_GET, require_POST
from django.db import transaction

from .forms import LabelingConfigForm
from data.historical_prices import load_adjusted_price_frames
from fmp.models import Symbol
from fmp.models import SymbolSectionHistorical
from fmp.models import WorkflowState
from labels.strategy_solver import solve_joint_trades_by_frequency
from data import FMPClient
from domain.labels.specs import LabelBuildSpec
from domain.trades.operations import apply_trade_deduplication, trade_return_pct
from utils.workflow import workflow_symbols_from_request
from workflows.labels import build_trade_results as workflow_build_trade_results


def _non_empty(value, default):
    return default if value in (None, "") else value


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _min_profit_percent_points(value, default_points: float = 1.0) -> float:
    points = _to_float(_non_empty(value, default_points), default_points)
    return max(0.0, points)


def _min_profit_decimal(value, default_points: float = 1.0) -> float:
    return _min_profit_percent_points(value, default_points=default_points) / 100.0


def _decimal_to_percent_points(value, default_points: float = 1.0) -> float:
    return max(0.0, _to_float(_non_empty(value, default_points / 100.0), default_points / 100.0) * 100.0)


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
            return cached.copy()
    qs = (
        SymbolSectionHistorical.objects.filter(symbol=symbol_obj, section_key="prices_div_adj")
        .order_by("record_date", "updated_at")
        .only("payload", "record_date")
    )
    if start_date:
        qs = qs.filter(record_date__gte=pd.to_datetime(start_date).date())
    if end_date:
        qs = qs.filter(record_date__lte=pd.to_datetime(end_date).date())
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
    return trade_return_pct(side, entry_px, exit_px)


def _build_normalized_config(cleaned_data: dict, symbols: list[str], k_params: dict[str, list[int]]) -> dict:
    return {
        "symbols": symbols,
        "k_params": k_params,
        "solver": {
            # UI value is percent points (10 means 10%), solver expects decimal (0.10).
            "min_profit_pct": _min_profit_decimal(cleaned_data.get("min_profit_pct"), default_points=1.0),
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
        "min_profit_pct": str(_decimal_to_percent_points(normalized["solver"]["min_profit_pct"])),
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
    download_missing_prices: bool = True,
    start_date: str | None = None,
    end_date: str | None = None,
    progress_callback=None,
    price_frames: dict[str, pd.DataFrame] | None = None,
) -> tuple[list[dict], list[dict]]:
    result = workflow_build_trade_results(
        symbols=symbols,
        spec=LabelBuildSpec(
            k_params={str(key): [int(value) for value in values] for key, values in dict(k_params or {}).items()},
            min_profit_pct=float(min_profit_pct),
            buy_execution=str(buy_col),
            sell_execution=str(sell_col),
            short_execution=str(short_col),
            cover_execution=str(cover_col),
            start_date=start_date,
            end_date=end_date,
            download_missing_prices=bool(download_missing_prices),
        ),
        progress_callback=progress_callback,
        price_frames=price_frames,
    )
    return result.trade_rows, result.completed_trades


def _sort_trade_rows(trades_rows: list[dict]) -> list[dict]:
    return sorted(trades_rows, key=lambda r: (r["symbol"], r["entry_date"], r["exit_date"]), reverse=True)


def _apply_trade_deduplication(
    trades_rows: list[dict],
    completed_trades: list[dict],
    mode: str,
) -> tuple[list[dict], list[dict]]:
    return apply_trade_deduplication(trades_rows, completed_trades, mode=mode)


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
    gdf = pd.DataFrame(completed_trades)
    rets = pd.Series([t["ret_dec"] for t in completed_trades], dtype=float)
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    breakevens = rets[rets == 0]
    hold_days = pd.Series([t["hold_days"] for t in completed_trades], dtype=float)
    long_count = int(sum(1 for t in completed_trades if t["side"] == "long"))
    short_count = int(sum(1 for t in completed_trades if t["side"] == "short"))
    trade_stats = {
        "total_trades": int(len(completed_trades)),
        "symbols_count": int(gdf["symbol"].nunique()) if "symbol" in gdf.columns else 0,
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


def _load_generated_label_results(symbols: list[str]) -> tuple[int, dict, list[dict], list[dict]]:
    if not symbols:
        return 0, {}, [], []
    section_key = "labels_generated"
    qs = (
        SymbolSectionHistorical.objects.filter(
            symbol__symbol__in=symbols,
            section_key=section_key,
        )
        .select_related("symbol")
        .only("symbol__symbol", "payload", "record_date")
        .order_by("-record_date", "-updated_at")
    )
    completed_trades: list[dict] = []
    for obj in qs.iterator():
        payload = obj.payload if isinstance(obj.payload, dict) else {}
        symbol = str(payload.get("symbol") or obj.symbol.symbol or "").strip().upper()
        entry_date = str(payload.get("date") or (obj.record_date.isoformat() if obj.record_date else ""))[:10]
        if not symbol or not entry_date:
            continue
        try:
            ret_dec = float(payload.get("trade_return") or 0.0)
        except Exception:
            ret_dec = 0.0
        try:
            hold_days = int(payload.get("hold_days") or 0)
        except Exception:
            hold_days = 0
        completed_trades.append(
            {
                "symbol": symbol,
                "side": str(payload.get("side") or ""),
                "freq": str(payload.get("freq") or ""),
                "k": int(payload.get("k") or 0),
                "entry_date": entry_date,
                "exit_date": entry_date,
                "ret_dec": ret_dec,
                "hold_days": hold_days,
            }
        )
    if not completed_trades:
        return 0, {}, [], []
    trade_stats, grouped_trade_stats, symbol_grouped_trade_stats = _build_trade_aggregates(completed_trades)
    return len(completed_trades), trade_stats, grouped_trade_stats, symbol_grouped_trade_stats


def _store_generated_labels(completed_trades: list[dict], symbols: list[str]) -> int:
    section_key = "labels_generated"
    symbol_set = {str(s).strip().upper() for s in symbols if str(s).strip()}
    if not symbol_set:
        return 0

    symbol_map = {
        str(row.symbol).strip().upper(): row
        for row in Symbol.objects.filter(symbol__in=list(symbol_set)).only("id", "symbol")
    }

    with transaction.atomic():
        SymbolSectionHistorical.objects.filter(
            symbol__symbol__in=list(symbol_set),
            section_key=section_key,
        ).delete()

        objects = []
        seen: set[tuple[str, str]] = set()
        for row in completed_trades:
            symbol = str(row.get("symbol") or "").strip().upper()
            entry_date = str(row.get("entry_date") or "")[:10]
            if not symbol or not entry_date:
                continue
            symbol_obj = symbol_map.get(symbol)
            if symbol_obj is None:
                continue
            side = str(row.get("side") or "").strip().lower()
            label_value = 1 if side == "long" else 0
            market_position = 1 if side == "long" else -1
            try:
                trade_return = float(row.get("ret_dec") or 0.0)
            except Exception:
                trade_return = 0.0
            try:
                hold_days = int(row.get("hold_days") or 0)
            except Exception:
                hold_days = 0
            payload = {
                "date": entry_date,
                "symbol": symbol,
                "label": label_value,
                "market_position": market_position,
                "trade_return": trade_return,
                "side": side,
                "freq": str(row.get("freq") or ""),
                "k": int(row.get("k") or 0),
                "hold_days": hold_days,
            }
            # Keep one label per symbol/date after deduplication.
            dedupe_key = (symbol, entry_date)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            objects.append(
                SymbolSectionHistorical(
                    symbol=symbol_obj,
                    section_key=section_key,
                    record_key=f"{entry_date}:{symbol}",
                    record_date=pd.to_datetime(entry_date).date(),
                    payload=payload,
                )
            )
        if objects:
            SymbolSectionHistorical.objects.bulk_create(objects, batch_size=1000)
        return len(objects)


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

@xframe_options_exempt
def labeling_config_form(request):
    normalized = None
    trades_rows = []
    trades_error = ""
    trade_stats = {}
    grouped_trade_stats = []
    symbol_grouped_trade_stats = []
    symbol_trade_groups = []
    generated_labels_count = 0
    build_message = ""
    force_default_exec = request.method != "POST"
    symbols = workflow_symbols_from_request(request)
    if request.method == "POST":
        form = LabelingConfigForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            k_params = {
                "W": _parse_k_list(cd.get("k_w_list")),
                "M": _parse_k_list(cd.get("k_m_list")),
                "QE": _parse_k_list(cd.get("k_qe_list")),
                "YE": _parse_k_list(cd.get("k_ye_list")),
            }
            k_params = {freq: ks for freq, ks in k_params.items() if ks}
            request.session["labels_model_target_col"] = str(cd.get("model_target_col") or "label")
            try:
                WorkflowState.objects.update_or_create(
                    key="default",
                    defaults={
                        "label_target_col": str(cd.get("model_target_col") or "label"),
                        "labels_config": normalized if "normalized" in locals() and isinstance(normalized, dict) else {},
                    },
                )
            except Exception:
                pass
            if not symbols:
                trades_error = "No symbols available. Run Universe Screener first."
            else:
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
                    generated_labels_count = _store_generated_labels(completed_trades, symbols)
                    if generated_labels_count > 0:
                        build_message = f"Generated and saved {generated_labels_count} label rows."
                    else:
                        build_message = "Config built, but no labels were generated for the current settings."
                    try:
                        WorkflowState.objects.update_or_create(
                            key="default",
                            defaults={
                                "universe_symbols": symbols,
                                "label_target_col": str(cd.get("model_target_col") or "label"),
                                "labels_config": normalized,
                                "labels_generated_count": int(generated_labels_count),
                            },
                        )
                    except Exception:
                        pass
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
            "session_symbols": symbols,
            "normalized": json.dumps(normalized, indent=2, sort_keys=True) if normalized else "",
            "trades_rows": trades_rows,
            "trades_error": trades_error,
            "trade_stats": trade_stats,
            "grouped_trade_stats": grouped_trade_stats,
            "symbol_grouped_trade_stats": symbol_grouped_trade_stats,
            "symbol_trade_groups": symbol_trade_groups,
            "generated_labels_count": generated_labels_count,
            "build_message": build_message,
            "force_default_exec": force_default_exec,
        },
    )


def labeling_symbol_detail(request, symbol: str, label_run_id: int | None = None):
    symbol_value = str(symbol or "").strip().upper()
    symbol_obj = Symbol.objects.filter(symbol__iexact=symbol_value).first()
    cleaned_data = None
    k_params = None

    label_run_id_raw = label_run_id if label_run_id is not None else request.GET.get("label_run_id")
    try:
        label_run_id = int(str(label_run_id_raw or "").strip())
    except Exception:
        label_run_id = 0

    if label_run_id > 0:
        return redirect(f"{reverse('pipeline-symbol-research', args=[symbol_value])}?label_run_id={label_run_id}")

    if cleaned_data is None or k_params is None:
        k_params = {
            "W": _parse_k_list(request.GET.get("k_w_list")),
            "M": _parse_k_list(request.GET.get("k_m_list")),
            "QE": _parse_k_list(request.GET.get("k_qe_list")),
            "YE": _parse_k_list(request.GET.get("k_ye_list")),
        }
        k_params = {freq: ks for freq, ks in k_params.items() if ks}
        cleaned_data = {
            "min_profit_pct": request.GET.get("min_profit_pct", 1.0),
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

    # Paginate symbol trades table to keep page responsive for large runs.
    try:
        page = int(request.GET.get("page") or 1)
    except Exception:
        page = 1
    page_size = 10
    page = max(1, page)
    total_trades = len(trades_rows)
    total_pages = max(1, (total_trades + page_size - 1) // page_size)
    page = min(page, total_pages)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    trades_rows_page = trades_rows[start_idx:end_idx]

    qs = request.GET.copy()
    qs.pop("page", None)
    qs.pop("page_size", None)
    base_qs = qs.urlencode()

    def _page_href(page_number: int) -> str:
        query = f"page={int(page_number)}&page_size={int(page_size)}"
        if base_qs:
            query = f"{query}&{base_qs}"
        return f"?{query}"

    chart_context = _build_labeling_chart_context(symbol_obj, trades_rows)
    return render(
        request,
        "labels/symbol_detail.html",
        {
            "symbol": symbol_value,
            "normalized": json.dumps(normalized, indent=2, sort_keys=True),
            "trades_rows": trades_rows_page,
            "trades_total_count": total_trades,
            "trades_page": page,
            "trades_page_size": page_size,
            "trades_total_pages": total_pages,
            "trades_has_prev": page > 1,
            "trades_has_next": page < total_pages,
            "trades_prev_href": _page_href(page - 1) if page > 1 else "",
            "trades_next_href": _page_href(page + 1) if page < total_pages else "",
            "trades_error": trades_error,
            "trade_stats": trade_stats,
            "grouped_trade_stats": grouped_trade_stats,
            **chart_context,
        },
    )
