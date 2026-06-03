from __future__ import annotations

import os
from datetime import date
from typing import Any

import pandas as pd
from django.contrib import messages
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.urls import reverse

from app.live_trade_leaderboard import (
    default_live_trade_config,
    latest_scored_staleness_reason,
    load_saved_leaderboard,
)
from app.optimal_trade_lookup import OptimalTradeQuery, find_nearest_optimal_trades


LEADERBOARD_PAGE_SIZE = 50


def _streamlit_trading_url() -> str:
    return str(os.getenv("STREAMLIT_TRADING_APP_URL") or "http://localhost:8502").rstrip("/")


def trading_leaderboard(request: HttpRequest) -> HttpResponse:
    default_cfg = default_live_trade_config()
    saved = load_saved_leaderboard()
    leaderboard = pd.DataFrame()
    meta: dict[str, Any] = {}
    if saved is not None:
        leaderboard, meta = saved
        strategy_cfg = (meta.get("config") or {}).get("strategy") or default_cfg.get("strategy") or {}
        threshold = float(strategy_cfg.get("component_threshold", 0.50))
        leaderboard = _recompute_eligibility(leaderboard, threshold=threshold)
        leaderboard = _rank_leaderboard_for_display(leaderboard)

    page = _coerce_int(request.GET.get("page"), default=1, minimum=1)
    page_rows, page_meta = _paginate_frame(leaderboard, page=page, page_size=LEADERBOARD_PAGE_SIZE)
    display_columns = [
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
    ]
    table = _frame_table(page_rows, display_columns, link_columns={"Similar Trades": "Similar Trades"})

    artifact_dir = (
        (meta.get("config") or {}).get("runtime", {}).get("artifact_dir")
        or meta.get("artifact_dir")
        or default_cfg["runtime"]["artifact_dir"]
    )
    stale_reason = latest_scored_staleness_reason(artifact_dir=artifact_dir)
    summary = _leaderboard_summary(leaderboard, meta)
    return render(
        request,
        "trading/leaderboard.html",
        {
            "leaderboard_table": table,
            "leaderboard_summary": summary,
            "page_meta": page_meta,
            "meta": meta,
            "artifact_dir": artifact_dir,
            "stale_reason": stale_reason,
            "has_leaderboard": not leaderboard.empty,
            "streamlit_url": _streamlit_trading_url(),
        },
    )


def similar_trades(request: HttpRequest) -> HttpResponse:
    form = _similar_trades_form(request)
    result = None
    error = ""
    if request.method == "POST" or str(request.GET.get("symbol") or "").strip():
        try:
            result = find_nearest_optimal_trades(
                OptimalTradeQuery(
                    symbol=str(form["symbol"]).strip().upper(),
                    as_of_date=str(form["as_of_date"] or "") or None,
                    top_k=int(form["top_k"]),
                    query_lookback_years=int(form["query_lookback_years"]),
                    reference_start_date=str(form["reference_start_date"]),
                    min_profit_pct_points=float(form["min_profit_pct_points"]),
                    download_missing_prices=False,
                    artifact_dir=str(form["artifact_dir"]),
                )
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            messages.error(request, f"Similar-trades lookup failed: {error}")

    context: dict[str, Any] = {"form": form, "error": error, "result": result}
    if result is not None:
        context.update(
            {
                "query_summary_table": _frame_table(result.query_summary),
                "indicator_table": _frame_table(result.indicator_summary),
                "nearest_trades_table": _frame_table(result.nearest_trades),
                "feature_attribution_table": _frame_table(result.feature_attribution),
                "model_predictions": result.model_predictions,
                "metadata": result.metadata,
            }
        )
    return render(request, "trading/similar_trades.html", context)


def _similar_trades_form(request: HttpRequest) -> dict[str, Any]:
    default_cfg = default_live_trade_config()
    source = request.POST if request.method == "POST" else request.GET
    return {
        "symbol": str(source.get("symbol") or "AAPL").strip().upper(),
        "as_of_date": str(source.get("as_of_date") or date.today().isoformat()),
        "top_k": _coerce_int(source.get("top_k"), default=10, minimum=1),
        "query_lookback_years": _coerce_int(source.get("query_lookback_years"), default=5, minimum=1),
        "reference_start_date": str(source.get("reference_start_date") or "2010-01-01"),
        "min_profit_pct_points": _coerce_float(source.get("min_profit_pct_points"), default=5.0, minimum=0.0),
        "artifact_dir": str(source.get("artifact_dir") or default_cfg["runtime"]["artifact_dir"]),
    }


def _recompute_eligibility(frame: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    component_cols = [column for column in out.columns if str(column).startswith("__component__")]
    if component_cols:
        eligible = pd.Series(True, index=out.index, dtype=bool)
        for component_col in component_cols:
            eligible &= pd.to_numeric(out[component_col], errors="coerce").gt(float(threshold)).fillna(False)
        out["Eligible"] = eligible & pd.to_numeric(out.get("Combined Score"), errors="coerce").notna()
        return out
    required_cols = ["Classifier Score", "Regressor Score", "Autoencoder Score"]
    if all(column in out.columns for column in required_cols):
        out["Eligible"] = (
            pd.to_numeric(out["Classifier Score"], errors="coerce").gt(float(threshold))
            & pd.to_numeric(out["Regressor Score"], errors="coerce").gt(float(threshold))
            & pd.to_numeric(out["Autoencoder Score"], errors="coerce").gt(float(threshold))
        )
    return out


def _rank_leaderboard_for_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out["Combined Score"] = pd.to_numeric(out.get("Combined Score"), errors="coerce")
    if "Eligible" in out.columns:
        out = out.sort_values(["Eligible", "Combined Score"], ascending=[False, False], kind="stable")
    else:
        out = out.sort_values(["Combined Score"], ascending=[False], kind="stable")
    out = out.reset_index(drop=True)
    out["Rank"] = range(1, len(out) + 1)
    out["Similar Trades"] = out["Symbol"].map(
        lambda symbol: reverse("trading-similar-trades") + f"?symbol={str(symbol).strip().upper()}"
    )
    return out


def _leaderboard_summary(frame: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "eligible": 0, "ineligible": 0}
    eligible = int(pd.Series(frame.get("Eligible"), dtype="boolean").fillna(False).sum()) if "Eligible" in frame.columns else 0
    return {
        "rows": int(len(frame)),
        "eligible": eligible,
        "ineligible": int(max(len(frame) - eligible, 0)),
        "latest_date": str(meta.get("latest_date") or ""),
        "universe_size": meta.get("universe_size"),
        "reference_trade_count": meta.get("reference_trade_count"),
        "vector_backend": (meta.get("vector_metadata") or {}).get("backend") or meta.get("vector_backend"),
    }


def _paginate_frame(frame: pd.DataFrame, *, page: int, page_size: int) -> tuple[pd.DataFrame, dict[str, int]]:
    total = int(len(frame))
    total_pages = max((total - 1) // page_size + 1, 1)
    safe_page = max(1, min(int(page), total_pages))
    start = (safe_page - 1) * page_size
    end = start + page_size
    return frame.iloc[start:end].copy(), {
        "page": safe_page,
        "page_size": int(page_size),
        "total_rows": total,
        "total_pages": total_pages,
        "prev_page": max(1, safe_page - 1),
        "next_page": min(total_pages, safe_page + 1),
    }


def _frame_table(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    link_columns: dict[str, str] | None = None,
) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {"columns": [], "rows": []}
    display = frame.copy()
    if columns is None:
        columns = [str(column) for column in display.columns if not str(column).startswith("__")]
    else:
        columns = [column for column in columns if column in display.columns]
    rows: list[dict[str, Any]] = []
    link_columns = dict(link_columns or {})
    for _, raw_row in display[columns].iterrows():
        row = {"cells": [], "is_eligible": bool(raw_row.get("Eligible", False))}
        for column in columns:
            value = raw_row.get(column)
            cell = {"column": column, "value": _format_cell(column, value), "href": ""}
            if column in link_columns:
                cell["href"] = str(value or "")
                cell["value"] = link_columns[column]
            row["cells"].append(cell)
        rows.append(row)
    return {"columns": columns, "rows": rows}


def _format_cell(column: str, value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if column in {"Classifier Score", "Regressor Score", "Autoencoder Score", "Combined Score"}:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return "" if pd.isna(numeric) else f"{float(numeric):.4f}"
    if column in {"Price", "__price", "Target Dollars"}:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return "" if pd.isna(numeric) else f"${float(numeric):,.2f}"
    if column in {"Target Weight"}:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return "" if pd.isna(numeric) else f"{float(numeric) * 100.0:.2f}%"
    if column == "Eligible":
        return "Yes" if bool(value) else "No"
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return str(value)


def _coerce_int(value: Any, *, default: int, minimum: int | None = None) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(default)
    if minimum is not None:
        out = max(int(minimum), out)
    return out


def _coerce_float(value: Any, *, default: float, minimum: float | None = None) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if minimum is not None:
        out = max(float(minimum), out)
    return out
