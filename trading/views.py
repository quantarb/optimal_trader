from __future__ import annotations

import os
from typing import Any

import pandas as pd
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from app.trading_app_v2_runtime import default_paths


LEADERBOARD_PAGE_SIZE = 50


def _streamlit_trading_url() -> str:
    return str(os.getenv("STREAMLIT_TRADING_APP_URL") or "http://localhost:8502").rstrip("/")


def trading_leaderboard(request: HttpRequest) -> HttpResponse:
    paths = default_paths()
    leaderboard_path = paths.live_artifact_dir / "leaderboard_latest.csv"
    metadata_path = paths.live_artifact_dir / "metadata.json"
    leaderboard = _load_v2_leaderboard(leaderboard_path)
    meta = _read_json(metadata_path)
    if not leaderboard.empty:
        leaderboard = _rank_v2_leaderboard_for_display(leaderboard)

    page = _coerce_int(request.GET.get("page"), default=1, minimum=1)
    page_rows, page_meta = _paginate_frame(leaderboard, page=page, page_size=LEADERBOARD_PAGE_SIZE)
    display_columns = [
        "Rank",
        "Scored Date",
        "Symbol",
        "Direction",
        "Eligible",
        "Classifier Score",
        "Combined Score",
        "Close",
        "Model Count",
    ]
    table = _frame_table(page_rows, display_columns)

    summary = _leaderboard_summary(leaderboard, meta)
    return render(
        request,
        "trading/leaderboard.html",
        {
            "leaderboard_table": table,
            "leaderboard_summary": summary,
            "page_meta": page_meta,
            "meta": meta,
            "artifact_dir": str(paths.live_artifact_dir),
            "stale_reason": "" if leaderboard_path.exists() else f"missing_v2_leaderboard={leaderboard_path}",
            "has_leaderboard": not leaderboard.empty,
            "streamlit_url": _streamlit_trading_url(),
        },
    )

def _load_v2_leaderboard(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    out = frame.copy()
    rename_map = {
        "rank": "Rank",
        "score_date": "Scored Date",
        "symbol": "Symbol",
        "eligible": "Eligible",
        "prob_buy": "Classifier Score",
        "close": "Close",
        "model_count": "Model Count",
    }
    out = out.rename(columns={source: target for source, target in rename_map.items() if source in out.columns})
    out["Direction"] = "Long"
    out["Combined Score"] = pd.to_numeric(out.get("Classifier Score"), errors="coerce")
    if "Eligible" in out.columns:
        out["Eligible"] = out["Eligible"].astype(bool)
    return out


def _read_json(path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import json

        return dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return {}


def _rank_v2_leaderboard_for_display(frame: pd.DataFrame) -> pd.DataFrame:
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
    if column in {"Price", "__price", "Target Dollars", "Close"}:
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
