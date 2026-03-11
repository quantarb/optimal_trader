from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from pipeline.feature_presentation import format_feature_value, get_feature_definition

from fmp.models import Symbol, SymbolSectionHistorical

from pipeline.contracts import normalize_prediction_row
from pipeline.models import Artifact


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except Exception:
        return None


def _read_artifact_rows(artifact: Artifact) -> list[dict[str, Any]]:
    uri = str(artifact.uri or "").strip()
    if not uri:
        return []
    path = Path(uri)
    if not path.exists() or not path.is_file():
        return []
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as fh:
                return [dict(row) for row in csv.DictReader(fh)]
        if suffix == ".json":
            blob = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(blob, list):
                return [dict(row) for row in blob if isinstance(row, dict)]
            if isinstance(blob, dict):
                rows = blob.get("rows")
                if isinstance(rows, list):
                    return [dict(row) for row in rows if isinstance(row, dict)]
    except Exception:
        return []
    return []


def artifact_rows_for_symbol(artifact: Artifact, symbol: str) -> list[dict[str, Any]]:
    symbol_value = str(symbol or "").strip().upper()
    rows = _read_artifact_rows(artifact)
    if not symbol_value:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("symbol") or "").strip().upper() != symbol_value:
            continue
        out.append(row)
    return out


def _artifact_type_values(artifact_type: str | Sequence[str]) -> list[str]:
    if isinstance(artifact_type, str):
        values = [artifact_type]
    else:
        values = list(artifact_type or [])
    out: list[str] = []
    for value in values:
        cleaned = str(value or "").strip().upper()
        if cleaned:
            out.append(cleaned)
    return out


def resolve_artifact(*, artifact_type: str | Sequence[str], artifact_id: int = 0, pipeline_run_id: int = 0) -> Artifact | None:
    types = _artifact_type_values(artifact_type)
    qs = Artifact.objects.filter(artifact_type__in=types)
    if artifact_id > 0:
        return qs.filter(pk=int(artifact_id)).select_related("pipeline_run").first()
    if pipeline_run_id > 0:
        return qs.filter(pipeline_run_id=int(pipeline_run_id)).select_related("pipeline_run").order_by("-created_at", "-id").first()
    return qs.select_related("pipeline_run").order_by("-created_at", "-id").first()


def recent_artifact_choices(artifact_type: str | Sequence[str], limit: int = 15) -> list[Artifact]:
    types = _artifact_type_values(artifact_type)
    return list(
        Artifact.objects.filter(artifact_type__in=types)
        .select_related("pipeline_run")
        .order_by("-created_at", "-id")[: max(1, limit)]
    )


def load_price_frame(symbol: str) -> pd.DataFrame:
    symbol_obj = Symbol.objects.filter(symbol__iexact=str(symbol or "").strip()).first()
    if symbol_obj is None:
        return pd.DataFrame()
    qs = (
        SymbolSectionHistorical.objects.filter(symbol=symbol_obj, section_key="prices_div_adj")
        .order_by("record_date", "updated_at")
        .only("record_date", "payload")
    )
    rows: list[dict[str, Any]] = []
    for item in qs.iterator():
        payload = item.payload if isinstance(item.payload, dict) else {}
        date_value = payload.get("date") or (item.record_date.isoformat() if item.record_date else None)
        if not date_value:
            continue
        rows.append(
            {
                "date": str(date_value)[:10],
                "open": payload.get("adjOpen"),
                "high": payload.get("adjHigh"),
                "low": payload.get("adjLow"),
                "close": payload.get("adjClose"),
                "volume": payload.get("volume"),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    return df[~df.index.duplicated(keep="last")]


def build_price_chart_context(symbol: str) -> dict[str, Any]:
    empty = {
        "price_points_count": 0,
        "labels_json": "[]",
        "opens_json": "[]",
        "highs_json": "[]",
        "lows_json": "[]",
        "closes_json": "[]",
        "volumes_json": "[]",
    }
    df = load_price_frame(symbol)
    if df.empty:
        return empty
    return {
        "price_points_count": int(len(df)),
        "labels_json": json.dumps([idx.strftime("%Y-%m-%d") for idx in df.index]),
        "opens_json": json.dumps([None if pd.isna(v) else float(v) for v in df["open"].tolist()]),
        "highs_json": json.dumps([None if pd.isna(v) else float(v) for v in df["high"].tolist()]),
        "lows_json": json.dumps([None if pd.isna(v) else float(v) for v in df["low"].tolist()]),
        "closes_json": json.dumps([None if pd.isna(v) else float(v) for v in df["close"].tolist()]),
        "volumes_json": json.dumps([None if pd.isna(v) else float(v) for v in df["volume"].tolist()]),
    }


def build_label_chart_context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    entry_markers: list[dict[str, Any]] = []
    exit_markers: list[dict[str, Any]] = []
    trade_lines: list[dict[str, Any]] = []
    for row in rows:
        entry_date = str(row.get("entry_date") or row.get("date") or "")[:10]
        exit_date = str(row.get("exit_date") or "")[:10]
        entry_px = _safe_float(row.get("entry_px"))
        exit_px = _safe_float(row.get("exit_px"))
        side = str(row.get("side") or "").strip().lower()
        trade_return = _safe_float(row.get("trade_return"))
        ret_pct = row.get("ret_pct")
        if ret_pct in (None, "") and trade_return is not None:
            ret_pct = f"{trade_return * 100.0:.2f}%"
        details = [
            f"Freq: {row.get('freq')}" if row.get("freq") not in (None, "") else "",
            f"k: {row.get('k')}" if row.get("k") not in (None, "") else "",
            f"Return: {ret_pct}" if ret_pct not in (None, "") else "",
        ]
        details = [value for value in details if value]
        if entry_date and entry_px is not None:
            entry_markers.append(
                {
                    "x": entry_date,
                    "y": entry_px,
                    "type": "Long Entry" if side == "long" else "Short Entry",
                    "details": details,
                }
            )
        if exit_date and exit_px is not None:
            exit_markers.append(
                {
                    "x": exit_date,
                    "y": exit_px,
                    "type": "Long Exit" if side == "long" else "Cover",
                    "details": details,
                }
            )
        if entry_date and exit_date and entry_px is not None and exit_px is not None:
            trade_lines.append(
                {
                    "entry_x": entry_date,
                    "entry_y": entry_px,
                    "exit_x": exit_date,
                    "exit_y": exit_px,
                    "side": "Long" if side == "long" else "Short",
                    "ret_pct": ret_pct or "",
                }
            )
    return {
        "label_rows_count": int(len(rows)),
        "entry_markers_json": json.dumps(entry_markers),
        "exit_markers_json": json.dumps(exit_markers),
        "trade_lines_json": json.dumps(trade_lines),
    }


def build_prediction_chart_context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    series: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (str(item.get("date") or ""), str(item.get("symbol") or ""))):
        date_value = str(row.get("date") or "")[:10]
        score = _safe_float(row.get("signal_score"))
        if score is None:
            score = _safe_float(row.get("prediction_score"))
        if score is None:
            score = _safe_float(row.get("raw_prediction"))
        if score is None:
            score = _safe_float(row.get("prediction"))
        if not date_value or score is None:
            continue
        series.append(
            {
                "x": date_value,
                "score": score,
                "prediction": _safe_float(row.get("raw_prediction"))
                if row.get("raw_prediction") not in (None, "")
                else _safe_float(row.get("prediction")),
                "predicted_class": _safe_int(row.get("predicted_class")),
                "label": _safe_float(row.get("label")),
                "ret_1": _safe_float(row.get("ret_1")),
                "trade_return": _safe_float(row.get("trade_return")),
                "market_position": _safe_int(row.get("market_position")),
            }
        )
    return {
        "prediction_rows_count": int(len(series)),
        "prediction_series_json": json.dumps(series),
    }


def build_prediction_chart_series(
    artifact_rows: list[tuple[Artifact, list[dict[str, Any]]]],
) -> dict[str, Any]:
    series_collection: list[dict[str, Any]] = []
    total_rows = 0
    for artifact, rows in artifact_rows:
        normalized_rows = normalize_prediction_rows(rows)
        payload = build_prediction_chart_context(normalized_rows)
        try:
            series = json.loads(payload["prediction_series_json"])
        except Exception:
            series = []
        total_rows += int(payload.get("prediction_rows_count") or 0)
        series_collection.append(
            {
                "artifact_id": int(artifact.id),
                "artifact_type": str(artifact.artifact_type),
                "name": f"{artifact.artifact_type} #{int(artifact.id)}",
                "series": series,
            }
        )
    return {
        "prediction_rows_count": int(total_rows),
        "prediction_series_collection_json": json.dumps(series_collection),
    }


def build_strategy_chart_context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    markers: list[dict[str, Any]] = []
    score_series: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (str(item.get("date") or ""), str(item.get("symbol") or ""))):
        date_value = str(row.get("date") or "")[:10]
        signal = _safe_int(row.get("strategy_signal"))
        score = _safe_float(row.get("strategy_score"))
        if date_value and score is not None:
            score_series.append(
                {
                    "x": date_value,
                    "score": score,
                    "signal": signal,
                }
            )
        if not date_value or signal in (None, 0):
            continue
        details = [
            f"Signal: {signal}",
            f"Score: {score:.4f}" if score is not None else "",
        ]
        markers.append(
            {
                "x": date_value,
                "signal": signal,
                "type": "Strategy Long" if signal > 0 else "Strategy Short",
                "details": [value for value in details if value],
            }
        )
    return {
        "strategy_rows_count": int(len(rows)),
        "strategy_markers_json": json.dumps(markers),
        "strategy_score_series_json": json.dumps(score_series),
    }


def build_backtest_chart_context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    markers: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (str(item.get("date") or ""), str(item.get("symbol") or ""))):
        date_value = str(row.get("date") or "")[:10]
        signal = _safe_int(row.get("strategy_signal"))
        realized = _safe_float(row.get("realized_return"))
        score = _safe_float(row.get("strategy_score"))
        if not date_value or signal in (None, 0):
            continue
        details = [
            f"Signal: {signal}",
            f"Realized: {realized:.4f}" if realized is not None else "",
            f"Score: {score:.4f}" if score is not None else "",
        ]
        markers.append(
            {
                "x": date_value,
                "signal": signal,
                "realized_return": realized,
                "type": "Backtest Long" if signal > 0 else "Backtest Short",
                "details": [value for value in details if value],
            }
        )
    return {
        "backtest_rows_count": int(len(rows)),
        "backtest_markers_json": json.dumps(markers),
    }


def normalize_prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(normalize_prediction_row(row))
    return normalized


def build_feature_table_context(rows: list[dict[str, Any]], limit: int = 20) -> dict[str, Any]:
    if not rows:
        return {
            "feature_rows": [],
            "feature_columns": [],
            "feature_display_columns": [],
            "feature_row_values": [],
            "feature_rendered_values": [],
            "feature_rows_count": 0,
        }
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
    preview_rows = rows[: max(1, limit)]
    return {
        "feature_rows": preview_rows,
        "feature_columns": keys,
        "feature_display_columns": [get_feature_definition(key).display_name for key in keys],
        "feature_row_values": [[row.get(key, "") for key in keys] for row in preview_rows],
        "feature_rendered_values": [[format_feature_value(key, row.get(key, "")) for key in keys] for row in preview_rows],
        "feature_rows_count": int(len(rows)),
    }
