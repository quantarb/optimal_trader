from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from django.utils import timezone

from fmp.models import Symbol, SymbolSectionHistorical
from settings import BASE_DIR

ARTIFACT_DIR = Path(BASE_DIR) / "data" / "pipeline_artifacts"
ARTIFACT_STORAGE_FORMAT_KEY = "storage_format"
ARTIFACT_STORAGE_KIND_KEY = "storage_kind"
ARTIFACT_STORAGE_ROW_COUNT_KEY = "storage_row_count"
ARTIFACT_STORAGE_COLUMNS_KEY = "storage_columns"
ARTIFACT_STORAGE_FORMAT_JSON = "json"
ARTIFACT_STORAGE_FORMAT_CSV = "csv"
ARTIFACT_STORAGE_FORMAT_PARQUET = "parquet"
FRAME_STORAGE_FORMATS = {ARTIFACT_STORAGE_FORMAT_CSV, ARTIFACT_STORAGE_FORMAT_PARQUET}
PAYLOAD_STORAGE_FORMATS = {ARTIFACT_STORAGE_FORMAT_JSON}


class PipelineExecutionError(RuntimeError):
    pass


@dataclass
class BuiltOutput:
    artifact_type: str
    content: dict[str, Any]
    metadata: dict[str, Any]
    uri: str


@dataclass(frozen=True)
class StoredArtifactFile:
    uri: str
    storage_format: str
    storage_kind: str
    row_count: int | None = None
    columns: list[str] | None = None

    def storage_metadata(self) -> dict[str, Any]:
        payload = {
            ARTIFACT_STORAGE_FORMAT_KEY: str(self.storage_format),
            ARTIFACT_STORAGE_KIND_KEY: str(self.storage_kind),
        }
        if self.row_count is not None:
            payload[ARTIFACT_STORAGE_ROW_COUNT_KEY] = int(self.row_count)
        if self.columns:
            payload[ARTIFACT_STORAGE_COLUMNS_KEY] = [str(column) for column in self.columns]
        return payload


def stable_payload_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def safe_numeric_series(df: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def as_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def ensure_artifact_dir() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_symbol_list(raw: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        symbol = str(item or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def artifact_payload_hash(content: dict[str, Any], uri: str) -> str:
    blob = json.dumps(content, sort_keys=True, separators=(",", ":")) + "|" + str(uri)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return value


def write_payload_artifact(
    name: str,
    payload: dict[str, Any],
    *,
    storage_format: str = ARTIFACT_STORAGE_FORMAT_JSON,
) -> StoredArtifactFile:
    normalized_format = _normalize_storage_format(storage_format, allowed=PAYLOAD_STORAGE_FORMATS)
    ensure_artifact_dir()
    suffix = _storage_suffix(normalized_format)
    path = ARTIFACT_DIR / f"{name}{suffix}"
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return StoredArtifactFile(
        uri=str(path),
        storage_format=normalized_format,
        storage_kind="payload",
    )


def write_frame_artifact(
    name: str,
    *,
    frame: pd.DataFrame | None = None,
    rows: list[dict[str, Any]] | None = None,
    fieldnames: list[str] | None = None,
    storage_format: str = ARTIFACT_STORAGE_FORMAT_CSV,
) -> StoredArtifactFile:
    normalized_format = _normalize_storage_format(storage_format, allowed=FRAME_STORAGE_FORMATS)
    frame_value = _coerce_frame(frame=frame, rows=rows, fieldnames=fieldnames)
    ensure_artifact_dir()
    suffix = _storage_suffix(normalized_format)
    path = ARTIFACT_DIR / f"{name}{suffix}"
    if normalized_format == ARTIFACT_STORAGE_FORMAT_CSV:
        frame_value.to_csv(path, index=False)
    elif normalized_format == ARTIFACT_STORAGE_FORMAT_PARQUET:
        try:
            parquet_frame = _coerce_frame_for_parquet(frame_value)
            parquet_frame.to_parquet(path, index=False)
        except Exception as exc:
            raise PipelineExecutionError(
                "Parquet artifact writing requires a pandas parquet engine such as pyarrow or fastparquet."
            ) from exc
    else:
        raise PipelineExecutionError(f"Unsupported frame artifact storage format: {normalized_format!r}")
    return StoredArtifactFile(
        uri=str(path),
        storage_format=normalized_format,
        storage_kind="frame",
        row_count=int(len(frame_value)),
        columns=[str(column) for column in list(frame_value.columns)],
    )


def write_json(name: str, payload: dict[str, Any]) -> str:
    return write_payload_artifact(name, payload).uri


def write_rows_csv(name: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    return write_frame_artifact(name, rows=rows, fieldnames=fieldnames).uri


def infer_artifact_storage_format(artifact_or_uri: Any, *, default: str = "") -> str:
    metadata = getattr(artifact_or_uri, "metadata", None)
    if isinstance(metadata, dict):
        value = str(metadata.get(ARTIFACT_STORAGE_FORMAT_KEY) or "").strip().lower()
        if value:
            return value
    uri = str(getattr(artifact_or_uri, "uri", artifact_or_uri) or "").strip()
    suffix = Path(uri).suffix.lower()
    if suffix == ".json":
        return ARTIFACT_STORAGE_FORMAT_JSON
    if suffix == ".csv":
        return ARTIFACT_STORAGE_FORMAT_CSV
    if suffix == ".parquet":
        return ARTIFACT_STORAGE_FORMAT_PARQUET
    return str(default or "").strip().lower()


def read_json_artifact(artifact) -> dict[str, Any]:
    storage_format = infer_artifact_storage_format(artifact, default=ARTIFACT_STORAGE_FORMAT_JSON)
    if storage_format != ARTIFACT_STORAGE_FORMAT_JSON:
        return dict(artifact.content or {})
    if artifact.uri:
        path = Path(artifact.uri)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return dict(artifact.content or {})
    return dict(artifact.content or {})


def read_csv_rows(path_value: str) -> list[dict[str, Any]]:
    return read_frame_rows(path_value)


def read_frame_artifact(
    artifact_or_uri: Any,
    *,
    parse_dates: bool = True,
    normalize_symbols: bool = True,
    limit: int | None = None,
) -> pd.DataFrame:
    uri = str(getattr(artifact_or_uri, "uri", artifact_or_uri) or "").strip()
    if not uri:
        return pd.DataFrame()
    path = Path(uri)
    if not path.exists() or not path.is_file():
        return pd.DataFrame()
    storage_format = infer_artifact_storage_format(artifact_or_uri, default=_storage_format_from_suffix(path))
    if storage_format == ARTIFACT_STORAGE_FORMAT_CSV:
        df = pd.read_csv(path, nrows=int(limit) if limit is not None and int(limit) > 0 else None)
    elif storage_format == ARTIFACT_STORAGE_FORMAT_PARQUET:
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            raise PipelineExecutionError(
                "Parquet artifact reading requires a pandas parquet engine such as pyarrow or fastparquet."
            ) from exc
        if limit is not None and int(limit) > 0:
            df = df.head(int(limit)).copy()
    else:
        raise PipelineExecutionError(f"Artifact at {uri!r} is not a supported frame format.")
    if df.empty:
        return df
    if parse_dates and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if normalize_symbols and "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    return df


def read_frame_rows(
    artifact_or_uri: Any,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    df = read_frame_artifact(
        artifact_or_uri,
        parse_dates=False,
        normalize_symbols=False,
        limit=limit,
    )
    if df.empty:
        return []
    return list(json_safe_value(df.to_dict(orient="records")))


def load_universe_symbols(universe_artifact) -> list[str]:
    payload = read_json_artifact(universe_artifact)
    return normalize_symbol_list(list(payload.get("symbols") or []))


def load_adjusted_close_rows(symbols: list[str]) -> dict[str, list[tuple[str, float]]]:
    rows_by_symbol: dict[str, dict[str, float]] = {symbol: {} for symbol in symbols}
    if not symbols:
        return {symbol: [] for symbol in symbols}
    qs = (
        SymbolSectionHistorical.objects.filter(symbol__symbol__in=symbols, section_key="prices_div_adj")
        .select_related("symbol")
        .only("symbol__symbol", "record_date", "payload")
        .order_by("symbol__symbol", "record_date", "updated_at")
    )
    for row in qs.iterator():
        symbol = str(row.symbol.symbol or "").strip().upper()
        payload = row.payload if isinstance(row.payload, dict) else {}
        date_value = str(payload.get("date") or (row.record_date.isoformat() if row.record_date else ""))[:10]
        if not date_value:
            continue
        close_value = payload.get("adjClose")
        if close_value is None:
            close_value = payload.get("close")
        try:
            close_num = float(close_value)
        except Exception:
            continue
        rows_by_symbol.setdefault(symbol, {})[date_value] = close_num
    return {
        symbol: sorted(list(date_map.items()), key=lambda item: item[0])
        for symbol, date_map in rows_by_symbol.items()
    }


__all__ = [
    "ARTIFACT_DIR",
    "ARTIFACT_STORAGE_COLUMNS_KEY",
    "ARTIFACT_STORAGE_FORMAT_CSV",
    "ARTIFACT_STORAGE_FORMAT_JSON",
    "ARTIFACT_STORAGE_FORMAT_KEY",
    "ARTIFACT_STORAGE_FORMAT_PARQUET",
    "ARTIFACT_STORAGE_KIND_KEY",
    "ARTIFACT_STORAGE_ROW_COUNT_KEY",
    "BuiltOutput",
    "PipelineExecutionError",
    "StoredArtifactFile",
    "artifact_payload_hash",
    "as_bool",
    "ensure_artifact_dir",
    "infer_artifact_storage_format",
    "json_safe_value",
    "load_adjusted_close_rows",
    "load_universe_symbols",
    "normalize_symbol_list",
    "read_csv_rows",
    "read_frame_artifact",
    "read_frame_rows",
    "read_json_artifact",
    "safe_numeric_series",
    "stable_payload_hash",
    "write_frame_artifact",
    "write_json",
    "write_payload_artifact",
    "write_rows_csv",
]


def _storage_suffix(storage_format: str) -> str:
    mapping = {
        ARTIFACT_STORAGE_FORMAT_JSON: ".json",
        ARTIFACT_STORAGE_FORMAT_CSV: ".csv",
        ARTIFACT_STORAGE_FORMAT_PARQUET: ".parquet",
    }
    try:
        return mapping[str(storage_format)]
    except KeyError as exc:
        raise PipelineExecutionError(f"Unsupported artifact storage format: {storage_format!r}") from exc


def _normalize_storage_format(storage_format: str, *, allowed: set[str]) -> str:
    value = str(storage_format or "").strip().lower()
    if value not in allowed:
        raise PipelineExecutionError(
            f"Unsupported artifact storage format {storage_format!r}. Allowed: {', '.join(sorted(allowed))}."
        )
    return value


def _storage_format_from_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return ARTIFACT_STORAGE_FORMAT_JSON
    if suffix == ".csv":
        return ARTIFACT_STORAGE_FORMAT_CSV
    if suffix == ".parquet":
        return ARTIFACT_STORAGE_FORMAT_PARQUET
    return ""


def _coerce_frame_for_parquet(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in list(out.columns):
        series = out[column]
        if pd.api.types.is_object_dtype(series):
            out[column] = series.map(lambda value: None if pd.isna(value) else str(value)).astype("string")
    return out


def _coerce_frame(
    *,
    frame: pd.DataFrame | None,
    rows: list[dict[str, Any]] | None,
    fieldnames: list[str] | None,
) -> pd.DataFrame:
    if frame is not None:
        frame_value = frame.copy()
    else:
        row_values = list(rows or [])
        if row_values:
            frame_value = pd.DataFrame(row_values)
        else:
            frame_value = pd.DataFrame(columns=list(fieldnames or []))
    ordered_columns = [str(column) for column in list(fieldnames or [])]
    if ordered_columns:
        for column in ordered_columns:
            if column not in frame_value.columns:
                frame_value[column] = ""
        frame_value = frame_value[ordered_columns].copy()
    return frame_value
