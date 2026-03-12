from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import REPO_ROOT
from ..models import ArtifactInventory, ArtifactRecord


PIPELINE_ARTIFACT_TYPES = ("UNIVERSE", "LABELS", "FEATURES", "REGRESSOR_PREDICTIONS", "STRATEGY_DATASET")


def _db_path(root: str | Path | None = None) -> Path:
    return (Path(root).resolve() if root is not None else REPO_ROOT) / "db.sqlite3"


def discover_artifact_inventory(*, root: str | Path | None = None, tiers: list[str] | tuple[str, ...] = ()) -> list[ArtifactInventory]:
    db_path = _db_path(root)
    if not db_path.exists():
        return []
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    inventories: list[ArtifactInventory] = []
    for tier in list(tiers or []):
        inventory = ArtifactInventory(tier=str(tier))
        for artifact_type in PIPELINE_ARTIFACT_TYPES:
            row = connection.execute(
                """
                select id, artifact_type, uri, created_at
                from pipeline_artifact
                where artifact_type = ? and uri like ?
                order by id desc
                limit 1
                """,
                (artifact_type, f"%/{tier}/%"),
            ).fetchone()
            if row is None:
                continue
            inventory.artifacts[artifact_type] = ArtifactRecord(
                artifact_id=int(row["id"]),
                artifact_type=str(row["artifact_type"]),
                tier=str(tier),
                uri=str(row["uri"]),
                created_at=str(row["created_at"] or ""),
            )
        strategy = inventory.artifacts.get("STRATEGY_DATASET")
        if strategy is not None:
            frame = load_artifact_frame(strategy.uri)
            inventory.symbol_count = int(frame["symbol"].nunique()) if "symbol" in frame.columns else 0
            inventory.date_count = int(frame["date"].nunique()) if "date" in frame.columns else 0
        inventories.append(inventory)
    connection.close()
    return inventories


def load_artifact_frame(uri: str) -> pd.DataFrame:
    path = Path(uri)
    if not path.is_absolute():
        path = (REPO_ROOT / uri).resolve()
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if isinstance(payload.get("symbols"), list):
                return pd.DataFrame({"symbol": payload.get("symbols")})
            if isinstance(payload.get("records"), list):
                return pd.DataFrame(payload.get("records"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        return pd.DataFrame()
    return pd.read_csv(path)


def latest_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns or "date" not in frame.columns:
        return frame.copy()
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values(["date", "symbol"])
    return work.groupby("symbol", as_index=False, sort=False).tail(1).reset_index(drop=True)


def symbol_examples(frame: pd.DataFrame, *, limit: int = 12) -> list[str]:
    if frame.empty or "symbol" not in frame.columns:
        return []
    return [str(value) for value in frame["symbol"].astype(str).dropna().drop_duplicates().sort_values().head(limit).tolist()]


def resolve_candidate_symbols(inventory: list[ArtifactInventory], *, preferred: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for symbol in preferred:
        code = str(symbol).strip().upper()
        if code and code not in seen:
            seen.add(code)
            out.append(code)
    for item in inventory:
        strategy = item.artifacts.get("STRATEGY_DATASET")
        if strategy is None:
            continue
        frame = latest_rows(load_artifact_frame(strategy.uri))
        for symbol in symbol_examples(frame):
            code = str(symbol).strip().upper()
            if code and code not in seen:
                seen.add(code)
                out.append(code)
    return out
