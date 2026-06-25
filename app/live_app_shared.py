from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd


def inspect_saved_artifacts(
    *,
    required_paths: Iterable[Path],
    artifact_dir: Path,
    max_age: pd.Timedelta,
    saved_score_date: str | pd.Timestamp | None = None,
    expected_score_date: str | pd.Timestamp | None = None,
    extra_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = [Path(path).resolve() for path in required_paths]
    status: dict[str, Any] = {
        "artifact_dir": str(Path(artifact_dir).resolve()),
        "age": pd.NaT,
        **dict(extra_status or {}),
    }
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        status["reason"] = "missing_files=" + ",".join(missing)
        status["reusable"] = False
        return status

    newest_mtime = max(path.stat().st_mtime for path in paths)
    oldest_mtime = min(path.stat().st_mtime for path in paths)
    now = pd.Timestamp.now(tz="UTC")
    status["age"] = now - pd.Timestamp(newest_mtime, unit="s", tz="UTC")
    status["oldest_artifact_age"] = now - pd.Timestamp(oldest_mtime, unit="s", tz="UTC")
    if status["oldest_artifact_age"] > pd.Timedelta(max_age):
        status["reason"] = f"older_than_{pd.Timedelta(max_age)}"
        status["reusable"] = False
        return status

    if saved_score_date is not None and expected_score_date is not None:
        saved_date = pd.Timestamp(saved_score_date).normalize()
        expected_date = pd.Timestamp(expected_score_date).normalize()
        if pd.isna(saved_date) or pd.isna(expected_date):
            status["reason"] = "invalid_score_date"
            status["reusable"] = False
            return status
        status["saved_latest_date"] = saved_date.date().isoformat()
        status["expected_score_date"] = expected_date.date().isoformat()
        if saved_date < expected_date:
            status["reason"] = (
                f"saved_latest_date_{saved_date.date().isoformat()}_lt_expected_"
                f"{expected_date.date().isoformat()}"
            )
            status["reusable"] = False
            return status

    status["reason"] = "fresh_saved_build"
    status["reusable"] = True
    return status


def artifact_age_hours(status: dict[str, Any]) -> float | None:
    age = status.get("age")
    if age is None or pd.isna(age):
        return None
    return round(float(pd.Timedelta(age) / pd.Timedelta(hours=1)), 2)


def log_artifact_decision(
    logger: Callable[[str], None],
    *,
    status: dict[str, Any],
    artifact_label: str,
    reused: bool,
) -> None:
    if reused:
        hours = artifact_age_hours(status)
        age_text = "unknown" if hours is None else f"{hours:.2f}h"
        logger(
            f"Reusing saved {artifact_label} artifacts from {status['artifact_dir']} | age {age_text}"
        )
        return
    logger(
        f"Saved {artifact_label} artifacts were not reusable; rebuilding | "
        f"reason={status.get('reason') or 'unknown'}"
    )


__all__ = ["artifact_age_hours", "inspect_saved_artifacts", "log_artifact_decision"]
