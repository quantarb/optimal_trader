from __future__ import annotations

from datetime import timedelta
import time
from typing import Any

from django.utils import timezone


def progress_from_config(config: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(config or {})
    progress = raw.get("progress")
    if isinstance(progress, dict) and progress:
        return dict(progress)

    total = _to_int(raw.get("progress_total_symbols"))
    completed = _to_int(raw.get("progress_completed_symbols"))
    current_item = str(raw.get("progress_current_symbol") or "").strip()
    if total <= 0 and completed <= 0 and not current_item:
        return {}
    return {
        "phase": "symbols",
        "phase_label": "Processing symbols",
        "phase_index": 1,
        "phase_total": 1,
        "unit_label": "symbols",
        "total_units": total if total > 0 else None,
        "completed_units": max(0, completed),
        "current_item": current_item,
        "percent_complete": _percent(completed=max(0, completed), total=total if total > 0 else None),
        "overall_percent_complete": _percent(completed=max(0, completed), total=total if total > 0 else None),
    }


def apply_progress_to_config(config: dict[str, Any] | None, progress: dict[str, Any] | None) -> dict[str, Any]:
    updated = dict(config or {})
    snapshot = dict(progress or {})
    updated["progress"] = snapshot
    if str(snapshot.get("unit_label") or "") == "symbols":
        updated["progress_total_symbols"] = int(snapshot.get("total_units") or 0)
        updated["progress_completed_symbols"] = int(snapshot.get("completed_units") or 0)
        updated["progress_current_symbol"] = str(snapshot.get("current_item") or "")
    return updated


class ProgressReporter:
    def __init__(
        self,
        *,
        pipeline_run=None,
        job_run=None,
        throttle_seconds: float = 1.0,
    ) -> None:
        self.pipeline_run = pipeline_run
        self.job_run = job_run
        self.throttle_seconds = max(0.0, float(throttle_seconds))
        self._last_write_monotonic = 0.0
        self._phase_started_at = None
        self._phase_key = ""

    def update(
        self,
        *,
        phase: str,
        phase_label: str = "",
        phase_index: int | None = None,
        phase_total: int | None = None,
        unit_label: str = "",
        total_units: int | None = None,
        completed_units: int | None = None,
        current_item: str = "",
        message: str = "",
        force: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = timezone.now()
        phase_key = str(phase or "").strip() or "progress"
        if self._phase_started_at is None or phase_key != self._phase_key:
            self._phase_started_at = now
            self._phase_key = phase_key
        run_started_at = self._resolve_run_started_at() or self._phase_started_at

        total_value = _to_int(total_units) if total_units not in (None, "") else None
        if total_value is not None and total_value <= 0:
            total_value = None
        completed_value = max(0, _to_int(completed_units)) if completed_units not in (None, "") else 0
        if total_value is not None:
            completed_value = min(completed_value, total_value)

        elapsed_seconds = max(0.0, (now - self._phase_started_at).total_seconds())
        run_elapsed_seconds = max(0.0, (now - run_started_at).total_seconds()) if run_started_at is not None else elapsed_seconds
        items_per_second = None
        eta_seconds = None
        eta_basis = None
        rate_elapsed_seconds = elapsed_seconds if elapsed_seconds > 0 else run_elapsed_seconds
        if total_value is not None and completed_value > 0 and rate_elapsed_seconds > 0:
            items_per_second = completed_value / rate_elapsed_seconds
            remaining = max(0, total_value - completed_value)
            eta_seconds = 0.0 if remaining == 0 else (remaining / items_per_second if items_per_second > 0 else None)
            if eta_seconds is not None:
                eta_basis = "throughput"

        phase_percent = _percent(completed=completed_value, total=total_value)
        overall_percent = phase_percent
        phase_index_value = _to_int(phase_index) if phase_index not in (None, "") else None
        phase_total_value = _to_int(phase_total) if phase_total not in (None, "") else None
        if phase_index_value is not None and phase_total_value and phase_total_value > 0:
            normalized_phase_index = min(max(1, phase_index_value), phase_total_value)
            phase_fraction = 0.0
            if total_value is not None and total_value > 0:
                phase_fraction = completed_value / float(total_value)
            overall_percent = round((((normalized_phase_index - 1) + phase_fraction) / float(phase_total_value)) * 100.0, 4)
        if eta_seconds is None:
            if overall_percent is not None and overall_percent >= 100.0:
                eta_seconds = 0.0
                eta_basis = "overall_progress"
            elif overall_percent is not None and overall_percent > 0.0 and run_elapsed_seconds > 0.0:
                eta_seconds = run_elapsed_seconds * ((100.0 - overall_percent) / overall_percent)
                eta_basis = "overall_progress"

        snapshot = {
            "phase": phase_key,
            "phase_label": str(phase_label or phase_key.replace("_", " ").title()),
            "phase_index": phase_index_value,
            "phase_total": phase_total_value,
            "unit_label": str(unit_label or ""),
            "total_units": total_value,
            "completed_units": completed_value,
            "remaining_units": (max(0, total_value - completed_value) if total_value is not None else None),
            "current_item": str(current_item or ""),
            "message": str(message or ""),
            "started_at": self._phase_started_at.isoformat() if self._phase_started_at else None,
            "run_started_at": run_started_at.isoformat() if run_started_at else None,
            "updated_at": now.isoformat(),
            "elapsed_seconds": round(float(elapsed_seconds), 6),
            "run_elapsed_seconds": round(float(run_elapsed_seconds), 6),
            "items_per_second": round(float(items_per_second), 6) if items_per_second is not None else None,
            "eta_seconds": round(float(eta_seconds), 6) if eta_seconds is not None else None,
            "eta_at": (now + timedelta(seconds=float(eta_seconds))).isoformat() if eta_seconds is not None else None,
            "eta_basis": eta_basis,
            "percent_complete": phase_percent,
            "overall_percent_complete": overall_percent,
        }
        if extra:
            snapshot["extra"] = dict(extra)

        if force or self._should_persist():
            self._persist(snapshot)
        return snapshot

    def complete(self, *, message: str = "Completed", force: bool = True) -> dict[str, Any]:
        current_config = {}
        if self.job_run is not None:
            current_config = dict(self.job_run.config or {})
        elif self.pipeline_run is not None:
            current_config = dict(self.pipeline_run.config or {})
        current_progress = progress_from_config(current_config)
        phase = str(current_progress.get("phase") or "progress")
        phase_label = str(current_progress.get("phase_label") or "Completed")
        total_units = current_progress.get("total_units")
        completed_units = current_progress.get("completed_units")
        if total_units not in (None, ""):
            completed_units = total_units
        return self.update(
            phase=phase,
            phase_label=phase_label,
            phase_index=current_progress.get("phase_index"),
            phase_total=current_progress.get("phase_total"),
            unit_label=str(current_progress.get("unit_label") or ""),
            total_units=total_units,
            completed_units=completed_units,
            current_item="",
            message=message,
            force=force,
            extra=dict(current_progress.get("extra") or {}),
        )

    def _should_persist(self) -> bool:
        now = time.monotonic()
        if self._last_write_monotonic <= 0.0:
            self._last_write_monotonic = now
            return True
        if (now - self._last_write_monotonic) < self.throttle_seconds:
            return False
        self._last_write_monotonic = now
        return True

    def _persist(self, snapshot: dict[str, Any]) -> None:
        self._last_write_monotonic = time.monotonic()
        for record in (self.pipeline_run, self.job_run):
            if record is None:
                continue
            record.config = apply_progress_to_config(getattr(record, "config", {}) or {}, snapshot)
            record.save(update_fields=["config", "updated_at"])

    def _resolve_run_started_at(self):
        started_values = []
        for record in (self.job_run, self.pipeline_run):
            started_at = getattr(record, "started_at", None) if record is not None else None
            if started_at is not None:
                started_values.append(started_at)
        if not started_values:
            return None
        return min(started_values)


def _percent(*, completed: int, total: int | None) -> float | None:
    if total is None or total <= 0:
        return None
    return round((min(max(0, completed), total) / float(total)) * 100.0, 4)


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


__all__ = ["ProgressReporter", "apply_progress_to_config", "progress_from_config"]
