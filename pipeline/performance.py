from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Any, Iterator

import psutil


@dataclass
class StageMeasurement:
    name: str
    category: str
    workload_type: str
    wall_seconds: float
    cpu_seconds: float
    rss_start_mb: float
    rss_end_mb: float
    rss_delta_mb: float
    read_bytes: int
    write_bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "category": str(self.category),
            "workload_type": str(self.workload_type),
            "wall_seconds": round(float(self.wall_seconds), 6),
            "cpu_seconds": round(float(self.cpu_seconds), 6),
            "cpu_ratio": round(float(self.cpu_seconds / self.wall_seconds), 6) if self.wall_seconds > 0 else 0.0,
            "rss_start_mb": round(float(self.rss_start_mb), 4),
            "rss_end_mb": round(float(self.rss_end_mb), 4),
            "rss_delta_mb": round(float(self.rss_delta_mb), 4),
            "read_bytes": int(self.read_bytes),
            "write_bytes": int(self.write_bytes),
            "metadata": dict(self.metadata or {}),
        }


class PerformanceTracer:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._process = psutil.Process()
        self._stages: list[StageMeasurement] = []

    @property
    def stages(self) -> list[StageMeasurement]:
        return list(self._stages)

    def add_stage(self, measurement: StageMeasurement) -> None:
        if self.enabled:
            self._stages.append(measurement)

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        category: str = "compute",
        workload_type: str = "batched",
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        rss_start = self._rss_mb()
        io_start = self._io_counters()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        try:
            yield
        finally:
            wall_seconds = time.perf_counter() - wall_start
            cpu_seconds = time.process_time() - cpu_start
            rss_end = self._rss_mb()
            io_end = self._io_counters()
            self._stages.append(
                StageMeasurement(
                    name=str(name),
                    category=str(category),
                    workload_type=str(workload_type),
                    wall_seconds=float(wall_seconds),
                    cpu_seconds=float(cpu_seconds),
                    rss_start_mb=float(rss_start),
                    rss_end_mb=float(rss_end),
                    rss_delta_mb=float(rss_end - rss_start),
                    read_bytes=max(0, int(io_end.get("read_bytes", 0) - io_start.get("read_bytes", 0))),
                    write_bytes=max(0, int(io_end.get("write_bytes", 0) - io_start.get("write_bytes", 0))),
                    metadata=dict(metadata or {}),
                )
            )

    def summary(self) -> dict[str, Any]:
        stage_rows = [stage.to_dict() for stage in self._stages]
        total_wall = sum(float(stage.wall_seconds) for stage in self._stages)
        total_cpu = sum(float(stage.cpu_seconds) for stage in self._stages)
        total_read = sum(int(stage.read_bytes) for stage in self._stages)
        total_write = sum(int(stage.write_bytes) for stage in self._stages)

        by_category: dict[str, dict[str, Any]] = {}
        by_workload: dict[str, dict[str, Any]] = {}
        for stage in self._stages:
            self._accumulate(by_category, key=stage.category, stage=stage)
            self._accumulate(by_workload, key=stage.workload_type, stage=stage)

        return {
            "total_runtime_seconds": round(float(total_wall), 6),
            "total_cpu_seconds": round(float(total_cpu), 6),
            "total_read_bytes": int(total_read),
            "total_write_bytes": int(total_write),
            "stages": stage_rows,
            "by_category": by_category,
            "by_workload_type": by_workload,
        }

    def _accumulate(self, bucket: dict[str, dict[str, Any]], *, key: str, stage: StageMeasurement) -> None:
        row = bucket.setdefault(
            str(key),
            {
                "wall_seconds": 0.0,
                "cpu_seconds": 0.0,
                "read_bytes": 0,
                "write_bytes": 0,
                "rss_delta_mb": 0.0,
                "stage_count": 0,
            },
        )
        row["wall_seconds"] += float(stage.wall_seconds)
        row["cpu_seconds"] += float(stage.cpu_seconds)
        row["read_bytes"] += int(stage.read_bytes)
        row["write_bytes"] += int(stage.write_bytes)
        row["rss_delta_mb"] += float(stage.rss_delta_mb)
        row["stage_count"] += 1
        row["wall_seconds"] = round(float(row["wall_seconds"]), 6)
        row["cpu_seconds"] = round(float(row["cpu_seconds"]), 6)
        row["rss_delta_mb"] = round(float(row["rss_delta_mb"]), 4)

    def _rss_mb(self) -> float:
        return float(self._process.memory_info().rss) / (1024.0 * 1024.0)

    def _io_counters(self) -> dict[str, int]:
        try:
            counters = self._process.io_counters()
        except Exception:
            return {"read_bytes": 0, "write_bytes": 0}
        return {
            "read_bytes": int(getattr(counters, "read_bytes", 0)),
            "write_bytes": int(getattr(counters, "write_bytes", 0)),
        }


__all__ = ["PerformanceTracer", "StageMeasurement"]
