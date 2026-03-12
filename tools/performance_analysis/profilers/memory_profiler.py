from __future__ import annotations

import resource
import sys
import tracemalloc
from pathlib import Path

from ..config import BenchmarkDefaults
from ..models import MemoryHotspot, MemoryProfileReport
from ..utils.path_utils import ensure_directory, safe_relative_path
from ..utils.report_utils import utc_timestamp, write_markdown
from .profile_targets import prepare_profile_environment, resolve_profile_target


def _peak_rss_mb() -> float:
    scale = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return round(float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / scale, 4)


def _stage_hotspots(result: dict) -> list[dict]:
    return sorted(list((result.get("performance") or {}).get("stages") or []), key=lambda row: float(row.get("rss_delta_mb") or 0.0), reverse=True)[:12]


def profile_memory(target: str, *, output_dir: str | Path, benchmark: BenchmarkDefaults | None = None) -> MemoryProfileReport:
    out = ensure_directory(output_dir)
    prepare_profile_environment()
    fn = resolve_profile_target(target, benchmark=benchmark)
    tracemalloc.start()
    result = fn()
    _current, peak = tracemalloc.get_traced_memory()
    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()
    hotspots = [MemoryHotspot(safe_relative_path(stat.traceback[0].filename, Path.cwd()), int(stat.traceback[0].lineno), round(float(stat.size) / (1024.0 * 1024.0), 6), int(stat.count), " | ".join(f"{frame.filename}:{frame.lineno}" for frame in stat.traceback[:3])) for stat in snapshot.statistics("lineno")[:20]]
    raw = write_markdown(out / f"memory_profile_{target}.txt", "\n".join(f"{row.path}:{row.line} {row.size_mb:.4f} MB ({row.count})" for row in hotspots))
    return MemoryProfileReport(utc_timestamp(), "tracemalloc", target, _peak_rss_mb(), round(float(peak) / (1024.0 * 1024.0), 6), str(raw), hotspots, _stage_hotspots(result), ["Fell back to tracemalloc because scalene/memory_profiler are not installed."])
