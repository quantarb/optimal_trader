from __future__ import annotations

import cProfile
import io
import pstats
import time
from pathlib import Path

from ..config import BenchmarkDefaults
from ..models import RuntimeHotspot, RuntimeProfileReport
from ..utils.path_utils import ensure_directory, safe_relative_path
from ..utils.report_utils import utc_timestamp, write_markdown
from .profile_targets import prepare_profile_environment, resolve_profile_target

try:
    from pyinstrument import Profiler
    HAS_PYINSTRUMENT = True
except ImportError:
    HAS_PYINSTRUMENT = False


def _stage_hotspots(result: dict) -> list[dict]:
    return sorted(list((result.get("performance") or {}).get("stages") or []), key=lambda row: float(row.get("wall_seconds") or 0.0), reverse=True)[:12]


def profile_runtime(target: str, *, output_dir: str | Path, benchmark: BenchmarkDefaults | None = None) -> RuntimeProfileReport:
    out = ensure_directory(output_dir)
    prepare_profile_environment()
    fn = resolve_profile_target(target, benchmark=benchmark)
    if HAS_PYINSTRUMENT:
        profiler = Profiler()
        started = time.perf_counter()
        profiler.start()
        result = fn()
        profiler.stop()
        total = time.perf_counter() - started
        raw = write_markdown(out / f"runtime_profile_{target}.txt", profiler.output_text(unicode=True, color=False, show_all=False))
        return RuntimeProfileReport(utc_timestamp(), "pyinstrument", target, round(total, 6), str(raw), [], _stage_hotspots(result), ["Used pyinstrument for whole-workflow profiling."])
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    result = fn()
    profiler.disable()
    total = time.perf_counter() - started
    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(30)
    raw = write_markdown(out / f"runtime_profile_{target}.txt", buf.getvalue())
    hotspots: list[RuntimeHotspot] = []
    for (filename, line, name), values in sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True):
        if filename.startswith("<") or "/lib/python" in filename:
            continue
        relative_path = safe_relative_path(filename, Path.cwd())
        if relative_path.startswith("tools/performance_analysis/") or relative_path == "~":
            continue
        ccalls, ncalls, total_seconds, cumulative_seconds = values[:4]
        hotspots.append(RuntimeHotspot(name, relative_path, int(line), f"{ccalls}/{ncalls}", round(float(cumulative_seconds), 6), round(float(total_seconds), 6), round((float(cumulative_seconds) / total) * 100.0, 4) if total > 0 else 0.0))
        if len(hotspots) >= 20:
            break
    return RuntimeProfileReport(utc_timestamp(), "cProfile", target, round(total, 6), str(raw), hotspots, _stage_hotspots(result), ["Fell back to cProfile because pyinstrument is not installed."])
