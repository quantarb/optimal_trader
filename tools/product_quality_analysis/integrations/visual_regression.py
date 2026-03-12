from __future__ import annotations

from pathlib import Path

from ..models import PageSnapshot
from ..utils.path_utils import ensure_directory, safe_slug
from ..utils.screenshot_utils import compare_screenshots, copy_baseline


def run_visual_regression(
    snapshots: list[PageSnapshot],
    *,
    baseline_dir: str | Path,
    current_label: str,
) -> dict:
    baseline_root = ensure_directory(baseline_dir)
    results: list[dict] = []
    for snapshot in snapshots:
        current_path = Path(snapshot.screenshot_path).resolve()
        baseline_path = baseline_root / f"{safe_slug(snapshot.name)}.png"
        if not baseline_path.exists() and current_path.exists():
            copy_baseline(current_path, baseline_path)
            results.append({"page": snapshot.name, "status": "baseline_created", "baseline_path": str(baseline_path)})
            continue
        results.append(
            {
                "page": snapshot.name,
                "baseline_path": str(baseline_path),
                "current_path": str(current_path),
                **compare_screenshots(baseline_path, current_path),
                "label": current_label,
            }
        )
    return {"results": results, "baseline_dir": str(baseline_root)}
