from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops

from .path_utils import ensure_directory


def compare_screenshots(baseline_path: str | Path, current_path: str | Path) -> dict[str, float | str | None]:
    baseline = Path(baseline_path).resolve()
    current = Path(current_path).resolve()
    if not baseline.exists() or not current.exists():
        return {"status": "missing", "diff_ratio": None}
    with Image.open(baseline) as left_image, Image.open(current) as right_image:
        if left_image.size != right_image.size:
            return {"status": "size_mismatch", "diff_ratio": 1.0}
        diff = ImageChops.difference(left_image.convert("RGB"), right_image.convert("RGB"))
        histogram = diff.histogram()
        total_pixels = float(left_image.size[0] * left_image.size[1] * 3)
        weighted = sum(index * count for index, count in enumerate(histogram))
        diff_ratio = min(1.0, weighted / max(total_pixels * 255.0, 1.0))
        return {"status": "ok", "diff_ratio": round(diff_ratio, 6)}


def copy_baseline(source_path: str | Path, target_path: str | Path) -> Path:
    source = Path(source_path).resolve()
    target = Path(target_path).resolve()
    ensure_directory(target.parent)
    target.write_bytes(source.read_bytes())
    return target
