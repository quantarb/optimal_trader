from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ..utils.path_utils import ensure_directory, safe_slug


def run_lighthouse(url: str, *, output_dir: str | Path) -> dict:
    binary = shutil.which("lighthouse")
    if binary is None:
        return {"status": "skipped", "reason": "lighthouse_not_installed"}
    output_path = ensure_directory(output_dir) / f"{safe_slug(url)}__lighthouse.json"
    try:
        completed = subprocess.run(
            [binary, url, "--quiet", "--chrome-flags=--headless", f"--output=json", f"--output-path={output_path}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}
    if completed.returncode != 0 or not output_path.exists():
        return {"status": "failed", "reason": (completed.stderr or completed.stdout).strip()}
    return {"status": "ok", "report": json.loads(output_path.read_text(encoding="utf-8")), "path": str(output_path)}
