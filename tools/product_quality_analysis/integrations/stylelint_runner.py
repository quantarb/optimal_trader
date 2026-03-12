from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def run_stylelint(*, root: str | Path) -> dict:
    binary = shutil.which("stylelint")
    if binary is None:
        return {"status": "skipped", "reason": "stylelint_not_installed"}
    try:
        completed = subprocess.run(
            [binary, "**/*.css", "**/*.html"],
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(root).resolve(),
            timeout=60,
        )
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}
    return {
        "status": "ok" if completed.returncode == 0 else "failed",
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
