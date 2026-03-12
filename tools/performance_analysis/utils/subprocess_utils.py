from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def project_python() -> str:
    return sys.executable


def project_env(*, extra_env: dict[str, Any] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items()})
    return env


def run_subprocess(
    command: list[str],
    *,
    cwd: str | Path,
    extra_env: dict[str, Any] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(Path(cwd).resolve()),
        env=project_env(extra_env=extra_env),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
