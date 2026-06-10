from __future__ import annotations

import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


class NotebookProgress:
    """Timestamped notebook logging with an optional idle heartbeat."""

    def __init__(self) -> None:
        self._last_log_time = time.time()

    def log(self, message: str) -> None:
        self._last_log_time = time.time()
        stamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{stamp}] {message}", flush=True)

    def run(
        self,
        label: str,
        fn: Callable[..., Any],
        *args,
        heartbeat_seconds: int = 30,
        **kwargs,
    ) -> Any:
        stop_event = threading.Event()
        start = time.time()
        interval = max(int(heartbeat_seconds), 1)

        def heartbeat() -> None:
            while not stop_event.wait(interval):
                if time.time() - self._last_log_time < interval:
                    continue
                elapsed = time.time() - start
                self.log(f"{label} still running... elapsed {elapsed / 60.0:.1f} min")

        self.log(f"Starting: {label}")
        thread = None
        if heartbeat_seconds and int(heartbeat_seconds) > 0:
            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
        try:
            value = fn(*args, **kwargs)
        except Exception:
            self.log(f"Failed: {label} after {(time.time() - start) / 60.0:.1f} min")
            raise
        finally:
            stop_event.set()
            if thread is not None:
                thread.join(timeout=1.0)
        self.log(f"Finished: {label} in {(time.time() - start) / 60.0:.1f} min")
        return value


def url_responds(url: str, *, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return int(response.status) < 500
    except Exception:
        return False


def _python_has_module(executable: str, module: str) -> bool:
    """Return True if the given Python executable can import the module."""
    try:
        subprocess.run(
            [executable, "-c", f"import {module}"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


class StreamlitProcess:
    """Own a Streamlit subprocess started from a notebook session."""

    def __init__(self, *, repo_root: Path, app_path: Path, logger: Callable[[str], None] = print) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.app_path = Path(app_path).resolve()
        self.logger = logger
        self.process: subprocess.Popen[str] | None = None

    def _ensure_streamlit_available(self) -> None:
        """Ensure the 'streamlit' package (and friends) are importable in the current interpreter.

        If missing, best-effort install from app/requirements_streamlit.txt using pip.
        On failure, raise a clear actionable error.
        """
        if _python_has_module(sys.executable, "streamlit"):
            return

        self.logger("streamlit not found in the current Python environment.")

        req_file = self.repo_root / "app" / "requirements_streamlit.txt"
        if req_file.exists():
            self.logger(f"Attempting to install streamlit requirements from {req_file} ...")
            self.logger("(This may take a minute the first time, especially for faiss-cpu wheels.)")
            try:
                # Use the same Python that the notebook (and future child) is using.
                install_cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
                # Run install and let its output go to the notebook logs for transparency.
                proc = subprocess.run(
                    install_cmd,
                    cwd=str(self.repo_root),
                    capture_output=True,
                    text=True,
                )
                if proc.stdout:
                    for line in proc.stdout.strip().splitlines()[-30:]:  # last bit of output
                        self.logger(f"  pip: {line}")
                if proc.returncode == 0 and _python_has_module(sys.executable, "streamlit"):
                    self.logger("Successfully installed streamlit requirements.")
                    return
                if proc.stderr:
                    for line in proc.stderr.strip().splitlines()[-20:]:
                        self.logger(f"  pip error: {line}")
            except Exception as exc:
                self.logger(f"Auto-install attempt raised: {exc}")

        # Still not available — give the user a precise command they can run.
        py = sys.executable
        raise RuntimeError(
            "streamlit is not installed in the Python environment used by this notebook "
            f"({py}).\n\n"
            "To fix, run one of the following in your terminal (with the same conda env active):\n\n"
            f"    {py} -m pip install -r app/requirements_streamlit.txt\n\n"
            "Or (conda):\n\n"
            "    conda install -c conda-forge streamlit faiss-cpu\n\n"
            "Then re-run the cell that calls streamlit_process.start(...)."
        )

    def start(
        self,
        *,
        host: str = "localhost",
        port: int = 8502,
        force_restart: bool = False,
        fallback_ports: int = 9,
        startup_timeout: int = 30,
    ) -> str:
        root_url = f"http://{host}:{int(port)}"
        if force_restart and self.process is not None:
            self.logger(f"Stopping notebook-owned Streamlit process before restart: {root_url}")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

        if force_restart and url_responds(root_url):
            for candidate_port in range(int(port) + 1, int(port) + int(fallback_ports) + 1):
                candidate_url = f"http://{host}:{candidate_port}"
                if not url_responds(candidate_url):
                    port = candidate_port
                    root_url = candidate_url
                    self.logger(f"Using fresh Streamlit fallback port: {root_url}")
                    break

        if url_responds(root_url):
            self.logger(f"Streamlit trading app already responding: {root_url}")
            return root_url

        self.logger(f"Starting Streamlit trading app: {self.app_path} on port {port}")

        self._ensure_streamlit_available()

        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(self.app_path),
                "--server.port",
                str(int(port)),
                "--server.headless",
                "true",
            ],
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for _ in range(max(int(startup_timeout), 1)):
            if url_responds(root_url):
                return root_url
            if self.process.poll() is not None:
                output = self.process.stdout.read() if self.process.stdout is not None else ""
                tail = (output or "")[-2000:]
                hint = ""
                if "No module named streamlit" in tail or "No module named 'streamlit'" in tail:
                    hint = (
                        "\n\nHint: streamlit is missing from the environment. "
                        "The launcher tried to auto-install from app/requirements_streamlit.txt. "
                        "You can also run:  python -m pip install -r app/requirements_streamlit.txt"
                    )
                raise RuntimeError(f"Streamlit exited before serving {root_url}: {tail}{hint}")
            time.sleep(1)
        raise RuntimeError(f"Streamlit did not respond at {root_url}.")


__all__ = ["NotebookProgress", "StreamlitProcess", "url_responds"]
