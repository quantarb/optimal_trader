from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from app.live_app_shared import artifact_age_hours, inspect_saved_artifacts


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_shared_artifact_inspection_accepts_fresh_current_build(tmp_path: Path):
    artifact = tmp_path / "artifact.pkl"
    artifact.write_bytes(b"artifact")

    status = inspect_saved_artifacts(
        required_paths=[artifact],
        artifact_dir=tmp_path,
        max_age=pd.Timedelta(days=1),
        saved_score_date="2026-06-12",
        expected_score_date="2026-06-12",
    )

    assert status["reusable"] is True
    assert status["reason"] == "fresh_saved_build"
    assert artifact_age_hours(status) is not None


def test_shared_artifact_inspection_rejects_oldest_required_file(tmp_path: Path):
    fresh = tmp_path / "fresh.pkl"
    stale = tmp_path / "stale.pkl"
    fresh.write_bytes(b"fresh")
    stale.write_bytes(b"stale")
    stale_time = pd.Timestamp.now().timestamp() - pd.Timedelta(days=2).total_seconds()
    os.utime(stale, (stale_time, stale_time))

    status = inspect_saved_artifacts(
        required_paths=[fresh, stale],
        artifact_dir=tmp_path,
        max_age=pd.Timedelta(days=1),
    )

    assert status["reusable"] is False
    assert status["reason"].startswith("older_than_")


def test_both_leaderboard_pages_use_shared_ui_components():
    regular = (REPO_ROOT / "app" / "pages" / "1_Leaderboard.py").read_text(encoding="utf-8")
    moe = (REPO_ROOT / "app" / "moe" / "streamlit_moe_paper_trading.py").read_text(
        encoding="utf-8"
    )

    for component in (
        "LEADERBOARD_CSS",
        "render_leaderboard_pager",
        "render_leaderboard_ribbon",
        "render_leaderboard_table",
    ):
        assert component in regular
        assert component in moe
    assert "def _render_leaderboard_html_table" not in regular
