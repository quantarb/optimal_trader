from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "trading_app_v2.ipynb"


def _source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_notebook_trades_only_the_equity_meta_stack():
    source = _source()

    assert "train_equity_meta_stack(" in source
    assert "strategy_scores = equity_meta_result.scores.copy()" in source
    assert "equity_family_scores=family_scores" in source
    assert "build_score_ensemble(" not in source
    assert '"training_prediction_scope": "in_sample_same_oracle_rows"' not in source
    assert 'ta_mode="curated"' in source


def test_historical_option_backfill_runs_after_live_artifacts_with_a_daily_bound():
    source = _source()

    save_position = source.index("saved_live_paths = save_live_artifacts(")
    frontend_position = source.index("streamlit_app = write_streamlit_leaderboard_app(")
    backfill_position = source.rindex("backfill_thetadata_for_oracle_trade_windows(")
    assert save_position < frontend_position < backfill_position
    assert "max_trades=THETADATA_ORACLE_BACKFILL_MAX_TRADES" in source
    assert 'TRADING_APP_V2_OPTION_BACKFILL_MAX_TRADES", "25"' in source
    assert "failed_after_live_artifacts" in source


def test_notebook_uses_only_requested_ye_k1_through_k3_labels():
    source = _source()

    assert 'oracle_trade_k_by_frequency={"YE": tuple(range(1, 4))}' in source
    assert "for k in range(1, 4)" in source
    assert "YE k=1..12" not in source
