from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "trading_app_v2.ipynb"
OPTION_NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "trading_app_v2_option_ml_ranker.ipynb"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def _option_source() -> str:
    notebook = json.loads(OPTION_NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_notebook_trades_only_the_equity_meta_stack():
    source = _source()

    assert "train_equity_meta_stack(" in source
    assert "strategy_scores = equity_meta_result.scores.copy()" in source
    assert "equity_family_scores=family_scores" in source
    assert "build_score_ensemble(" not in source
    assert '"training_prediction_scope": "in_sample_same_oracle_rows"' not in source
    assert '"100" if MIN_MARKET_CAP >= 1_000_000_000_000 else "250"' in source
    assert "min_train_rows=MIN_FAMILY_TRAIN_ROWS" in source


def test_option_downloads_are_limited_to_score_date_refresh():
    source = _source()

    assert "backfill_thetadata_eod_for_score_date(" in source
    assert "backfill_thetadata_for_oracle_trade_windows(" not in source
    assert "backfill_thetadata_options_for_oracle_trade_ranges(" not in source
    assert "THETADATA_ORACLE_BACKFILL_MAX_TRADES" not in source
    assert "thetadata_historical_backfill_summary.json" not in source


def test_notebook_uses_only_requested_ye_k1_through_k3_labels():
    source = _source()

    assert 'oracle_trade_k_by_frequency={"YE": tuple(range(1, 4))}' in source
    assert "for k in range(1, 4)" in source
    assert "YE k=1..12" not in source


def test_option_notebook_preserves_unified_targets_and_uses_cuda():
    source = _option_source()

    assert "ORACLE_YE_K = (1, 2, 3)" in source
    assert "OPTION_MAX_DTE = None" in source
    assert 'dropna(subset=["trade_id", "entry_date", "rank_y", "symbol"])' in source
    assert 'dropna(subset=["trade_id", "entry_date", "option_return", "symbol"])' not in source
    assert "rank_y is the immutable unified target" in source
    assert '_select_diverse_option_candidates(group, int(train_top_k_by_return))' in source
    assert 'model_backend="rapids_random_forest"' in source
    assert '"option_target_contract": OPTION_TARGET_CONTRACT' in source


def test_scale_runner_is_parameterized_and_technicals_are_mandatory():
    tier_names = ("1t", "100b", "10b")
    script_names = [path.name.lower() for path in SCRIPTS.glob("*.py")]
    assert not any(tier in name for tier in tier_names for name in script_names)

    runner = (SCRIPTS / "run_option_meta.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--min-market-cap"' in runner
    assert "build_fundamental_feature_panel(" in runner
    assert "build_technical_feature_panel(" in runner
    assert "build_technical_feature_panel(" in runner
    assert "Expected all six curated technical families" in runner
    assert "_build_symbol_fundamental_panel" not in runner
    assert "build_price_ta_classic_feature_families" not in runner
