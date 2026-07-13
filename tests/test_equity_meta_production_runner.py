from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_equity_meta_notebook.py"


def test_production_runner_uses_all_oracle_labels_and_latest_only_scoring():
    source = RUNNER.read_text(encoding="utf-8")

    assert 'choices=("oracle_only",)' in source
    assert 'g["RUN_BACKTESTS"] = False' in source
    assert 'g["RUN_ANCHORED_WFO"] = False' in source
    assert 'g["RUN_SYMBOL_LEVEL_BACKTESTING_PY"] = False' in source
    assert 'value_cols = ["long_score"]' in source
    assert 'strategy_scores["score_rank"]' in source
    assert '"training_scope": "all_available_labels"' in source
    assert '"model_validation_run": False' in source

