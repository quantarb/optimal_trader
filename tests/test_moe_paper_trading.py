from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.moe_paper_trading import (
    build_alpaca_option_trade_plan,
    build_latest_moe_scored,
    build_moe_ranked_scores,
    load_moe_paper_artifacts,
    load_recent_moe_paper_build,
    save_moe_paper_artifacts,
)
from app.live_trade_leaderboard import build_latest_scoring_panel
REPO_ROOT = Path(__file__).resolve().parents[1]


def _notebook_source(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source") or [])
        for cell in payload.get("cells") or []
    )


def test_moe_notebook_trains_full_history_and_launches_dedicated_option_app():
    source = _notebook_source(REPO_ROOT / "notebooks" / "moe_paper_trading.ipynb")
    research_source = _notebook_source(
        REPO_ROOT / "notebook_traditional_ml_synthetic_options_backtest.ipynb"
    )

    assert "PAPER_INSTRUMENT = 'otm_option'" in source
    assert "PAPER_TOP_K = 40" in source
    assert "DATA_START = '1900-01-01'" in source
    assert "os.environ['MOE_TRAIN_CUTOFF'] = DATA_END" in source
    assert "os.environ['MOE_PAPER_TRADING_MODE'] = '1'" in source
    assert "research_code_cells[:6]" in source
    assert "strategy backtests skipped" in source
    assert '%run "$MOE_RESEARCH_NOTEBOOK"' not in source
    assert "load_recent_moe_paper_build" in source
    assert "log_artifact_decision" in source
    assert "model_source = 'artifacts'" in source
    assert "create_live_app_runtime" in source
    assert "streamlit_optimal_trade_finder.py" in source
    assert (REPO_ROOT / "app" / "pages" / "2_MoE_Paper_Trading.py").exists()
    assert 'os.getenv("MOE_TRAIN_CUTOFF", "2020-12-31")' in research_source
    assert "Backtest start must be after train_cutoff" not in research_source
    assert "variant_runs" not in source
    assert "latest_targets" not in source
    assert "build_latest_moe_scored" in source


def test_moe_streamlit_page_matches_leaderboard_layout_and_uses_alpaca():
    page_source = (REPO_ROOT / "app" / "moe" / "streamlit_moe_paper_trading.py").read_text(
        encoding="utf-8"
    )

    assert "LEADERBOARD_CSS" in page_source
    assert 'class="leaderboard-hero"' in page_source
    assert "render_leaderboard_pager" in page_source
    assert "Leaderboard Table" in page_source
    assert "Alpaca Option Automation" in page_source
    assert "Option Bucket" in page_source
    assert "Option Tenor (Days)" in page_source
    assert "Current Alpaca Option Positions" in page_source
    assert "Outstanding Alpaca Option Orders" in page_source
    assert "Submit Alpaca Option Orders" in page_source
    assert "Robinhood" not in page_source
    assert 'leaderboard.sort_values("Combined Score"' in page_source
    assert 'leaderboard["Classifier Score"]' in page_source
    assert 'leaderboard["Regressor Score"]' not in page_source
    assert 'leaderboard["Autoencoder Score"]' not in page_source
    assert 'leaderboard["Similar Trades"]' in page_source
    assert 'leaderboard["Selected"]' not in page_source
    assert "Alpaca Equity Paper" not in page_source
    assert "render_leaderboard_ribbon" in page_source


def test_moe_artifacts_round_trip(tmp_path: Path):
    scored = pd.DataFrame(
        {"close": [100.0, 200.0], "prob_buy": [0.8, 0.7]},
        index=pd.Index(["aapl", "msft"], name="symbol"),
    )
    save_moe_paper_artifacts(
        artifact_dir=tmp_path,
        latest_scored=scored,
        metadata={"strategy_date": "2026-06-11", "instrument": "otm_option"},
    )
    loaded = load_moe_paper_artifacts(tmp_path)

    assert loaded.latest_scored.index.tolist() == ["AAPL", "MSFT"]
    assert loaded.metadata["instrument"] == "otm_option"


def test_recent_moe_build_reuses_saved_models_and_scores(tmp_path: Path):
    score_dir = tmp_path / "scores"
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "classifier_families.pkl").write_bytes(b"models")
    (model_dir / "classifier_families_meta.json").write_text("{}", encoding="utf-8")
    save_moe_paper_artifacts(
        artifact_dir=score_dir,
        latest_scored=pd.DataFrame(
            {"close": [100.0], "prob_buy": [0.8]},
            index=pd.Index(["AAPL"], name="symbol"),
        ),
        metadata={"strategy_date": "2026-06-12", "instrument": "otm_option"},
    )

    loaded, status = load_recent_moe_paper_build(
        artifact_dir=score_dir,
        model_artifact_dir=model_dir,
        expected_score_date="2026-06-12",
    )

    assert loaded is not None
    assert loaded.latest_scored.index.tolist() == ["AAPL"]
    assert status["reason"] == "fresh_saved_build"


def test_recent_moe_build_retrains_when_score_date_is_stale(tmp_path: Path):
    score_dir = tmp_path / "scores"
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "classifier_families.pkl").write_bytes(b"models")
    (model_dir / "classifier_families_meta.json").write_text("{}", encoding="utf-8")
    save_moe_paper_artifacts(
        artifact_dir=score_dir,
        latest_scored=pd.DataFrame(
            {"close": [100.0], "prob_buy": [0.8]},
            index=pd.Index(["AAPL"], name="symbol"),
        ),
        metadata={"strategy_date": "2026-06-11", "instrument": "otm_option"},
    )

    loaded, status = load_recent_moe_paper_build(
        artifact_dir=score_dir,
        model_artifact_dir=model_dir,
        expected_score_date="2026-06-12",
    )

    assert loaded is None
    assert "saved_latest_date_2026-06-11_lt_expected_2026-06-12" == status["reason"]


def test_ranked_scores_select_top_k_by_moe_score():
    scored = pd.DataFrame(
        {"close": [100.0, 200.0], "prob_buy": [0.8, 0.7]},
        index=pd.Index(["AAPL", "MSFT"], name="symbol"),
    )

    ranked = build_moe_ranked_scores(scored, top_k=1, threshold=0.5)

    assert ranked.index.tolist() == ["AAPL", "MSFT"]
    assert ranked["selected"].tolist() == [True, False]


def test_latest_moe_scores_exclude_symbols_without_current_date_rows():
    panel = pd.DataFrame(
        {
            "close": [100.0, 101.0, 200.0],
            "prob_buy": [0.6, 0.7, 0.8],
        },
        index=pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2026-06-11"), "AAPL"),
                (pd.Timestamp("2026-06-12"), "AAPL"),
                (pd.Timestamp("2026-06-11"), "MSFT"),
            ],
            names=["date", "symbol"],
        ),
    )

    latest, stats = build_latest_moe_scored(panel, scoring_date="2026-06-12")

    assert latest.index.tolist() == ["AAPL"]
    assert latest.loc["AAPL", "prob_buy"] == 0.7
    assert stats == {
        "symbol_count": 1,
        "exact_date_count": 1,
        "carry_forward_count": 0,
        "inactive_count": 1,
    }


def test_leaderboard_scores_exclude_symbols_without_current_date_rows():
    panel = pd.DataFrame(
        {"close": [101.0, 200.0]},
        index=pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2026-06-12"), "AAPL"),
                (pd.Timestamp("2026-06-11"), "MSFT"),
            ],
            names=["date", "symbol"],
        ),
    )

    latest, stats = build_latest_scoring_panel(
        feature_df=panel,
        scoring_date=pd.Timestamp("2026-06-12"),
    )

    assert latest.index.get_level_values("symbol").tolist() == ["AAPL"]
    assert stats["symbol_count"] == 1
    assert stats["inactive_count"] == 1


def test_stateless_alpaca_option_plan_accounts_for_positions_and_open_orders():
    ranked = build_moe_ranked_scores(
        pd.DataFrame(
            {"close": [100.0, 200.0], "prob_buy": [0.9, 0.8]},
            index=pd.Index(["AAPL", "MSFT"], name="symbol"),
        ),
        top_k=1,
    )
    plan = build_alpaca_option_trade_plan(
        ranked_scores=ranked,
        current_option_positions=[{
            "symbol": "MSFT260821C00200000", "underlying_symbol": "MSFT", "qty": "3"
        }],
        open_orders=[{
            "id": "1", "symbol": "AAPL260821C00105000", "underlying_symbol": "AAPL",
            "side": "buy", "qty": "2", "filled_qty": "0"
        }],
        option_contracts={},
        option_snapshots={},
        strategy_allocation=1_000.0,
        as_of_date="2026-06-12",
        option_bucket="otm_option",
        tenor_days=60,
    )

    orders = plan["actionable_orders"].set_index("symbol")
    assert "AAPL260821C00105000" not in orders.index
    assert orders.loc["MSFT260821C00200000", "side"] == "sell"
    assert plan["summary"].iloc[0]["pending_buy_underlyings"] == 1


def test_alpaca_option_plan_selects_otm_call_and_sizes_contracts():
    ranked = build_moe_ranked_scores(
        pd.DataFrame(
            {"close": [100.0], "prob_buy": [0.9]},
            index=pd.Index(["AAPL"], name="symbol"),
        ),
        top_k=1,
    )
    option_symbol = "AAPL260814C00105000"
    plan = build_alpaca_option_trade_plan(
        ranked_scores=ranked,
        current_option_positions=[],
        open_orders=[],
        option_contracts={"AAPL": [
            {"symbol": "AAPL260814C00100000", "expiration_date": "2026-08-14", "strike_price": "100"},
            {"symbol": option_symbol, "expiration_date": "2026-08-14", "strike_price": "105"},
        ]},
        option_snapshots={option_symbol: {"latestQuote": {"bp": 2.0, "ap": 2.5}}},
        strategy_allocation=1_000.0,
        as_of_date="2026-06-12",
        option_bucket="otm_option",
        tenor_days=60,
    )

    order = plan["actionable_orders"].iloc[0]
    assert order["symbol"] == option_symbol
    assert order["action"] == "buy_to_open_call"
    assert order["qty"] == 5
    assert order["limit_price"] == 2.0
