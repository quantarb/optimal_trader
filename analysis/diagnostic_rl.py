from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from ml import RLConfig, run_a2c_workflow, run_ppo_workflow

from pipeline.models import Artifact
from pipeline.service_runtime import read_frame_artifact


DEFAULT_RL_YEARS = (2024, 2025)
DEFAULT_RL_ALGORITHMS = ("ppo",)
DEFAULT_RL_ELIGIBILITY_QUANTILES = (0.5, 0.6)
DEFAULT_RL_MAX_STOCKS = (2, 3, 5)
RL_LOOKBACK_WINDOW = 20
RL_REBALANCE_FREQ = "W"
RL_INITIAL_BALANCE = 100000.0
RL_DRAWDOWN_PENALTY_LAMBDA = 0.10
RL_DEFAULT_SEED = 42


def _read_csv_artifact(artifact: Artifact) -> pd.DataFrame:
    path = Path(str(artifact.uri or ""))
    if not path.exists():
        raise ValueError(f"Artifact #{artifact.id} file does not exist.")
    return read_frame_artifact(artifact)


def _feature_artifact_for_strategy(strategy_artifact: Artifact) -> Artifact:
    feature_artifact_id = int((strategy_artifact.metadata or {}).get("source_features_artifact_id") or 0)
    feature_artifact = Artifact.objects.filter(pk=feature_artifact_id, artifact_type="FEATURES").first()
    if feature_artifact is None:
        raise ValueError("RL diagnostics require the source FEATURES artifact.")
    return feature_artifact


def _rl_backtest_panel(strategy_artifact: Artifact, panel: pd.DataFrame) -> pd.DataFrame:
    features = _read_csv_artifact(_feature_artifact_for_strategy(strategy_artifact))
    if "close" not in features.columns:
        raise ValueError("RL diagnostics require 'close' in the FEATURES artifact.")
    bt_panel = features[["date", "symbol", "close"]].copy()
    bt_panel = bt_panel.merge(
        panel[["date", "symbol", "prob_buy", "pred_rf_reg", "ae_familiarity"]],
        on=["date", "symbol"],
        how="inner",
    )
    return bt_panel.dropna().set_index(["date", "symbol"]).sort_index()


def _rl_config(
    *,
    backtest_cfg: dict[str, Any],
    eligibility_quantile: float,
    max_stocks_per_day: int,
    rl_episodes: int,
) -> RLConfig:
    return RLConfig(
        lookback_window=RL_LOOKBACK_WINDOW,
        eligibility_quantile=float(eligibility_quantile),
        rebalance_freq=RL_REBALANCE_FREQ,
        max_stocks_per_day=int(max_stocks_per_day),
        initial_balance=RL_INITIAL_BALANCE,
        fee_bps=float(backtest_cfg.get("fee_bps") or 5.0),
        slippage_bps=float(backtest_cfg.get("slippage_bps") or 5.0),
        ppo_episodes=int(rl_episodes),
        drawdown_penalty_lambda=RL_DRAWDOWN_PENALTY_LAMBDA,
        seed=RL_DEFAULT_SEED,
    )


def _rl_result_row(
    *,
    algorithm: str,
    eligibility_quantile: float,
    max_stocks_per_day: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    summary = result["rl_summary_df"].iloc[0].to_dict()
    return {
        "algorithm": algorithm,
        "eligibility_quantile": float(eligibility_quantile),
        "max_stocks_per_day": int(max_stocks_per_day),
        "combined_total_return_pct": float(summary.get("combined_total_return_pct") or 0.0),
        "combined_sharpe": float(summary.get("combined_sharpe") or 0.0),
        "combined_max_drawdown_pct": float(summary.get("combined_max_drawdown_pct") or 0.0),
        "rebalance_days": int(summary.get("rebalance_days") or 0),
        "executed_buys": int(result["executed_action_counts"].get("buy", 0)),
        "executed_sells": int(result["executed_action_counts"].get("sell", 0)),
    }


def run_rl_diagnostics(
    *,
    panel: pd.DataFrame,
    strategy_artifact: Artifact | None,
    backtest_cfg: dict[str, Any],
    rl_train_split_date: str,
    rl_years: list[int] | None,
    rl_algorithms: list[str] | None,
    rl_eligibility_quantiles: list[float] | None,
    rl_max_stocks: list[int] | None,
    rl_episodes: int,
) -> list[dict[str, Any]]:
    if strategy_artifact is None:
        raise ValueError("RL diagnostics require a strategy or scored panel reference.")
    bt_panel = _rl_backtest_panel(strategy_artifact, panel)
    split_ts = pd.Timestamp(str(rl_train_split_date))
    rl_rows: list[dict[str, Any]] = []
    rl_year_values = list(rl_years or DEFAULT_RL_YEARS)
    rl_algo_values = [str(value).lower() for value in list(rl_algorithms or DEFAULT_RL_ALGORITHMS)]
    rl_quantile_values = list(rl_eligibility_quantiles or DEFAULT_RL_ELIGIBILITY_QUANTILES)
    rl_max_stock_values = list(rl_max_stocks or DEFAULT_RL_MAX_STOCKS)
    for algo, eligibility_quantile, max_stocks_per_day in product(
        rl_algo_values,
        rl_quantile_values,
        rl_max_stock_values,
    ):
        cfg = _rl_config(
            backtest_cfg=backtest_cfg,
            eligibility_quantile=float(eligibility_quantile),
            max_stocks_per_day=int(max_stocks_per_day),
            rl_episodes=rl_episodes,
        )
        runner = run_ppo_workflow if algo == "ppo" else run_a2c_workflow
        result = runner(bt_panel=bt_panel, cfg=cfg, train_split_date=split_ts, years=rl_year_values)
        rl_rows.append(
            _rl_result_row(
                algorithm=algo,
                eligibility_quantile=float(eligibility_quantile),
                max_stocks_per_day=int(max_stocks_per_day),
                result=result,
            )
        )
    rl_rows.sort(key=lambda row: float(row.get("combined_total_return_pct") or 0.0), reverse=True)
    return rl_rows
