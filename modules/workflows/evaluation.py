from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from modules.api import build_label_dataframe
from modules.data.preparation import MLDatasetConfig, prepare_ml_dataset
from modules.engine.backtest import backtest_panel, ExecutionConfig
from modules.engine.latest import run_panel_prediction_custom, make_autoencoder_familiarity_predictor
from modules.workflows.training import train_rf_models, train_ae
from modules.strategies.stateful import StatefulModelExitStrategy, EqualWeightStatefulStrategy


@dataclass(frozen=True)
class StrategyCase:
    case_name: str
    top_k: int
    gate_q: float
    long_budget: float
    short_budget: float
    use_vol_scaling: bool
    min_weight_change: float
    hold_gate_q: float = 0.50
    hold_drop_pct: float | None = None
    rebalance_freq: str = "W"


@dataclass(frozen=True)
class ProbabilityColumnConfig:
    buy_col: str = "clf__prob_buy"
    short_col: str | None = "clf__prob_short"
    infer_short_from_buy: bool = False


def enrich_scored_panel(
    scored_panel: pd.DataFrame,
    prob_config: ProbabilityColumnConfig | None = None,
) -> pd.DataFrame:
    cfg = prob_config or ProbabilityColumnConfig()
    out = scored_panel.copy()

    if cfg.buy_col not in out.columns:
        raise KeyError(f"Configured buy probability column '{cfg.buy_col}' not found.")
    out["prob_buy"] = pd.to_numeric(out[cfg.buy_col], errors="coerce").fillna(0.0)

    if cfg.short_col is not None:
        if cfg.short_col not in out.columns:
            raise KeyError(f"Configured short probability column '{cfg.short_col}' not found.")
        out["prob_short"] = pd.to_numeric(out[cfg.short_col], errors="coerce").fillna(0.0)
    elif cfg.infer_short_from_buy:
        out["prob_short"] = (1.0 - out["prob_buy"]).clip(0.0, 1.0)
    else:
        raise KeyError("Short probability column not configured. Set short_col or infer_short_from_buy=True.")

    out["pred_rf_reg"] = pd.to_numeric(out["ranking"], errors="coerce").fillna(0.0)
    out["ae_familiarity"] = pd.to_numeric(out["ae_familiarity"], errors="coerce").fillna(1.0)
    out["buy_score"] = out["prob_buy"] * out["pred_rf_reg"] * out["ae_familiarity"]
    out["short_score"] = out["prob_short"] * out["pred_rf_reg"] * out["ae_familiarity"]
    return out


def resolve_buy_probability_series(
    df: pd.DataFrame,
    prob_config: ProbabilityColumnConfig | None = None,
) -> pd.Series:
    cfg = prob_config or ProbabilityColumnConfig()
    if cfg.buy_col not in df.columns:
        raise KeyError(f"Configured buy probability column '{cfg.buy_col}' not found.")
    return pd.to_numeric(df[cfg.buy_col], errors="coerce").fillna(0.0)


def resolve_price_column(technical_df: pd.DataFrame) -> str:
    lower_to_orig = {str(c).lower(): c for c in technical_df.columns}
    price_col = lower_to_orig.get("close") or lower_to_orig.get("adj_close") or lower_to_orig.get("adjusted_close") or lower_to_orig.get("adjclose")
    if price_col is not None:
        return price_col

    close_like = [c for c in technical_df.columns if "close" in str(c).lower()]
    if not close_like:
        raise KeyError("No close-like price column found")
    return close_like[0]


def make_backtest_panel(*, scored_panel: pd.DataFrame, technical_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    price_col = resolve_price_column(technical_df)
    price_df = technical_df[[price_col]].rename(columns={price_col: "close"})
    price_df = price_df.loc[
        (price_df.index.get_level_values("date") >= start)
        & (price_df.index.get_level_values("date") <= end)
    ]
    return scored_panel.join(price_df, how="left")


def make_exec_cfg(
    *,
    fee_bps: float,
    slippage_bps: float,
    execution_mode: str = "rl_env",
) -> ExecutionConfig:
    return ExecutionConfig(
        price_col="close",
        fee_bps=float(fee_bps),
        slippage_bps=float(slippage_bps),
        use_lagged_weights=True,
        execution_mode=str(execution_mode),
    )


def run_case(
    *,
    clf_model: Any,
    panel: pd.DataFrame,
    case: StrategyCase,
    exec_cfg: ExecutionConfig,
    compute_weight_metrics: bool = True,
) -> tuple[Any, Any, dict[str, Any]]:
    strat_cls = StatefulModelExitStrategy if case.use_vol_scaling else EqualWeightStatefulStrategy
    strategy = strat_cls(
        clf_model=clf_model,
        top_k=int(case.top_k),
        hold_top_k=max(int(case.top_k) * 2, int(case.top_k) + 5),
        long_budget=float(case.long_budget),
        short_budget=float(case.short_budget),
        gate_quantile=float(case.gate_q),
        hold_gate_quantile=float(case.hold_gate_q),
        vol_window=20,
        min_weight_change=float(case.min_weight_change),
        hold_score_drop_pct=case.hold_drop_pct,
        rebalance_freq=str(case.rebalance_freq),
    )
    res = backtest_panel(panel=panel, strategy=strategy, cfg=exec_cfg)
    if compute_weight_metrics:
        w = strategy.compute_weights(panel)
        avg_gross_exposure = float(w.abs().sum(axis=1).mean())
        median_gross_exposure = float(w.abs().sum(axis=1).median())
        avg_active_names = float(((w > 0).sum(axis=1) + (w < 0).sum(axis=1)).mean())
    else:
        avg_gross_exposure = np.nan
        median_gross_exposure = np.nan
        avg_active_names = np.nan
    row = {
        "case": case.case_name,
        "top_k": case.top_k,
        "gate_q": case.gate_q,
        "hold_gate_q": case.hold_gate_q,
        "long_budget": case.long_budget,
        "short_budget": case.short_budget,
        "gross_target": case.long_budget + case.short_budget,
        "rebalance": case.rebalance_freq,
        "vol_scaling": case.use_vol_scaling,
        "min_weight_change": case.min_weight_change,
        "hold_drop_pct": case.hold_drop_pct,
        "avg_gross_exposure": avg_gross_exposure,
        "median_gross_exposure": median_gross_exposure,
        "avg_active_names": avg_active_names,
        **res.stats,
    }
    return strategy, res, row


def strategy_diagnostics(strategy: Any, panel: pd.DataFrame) -> pd.DataFrame:
    """Return per-day position/turnover diagnostics for a strategy on a panel."""
    w = strategy.compute_weights(panel)
    active_long = (w > 0).sum(axis=1)
    active_short = (w < 0).sum(axis=1)
    gross_exposure = w.abs().sum(axis=1)
    turnover_est = 0.5 * w.diff().abs().fillna(0.0).sum(axis=1)
    return pd.DataFrame(
        {
            "n_long": active_long,
            "n_short": active_short,
            "n_total": active_long + active_short,
            "gross_exposure": gross_exposure,
            "turnover_est": turnover_est,
        }
    )


def build_anchored_fold(
    *,
    test_year: int,
    anchor_train_start: pd.Timestamp,
    universe: tuple[str, ...],
    technical_df: pd.DataFrame,
    final_df: pd.DataFrame,
    k_params: dict[str, Any],
    execution_params: dict[str, Any],
    weighting_params: dict[str, Any],
    prob_config: ProbabilityColumnConfig | None = None,
) -> tuple[pd.DataFrame, Any, pd.DataFrame, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    train_cutoff = pd.Timestamp(f"{test_year - 1}-12-31")
    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end = pd.Timestamp(f"{test_year}-12-31")

    tech_train = technical_df.loc[
        (technical_df.index.get_level_values("date") >= anchor_train_start)
        & (technical_df.index.get_level_values("date") <= train_cutoff)
    ]
    symbols_in_train = set(tech_train.index.get_level_values("symbol"))
    daily_map_train = {
        s: tech_train.xs(s, level="symbol").copy()
        for s in universe
        if s in symbols_in_train
    }

    label_df_train = build_label_dataframe(
        daily_by_symbol=daily_map_train,
        k_params=k_params,
        execution_params=execution_params,
        weighting=weighting_params,
        add_rank_labels=True,
        verbose=False,
    )

    features_train = final_df.loc[
        (final_df.index.get_level_values("date") >= anchor_train_start)
        & (final_df.index.get_level_values("date") <= train_cutoff)
    ].copy()

    train_df, feature_list, _ = prepare_ml_dataset(
        features_df=features_train,
        labels_df=label_df_train,
        target_cols=["target", "trade_return", "trade_duration_days"],
        weight_col="sample_weight",
        config=MLDatasetConfig(drop_nan_features=False),
        verbose=False,
    )

    rf = train_rf_models(
        train_df,
        feature_list,
        split_ratio=1.0,
        classifier_target_col="target",
        ranking_target_col="rank_y",
        classifier_market_position_col=None,
        train_trade_return_model=True,
        trade_return_target_col="trade_return",
        train_duration_model=False,
    )
    clf = rf.clf
    reg = rf.trade_return_reg if rf.trade_return_reg is not None else rf.ranking_reg
    ae, ae_num = train_ae(train_df, feature_list)

    panel_test = final_df.loc[
        (final_df.index.get_level_values("date") >= test_start)
        & (final_df.index.get_level_values("date") <= test_end)
    ].copy()

    ae_predict = make_autoencoder_familiarity_predictor(ae_num)
    scored = run_panel_prediction_custom(
        train_data=panel_test,
        model_specs=[
            {"model": clf, "pred_col": "clf", "include_class_probs": True},
            {"model": reg, "pred_col": "ranking"},
            {"model": ae, "pred_col": "ae_familiarity", "predict_fn": lambda df, m: ae_predict(df, m)},
        ],
        market_position_value=None,
        combine_scores_fn=lambda df: resolve_buy_probability_series(df, prob_config=prob_config)
        * pd.to_numeric(df.get("ranking", 0.0), errors="coerce").fillna(0.0)
        * pd.to_numeric(df.get("ae_familiarity", 1.0), errors="coerce").fillna(1.0),
        row_filter_fn=None,
        round_decimals=None,
    )
    scored = enrich_scored_panel(scored, prob_config=prob_config)
    bt_panel = make_backtest_panel(scored_panel=scored, technical_df=technical_df, start=test_start, end=test_end)
    return bt_panel, clf, train_df, train_cutoff, test_start, test_end
