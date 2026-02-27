# ============================================================
# modules/engine/backtest.py
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from modules.signals.predictors import run_signal_producers, SignalProducer
from modules.utils.panel import ensure_panel_index


# ============================================================
# Strategy protocol
# ============================================================
class Strategy(Protocol):
    @property
    def name(self) -> str: ...

    def compute_weights(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Return weights as DataFrame indexed by date, columns are symbols.
        Weights are target weights at end of day t (applied on t+1 if lagging).
        """
        ...


# ============================================================
# Config + Results
# ============================================================
class ExecutionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    price_col: str = "close"
    fee_bps: float = 2.0
    slippage_bps: float = 2.0

    # Common backtest conventions:
    # - If True: weights(t) are applied to returns(t+1) (avoids lookahead)
    use_lagged_weights: bool = True

    # Turnover definition:
    # - If True: turnover = 0.5 * sum(|w_t - w_{t-1}|)
    # - Else: turnover = sum(|w_t - w_{t-1}|)
    turnover_half_l1: bool = True
    # Execution engine:
    # - "weights": legacy weight x return engine
    # - "rl_env": discrete action engine used by SB3 workflows
    execution_mode: str = "weights"


@dataclass(frozen=True)
class BacktestResult:
    strategy_name: str
    stats: Dict[str, Any]
    equity_curve: pd.Series  # index=date
    returns: pd.Series  # index=date
    turnover: pd.Series  # index=date
    costs: pd.Series  # index=date


# ============================================================
# Panel builder
# ============================================================
def build_panel_from_daily_by_symbol(
        daily_by_symbol: Dict[str, pd.DataFrame],
        *,
        include_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    frames = []
    for sym, df in daily_by_symbol.items():
        if df is None or len(df) == 0:
            continue
        d = df.copy()
        d.columns = [str(c).strip().lower() for c in d.columns]

        if not isinstance(d.index, pd.DatetimeIndex):
            if "date" in d.columns:
                d["date"] = pd.to_datetime(d["date"], errors="coerce")
                d = d.set_index("date")
            else:
                d.index = pd.to_datetime(d.index, errors="coerce")
        d = d.sort_index()

        # Ensure per-symbol unique dates (required for later unstack)
        if d.index.has_duplicates:
            d = d[~d.index.duplicated(keep="last")]

        if include_cols is not None:
            keep = [c for c in include_cols if c in d.columns]
            d = d[keep].copy()

        d["__sym__"] = str(sym)
        d = d.set_index("__sym__", append=True)
        d.index = d.index.set_names(["date", "symbol"])
        frames.append(d)

    if not frames:
        out = pd.DataFrame()
        out.index = pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"])
        return out

    panel = pd.concat(frames, axis=0).sort_index()
    return ensure_panel_index(panel)


# ============================================================
# Backtest core
# ============================================================
def _align_prices(panel: pd.DataFrame, price_col: str) -> pd.DataFrame:
    if price_col not in panel.columns:
        raise ValueError(f"panel missing required price column '{price_col}'")
    px = panel[price_col].unstack("symbol").sort_index()
    px = px.apply(pd.to_numeric, errors="coerce")
    return px


def _compute_symbol_returns(px: pd.DataFrame) -> pd.DataFrame:
    """Computes daily pct change without lookahead padding."""
    # Setting fill_method=None fixes the FutureWarning and prevents
    # synthetic returns from missing price data.
    r = px.pct_change(fill_method=None)
    return r.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _compute_turnover(weights: pd.DataFrame, *, half_l1: bool) -> pd.Series:
    """Computes turnover (L1 norm of weight changes)."""
    w = weights.fillna(0.0)
    # diff() computes w[t] - w[t-1]. fillna(0.0) treats start as cash.
    dw = w.diff().abs().fillna(0.0)
    t = dw.sum(axis=1)
    if half_l1:
        t = 0.5 * t
    return t


def _portfolio_returns(
        w: pd.DataFrame,
        sym_ret: pd.DataFrame,
        *,
        lag_weights: bool,
) -> pd.Series:
    """Computes the dot product of weights and asset returns."""
    #
    w = w.reindex(sym_ret.index).reindex(columns=sym_ret.columns).fillna(0.0)

    if lag_weights:
        # Avoid lookahead bias: weights from end of day t apply to day t+1 returns
        w_use = w.shift(1).fillna(0.0)
    else:
        w_use = w

    pret = (w_use * sym_ret).sum(axis=1)
    pret.name = "portfolio_return"
    return pret


def _weights_to_discrete_actions(
        weights: pd.DataFrame,
        *,
        eps: float = 1e-12,
) -> np.ndarray:
    """
    Convert target weights into per-symbol discrete actions:
      0 = hold, 1 = buy/increase long, 2 = sell/decrease/exit long.
    """
    w = weights.fillna(0.0).to_numpy(dtype=float)
    if w.shape[0] == 0:
        return np.zeros_like(w, dtype=int)
    prev = np.zeros(w.shape[1], dtype=float)
    actions = np.zeros_like(w, dtype=int)
    for t in range(w.shape[0]):
        cur = w[t]
        actions[t] = np.where(cur > prev + eps, 1, np.where(cur < prev - eps, 2, 0))
        prev = cur
    return actions.astype(int)


def backtest_panel(
        panel: pd.DataFrame,
        *,
        strategy: Strategy,
        cfg: ExecutionConfig,
        signal_producers: Optional[Sequence[SignalProducer]] = None,
        keep_panel_in_result: bool = False,
) -> BacktestResult:
    panel0 = ensure_panel_index(panel)
    if signal_producers:
        panel0 = run_signal_producers(panel0, signal_producers)

    px = _align_prices(panel0, cfg.price_col)
    sym_ret = _compute_symbol_returns(px)

    w = strategy.compute_weights(panel0)
    w = w.reindex(sym_ret.index).reindex(columns=sym_ret.columns).fillna(0.0)

    if str(cfg.execution_mode).lower() == "rl_env":
        # Reuse the same discrete per-stock execution engine as RL.
        from modules.models.stable_baselines3 import backtest_strategy_per_stock_discrete

        if (w < -1e-12).any().any():
            # Current RL environment implementation is long-only.
            # Fall back to legacy engine for strategies with short weights.
            pass
        else:
            # Keep execution numerically stable with sparse panels.
            px_exec = px.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
            action_type = _weights_to_discrete_actions(w)
            eq_abs, ret_abs, _cash = backtest_strategy_per_stock_discrete(
                action_type=action_type,
                close_by_day=px_exec,
                eligible_by_day=None,
                buy_score_by_day=None,
                initial_balance=1.0,
                fee_bps=float(cfg.fee_bps),
                slippage_bps=float(cfg.slippage_bps),
                max_buys_per_day=None,
                rebalance_mask=None,
            )
            turnover = _compute_turnover(w, half_l1=cfg.turnover_half_l1)
            costs = pd.Series(0.0, index=eq_abs.index, name="costs")
            equity = eq_abs.rename("equity")
            pret_net = ret_abs.rename("portfolio_return_net")
            stats = _summarize_stats(pret_net, equity, turnover, cfg)
            return BacktestResult(
                strategy_name=strategy.name,
                stats=stats,
                equity_curve=equity,
                returns=pret_net,
                turnover=turnover,
                costs=costs,
            )

    turnover = _compute_turnover(w, half_l1=cfg.turnover_half_l1)

    # Calculate costs (bps converted to decimal)
    cost_bps = float(cfg.fee_bps) + float(cfg.slippage_bps)
    costs = (turnover * (cost_bps / 10000.0)).astype(float)
    costs.name = "costs"

    pret_gross = _portfolio_returns(w, sym_ret, lag_weights=cfg.use_lagged_weights)
    pret_net = (pret_gross - costs).astype(float)
    pret_net.name = "portfolio_return_net"

    # Cumulative growth
    equity = (1.0 + pret_net).cumprod()
    equity.name = "equity"

    stats = _summarize_stats(pret_net, equity, turnover, cfg)

    return BacktestResult(
        strategy_name=strategy.name,
        stats=stats,
        equity_curve=equity,
        returns=pret_net,
        turnover=turnover,
        costs=costs,
    )


def _summarize_stats(
        ret: pd.Series,
        equity: pd.Series,
        turnover: pd.Series,
        cfg: ExecutionConfig,
) -> Dict[str, Any]:
    r = ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) > 0 else 0.0

    # Annualization assuming daily frequency
    ann_factor = 252.0
    mu = float(r.mean())
    sig = float(r.std(ddof=0))
    sharpe = (mu / sig * np.sqrt(ann_factor)) if sig > 1e-12 else np.nan

    # Max Drawdown
    if len(equity) > 0:
        peak = equity.cummax()
        dd = (equity / peak) - 1.0
        mdd = float(dd.min())
    else:
        mdd = 0.0

    avg_turnover = float(turnover.mean()) if len(turnover) > 0 else 0.0

    return {
        "total_return_pct": 100.0 * total_return,
        "sharpe": sharpe,
        "max_drawdown_pct": 100.0 * mdd,
        "avg_turnover": avg_turnover,
        "fee_bps": float(cfg.fee_bps),
        "slippage_bps": float(cfg.slippage_bps),
    }

# ============================================================
# Canonical public entrypoint used by experiment recipes
# ============================================================
def run_backtest(*, panel: pd.DataFrame, spec: Any, title: str | None = None) -> tuple[Any, pd.DataFrame]:
    """Run a backtest using the strategy stored on `spec`.

    This is the canonical entrypoint for callers that want a simple:
        (result, one_row_table)

    Notes:
    - `spec.strategy` must be set by the caller (see backtest_recipes).
    - We intentionally keep this thin and delegate core mechanics to `backtest_panel`.
    """
    strategy = getattr(spec, "strategy", None)
    if strategy is None:
        raise ValueError("spec.strategy is required for run_backtest(panel=..., spec=...).")

    # Most callers store execution config on `spec.execution` (pydantic model or dict-like).
    cfg = getattr(spec, "execution", None)
    if cfg is None:
        cfg = getattr(spec, "execution_params", None)

    res = backtest_panel(panel=panel, strategy=strategy, cfg=cfg)

    # One-row stats table (what notebooks typically display / concatenate)
    tbl = pd.DataFrame([res.stats])
    if title is not None:
        tbl.insert(0, "title", title)
    return res, tbl
