from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RLConfig:
    lookback_window: int = 20
    eligibility_quantile: float = 0.50
    rebalance_freq: str | None = "W"
    max_stocks_per_day: int | None = 5
    initial_balance: float = 100000.0
    fee_bps: float = 5.0
    slippage_bps: float = 5.0
    ppo_episodes: int = 3000
    drawdown_penalty_lambda: float = 0.10
    seed: int = 42
    force_buy_sell_everything: bool = False


def trade_cost_from_bps(gross: float, fee_bps: float, slippage_bps: float) -> float:
    return float(gross) * (float(fee_bps) + float(slippage_bps)) / 10000.0


def apply_buy_cap(action_type: np.ndarray, max_buys_per_day: int | None, scores: np.ndarray | None = None) -> np.ndarray:
    at = np.asarray(action_type, dtype=int).copy()
    if max_buys_per_day is None:
        return at

    k = int(max_buys_per_day)
    if k < 0:
        raise ValueError("max_buys_per_day must be >= 0 or None.")

    buy_idx = np.where(at == 1)[0]
    if len(buy_idx) <= k:
        return at
    if k == 0:
        at[buy_idx] = 0
        return at

    if scores is None:
        keep = buy_idx[:k]
    else:
        s = np.asarray(scores, dtype=float)
        keep_local = np.argsort(-s[buy_idx], kind="stable")[:k]
        keep = buy_idx[keep_local]

    drop = np.setdiff1d(buy_idx, keep, assume_unique=False)
    at[drop] = 0
    return at


def make_rebalance_mask(dates: pd.DatetimeIndex, freq: str | None) -> np.ndarray:
    if freq is None:
        return np.ones(len(dates), dtype=bool)
    freq_u = str(freq).upper()
    period_freq = "W-FRI" if freq_u == "W" else freq_u
    ds = pd.Series(dates, index=dates)
    rebalance_dates = ds.groupby(dates.to_period(period_freq)).max().values
    return dates.isin(pd.DatetimeIndex(rebalance_dates))


def backtest_strategy_per_stock_discrete(
    *,
    action_type: np.ndarray,
    close_by_day: pd.DataFrame,
    eligible_by_day: np.ndarray | None,
    buy_score_by_day: np.ndarray | None,
    initial_balance: float,
    fee_bps: float,
    slippage_bps: float,
    max_buys_per_day: int | None,
    rebalance_mask: np.ndarray | None,
    return_execution_stats: bool = False,
    return_trade_log: bool = False,
) -> tuple[pd.Series, pd.Series, pd.Series] | tuple[pd.Series, pd.Series, pd.Series, dict[str, Any]]:
    prices = close_by_day.to_numpy(dtype=float)
    t_n, s_n = prices.shape

    balance = float(initial_balance)
    shares = np.zeros(s_n, dtype=float)
    cost_rate = (float(fee_bps) + float(slippage_bps)) / 10000.0

    equity = np.zeros(t_n, dtype=float)
    cash = np.zeros(t_n, dtype=float)
    exec_buy_count = 0
    exec_sell_count = 0
    trade_rows: list[dict[str, Any]] = []

    for t in range(t_n):
        px = prices[t]
        at = np.asarray(action_type[t], dtype=int)

        if rebalance_mask is not None and (not bool(rebalance_mask[t])):
            at = np.zeros_like(at)

        if eligible_by_day is not None:
            elig_t = np.asarray(eligible_by_day[t], dtype=bool)
            at = np.where((at == 1) & (~elig_t), 0, at)

        score_t = None if buy_score_by_day is None else np.asarray(buy_score_by_day[t], dtype=float)
        at = apply_buy_cap(at, max_buys_per_day, scores=score_t)

        for j in np.where(at == 2)[0]:
            if px[j] <= 0:
                continue
            shares_to_sell = shares[j]
            if shares_to_sell <= 0:
                continue
            gross = shares_to_sell * px[j]
            fee = trade_cost_from_bps(gross, fee_bps, slippage_bps)
            shares[j] = 0.0
            balance += gross - fee
            exec_sell_count += 1
            if return_trade_log:
                trade_rows.append(
                    {
                        "date": close_by_day.index[t],
                        "symbol": close_by_day.columns[j],
                        "side": "sell",
                        "price": float(px[j]),
                        "shares": float(shares_to_sell),
                        "gross": float(gross),
                        "fee": float(fee),
                        "net_cash_flow": float(gross - fee),
                    }
                )

        net_worth = balance + float(np.sum(shares * px))
        buy_idx_today = np.where(at == 1)[0]
        target_slots_today = int(len(buy_idx_today))
        day_buy_budget = max(0.0, float(balance))
        if target_slots_today > 0:
            target_notional = net_worth / float(target_slots_today)
        for j in buy_idx_today:
            if px[j] <= 0 or day_buy_budget <= 0:
                continue
            current_notional = shares[j] * px[j]
            need_gross = max(0.0, target_notional - current_notional)
            if need_gross <= 0:
                continue

            max_affordable_gross = day_buy_budget / (1.0 + cost_rate)
            gross = min(need_gross, max_affordable_gross)
            if gross <= 0:
                continue

            fee = trade_cost_from_bps(gross, fee_bps, slippage_bps)
            total_spend = gross + fee
            shares_to_buy = gross / px[j] if px[j] > 0 else 0.0
            if shares_to_buy <= 0:
                continue
            shares[j] += shares_to_buy
            day_buy_budget -= total_spend
            exec_buy_count += 1
            if return_trade_log:
                trade_rows.append(
                    {
                        "date": close_by_day.index[t],
                        "symbol": close_by_day.columns[j],
                        "side": "buy",
                        "price": float(px[j]),
                        "shares": float(shares_to_buy),
                        "gross": float(gross),
                        "fee": float(fee),
                        "net_cash_flow": float(-total_spend),
                    }
                )

        balance = day_buy_budget if day_buy_budget > 1e-9 else 0.0

        net_worth = balance + float(np.sum(shares * px))
        equity[t] = net_worth
        cash[t] = balance

    eq_s = pd.Series(equity, index=close_by_day.index, name="equity")
    ret_s = eq_s.pct_change().fillna(0.0).rename("returns")
    cash_s = pd.Series(cash, index=close_by_day.index, name="cash")
    if return_execution_stats or return_trade_log:
        details: dict[str, Any] = {}
        if return_execution_stats:
            details.update(
                {
                    "executed_buy_count": int(exec_buy_count),
                    "executed_sell_count": int(exec_sell_count),
                }
            )
        if return_trade_log:
            trade_log = pd.DataFrame(trade_rows)
            if len(trade_log):
                trade_log = trade_log.sort_values(["date", "symbol", "side"]).reset_index(drop=True)
            details["trade_log"] = trade_log
        return eq_s, ret_s, cash_s, details
    return eq_s, ret_s, cash_s


def backtest_buy_and_hold_equal_weight(
    *,
    close_by_day: pd.DataFrame,
    initial_balance: float,
    fee_bps: float,
    slippage_bps: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    prices = close_by_day.to_numpy(dtype=float)
    dates = close_by_day.index
    t_n, s_n = prices.shape
    if t_n == 0 or s_n == 0:
        raise RuntimeError("Buy-and-hold benchmark received empty price panel.")

    cost_rate = (float(fee_bps) + float(slippage_bps)) / 10000.0
    balance = float(initial_balance)
    shares = np.zeros(s_n, dtype=float)

    p0 = prices[0]
    valid0 = p0 > 0
    n_valid = int(valid0.sum())
    if n_valid == 0:
        raise RuntimeError("No valid positive prices on first day for buy-and-hold benchmark.")

    gross_per_name = (balance / float(n_valid)) / (1.0 + cost_rate)
    for j in np.where(valid0)[0]:
        fee = trade_cost_from_bps(gross_per_name, fee_bps, slippage_bps)
        total_spend = gross_per_name + fee
        if total_spend > balance:
            gross = balance / (1.0 + cost_rate)
            fee = trade_cost_from_bps(gross, fee_bps, slippage_bps)
            total_spend = gross + fee
        else:
            gross = gross_per_name
        sh = gross / p0[j] if p0[j] > 0 else 0.0
        if sh <= 0:
            continue
        shares[j] += sh
        balance -= total_spend

    equity = np.zeros(t_n, dtype=float)
    cash = np.zeros(t_n, dtype=float)
    for t in range(t_n):
        net_worth = balance + float(np.sum(shares * prices[t]))
        equity[t] = net_worth
        cash[t] = balance

    p_last = prices[-1]
    for j in np.where(shares > 0)[0]:
        if p_last[j] <= 0:
            continue
        gross = shares[j] * p_last[j]
        fee = trade_cost_from_bps(gross, fee_bps, slippage_bps)
        balance += gross - fee
        shares[j] = 0.0

    equity[-1] = balance
    cash[-1] = balance
    eq_s = pd.Series(equity, index=dates, name="equity")
    ret_s = eq_s.pct_change().fillna(0.0).rename("returns")
    cash_s = pd.Series(cash, index=dates, name="cash")
    return eq_s, ret_s, cash_s


def summarize_returns(returns: pd.Series, years: list[int], mode: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for yr in years:
        yret = returns.loc[
            (returns.index >= pd.Timestamp(f"{yr}-01-01"))
            & (returns.index <= pd.Timestamp(f"{yr}-12-31"))
        ]
        yeq = (1.0 + yret).cumprod()
        y_total = float((yeq.iloc[-1] - 1.0) * 100.0) if len(yeq) else np.nan
        y_sharpe = float((yret.mean() / yret.std(ddof=0)) * np.sqrt(252.0)) if len(yret) and yret.std(ddof=0) > 1e-12 else np.nan
        y_mdd = float((((yeq / yeq.cummax()) - 1.0).min()) * 100.0) if len(yeq) else np.nan
        rows.append(
            {
                "mode": mode,
                "test_year": int(yr),
                "total_return_pct": y_total,
                "sharpe": y_sharpe,
                "max_drawdown_pct": y_mdd,
            }
        )
    return pd.DataFrame(rows)


def run_sb3_workflow(
    *,
    bt_panel: pd.DataFrame,
    cfg: RLConfig,
    train_split_date: pd.Timestamp,
    years: list[int],
    algorithm: str = "a2c",
) -> dict[str, Any]:
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import A2C, PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as e:
        raise ImportError("This workflow requires gymnasium + stable-baselines3.") from e

    symbols = sorted(bt_panel.index.get_level_values("symbol").unique())
    n_symbols = len(symbols)
    action_names = np.array(["hold", "buy", "sell"])

    def pivot(col: str) -> pd.DataFrame:
        return (
            bt_panel[[col]]
            .reset_index()
            .pivot(index="date", columns="symbol", values=col)
            .reindex(columns=symbols)
            .sort_index()
        )

    p_buy = pivot("prob_buy").shift(1)
    p_reg = pivot("pred_rf_reg").shift(1)
    ae_fam = pivot("ae_familiarity").shift(1)
    close = pivot("close")

    common_dates = p_buy.index.intersection(p_reg.index).intersection(ae_fam.index).intersection(close.index)
    p_buy = p_buy.loc[common_dates].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    p_reg = p_reg.loc[common_dates].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ae_fam = ae_fam.loc[common_dates].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    close = close.loc[common_dates].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    q = float(cfg.eligibility_quantile)
    buy_thr = p_buy.quantile(q, axis=1)
    reg_thr = p_reg.quantile(q, axis=1)
    fam_thr = ae_fam.quantile(q, axis=1)
    eligible = (p_buy.ge(buy_thr, axis=0) & p_reg.ge(reg_thr, axis=0) & ae_fam.ge(fam_thr, axis=0)).fillna(False)

    feat_t = np.stack(
        [
            p_buy.to_numpy(dtype=np.float32),
            p_reg.to_numpy(dtype=np.float32),
            ae_fam.to_numpy(dtype=np.float32),
            close.to_numpy(dtype=np.float32),
        ],
        axis=-1,
    )

    lookback = int(cfg.lookback_window)
    if len(common_dates) <= lookback:
        raise RuntimeError(f"Not enough rows ({len(common_dates)}) for lookback_window={lookback}.")

    x_seq, c_seq, e_seq, b_seq, d_seq = [], [], [], [], []
    close_np = close.to_numpy(dtype=np.float32)
    elig_np = eligible.to_numpy(dtype=bool)
    p_buy_np = p_buy.to_numpy(dtype=np.float32)
    for t in range(lookback - 1, len(common_dates)):
        x_seq.append(feat_t[t - lookback + 1 : t + 1])
        c_seq.append(close_np[t])
        e_seq.append(elig_np[t])
        b_seq.append(p_buy_np[t])
        d_seq.append(common_dates[t])

    x_seq = np.asarray(x_seq, dtype=np.float32)
    c_seq = np.asarray(c_seq, dtype=np.float32)
    e_seq = np.asarray(e_seq, dtype=bool)
    b_seq = np.asarray(b_seq, dtype=np.float32)
    d_seq = pd.DatetimeIndex(d_seq)

    rebalance_mask = make_rebalance_mask(d_seq, cfg.rebalance_freq)
    split_ts = pd.Timestamp(train_split_date)
    train_mask_full = np.asarray(d_seq <= split_ts, dtype=bool)
    eval_mask_full = np.asarray(d_seq > split_ts, dtype=bool)
    if train_mask_full.sum() < 30:
        raise RuntimeError(
            f"Too few RL training rows before train_split_date={split_ts.date()} "
            f"(found {int(train_mask_full.sum())}). Provide a panel that includes pre-split history."
        )
    if eval_mask_full.sum() < 5:
        raise RuntimeError(
            f"Too few RL evaluation rows after train_split_date={split_ts.date()} "
            f"(found {int(eval_mask_full.sum())})."
        )

    flat_train = x_seq[train_mask_full].reshape(-1, x_seq.shape[-1])
    x_mu = flat_train.mean(axis=0, keepdims=True)
    x_sd = flat_train.std(axis=0, keepdims=True)
    x_sd = np.where(x_sd < 1e-9, 1.0, x_sd)
    xn = ((x_seq - x_mu) / x_sd).astype(np.float32)

    x_agent = xn[rebalance_mask]
    c_agent = c_seq[rebalance_mask]
    e_agent = e_seq[rebalance_mask]
    b_agent = b_seq[rebalance_mask]
    train_mask_agent = train_mask_full[rebalance_mask]
    if train_mask_agent.sum() < 20:
        raise RuntimeError("Too few rebalance training rows.")

    def _build_obs_with_portfolio_state(
        x_t: np.ndarray,
        px_t: np.ndarray,
        shares_t: np.ndarray,
        balance_t: float,
    ) -> np.ndarray:
        base = x_t.reshape(-1).astype(np.float32)
        px = np.asarray(px_t, dtype=float)
        sh = np.asarray(shares_t, dtype=float)
        net = float(balance_t + np.sum(sh * px))
        denom = max(net, 1e-9)
        pos_notional_pct = (sh * px) / denom
        cash_pct = np.array([float(balance_t) / denom], dtype=float)
        return np.concatenate([base, pos_notional_pct.astype(np.float32), cash_pct.astype(np.float32)], axis=0)

    def _execute_action_step(
        *,
        action_type: np.ndarray,
        px: np.ndarray,
        elig_t: np.ndarray,
        score_t: np.ndarray,
        shares: np.ndarray,
        balance: float,
        max_buys: int | None,
        fee_bps: float,
        slippage_bps: float,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        at = np.asarray(action_type, dtype=int).copy()
        px = np.asarray(px, dtype=float)
        shares_new = np.asarray(shares, dtype=float).copy()
        bal = float(balance)
        cost_rate = (float(fee_bps) + float(slippage_bps)) / 10000.0

        at = np.where((at == 1) & (~np.asarray(elig_t, dtype=bool)), 0, at)
        at = apply_buy_cap(at, max_buys, scores=np.asarray(score_t, dtype=float))
        at = np.where((at == 2) & (shares_new <= 0.0), 0, at)

        for j in np.where(at == 2)[0]:
            if px[j] <= 0:
                continue
            sh = shares_new[j]
            if sh <= 0:
                continue
            gross = sh * px[j]
            fee = trade_cost_from_bps(gross, fee_bps, slippage_bps)
            shares_new[j] = 0.0
            bal += gross - fee

        net_worth = bal + float(np.sum(shares_new * px))
        buy_idx = np.where(at == 1)[0]
        n_buy = int(len(buy_idx))
        day_buy_budget = max(0.0, float(bal))
        if n_buy > 0:
            target_notional = net_worth / float(n_buy)
            for j in buy_idx:
                if px[j] <= 0 or day_buy_budget <= 0:
                    continue
                current_notional = shares_new[j] * px[j]
                need_gross = max(0.0, target_notional - current_notional)
                if need_gross <= 0:
                    continue
                max_affordable_gross = day_buy_budget / (1.0 + cost_rate)
                gross = min(need_gross, max_affordable_gross)
                if gross <= 0:
                    continue
                fee = trade_cost_from_bps(gross, fee_bps, slippage_bps)
                total_spend = gross + fee
                shares_to_buy = gross / px[j] if px[j] > 0 else 0.0
                if shares_to_buy <= 0:
                    continue
                shares_new[j] += shares_to_buy
                day_buy_budget -= total_spend

        bal = day_buy_budget if day_buy_budget > 1e-9 else 0.0
        net_worth = bal + float(np.sum(shares_new * px))
        return at, shares_new, bal, net_worth

    class PerStockDiscreteEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, xa: np.ndarray, ca: np.ndarray, ea: np.ndarray, ba: np.ndarray):
            super().__init__()
            self.x = np.asarray(xa, dtype=np.float32)
            self.close_arr = np.asarray(ca, dtype=np.float32)
            self.elig_arr = np.asarray(ea, dtype=bool)
            self.buy_score_arr = np.asarray(ba, dtype=np.float32)
            self.balance = float(cfg.initial_balance)
            self.fee_bps = float(cfg.fee_bps)
            self.slippage_bps = float(cfg.slippage_bps)
            self.cost_rate = (self.fee_bps + self.slippage_bps) / 10000.0
            self.max_buys = cfg.max_stocks_per_day
            self.n_symbols = self.x.shape[2]
            self.obs_dim = int(self.x.shape[1] * self.x.shape[2] * self.x.shape[3]) + self.n_symbols + 1
            self.n = len(self.x)
            self.action_space = spaces.MultiDiscrete(np.full(self.n_symbols, 3, dtype=np.int64))
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            self.reset()

        def _obs(self) -> np.ndarray:
            return _build_obs_with_portfolio_state(
                self.x[self.t],
                self.close_arr[self.t],
                self.shares,
                self.balance,
            )

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.t = 0
            self.balance = float(cfg.initial_balance)
            self.shares = np.zeros(self.n_symbols, dtype=float)
            self.prev_net = float(cfg.initial_balance)
            self.peak_net = float(cfg.initial_balance)
            return self._obs(), {}

        def step(self, action):
            at = np.asarray(np.clip(np.rint(np.asarray(action, dtype=float)), 0, 2), dtype=int)
            px = self.close_arr[self.t].astype(float)
            elig_t = self.elig_arr[self.t]
            score_t = self.buy_score_arr[self.t]
            at, self.shares, self.balance, net_worth = _execute_action_step(
                action_type=at,
                px=px,
                elig_t=elig_t,
                score_t=score_t,
                shares=self.shares,
                balance=self.balance,
                max_buys=self.max_buys,
                fee_bps=self.fee_bps,
                slippage_bps=self.slippage_bps,
            )
            self.peak_net = max(self.peak_net, net_worth)
            daily_ret = (net_worth - self.prev_net) / max(self.prev_net, 1e-9)
            drawdown = max(0.0, (self.peak_net - net_worth) / max(self.peak_net, 1e-9))
            reward = daily_ret - float(cfg.drawdown_penalty_lambda) * drawdown
            self.prev_net = net_worth

            self.t += 1
            terminated = self.t >= self.n
            truncated = False
            obs = np.zeros(self.obs_dim, dtype=np.float32) if terminated else self._obs()
            info = {
                "net_worth": float(net_worth),
                "balance": float(self.balance),
                "n_buy": int(np.sum(at == 1)),
                "n_hold": int(np.sum(at == 0)),
                "n_sell": int(np.sum(at == 2)),
                "n_eligible": int(np.sum(elig_t)),
                "drawdown": float(drawdown),
            }
            return obs, float(reward), terminated, truncated, info

    x_train = x_agent[train_mask_agent]
    c_train = c_agent[train_mask_agent]
    e_train = e_agent[train_mask_agent]
    b_train = b_agent[train_mask_agent]

    train_env = DummyVecEnv([lambda: PerStockDiscreteEnv(x_train, c_train, e_train, b_train)])

    train_steps_per_episode = int(len(x_train))
    if train_steps_per_episode <= 0:
        raise RuntimeError("No training steps available.")
    total_timesteps = int(int(cfg.ppo_episodes) * train_steps_per_episode)

    algo = str(algorithm).strip().lower()
    if algo == "a2c":
        model = A2C("MlpPolicy", train_env, seed=int(cfg.seed), verbose=0)
    elif algo == "ppo":
        model = PPO("MlpPolicy", train_env, seed=int(cfg.seed), verbose=0)
    else:
        raise ValueError("algorithm must be 'a2c' or 'ppo'.")
    model.learn(total_timesteps=total_timesteps)

    policy_action_type = np.zeros((len(xn), n_symbols), dtype=int)
    sim_shares = np.zeros(n_symbols, dtype=float)
    sim_balance = float(cfg.initial_balance)
    for t in range(len(xn)):
        if not bool(rebalance_mask[t]):
            policy_action_type[t] = 0
            continue
        obs_t = _build_obs_with_portfolio_state(
            xn[t],
            c_seq[t],
            sim_shares,
            sim_balance,
        ).reshape(1, -1)
        act_t, _ = model.predict(obs_t, deterministic=True)
        at = np.asarray(np.clip(np.rint(np.asarray(act_t[0], dtype=float)), 0, 2), dtype=int)
        at, sim_shares, sim_balance, _ = _execute_action_step(
            action_type=at,
            px=c_seq[t],
            elig_t=e_seq[t],
            score_t=b_seq[t],
            shares=sim_shares,
            balance=sim_balance,
            max_buys=cfg.max_stocks_per_day,
            fee_bps=float(cfg.fee_bps),
            slippage_bps=float(cfg.slippage_bps),
        )
        policy_action_type[t] = at

    # Evaluate strictly on post-split dates (no leakage into reported metrics).
    d_eval = d_seq[eval_mask_full]
    c_eval = c_seq[eval_mask_full]
    e_eval = e_seq[eval_mask_full]
    b_eval = b_seq[eval_mask_full]
    rb_eval = rebalance_mask[eval_mask_full]
    action_eval = policy_action_type[eval_mask_full]

    close_bt = pd.DataFrame(c_eval, index=d_eval, columns=symbols)
    rl_eq, rl_ret, rl_cash, exec_details = backtest_strategy_per_stock_discrete(
        action_type=action_eval,
        close_by_day=close_bt,
        eligible_by_day=e_eval,
        buy_score_by_day=b_eval,
        initial_balance=float(cfg.initial_balance),
        fee_bps=float(cfg.fee_bps),
        slippage_bps=float(cfg.slippage_bps),
        max_buys_per_day=cfg.max_stocks_per_day,
        rebalance_mask=rb_eval,
        return_execution_stats=True,
        return_trade_log=True,
    )
    rl_cash = rl_cash.mask(rl_cash.abs() < 1e-9, 0.0)

    rl_total_return_pct = float((rl_eq.iloc[-1] / rl_eq.iloc[0] - 1.0) * 100.0) if len(rl_eq) else np.nan
    rl_sharpe = float((rl_ret.mean() / rl_ret.std(ddof=0)) * np.sqrt(252.0)) if rl_ret.std(ddof=0) > 1e-12 else np.nan
    rl_mdd = float((((rl_eq / rl_eq.cummax()) - 1.0).min()) * 100.0) if len(rl_eq) else np.nan

    mode_name = f"rl_agent_{algo}_framework_backtest"
    yearly_df = summarize_returns(rl_ret, years, mode=mode_name)
    summary_df = pd.DataFrame(
        [
            {
                "mode": mode_name,
                "years": f"{years[0]}-{years[-1]}",
                "combined_total_return_pct": rl_total_return_pct,
                "combined_sharpe": rl_sharpe,
                "combined_max_drawdown_pct": rl_mdd,
                "initial_balance": float(cfg.initial_balance),
                "fee_bps": float(cfg.fee_bps),
                "slippage_bps": float(cfg.slippage_bps),
                "symbols": len(symbols),
                "avg_eligible_names": float(np.mean(e_eval.sum(axis=1))),
                "eligibility_quantile": float(cfg.eligibility_quantile),
                "max_stocks_per_day": cfg.max_stocks_per_day,
                "weighting": "dynamic_1_over_n_buys_today",
                "rebalance_freq": cfg.rebalance_freq,
                "rebalance_days": int(np.sum(rb_eval)),
            }
        ]
    )

    flat_types = action_eval.reshape(-1)
    action_counts = pd.Series(action_names[flat_types]).value_counts().reindex(action_names, fill_value=0)
    executed_action_counts = pd.Series(
        {
            "buy": int(exec_details["executed_buy_count"]),
            "sell": int(exec_details["executed_sell_count"]),
        },
        dtype=int,
    )
    trade_log = exec_details.get("trade_log", pd.DataFrame())
    last_actions = pd.DataFrame(
        {
            "symbol": symbols,
            "eligible": e_eval[-1],
            "action_type": action_eval[-1],
            "action_name": action_names[action_eval[-1]],
            "target_weight_if_buy": np.nan,
        }
    )

    return {
        "model": model,
        "symbols": symbols,
        "close_bt": close_bt,
        "policy_action_type": action_eval,
        "eligible_by_day": e_eval,
        "buy_score_by_day": b_eval,
        "dates": d_eval,
        "rebalance_mask": rb_eval,
        "rl_equity": rl_eq,
        "rl_returns": rl_ret,
        "rl_cash": rl_cash,
        "rl_yearly_df": yearly_df,
        "rl_summary_df": summary_df,
        "action_counts": action_counts,
        "executed_action_counts": executed_action_counts,
        "trade_log": trade_log,
        "last_actions": last_actions,
        "train_steps_per_episode": train_steps_per_episode,
        "total_timesteps": total_timesteps,
    }
