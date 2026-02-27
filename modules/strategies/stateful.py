from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class StatefulModelExitStrategy:
    """Stateful long/short strategy that feeds current position state into classifier.

    Notes:
    - Supports D/W/M rebalance cadence.
    - Uses entry and hold gates (quantile based) on classifier prob, reg score, and AE familiarity.
    - Optional inverse-vol weighting and no-trade buffer (`min_weight_change`).
    """

    def __init__(
        self,
        clf_model: Any,
        *,
        top_k: int = 10,
        hold_top_k: int = 20,
        long_budget: float = 0.4,
        short_budget: float = 0.4,
        gate_quantile: float = 0.75,
        hold_gate_quantile: float = 0.60,
        reg_col: str = "pred_rf_reg",
        familiarity_col: str = "ae_familiarity",
        price_col: str = "close",
        vol_window: int = 20,
        vol_floor: float = 1e-4,
        min_weight_change: float = 0.01,
        hold_score_drop_pct: float | None = None,
        rebalance_freq: str = "W",
        rebalance_anchor: str = "period_start",
    ):
        self.clf_model = clf_model
        self.top_k = int(top_k)
        self.hold_top_k = int(hold_top_k)
        self.long_budget = float(long_budget)
        self.short_budget = float(short_budget)
        self.gate_quantile = float(gate_quantile)
        self.hold_gate_quantile = float(hold_gate_quantile)
        self.reg_col = str(reg_col)
        self.familiarity_col = str(familiarity_col)
        self.price_col = str(price_col)
        self.vol_window = int(vol_window)
        self.vol_floor = float(vol_floor)
        self.min_weight_change = float(min_weight_change)
        self.hold_score_drop_pct = None if hold_score_drop_pct is None else float(hold_score_drop_pct)
        self.rebalance_freq = str(rebalance_freq)
        self.rebalance_anchor = str(rebalance_anchor)

    @property
    def name(self) -> str:
        return (
            f"StatefulExit(top_k={self.top_k},hold_top_k={self.hold_top_k},"
            f"freq={self.rebalance_freq},gate={self.gate_quantile},"
            f"vol_win={self.vol_window},min_dw={self.min_weight_change})"
        )

    def _rebalance_mask(self, idx: pd.DatetimeIndex) -> pd.Series:
        if self.rebalance_freq == "D":
            return pd.Series(True, index=idx)

        if self.rebalance_freq == "W":
            period = idx.to_period("W")
        elif self.rebalance_freq == "M":
            period = idx.to_period("M")
        else:
            raise ValueError("rebalance_freq must be D/W/M")

        s = idx.to_series(index=idx)
        if self.rebalance_anchor == "period_start":
            rb_dates = set(s.groupby(period).min().tolist())
        elif self.rebalance_anchor == "period_end":
            rb_dates = set(s.groupby(period).max().tolist())
        else:
            raise ValueError("rebalance_anchor must be period_start/period_end")

        return pd.Series(idx.isin(rb_dates), index=idx)

    def _class_probs(self, d: pd.DataFrame, pos_state: pd.Series) -> tuple[pd.Series, pd.Series]:
        clf = self.clf_model
        raw = getattr(clf, "model", clf)
        used = list(getattr(clf, "_used_features", d.columns))

        x = d.copy()
        if "market_position" in used:
            x["market_position"] = x.index.to_series().map(pos_state).fillna(0).astype(int)

        for c in used:
            if c not in x.columns:
                x[c] = 0.0

        X = x[used].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        proba = np.asarray(raw.predict_proba(X), dtype=float)

        classes = list(getattr(raw, "classes_", []))
        class_mapping = getattr(clf, "_class_mapping", {}) or {}
        labels = [str(class_mapping.get(c, c)).strip().lower() for c in classes]

        def find_idx(cands: set[str]):
            for i, lab in enumerate(labels):
                if lab in cands:
                    return i
            return None

        i_buy = find_idx({"buy", "long", "1"})
        i_short = find_idx({"short", "-1"})

        p_buy = (
            pd.Series(np.max(proba, axis=1), index=d.index, dtype=float)
            if i_buy is None
            else pd.Series(proba[:, i_buy], index=d.index, dtype=float)
        )
        p_short = (
            pd.Series(np.min(proba, axis=1), index=d.index, dtype=float)
            if i_short is None
            else pd.Series(proba[:, i_short], index=d.index, dtype=float)
        )
        return p_buy, p_short

    @staticmethod
    def _safe_quantile(s: pd.Series, q: float, default: float = np.nan) -> float:
        x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) == 0:
            return float(default)
        return float(x.quantile(q))

    @staticmethod
    def _safe_median(s: pd.Series, default: float = np.nan) -> float:
        x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) == 0:
            return float(default)
        return float(x.median())

    def _vol_scaled_weights(self, names: list[str], vol_s: pd.Series, budget: float, side: int) -> dict[str, float]:
        if len(names) == 0:
            return {}

        v = pd.to_numeric(vol_s.reindex(names), errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = self._safe_median(v, default=np.nan)
        if pd.notna(med):
            v = v.fillna(med)

        if v.isna().all():
            w = pd.Series(1.0, index=names, dtype=float)
        else:
            w = 1.0 / np.maximum(v.astype(float), self.vol_floor)

        denom = float(w.sum())
        if not np.isfinite(denom) or denom <= 0:
            w = pd.Series(1.0, index=names, dtype=float)
            denom = float(w.sum())

        w = w / denom
        sgn = 1.0 if side > 0 else -1.0
        return {str(k): float(sgn * budget * wv) for k, wv in w.items()}

    def compute_weights(self, panel: pd.DataFrame) -> pd.DataFrame:
        dates = sorted(pd.Index(panel.index.get_level_values("date")).unique())
        dti = pd.DatetimeIndex(dates)
        symbols = sorted(pd.Index(panel.index.get_level_values("symbol")).unique())
        w = pd.DataFrame(0.0, index=dti, columns=symbols)

        if self.price_col not in panel.columns:
            raise KeyError(f"Missing price column '{self.price_col}' in panel")

        px = panel[self.price_col].unstack("symbol").sort_index().apply(pd.to_numeric, errors="coerce")
        r = px.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        vol = r.rolling(self.vol_window, min_periods=max(5, self.vol_window // 2)).std()

        rebalance_mask = self._rebalance_mask(dti)
        pos_state = pd.Series(0, index=pd.Index(symbols, dtype=object), dtype=int)
        entry_long_score = pd.Series(np.nan, index=pd.Index(symbols, dtype=object), dtype=float)
        entry_short_score = pd.Series(np.nan, index=pd.Index(symbols, dtype=object), dtype=float)

        for i, dt in enumerate(dti):
            prev = w.iloc[i - 1].copy() if i > 0 else pd.Series(0.0, index=w.columns)
            if i > 0 and not rebalance_mask.iloc[i]:
                w.loc[dt] = prev.values
                continue

            day = panel.xs(dt, level="date").copy()
            p_buy, p_short = self._class_probs(day, pos_state)
            rg = pd.to_numeric(day.get(self.reg_col), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            fm = pd.to_numeric(day.get(self.familiarity_col), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

            buy_score = p_buy * rg * fm
            short_score = p_short * rg * fm

            q = self.gate_quantile
            pb_q = self._safe_quantile(p_buy, q, default=np.nan)
            ps_q = self._safe_quantile(p_short, q, default=np.nan)
            rg_q = self._safe_quantile(rg, q, default=np.nan)
            fm_q = self._safe_quantile(fm, q, default=np.nan)
            long_gate = ((p_buy >= pb_q) & (rg >= rg_q) & (fm >= fm_q)).fillna(False)
            short_gate = ((p_short >= ps_q) & (rg >= rg_q) & (fm >= fm_q)).fillna(False)

            long_rank = buy_score.sort_values(ascending=False)
            short_rank = short_score.sort_values(ascending=False)

            current_long = pos_state[pos_state > 0].index.astype(str)
            current_short = pos_state[pos_state < 0].index.astype(str)

            hq = self.hold_gate_quantile
            pb_hq = self._safe_quantile(p_buy, hq, default=np.nan)
            ps_hq = self._safe_quantile(p_short, hq, default=np.nan)
            rg_hq = self._safe_quantile(rg, hq, default=np.nan)
            fm_hq = self._safe_quantile(fm, hq, default=np.nan)
            long_hold_gate = ((p_buy >= pb_hq) & (rg >= rg_hq) & (fm >= fm_hq)).fillna(False)
            short_hold_gate = ((p_short >= ps_hq) & (rg >= rg_hq) & (fm >= fm_hq)).fillna(False)

            keep_long = [s for s in current_long if s in long_rank.index[: self.hold_top_k] and bool(long_hold_gate.get(s, False))]
            keep_short = [s for s in current_short if s in short_rank.index[: self.hold_top_k] and bool(short_hold_gate.get(s, False))]

            if self.hold_score_drop_pct is not None and self.hold_score_drop_pct > 0:
                drop_mult = max(0.0, 1.0 - float(self.hold_score_drop_pct))
                keep_long = [
                    s for s in keep_long
                    if float(buy_score.get(s, 0.0)) >= float(entry_long_score.get(s, np.inf)) * drop_mult
                ]
                keep_short = [
                    s for s in keep_short
                    if float(short_score.get(s, 0.0)) >= float(entry_short_score.get(s, np.inf)) * drop_mult
                ]

            long_candidates = [s for s in long_rank.index if bool(long_gate.get(s, False)) and s not in keep_long and s not in keep_short]
            short_candidates = [s for s in short_rank.index if bool(short_gate.get(s, False)) and s not in keep_short and s not in keep_long]

            need_long = max(0, self.top_k - len(keep_long))
            need_short = max(0, self.top_k - len(keep_short))

            new_long = keep_long + long_candidates[:need_long]
            new_short = keep_short + short_candidates[:need_short]

            overlap = set(new_long).intersection(set(new_short))
            if overlap:
                new_short = [s for s in new_short if s not in overlap]

            vol_dt = vol.loc[dt] if dt in vol.index else pd.Series(index=px.columns, dtype=float)
            target = pd.Series(0.0, index=w.columns)

            lw = self._vol_scaled_weights(list(map(str, new_long)), vol_dt, self.long_budget, side=1)
            sw = self._vol_scaled_weights(list(map(str, new_short)), vol_dt, self.short_budget, side=-1)
            for k, v in {**lw, **sw}.items():
                if k in target.index:
                    target.loc[k] = v

            if self.min_weight_change > 0:
                small = (target - prev).abs() < self.min_weight_change
                target.loc[small] = prev.loc[small]

            w.loc[dt] = target.values

            pos_state[:] = 0
            pos_state.loc[target[target > 0].index] = 1
            pos_state.loc[target[target < 0].index] = -1

            # Track entry scores for positions opened this rebalance.
            new_long_only = set(new_long) - set(keep_long)
            new_short_only = set(new_short) - set(keep_short)
            for s in new_long_only:
                entry_long_score.loc[s] = float(buy_score.get(s, np.nan))
            for s in new_short_only:
                entry_short_score.loc[s] = float(short_score.get(s, np.nan))

            # Clear stale entry scores for exited names.
            exited_long = set(current_long) - set(new_long)
            exited_short = set(current_short) - set(new_short)
            for s in exited_long:
                entry_long_score.loc[s] = np.nan
            for s in exited_short:
                entry_short_score.loc[s] = np.nan

        return w


class EqualWeightStatefulStrategy(StatefulModelExitStrategy):
    """Same stateful logic, but equal-weights selected names per side."""

    def _vol_scaled_weights(self, names: list[str], vol_s: pd.Series, budget: float, side: int) -> dict[str, float]:
        if len(names) == 0:
            return {}
        sgn = 1.0 if side > 0 else -1.0
        per = float(budget) / float(len(names))
        return {str(k): sgn * per for k in names}
