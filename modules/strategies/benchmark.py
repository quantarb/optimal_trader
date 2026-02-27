from __future__ import annotations

import numpy as np
import pandas as pd


class BuyAndHoldEqualWeightStrategy:
    """Equal-weight buy-and-hold benchmark as a strategy.

    Buys selected symbols at the first date, holds through time, and optionally
    liquidates on the last date by setting target weights to zero.
    """

    def __init__(
        self,
        *,
        price_col: str = "close",
        gross_exposure: float = 1.0,
        top_k: int | None = None,
        liquidate_on_last_day: bool = True,
    ):
        self.price_col = str(price_col)
        self.gross_exposure = float(gross_exposure)
        self.top_k = None if top_k is None else int(top_k)
        self.liquidate_on_last_day = bool(liquidate_on_last_day)

    @property
    def name(self) -> str:
        return (
            f"BuyAndHoldEqualWeight(gross={self.gross_exposure},"
            f"top_k={self.top_k},liquidate={self.liquidate_on_last_day})"
        )

    def compute_weights(self, panel: pd.DataFrame) -> pd.DataFrame:
        dates = sorted(pd.Index(panel.index.get_level_values("date")).unique())
        dti = pd.DatetimeIndex(dates)
        symbols = sorted(pd.Index(panel.index.get_level_values("symbol")).unique())
        w = pd.DataFrame(0.0, index=dti, columns=symbols)

        if len(dti) == 0 or len(symbols) == 0:
            return w
        if self.price_col not in panel.columns:
            raise KeyError(f"Missing price column '{self.price_col}' in panel")

        px = panel[self.price_col].unstack("symbol").reindex(index=dti, columns=symbols)
        # Enter on the first day that has at least one tradable symbol.
        valid_counts = (px > 0).sum(axis=1)
        if int(valid_counts.max()) <= 0:
            return w
        entry_dt = valid_counts[valid_counts > 0].index[0]

        entry_prices = pd.to_numeric(px.loc[entry_dt], errors="coerce")
        selected = entry_prices[entry_prices > 0].index.tolist()
        if self.top_k is not None:
            selected = selected[: self.top_k]
        if len(selected) == 0:
            return w

        per_weight = self.gross_exposure / float(len(selected))
        w.loc[entry_dt:, selected] = per_weight
        if self.liquidate_on_last_day:
            w.iloc[-1, :] = 0.0
        return w
