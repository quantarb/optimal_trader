from __future__ import annotations

from typing import Set

import pandas as pd


def ensure_panel_index(panel: pd.DataFrame) -> pd.DataFrame:
    """Ensure the panel is indexed by (date, symbol) and is reshape-safe."""
    if panel is None:
        raise ValueError("panel is None")

    out = panel

    if isinstance(out.index, pd.MultiIndex) and out.index.nlevels >= 2:
        names = [str(n) if n is not None else None for n in out.index.names]

        if not (len(names) >= 2 and names[0] == "date" and names[1] == "symbol"):
            if "date" in names and "symbol" in names:
                date_pos = names.index("date")
                sym_pos = names.index("symbol")
                if [date_pos, sym_pos] != [0, 1]:
                    out = out.reorder_levels(["date", "symbol"]).sort_index()
            else:
                lvl0 = out.index.get_level_values(0)
                lvl1 = out.index.get_level_values(1)
                dt0 = pd.to_datetime(lvl0, errors="coerce")
                dt1 = pd.to_datetime(lvl1, errors="coerce")
                nat0 = float(dt0.isna().mean())
                nat1 = float(dt1.isna().mean())

                if nat0 < nat1:
                    out.index = out.index.set_names(["date", "symbol"])
                else:
                    out.index = out.index.set_names(["symbol", "date"])
                    out = out.reorder_levels(["date", "symbol"]).sort_index()

        try:
            dt = pd.to_datetime(out.index.get_level_values("date"), errors="coerce")
            dt = pd.DatetimeIndex(dt).tz_localize(None)
            out = out.copy()
            out.index = pd.MultiIndex.from_arrays(
                [dt, out.index.get_level_values("symbol").astype(str)],
                names=["date", "symbol"],
            )
        except Exception:
            out = out.reset_index()

    if not (isinstance(out.index, pd.MultiIndex) and out.index.names[:2] == ["date", "symbol"]):
        date_col = None
        sym_col = None
        for c in out.columns:
            lc = str(c).lower()
            if lc in ("date", "dt", "time"):
                date_col = c
            if lc in ("symbol", "ticker", "sym"):
                sym_col = c

        if date_col is None or sym_col is None:
            raise ValueError(
                "Panel must have MultiIndex (date,symbol) or columns ['date','symbol']. "
                f"Got index={getattr(out.index, 'names', None)}, cols={list(out.columns)[:20]}..."
            )

        out = out.copy()
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.tz_localize(None)
        out[sym_col] = out[sym_col].astype(str)

        out = out.dropna(subset=[date_col, sym_col])
        out = out.sort_values([date_col, sym_col])
        out = out.set_index([date_col, sym_col])
        out.index = out.index.set_names(["date", "symbol"])
        out = out.sort_index()

    if out.index.has_duplicates:
        out = out.sort_index().groupby(level=["date", "symbol"]).last().sort_index()

    return out


def ensure_panel_index_strict(panel: pd.DataFrame) -> pd.DataFrame:
    """Strict version used by engine code: requires MultiIndex(date, symbol) with 2 levels."""
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("panel must be indexed by MultiIndex(date, symbol)")
    if panel.index.nlevels != 2:
        raise ValueError("panel index must have 2 levels: (date, symbol)")
    names: Set[str] = set(panel.index.names)
    if names != {"date", "symbol"}:
        # common when created by concat/stack
        panel = panel.copy()
        panel.index = panel.index.set_names(["date", "symbol"])
    return panel


def panel_dates_symbols(panel: pd.DataFrame) -> tuple[pd.DatetimeIndex, list[str]]:
    """Return (sorted unique dates, sorted symbols) from a panel."""
    p = ensure_panel_index(panel)
    dates = pd.DatetimeIndex(p.index.get_level_values("date").unique()).sort_values()
    symbols = sorted(p.index.get_level_values("symbol").astype(str).unique().tolist())
    return dates, symbols
