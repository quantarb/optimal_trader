"""Adapters for the existing quant-warehouse feature-family panel contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TA_FAMILIES = {
    "equity-historical-price-eod",
    "technical_candles",
    "technical_cycles",
    "technical_math",
    "technical_momentum",
    "technical_overlap",
    "technical_performance",
}


def load_existing_feature_family_panel(
    index_path: str | Path,
    *,
    exclude_ta: bool = True,
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    """Load the same outer-joined family panels used by current models.

    The persisted family panels are already aligned to warehouse observation
    dates. Families are outer-joined so narrower coverage is not discarded.
    """
    index = pd.read_csv(index_path)
    parts: list[pd.DataFrame] = []
    families: dict[str, tuple[str, ...]] = {}
    for _, row in index.iterrows():
        family = str(row.family)
        if exclude_ta and (family.lower() in TA_FAMILIES or family.lower().startswith(("technical_", "ta_"))):
            continue
        panel = pd.read_parquet(row.panel_path)
        metadata = pd.read_parquet(row.metadata_path)
        columns = [str(value) for value in metadata.feature if str(value) in panel.columns]
        if not columns:
            continue
        frame = panel[["symbol", "date", *columns]].copy()
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        renamed = {column: f"{family}__{column}" if not str(column).startswith(f"{family}__") else str(column) for column in columns}
        frame = frame.rename(columns=renamed).sort_values(["symbol", "date"])
        selected = tuple(renamed.values())
        families[family] = selected
        parts.append(frame.set_index(["symbol", "date"]))
    if not parts:
        raise RuntimeError(f"no feature families were loaded from {index_path}")
    fused = pd.concat(parts, axis=1, join="outer").reset_index()
    return fused.sort_values(["symbol", "date"]).reset_index(drop=True), families


def build_rate_views(
    daily_panel: pd.DataFrame,
    *,
    symbol: str,
    prediction_date: pd.Timestamp,
    daily_length: int = 256,
    annual_length: int = 5,
    quarterly_length: int = 12,
) -> dict[str, pd.DataFrame]:
    """Create causal daily, annual-cadence, and quarterly-cadence views.

    This follows the existing warehouse panel semantics: ``date`` is the
    observation/as-of date after sparse warehouse rows have been forward-filled.
    No future rows or fiscal-period-only alignment is used.
    """
    required = {"symbol", "date"}
    if missing := required.difference(daily_panel.columns):
        raise KeyError(f"feature panel missing columns: {sorted(missing)}")
    frame = daily_panel.loc[
        daily_panel.symbol.astype(str).str.upper().eq(str(symbol).upper())
    ].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame = frame.loc[frame.date.le(pd.Timestamp(prediction_date).normalize())].sort_values("date")
    if frame.empty:
        return {"daily": frame, "annual": frame.copy(), "quarterly": frame.copy()}
    feature_columns = [column for column in frame.columns if column not in {"symbol", "date"}]
    daily = frame.tail(daily_length).reset_index(drop=True)
    dated = frame.set_index("date")
    annual = dated.groupby(dated.index.year, sort=True).tail(1).tail(annual_length).reset_index()
    quarterly = dated.groupby([dated.index.year, dated.index.quarter], sort=True).tail(1).tail(quarterly_length).reset_index()
    for view in (daily, annual, quarterly):
        view["available_date"] = view["date"]
        view[feature_columns] = view[feature_columns].apply(pd.to_numeric, errors="coerce")
    return {"daily": daily, "annual": annual, "quarterly": quarterly}
