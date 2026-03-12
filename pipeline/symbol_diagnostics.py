from __future__ import annotations

from math import sqrt
from typing import Any, Mapping, Sequence

import pandas as pd

from pipeline.service_runtime import read_frame_artifact


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _round_float(value: float, digits: int = 8) -> float:
    return round(float(value), digits)


def _resolve_backtest_config(
    artifact_or_frame: Any,
    *,
    backtest_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = dict(backtest_config or {})
    metadata = dict(getattr(artifact_or_frame, "metadata", {}) or {})
    runtime_config = dict(metadata.get("backtest_config") or {})
    merged = {**runtime_config, **resolved}
    return {
        "fee_bps": _safe_float(merged.get("fee_bps"), 0.0),
        "slippage_bps": _safe_float(merged.get("slippage_bps"), _safe_float(merged.get("transaction_cost_bps"), 0.0)),
        "short_borrow_bps_annual": _safe_float(merged.get("short_borrow_bps_annual"), 0.0),
        "turnover_half_l1": bool(merged.get("turnover_half_l1", True)),
    }


def load_backtest_trade_frame(artifact_or_frame: Any) -> pd.DataFrame:
    if isinstance(artifact_or_frame, pd.DataFrame):
        frame = artifact_or_frame.copy()
    else:
        frame = read_frame_artifact(
            artifact_or_frame,
            parse_dates=False,
            normalize_symbols=True,
        )
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    frame = frame.dropna(subset=["date", "symbol"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    return frame


def _artifact_dates(artifact_or_frame: Any, trade_df: pd.DataFrame) -> pd.DatetimeIndex:
    content = dict(getattr(artifact_or_frame, "content", {}) or {})
    daily_rows = list(content.get("daily_rows") or [])
    if daily_rows:
        dates = pd.to_datetime(
            [row.get("date") for row in daily_rows],
            errors="coerce",
        )
        dates = dates.dropna()
        if len(dates):
            return pd.DatetimeIndex(sorted(dates.unique()))
    return pd.DatetimeIndex(sorted(trade_df["date"].dropna().unique().tolist()))


def _symbol_trade_returns(effective_weight: pd.Series, net_daily_return: pd.Series) -> pd.Series:
    active = effective_weight.abs() > 1e-12
    sign = effective_weight.gt(0.0).astype(int) - effective_weight.lt(0.0).astype(int)
    trade_start = active & (
        (~active.shift(fill_value=False))
        | (sign != sign.shift(fill_value=0))
    )
    trade_ids = trade_start.cumsum().where(active, 0)
    trade_returns = net_daily_return.groupby(trade_ids).sum()
    return trade_returns[trade_returns.index > 0]


def compute_symbol_strategy_diagnostics(
    artifact_or_frame: Any,
    *,
    strategy_name: str = "",
    filter_name: str = "",
    evaluation_scope: str = "",
    fold_name: str = "",
    backtest_start_date: str = "",
    backtest_end_date: str = "",
    backtest_config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    trade_df = load_backtest_trade_frame(artifact_or_frame)
    if trade_df.empty:
        return []

    config = _resolve_backtest_config(artifact_or_frame, backtest_config=backtest_config)
    all_dates = _artifact_dates(artifact_or_frame, trade_df)
    if all_dates.empty:
        return []

    cost_rate = (_safe_float(config.get("fee_bps")) + _safe_float(config.get("slippage_bps"))) / 10000.0
    borrow_rate = _safe_float(config.get("short_borrow_bps_annual")) / 10000.0 / 252.0
    apply_half_turnover = bool(config.get("turnover_half_l1", True))

    rows: list[dict[str, Any]] = []
    for symbol, group in trade_df.groupby("symbol", sort=True):
        panel = (
            group.drop_duplicates(subset=["date"], keep="last")
            .set_index("date")
            .reindex(all_dates)
        )
        effective_weight = pd.to_numeric(panel.get("effective_weight"), errors="coerce").fillna(0.0)
        target_weight = pd.to_numeric(panel.get("target_weight"), errors="coerce").fillna(0.0)
        asset_return = pd.to_numeric(panel.get("asset_return"), errors="coerce").fillna(0.0)
        realized_return = effective_weight * asset_return
        turnover = effective_weight.diff().abs().fillna(effective_weight.abs())
        if apply_half_turnover:
            turnover = turnover * 0.5
        turnover_cost = turnover * cost_rate
        short_borrow_cost = effective_weight.clip(upper=0.0).abs() * borrow_rate
        net_daily_return = realized_return - turnover_cost - short_borrow_cost

        equity_curve = (1.0 + net_daily_return).cumprod()
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max) - 1.0
        volatility = float(net_daily_return.std(ddof=0)) if len(net_daily_return) > 1 else 0.0
        sharpe = (
            (float(net_daily_return.mean()) / volatility) * sqrt(252.0)
            if volatility > 1e-12
            else 0.0
        )

        trade_returns = _symbol_trade_returns(effective_weight, net_daily_return)
        active_days = int((effective_weight.abs() > 1e-12).sum())
        selected_days = int((target_weight.abs() > 1e-12).sum())
        trade_count = int(len(trade_returns))
        hit_rate = float((trade_returns > 0.0).mean()) if trade_count else 0.0
        avg_trade_return = float(trade_returns.mean()) if trade_count else 0.0

        rows.append(
            {
                "strategy_name": str(strategy_name or ""),
                "filter_name": str(filter_name or ""),
                "evaluation_scope": str(evaluation_scope or ""),
                "fold_name": str(fold_name or ""),
                "backtest_start_date": str(backtest_start_date or ""),
                "backtest_end_date": str(backtest_end_date or ""),
                "symbol": str(symbol),
                "sharpe": _round_float(sharpe),
                "avg_trade_return": _round_float(avg_trade_return),
                "hit_rate": _round_float(hit_rate, 6),
                "max_drawdown": _round_float(float(drawdown.min()) if len(drawdown) else 0.0),
                "trade_count": trade_count,
                "turnover": _round_float(float(turnover.sum())),
                "active_days": active_days,
                "selected_days": selected_days,
                "avg_abs_weight": _round_float(float(effective_weight.abs().mean())),
                "cumulative_return": _round_float(float(equity_curve.iloc[-1] - 1.0)),
                "final_equity": _round_float(float(equity_curve.iloc[-1])),
            }
        )
    return rows


def compute_symbol_buy_hold_diagnostics(
    artifact_or_frame: Any,
    *,
    evaluation_scope: str = "",
    fold_name: str = "",
    backtest_start_date: str = "",
    backtest_end_date: str = "",
) -> list[dict[str, Any]]:
    trade_df = load_backtest_trade_frame(artifact_or_frame)
    if trade_df.empty:
        return []

    all_dates = _artifact_dates(artifact_or_frame, trade_df)
    if all_dates.empty:
        return []

    rows: list[dict[str, Any]] = []
    for symbol, group in trade_df.groupby("symbol", sort=True):
        panel = (
            group.drop_duplicates(subset=["date"], keep="last")
            .set_index("date")
            .reindex(all_dates)
        )
        asset_return = pd.to_numeric(panel.get("asset_return"), errors="coerce").fillna(0.0)
        equity_curve = (1.0 + asset_return).cumprod()
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max) - 1.0
        rows.append(
            {
                "evaluation_scope": str(evaluation_scope or ""),
                "fold_name": str(fold_name or ""),
                "backtest_start_date": str(backtest_start_date or ""),
                "backtest_end_date": str(backtest_end_date or ""),
                "symbol": str(symbol),
                "buy_and_hold_return": _round_float(float(equity_curve.iloc[-1] - 1.0)),
                "buy_and_hold_final_equity": _round_float(float(equity_curve.iloc[-1])),
                "buy_and_hold_max_drawdown": _round_float(float(drawdown.min()) if len(drawdown) else 0.0),
                "observed_days": int(len(asset_return)),
            }
        )
    return rows


def aggregate_symbol_diagnostic_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_keys: Sequence[str] = ("strategy_name", "filter_name", "symbol"),
) -> list[dict[str, Any]]:
    if not rows:
        return []

    df = pd.DataFrame([dict(row) for row in rows])
    if df.empty:
        return []
    for column in [
        "sharpe",
        "avg_trade_return",
        "hit_rate",
        "max_drawdown",
        "trade_count",
        "turnover",
        "active_days",
        "selected_days",
        "avg_abs_weight",
        "cumulative_return",
        "final_equity",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    grouped = (
        df.groupby(list(group_keys), dropna=False)
        .agg(
            sharpe=("sharpe", "mean"),
            avg_trade_return=("avg_trade_return", "mean"),
            hit_rate=("hit_rate", "mean"),
            max_drawdown=("max_drawdown", "min"),
            trade_count=("trade_count", "sum"),
            turnover=("turnover", "sum"),
            active_days=("active_days", "sum"),
            selected_days=("selected_days", "sum"),
            avg_abs_weight=("avg_abs_weight", "mean"),
            cumulative_return=("cumulative_return", "mean"),
            final_equity=("final_equity", "mean"),
            folds=("fold_name", lambda values: len({str(value) for value in values if str(value).strip()})),
        )
        .reset_index()
    )
    sort_spec = [
        ("strategy_name", True),
        ("filter_name", True),
        ("sharpe", False),
        ("avg_trade_return", False),
        ("trade_count", False),
        ("symbol", True),
    ]
    sort_fields = [field for field, _ascending in sort_spec if field in grouped.columns]
    if sort_fields:
        grouped = grouped.sort_values(
            sort_fields,
            ascending=[ascending for field, ascending in sort_spec if field in grouped.columns],
        )
    return [
        {
            **dict(row),
            "sharpe": _round_float(row["sharpe"]),
            "avg_trade_return": _round_float(row["avg_trade_return"]),
            "hit_rate": _round_float(row["hit_rate"], 6),
            "max_drawdown": _round_float(row["max_drawdown"]),
            "turnover": _round_float(row["turnover"]),
            "avg_abs_weight": _round_float(row["avg_abs_weight"]),
            "cumulative_return": _round_float(row["cumulative_return"]),
            "final_equity": _round_float(row["final_equity"]),
            "trade_count": int(row["trade_count"]),
            "active_days": int(row["active_days"]),
            "selected_days": int(row["selected_days"]),
            "folds": int(row["folds"]),
        }
        for row in grouped.to_dict(orient="records")
    ]


__all__ = [
    "aggregate_symbol_diagnostic_rows",
    "compute_symbol_buy_hold_diagnostics",
    "compute_symbol_strategy_diagnostics",
    "load_backtest_trade_frame",
]
